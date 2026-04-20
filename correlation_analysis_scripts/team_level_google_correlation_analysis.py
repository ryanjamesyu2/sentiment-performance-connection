from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

google_df = pd.read_csv(
    PROJECT_ROOT / "sentiment_indices" / "google_local_index_sprint2.csv"
)

google_df["team_abbreviation"] = google_df["subject"].map(TEAM_MAP)
google_df["game_week"]         = google_df["game_id"].str.extract(r"W(\d+)").astype(int)
google_df = google_df.dropna(subset=["team_abbreviation"])

google_sentiment = google_df[["team_abbreviation", "game_week", "local_index"]].copy()
google_sentiment.columns = ["team_abbreviation", "game_week", "sentiment_index"]

coverage = (
    google_sentiment
    .groupby("team_abbreviation")["game_week"]
    .agg(weeks_covered="count", first_week="min", last_week="max")
    .sort_values("weeks_covered", ascending=False)
)
print("📡  Google Sentiment Coverage (teams × weeks):")
print(coverage.to_string())
print(f"\n   Total team-week observations: {len(google_sentiment)}")
print(f"   Teams covered: {google_sentiment['team_abbreviation'].nunique()}\n")

stats_df = pd.read_csv(
    Path('~/Downloads/nfl_sentiment_2025_cleaned.csv').expanduser()
)

team_stats = (
    stats_df
    .groupby(["team_abbreviation", "game_week"])
    .agg(
        pass_yards       =("stats_passing_pass_yards",           "sum"),
        pass_attempts    =("stats_passing_pass_attempts",        "sum"),
        pass_completions =("stats_passing_pass_completions",     "sum"),
        pass_td          =("stats_passing_pass_td",              "sum"),
        pass_int         =("stats_passing_pass_int",             "sum"),
        rush_yards       =("stats_rushing_rush_yards",           "sum"),
        rush_attempts    =("stats_rushing_rush_attempts",        "sum"),
        rush_td          =("stats_rushing_rush_td",              "sum"),
        targets          =("stats_receiving_targets",            "sum"),
        receptions       =("stats_receiving_receptions",         "sum"),
        sacks            =("stats_tackles_sacks",                "sum"),
        interceptions    =("stats_interceptions_interceptions",  "sum"),
        passes_defended  =("stats_interceptions_passes_defended","sum"),
        tackles          =("stats_tackles_tackle_total",         "sum"),
        total_fantasy_pts=("fantasy_points_ppr",                "sum"),
        total_touches    =("total_touches",                     "sum"),
        total_opps       =("total_opportunities",               "sum"),
        opp_offense_dvoa =("opp_offense_dvoa",                  "mean"),
        opp_defense_dvoa =("opp_defense_dvoa",                  "mean"),
        snap_count       =("stats_snap_counts_offense_snaps",   "sum"),
    )
    .reset_index()
)

team_stats["completion_pct"]  = team_stats["pass_completions"] / team_stats["pass_attempts"].replace(0, np.nan)
team_stats["yards_per_rush"]  = team_stats["rush_yards"]       / team_stats["rush_attempts"].replace(0, np.nan)
team_stats["total_yards"]     = team_stats["pass_yards"] + team_stats["rush_yards"]
team_stats["total_td"]        = team_stats["pass_td"] + team_stats["rush_td"]
team_stats["turnover_sum"]    = team_stats["pass_int"]
team_stats["defensive_stops"] = team_stats["sacks"] + team_stats["interceptions"]

def merge_with_lag(stats, sentiment, lag: int) -> pd.DataFrame:
    s = sentiment.copy()
    s["game_week"] = s["game_week"] + lag
    return stats.merge(s, on=["team_abbreviation", "game_week"], how="inner").assign(lag=lag)

google_concurrent    = merge_with_lag(team_stats, google_sentiment, lag=0)
google_sent_leads    = merge_with_lag(team_stats, google_sentiment, lag=1)
google_perf_leads    = merge_with_lag(team_stats, google_sentiment, lag=-1)

for label, df in [("concurrent", google_concurrent),
                  ("sentiment leads (+1)", google_sent_leads),
                  ("perf leads (-1)", google_perf_leads)]:
    print(f"   {label}: {len(df)} team-week pairs | {df['team_abbreviation'].nunique()} teams")

PERF_COLS = [
    "total_fantasy_pts", "total_yards", "total_td",
    "pass_yards", "pass_td", "pass_int",
    "rush_yards", "rush_td",
    "completion_pct", "yards_per_rush",
    "targets", "receptions",
    "sacks", "interceptions", "passes_defended", "tackles", "defensive_stops",
    "total_touches", "total_opps",
    "opp_offense_dvoa", "opp_defense_dvoa",
]

def dcor(a, b):
    def cent_dist(v):
        d = np.abs(v[:, None] - v[None, :]).astype(float)
        return d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
    A, B = cent_dist(a), cent_dist(b)
    denom = np.sqrt(abs((A*A).mean()) * abs((B*B).mean()))
    return np.sqrt(abs((A*B).mean()) / denom) if denom > 0 else 0.0

def compute_correlations(df: pd.DataFrame, label: str, min_n: int = 10) -> pd.DataFrame:
    results = []
    x = df["sentiment_index"].values
    for col in PERF_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["sentiment_index"].notna()
        xi, yi = x[mask], df.loc[mask, col].values
        n = mask.sum()
        if n < min_n:
            continue
        r_p, p_p = stats.pearsonr(xi, yi)
        r_s, p_s = stats.spearmanr(xi, yi)
        dc       = dcor(xi, yi)
        results.append({
            "source":       "Google",
            "alignment":    label,
            "metric":       col,
            "n":            n,
            "pearson_r":    round(r_p, 4),
            "pearson_p":    round(p_p, 4),
            "spearman_r":   round(r_s, 4),
            "spearman_p":   round(p_s, 4),
            "distance_cor": round(dc,  4),
            "pearson_sig":  "***" if p_p<0.001 else ("**" if p_p<0.01 else ("*" if p_p<0.05 else "")),
            "spearman_sig": "***" if p_s<0.001 else ("**" if p_s<0.01 else ("*" if p_s<0.05 else "")),
        })
    return pd.DataFrame(results).sort_values("spearman_r", key=abs, ascending=False)

google_corr_concurrent  = compute_correlations(google_concurrent,  "concurrent (W vs W)")
google_corr_sent_leads  = compute_correlations(google_sent_leads,  "sentiment leads (sent W → perf W+1)")
google_corr_perf_leads  = compute_correlations(google_perf_leads,  "perf leads (perf W → sent W+1)")

google_all_corrs = pd.concat(
    [google_corr_concurrent, google_corr_sent_leads, google_corr_perf_leads],
    ignore_index=True
)

print("\n" + "=" * 70)
for label, grp in google_all_corrs.groupby("alignment"):
    print(f"\n📊  {label}")
    print("-" * 70)
    print(grp[["metric", "n", "pearson_r", "pearson_sig",
               "spearman_r", "spearman_sig", "distance_cor"]].head(10).to_string(index=False))

google_all_corrs.to_csv(
    PROJECT_ROOT
    / "correlation_results"
    / "google_team_sentiment_correlations_mean_stats.csv",
    index=False,
)
print("\nSaved to google_team_sentiment_correlations_mean_stats.csv")



stats_df = pd.read_csv(Path('~/Downloads/nfl_running_means.csv').expanduser())

team_stats = (
    stats_df
    .groupby(["team_abbreviation", "game_week"])
    .agg(
        pass_yards       =("stats_passing_pass_yards",           "sum"),
        pass_attempts    =("stats_passing_pass_attempts",        "sum"),
        pass_completions =("stats_passing_pass_completions",     "sum"),
        pass_td          =("stats_passing_pass_td",              "sum"),
        pass_int         =("stats_passing_pass_int",             "sum"),
        rush_yards       =("stats_rushing_rush_yards",           "sum"),
        rush_attempts    =("stats_rushing_rush_attempts",        "sum"),
        rush_td          =("stats_rushing_rush_td",              "sum"),
        targets          =("stats_receiving_targets",            "sum"),
        receptions       =("stats_receiving_receptions",         "sum"),
        sacks            =("stats_tackles_sacks",                "sum"),
        interceptions    =("stats_interceptions_interceptions",  "sum"),
        passes_defended  =("stats_interceptions_passes_defended","sum"),
        tackles          =("stats_tackles_tackle_total",         "sum"),
        total_fantasy_pts=("fantasy_points_ppr",                "sum"),
        total_touches    =("total_touches",                     "sum"),
        total_opps       =("total_opportunities",               "sum"),
        opp_offense_dvoa =("opp_offense_dvoa",                  "mean"),
        opp_defense_dvoa =("opp_defense_dvoa",                  "mean"),
        snap_count       =("stats_snap_counts_offense_snaps",   "sum"),
    )
    .reset_index()
)

team_stats["completion_pct"]  = team_stats["pass_completions"] / team_stats["pass_attempts"].replace(0, np.nan)
team_stats["yards_per_rush"]  = team_stats["rush_yards"]       / team_stats["rush_attempts"].replace(0, np.nan)
team_stats["total_yards"]     = team_stats["pass_yards"] + team_stats["rush_yards"]
team_stats["total_td"]        = team_stats["pass_td"] + team_stats["rush_td"]
team_stats["turnover_sum"]    = team_stats["pass_int"]
team_stats["defensive_stops"] = team_stats["sacks"] + team_stats["interceptions"]

def merge_with_lag(stats, sentiment, lag: int) -> pd.DataFrame:
    s = sentiment.copy()
    s["game_week"] = s["game_week"] + lag
    return stats.merge(s, on=["team_abbreviation", "game_week"], how="inner").assign(lag=lag)

google_concurrent    = merge_with_lag(team_stats, google_sentiment, lag=0)
google_sent_leads    = merge_with_lag(team_stats, google_sentiment, lag=1)
google_perf_leads    = merge_with_lag(team_stats, google_sentiment, lag=-1)

for label, df in [("concurrent", google_concurrent),
                  ("sentiment leads (+1)", google_sent_leads),
                  ("perf leads (-1)", google_perf_leads)]:
    print(f"   {label}: {len(df)} team-week pairs | {df['team_abbreviation'].nunique()} teams")

PERF_COLS = [
    "total_fantasy_pts", "total_yards", "total_td",
    "pass_yards", "pass_td", "pass_int",
    "rush_yards", "rush_td",
    "completion_pct", "yards_per_rush",
    "targets", "receptions",
    "sacks", "interceptions", "passes_defended", "tackles", "defensive_stops",
    "total_touches", "total_opps",
    "opp_offense_dvoa", "opp_defense_dvoa",
]

def dcor(a, b):
    def cent_dist(v):
        d = np.abs(v[:, None] - v[None, :]).astype(float)
        return d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
    A, B = cent_dist(a), cent_dist(b)
    denom = np.sqrt(abs((A*A).mean()) * abs((B*B).mean()))
    return np.sqrt(abs((A*B).mean()) / denom) if denom > 0 else 0.0

def compute_correlations(df: pd.DataFrame, label: str, min_n: int = 10) -> pd.DataFrame:
    results = []
    x = df["sentiment_index"].values
    for col in PERF_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["sentiment_index"].notna()
        xi, yi = x[mask], df.loc[mask, col].values
        n = mask.sum()
        if n < min_n:
            continue
        r_p, p_p = stats.pearsonr(xi, yi)
        r_s, p_s = stats.spearmanr(xi, yi)
        dc       = dcor(xi, yi)
        results.append({
            "source":       "Google",
            "alignment":    label,
            "metric":       col,
            "n":            n,
            "pearson_r":    round(r_p, 4),
            "pearson_p":    round(p_p, 4),
            "spearman_r":   round(r_s, 4),
            "spearman_p":   round(p_s, 4),
            "distance_cor": round(dc,  4),
            "pearson_sig":  "***" if p_p<0.001 else ("**" if p_p<0.01 else ("*" if p_p<0.05 else "")),
            "spearman_sig": "***" if p_s<0.001 else ("**" if p_s<0.01 else ("*" if p_s<0.05 else "")),
        })
    return pd.DataFrame(results).sort_values("spearman_r", key=abs, ascending=False)

google_corr_concurrent  = compute_correlations(google_concurrent,  "concurrent (W vs W)")
google_corr_sent_leads  = compute_correlations(google_sent_leads,  "sentiment leads (sent W → perf W+1)")
google_corr_perf_leads  = compute_correlations(google_perf_leads,  "perf leads (perf W → sent W+1)")

google_all_corrs = pd.concat(
    [google_corr_concurrent, google_corr_sent_leads, google_corr_perf_leads],
    ignore_index=True
)

print("\n" + "=" * 70)
for label, grp in google_all_corrs.groupby("alignment"):
    print(f"\n📊  {label}")
    print("-" * 70)
    print(grp[["metric", "n", "pearson_r", "pearson_sig",
               "spearman_r", "spearman_sig", "distance_cor"]].head(10).to_string(index=False))

google_all_corrs.to_csv(
    PROJECT_ROOT
    / "correlation_results"
    / "google_team_sentiment_correlations_running_means.csv",
    index=False,
)
print("\nSaved to google_team_sentiment_correlations_running_means.csv")