from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# 1. TEAM NAME → ABBREVIATION MAPPING
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 2. LOAD & PREP SENTIMENT DATA
# ─────────────────────────────────────────────
sentiment_df = pd.read_csv(
    PROJECT_ROOT / "sentiment_indices" / "reddit_local_index.csv"
)

sentiment_df["team_abbreviation"] = sentiment_df["subject"].map(TEAM_MAP)
# Parse "W1" → integer 1
sentiment_df["game_week"] = sentiment_df["game_id"].str.extract(r"W(\d+)").astype(int)
sentiment_df = sentiment_df.dropna(subset=["team_abbreviation"])

sentiment_clean = sentiment_df[["team_abbreviation", "game_week", "local_index"]].copy()
sentiment_clean.columns = ["team_abbreviation", "game_week", "sentiment_index"]

# ─────────────────────────────────────────────
# 3. AGGREGATE STATS TO TEAM × WEEK LEVEL
#    Careful not to double-count:
#      • pass_yards  = QB perspective (don't also sum rec_yards → same yards)
#      • rush_yards  = separate and additive
#      • TDs: pass_td (QBs) + rush_td (all positions) covers all scoring TDs
#        rec_td would double-count pass_td, so skip it
#    Fantasy points summed across ALL players = best holistic offensive proxy
# ─────────────────────────────────────────────

stats_df = pd.read_csv(
    Path('~/Downloads/nfl_sentiment_2025_cleaned.csv').expanduser()
)

team_stats = (
    stats_df
    .groupby(["team_abbreviation", "game_week"])
    .agg(
        # Offense – volume
        pass_yards      =("stats_passing_pass_yards",           "sum"),
        pass_attempts   =("stats_passing_pass_attempts",        "sum"),
        pass_completions=("stats_passing_pass_completions",     "sum"),
        pass_td         =("stats_passing_pass_td",              "sum"),
        pass_int        =("stats_passing_pass_int",             "sum"),
        rush_yards      =("stats_rushing_rush_yards",           "sum"),
        rush_attempts   =("stats_rushing_rush_attempts",        "sum"),
        rush_td         =("stats_rushing_rush_td",              "sum"),
        # Receiving targets / receptions – useful for pass-game volume
        targets         =("stats_receiving_targets",            "sum"),
        receptions      =("stats_receiving_receptions",         "sum"),
        # Defense
        sacks           =("stats_tackles_sacks",                "sum"),
        interceptions   =("stats_interceptions_interceptions",  "sum"),
        passes_defended =("stats_interceptions_passes_defended","sum"),
        tackles         =("stats_tackles_tackle_total",         "sum"),
        # Holistic
        total_fantasy_pts=("fantasy_points_ppr",                "sum"),
        total_touches    =("total_touches",                     "sum"),
        total_opps       =("total_opportunities",               "sum"),
        # Opponent strength (use mean — same for all players on same team that week)
        opp_offense_dvoa =("opp_offense_dvoa",                  "mean"),
        opp_defense_dvoa =("opp_defense_dvoa",                  "mean"),
        snap_count       =("stats_snap_counts_offense_snaps",   "sum"),
    )
    .reset_index()
)

# Derived efficiency metrics
team_stats["completion_pct"]   = team_stats["pass_completions"] / team_stats["pass_attempts"].replace(0, np.nan)
team_stats["yards_per_rush"]   = team_stats["rush_yards"]       / team_stats["rush_attempts"].replace(0, np.nan)
team_stats["total_yards"]      = team_stats["pass_yards"] + team_stats["rush_yards"]
team_stats["total_td"]         = team_stats["pass_td"] + team_stats["rush_td"]
team_stats["turnover_sum"]     = team_stats["pass_int"]   # add fumbles if available
team_stats["defensive_stops"]  = team_stats["sacks"] + team_stats["interceptions"]

# ─────────────────────────────────────────────
# 4. MERGE — THREE TEMPORAL ALIGNMENTS
# ─────────────────────────────────────────────
def merge_with_lag(stats, sentiment, lag: int) -> pd.DataFrame:
    """
    lag = 0  → same week (concurrent)
    lag = 1  → sentiment(W) predicts performance(W+1)   [sentiment leads]
    lag = -1 → performance(W) predicts sentiment(W+1)   [performance leads]
    """
    s = sentiment.copy()
    s["game_week_shifted"] = s["game_week"] + lag  # shift the sentiment week
    merged = stats.merge(
        s.rename(columns={"game_week": "sentiment_week",
                          "game_week_shifted": "game_week"}),
        on=["team_abbreviation", "game_week"],
        how="inner",
    )
    merged["lag"] = lag
    return merged

concurrent_df     = merge_with_lag(team_stats, sentiment_clean, lag=0)
sentiment_leads_df = merge_with_lag(team_stats, sentiment_clean, lag=1)   # sent(W) vs perf(W+1)
perf_leads_df      = merge_with_lag(team_stats, sentiment_clean, lag=-1)  # perf(W) vs sent(W+1)

# ─────────────────────────────────────────────
# 5. CORRELATION FUNCTIONS
# ─────────────────────────────────────────────
PERF_COLS = [
    # Offense
    "total_fantasy_pts", "total_yards", "total_td",
    "pass_yards", "pass_td", "pass_int",
    "rush_yards", "rush_td",
    "completion_pct", "yards_per_rush",
    "targets", "receptions",
    # Defense
    "sacks", "interceptions", "passes_defended", "tackles", "defensive_stops",
    # Volume/efficiency context
    "total_touches", "total_opps",
    "opp_offense_dvoa", "opp_defense_dvoa",
]

def compute_correlations(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute Pearson, Spearman, and distance correlation vs sentiment_index."""
    results = []
    x = df["sentiment_index"].values

    for col in PERF_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["sentiment_index"].notna()
        xi, yi = x[mask], df.loc[mask, col].values
        n = mask.sum()
        if n < 10:
            continue

        # Pearson
        r_p, p_p   = stats.pearsonr(xi, yi)
        # Spearman
        r_s, p_s   = stats.spearmanr(xi, yi)
        # Distance correlation (dcor) — detects non-linear associations
        # Computed manually to avoid extra dependency
        def dcor(a, b):
            """Unbiased distance correlation."""
            def cent_dist(v):
                d = np.abs(v[:, None] - v[None, :]).astype(float)
                row_mean = d.mean(axis=1, keepdims=True)
                col_mean = d.mean(axis=0, keepdims=True)
                grand    = d.mean()
                return d - row_mean - col_mean + grand
            A, B = cent_dist(a), cent_dist(b)
            dcov2_ab = (A * B).mean()
            dcov2_aa = (A * A).mean()
            dcov2_bb = (B * B).mean()
            denom    = np.sqrt(abs(dcov2_aa) * abs(dcov2_bb))
            return np.sqrt(abs(dcov2_ab) / denom) if denom > 0 else 0.0

        dc = dcor(xi, yi)

        results.append({
            "alignment":    label,
            "metric":       col,
            "n":            n,
            "pearson_r":    round(r_p,  4),
            "pearson_p":    round(p_p,  4),
            "spearman_r":   round(r_s,  4),
            "spearman_p":   round(p_s,  4),
            "distance_cor": round(dc,   4),
            "pearson_sig":  "***" if p_p<0.001 else ("**" if p_p<0.01 else ("*" if p_p<0.05 else "")),
            "spearman_sig": "***" if p_s<0.001 else ("**" if p_s<0.01 else ("*" if p_s<0.05 else "")),
        })

    return pd.DataFrame(results).sort_values("spearman_r", key=abs, ascending=False)

# ─────────────────────────────────────────────
# 6. RUN & DISPLAY
# ─────────────────────────────────────────────
corr_concurrent      = compute_correlations(concurrent_df,      "concurrent (W vs W)")
corr_sent_leads      = compute_correlations(sentiment_leads_df,  "sentiment leads (sent W → perf W+1)")
corr_perf_leads      = compute_correlations(perf_leads_df,       "perf leads (perf W → sent W+1)")

all_corrs = pd.concat([corr_concurrent, corr_sent_leads, corr_perf_leads], ignore_index=True)

# Pretty summary — top movers per alignment
print("=" * 70)
for label, grp in all_corrs.groupby("alignment"):
    print(f"\n📊  {label}")
    print("-" * 70)
    display_cols = ["metric", "n", "pearson_r", "pearson_sig",
                    "spearman_r", "spearman_sig", "distance_cor"]
    print(grp[display_cols].head(10).to_string(index=False))

# Save full results
all_corrs.to_csv(
    PROJECT_ROOT
    / "correlation_results"
    / "reddit_team_sentiment_correlations_mean_stats.csv",
    index=False,
)
print("\nFull results saved to reddit_team_sentiment_correlations_mean_stats.csv")



stats_df = pd.read_csv(Path('~/Downloads/nfl_running_means.csv').expanduser())

team_stats = (
    stats_df
    .groupby(["team_abbreviation", "game_week"])
    .agg(
        # Offense – volume
        pass_yards      =("stats_passing_pass_yards",           "sum"),
        pass_attempts   =("stats_passing_pass_attempts",        "sum"),
        pass_completions=("stats_passing_pass_completions",     "sum"),
        pass_td         =("stats_passing_pass_td",              "sum"),
        pass_int        =("stats_passing_pass_int",             "sum"),
        rush_yards      =("stats_rushing_rush_yards",           "sum"),
        rush_attempts   =("stats_rushing_rush_attempts",        "sum"),
        rush_td         =("stats_rushing_rush_td",              "sum"),
        # Receiving targets / receptions – useful for pass-game volume
        targets         =("stats_receiving_targets",            "sum"),
        receptions      =("stats_receiving_receptions",         "sum"),
        # Defense
        sacks           =("stats_tackles_sacks",                "sum"),
        interceptions   =("stats_interceptions_interceptions",  "sum"),
        passes_defended =("stats_interceptions_passes_defended","sum"),
        tackles         =("stats_tackles_tackle_total",         "sum"),
        # Holistic
        total_fantasy_pts=("fantasy_points_ppr",                "sum"),
        total_touches    =("total_touches",                     "sum"),
        total_opps       =("total_opportunities",               "sum"),
        # Opponent strength (use mean — same for all players on same team that week)
        opp_offense_dvoa =("opp_offense_dvoa",                  "mean"),
        opp_defense_dvoa =("opp_defense_dvoa",                  "mean"),
        snap_count       =("stats_snap_counts_offense_snaps",   "sum"),
    )
    .reset_index()
)

# Derived efficiency metrics
team_stats["completion_pct"]   = team_stats["pass_completions"] / team_stats["pass_attempts"].replace(0, np.nan)
team_stats["yards_per_rush"]   = team_stats["rush_yards"]       / team_stats["rush_attempts"].replace(0, np.nan)
team_stats["total_yards"]      = team_stats["pass_yards"] + team_stats["rush_yards"]
team_stats["total_td"]         = team_stats["pass_td"] + team_stats["rush_td"]
team_stats["turnover_sum"]     = team_stats["pass_int"]   # add fumbles if available
team_stats["defensive_stops"]  = team_stats["sacks"] + team_stats["interceptions"]

# ─────────────────────────────────────────────
# 4. MERGE — THREE TEMPORAL ALIGNMENTS
# ─────────────────────────────────────────────
def merge_with_lag(stats, sentiment, lag: int) -> pd.DataFrame:
    """
    lag = 0  → same week (concurrent)
    lag = 1  → sentiment(W) predicts performance(W+1)   [sentiment leads]
    lag = -1 → performance(W) predicts sentiment(W+1)   [performance leads]
    """
    s = sentiment.copy()
    s["game_week_shifted"] = s["game_week"] + lag  # shift the sentiment week
    merged = stats.merge(
        s.rename(columns={"game_week": "sentiment_week",
                          "game_week_shifted": "game_week"}),
        on=["team_abbreviation", "game_week"],
        how="inner",
    )
    merged["lag"] = lag
    return merged

concurrent_df     = merge_with_lag(team_stats, sentiment_clean, lag=0)
sentiment_leads_df = merge_with_lag(team_stats, sentiment_clean, lag=1)   # sent(W) vs perf(W+1)
perf_leads_df      = merge_with_lag(team_stats, sentiment_clean, lag=-1)  # perf(W) vs sent(W+1)

# ─────────────────────────────────────────────
# 5. CORRELATION FUNCTIONS
# ─────────────────────────────────────────────
PERF_COLS = [
    # Offense
    "total_fantasy_pts", "total_yards", "total_td",
    "pass_yards", "pass_td", "pass_int",
    "rush_yards", "rush_td",
    "completion_pct", "yards_per_rush",
    "targets", "receptions",
    # Defense
    "sacks", "interceptions", "passes_defended", "tackles", "defensive_stops",
    # Volume/efficiency context
    "total_touches", "total_opps",
    "opp_offense_dvoa", "opp_defense_dvoa",
]

def compute_correlations(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute Pearson, Spearman, and distance correlation vs sentiment_index."""
    results = []
    x = df["sentiment_index"].values

    for col in PERF_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["sentiment_index"].notna()
        xi, yi = x[mask], df.loc[mask, col].values
        n = mask.sum()
        if n < 10:
            continue

        # Pearson
        r_p, p_p   = stats.pearsonr(xi, yi)
        # Spearman
        r_s, p_s   = stats.spearmanr(xi, yi)
        # Distance correlation (dcor) — detects non-linear associations
        # Computed manually to avoid extra dependency
        def dcor(a, b):
            """Unbiased distance correlation."""
            def cent_dist(v):
                d = np.abs(v[:, None] - v[None, :]).astype(float)
                row_mean = d.mean(axis=1, keepdims=True)
                col_mean = d.mean(axis=0, keepdims=True)
                grand    = d.mean()
                return d - row_mean - col_mean + grand
            A, B = cent_dist(a), cent_dist(b)
            dcov2_ab = (A * B).mean()
            dcov2_aa = (A * A).mean()
            dcov2_bb = (B * B).mean()
            denom    = np.sqrt(abs(dcov2_aa) * abs(dcov2_bb))
            return np.sqrt(abs(dcov2_ab) / denom) if denom > 0 else 0.0

        dc = dcor(xi, yi)

        results.append({
            "alignment":    label,
            "metric":       col,
            "n":            n,
            "pearson_r":    round(r_p,  4),
            "pearson_p":    round(p_p,  4),
            "spearman_r":   round(r_s,  4),
            "spearman_p":   round(p_s,  4),
            "distance_cor": round(dc,   4),
            "pearson_sig":  "***" if p_p<0.001 else ("**" if p_p<0.01 else ("*" if p_p<0.05 else "")),
            "spearman_sig": "***" if p_s<0.001 else ("**" if p_s<0.01 else ("*" if p_s<0.05 else "")),
        })

    return pd.DataFrame(results).sort_values("spearman_r", key=abs, ascending=False)

# ─────────────────────────────────────────────
# 6. RUN & DISPLAY
# ─────────────────────────────────────────────
corr_concurrent      = compute_correlations(concurrent_df,      "concurrent (W vs W)")
corr_sent_leads      = compute_correlations(sentiment_leads_df,  "sentiment leads (sent W → perf W+1)")
corr_perf_leads      = compute_correlations(perf_leads_df,       "perf leads (perf W → sent W+1)")

all_corrs = pd.concat([corr_concurrent, corr_sent_leads, corr_perf_leads], ignore_index=True)

# Pretty summary — top movers per alignment
print("=" * 70)
for label, grp in all_corrs.groupby("alignment"):
    print(f"\n📊  {label}")
    print("-" * 70)
    display_cols = ["metric", "n", "pearson_r", "pearson_sig",
                    "spearman_r", "spearman_sig", "distance_cor"]
    print(grp[display_cols].head(10).to_string(index=False))

# Save full results
all_corrs.to_csv(
    PROJECT_ROOT
    / "correlation_results"
    / "reddit_team_sentiment_correlations_running_means.csv",
    index=False,
)
print("\nFull results saved to reddit_team_sentiment_correlations_running_means.csv")
