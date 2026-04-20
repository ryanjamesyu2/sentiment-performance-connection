import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

global_index_df = pd.read_csv(PROJECT_ROOT / "sentiment_indices" / "global_index.csv")
global_index_df = global_index_df.rename(
    columns={
        "player": "player_name_sentiment",
        "week": "week_label",
        "global_index": "sentiment_index",
    }
)
global_index_df["game_week"] = global_index_df["week_label"].str.extract(r"W(\d+)").astype(int)
global_index_df = global_index_df[["player_name_sentiment", "game_week", "sentiment_index"]]


def build_name_map(source_names, stats_names, score_cutoff=85):
    """Fuzzy-match sentiment names to stats names when score >= cutoff."""
    mapping = {}
    low_confidence = []
    for name in source_names:
        match, score, _ = process.extractOne(name, stats_names, scorer=fuzz.token_sort_ratio)
        if score >= score_cutoff:
            mapping[name] = match
        else:
            low_confidence.append((name, match, score))

    if low_confidence:
        print(f"{len(low_confidence)} low-confidence name matches found.")
    return mapping


POSITION_CONFIG = {
    "QB": {
        "positions": ["QB"],
        "metrics": [
            "stats_passing_pass_yards",
            "stats_passing_pass_td",
            "stats_passing_pass_int",
            "stats_passing_qb_rating",
            "stats_passing_pass_attempts",
            "stats_passing_pass_completions",
            "stats_rushing_rush_yards",
            "stats_rushing_rush_td",
            "fantasy_points_ppr",
            "stats_snap_counts_offense_snaps",
        ],
    },
    "RB": {
        "positions": ["RB", "FB"],
        "metrics": [
            "stats_rushing_rush_yards",
            "stats_rushing_rush_td",
            "stats_rushing_rush_attempts",
            "stats_receiving_targets",
            "stats_receiving_receptions",
            "stats_receiving_rec_yards",
            "stats_receiving_rec_td",
            "total_touches",
            "total_opportunities",
            "fantasy_points_ppr",
            "stats_snap_counts_offense_snaps",
        ],
    },
    "WR_TE": {
        "positions": ["WR", "TE"],
        "metrics": [
            "stats_receiving_targets",
            "stats_receiving_receptions",
            "stats_receiving_rec_yards",
            "stats_receiving_rec_td",
            "total_touches",
            "total_opportunities",
            "fantasy_points_ppr",
            "stats_snap_counts_offense_snaps",
        ],
    },
    "DEF": {
        "positions": ["DE", "DT", "LB", "CB", "S", "ILB", "OLB", "MLB", "FS", "SS", "NT"],
        "metrics": [
            "stats_tackles_tackle_total",
            "stats_tackles_sacks",
            "stats_interceptions_interceptions",
            "stats_interceptions_passes_defended",
            "stats_fumbles_fum_forced",
            "stats_snap_counts_offense_snaps",
        ],
    },
}


def dcor(a, b):
    def cent_dist(v):
        d = np.abs(v[:, None] - v[None, :]).astype(float)
        return d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()

    A, B = cent_dist(a), cent_dist(b)
    denom = np.sqrt(abs((A * A).mean()) * abs((B * B).mean()))
    return np.sqrt(abs((A * B).mean()) / denom) if denom > 0 else 0.0


def correlate_series(xi, yi, min_n=15):
    mask = ~(np.isnan(xi) | np.isnan(yi))
    xi, yi = xi[mask], yi[mask]
    if len(xi) < min_n:
        return None
    r_p, p_p = stats.pearsonr(xi, yi)
    r_s, p_s = stats.spearmanr(xi, yi)
    dc = dcor(xi, yi)
    return dict(
        n=len(xi),
        pearson_r=round(r_p, 4),
        pearson_p=round(p_p, 4),
        spearman_r=round(r_s, 4),
        spearman_p=round(p_s, 4),
        distance_cor=round(dc, 4),
        pearson_sig="***" if p_p < 0.001 else ("**" if p_p < 0.01 else ("*" if p_p < 0.05 else "")),
        spearman_sig="***" if p_s < 0.001 else ("**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")),
    )


def apply_lag(df, lag, sentiment_df):
    out = df.copy()
    shifted_index = sentiment_df[["player_name", "game_week", "sentiment_index"]].assign(
        game_week=lambda x: x["game_week"] + lag
    )
    out = out.drop(columns=["sentiment_index"])
    out = out.merge(shifted_index, on=["player_name", "game_week"], how="inner")
    out["lag"] = lag
    return out


def run_analysis(stats_file_path, output_suffix):
    stats_df = pd.read_csv(Path(stats_file_path).expanduser())
    stats_names = stats_df["player_name"].unique().tolist()
    global_index_names = global_index_df["player_name_sentiment"].unique().tolist()

    sentiment_df = global_index_df.copy()
    name_map = build_name_map(global_index_names, stats_names)
    sentiment_df["player_name"] = sentiment_df["player_name_sentiment"].map(name_map)
    sentiment_df = sentiment_df.dropna(subset=["player_name"])

    player_df = stats_df.merge(
        sentiment_df[["player_name", "game_week", "sentiment_index"]],
        on=["player_name", "game_week"],
        how="inner",
    )

    concurrent_pdf = apply_lag(player_df, lag=0, sentiment_df=sentiment_df)
    sent_leads_pdf = apply_lag(player_df, lag=1, sentiment_df=sentiment_df)
    perf_leads_pdf = apply_lag(player_df, lag=-1, sentiment_df=sentiment_df)

    lag_frames = {
        "concurrent (W vs W)": concurrent_pdf,
        "sentiment index leads (sent W -> perf W+1)": sent_leads_pdf,
        "perf leads (perf W -> sent W+1)": perf_leads_pdf,
    }

    pop_results = []
    for lag_label, lag_df in lag_frames.items():
        for pos_group, cfg in POSITION_CONFIG.items():
            pos_df = lag_df[lag_df["player_position"].isin(cfg["positions"])]
            for metric in cfg["metrics"]:
                if metric not in pos_df.columns:
                    continue
                res = correlate_series(
                    pos_df["sentiment_index"].values.astype(float),
                    pos_df[metric].values.astype(float),
                    min_n=15,
                )
                if res:
                    pop_results.append(
                        {
                            "level": "population",
                            "alignment": lag_label,
                            "pos_group": pos_group,
                            "metric": metric,
                            **res,
                        }
                    )

    pop_corr_df = pd.DataFrame(pop_results).sort_values("spearman_r", key=abs, ascending=False)

    min_player_weeks = 5
    per_player_results = []
    for lag_label, lag_df in lag_frames.items():
        for pos_group, cfg in POSITION_CONFIG.items():
            pos_df = lag_df[lag_df["player_position"].isin(cfg["positions"])]
            anchor_metric = "fantasy_points_ppr"
            if anchor_metric not in pos_df.columns:
                continue
            for player, pdata in pos_df.groupby("player_name"):
                res = correlate_series(
                    pdata["sentiment_index"].values.astype(float),
                    pdata[anchor_metric].values.astype(float),
                    min_n=min_player_weeks,
                )
                if res:
                    per_player_results.append(
                        {
                            "level": "per_player",
                            "alignment": lag_label,
                            "pos_group": pos_group,
                            "player": player,
                            "team": pdata["team_abbreviation"].iloc[0],
                            "position": pdata["player_position"].iloc[0],
                            "metric": anchor_metric,
                            **res,
                        }
                    )

    per_player_df = pd.DataFrame(per_player_results).sort_values("spearman_r", key=abs, ascending=False)

    pop_output_path = (
        PROJECT_ROOT
        / "correlation_results"
        / f"global_index_player_population_correlations_{output_suffix}.csv"
    )
    per_player_output_path = (
        PROJECT_ROOT
        / "correlation_results"
        / f"global_index_player_individual_correlations_{output_suffix}.csv"
    )
    pop_corr_df.to_csv(pop_output_path, index=False)
    per_player_df.to_csv(per_player_output_path, index=False)
    print(
        f"Saved {output_suffix}: "
        f"{len(pop_corr_df)} population rows, {len(per_player_df)} per-player rows."
    )


run_analysis("~/Downloads/nfl_running_means.csv", "running_means")
run_analysis("~/Downloads/nfl_sentiment_2025_cleaned.csv", "mean_stats")
