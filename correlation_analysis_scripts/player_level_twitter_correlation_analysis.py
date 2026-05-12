from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
        "positions": [
            "DE",
            "DT",
            "LB",
            "CB",
            "S",
            "ILB",
            "OLB",
            "MLB",
            "FS",
            "SS",
            "NT",
        ],
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

    a_c, b_c = cent_dist(a), cent_dist(b)
    denom = np.sqrt(abs((a_c * a_c).mean()) * abs((b_c * b_c).mean()))
    return np.sqrt(abs((a_c * b_c).mean()) / denom) if denom > 0 else 0.0


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
        pearson_sig="***"
        if p_p < 0.001
        else ("**" if p_p < 0.01 else ("*" if p_p < 0.05 else "")),
        spearman_sig="***"
        if p_s < 0.001
        else ("**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")),
    )


def build_name_map(twitter_names, stats_names, score_cutoff=85):
    mapping = {}
    low_confidence = []
    for name in twitter_names:
        match, score, _ = process.extractOne(
            name, stats_names, scorer=fuzz.token_sort_ratio
        )
        if score >= score_cutoff:
            mapping[name] = match
        else:
            low_confidence.append((name, match, score))

    if low_confidence:
        print(f"\n{len(low_confidence)} low-confidence matches (review):")
        for t, s, sc in sorted(low_confidence, key=lambda x: x[2]):
            print(f"   '{t}' -> '{s}'  (score={sc})")
    print(
        f"\n{len(mapping)} / {len(twitter_names)} names matched at score >= {score_cutoff}"
    )
    return mapping


def apply_lag(df, lag, sentiment_df):
    out = df.copy()
    sent_shifted = sentiment_df[
        ["player_name", "game_week", "sentiment_index"]
    ].assign(game_week=lambda x: x["game_week"] + lag)
    out = out.drop(columns=["sentiment_index"])
    out = out.merge(sent_shifted, on=["player_name", "game_week"], how="inner")
    out["lag"] = lag
    return out


def run(stats_csv: Path, pop_filename: str, per_player_filename: str):
    stats_df = pd.read_csv(stats_csv.expanduser(), encoding="utf-8")
    twitter_df = pd.read_csv(
        PROJECT_ROOT / "sentiment_indices" / "twitter_local_index_sprint2.csv",
        encoding="utf-8",
    )
    twitter_df["game_week"] = twitter_df["game_id"].str.extract(r"W(\d+)").astype(int)
    twitter_df = twitter_df.rename(
        columns={"subject": "player_name_sentiment", "local_index": "sentiment_index"}
    )
    twitter_df = twitter_df[
        ["player_name_sentiment", "game_week", "sentiment_index"]
    ]

    stats_names = stats_df["player_name"].unique().tolist()
    twitter_names = twitter_df["player_name_sentiment"].unique().tolist()
    name_map = build_name_map(twitter_names, stats_names)

    twitter_df["player_name"] = twitter_df["player_name_sentiment"].map(name_map)
    twitter_df = twitter_df.dropna(subset=["player_name"])
    sentiment_for_lag = twitter_df[
        ["player_name", "game_week", "sentiment_index"]
    ].copy()

    player_df = stats_df.merge(
        sentiment_for_lag,
        on=["player_name", "game_week"],
        how="inner",
    )

    print(
        f"\nMerged dataset: {len(player_df)} player-week rows "
        f"| {player_df['player_name'].nunique()} players "
        f"| {player_df['game_week'].nunique()} weeks"
    )

    lag_frames = {
        "concurrent (W vs W)": apply_lag(player_df, 0, sentiment_for_lag),
        "sentiment leads (sent W -> perf W+1)": apply_lag(player_df, 1, sentiment_for_lag),
        "perf leads (perf W -> sent W+1)": apply_lag(player_df, -1, sentiment_for_lag),
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

    pop_corr_df = pd.DataFrame(pop_results).sort_values(
        "spearman_r", key=abs, ascending=False
    )

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

    per_player_df = pd.DataFrame(per_player_results).sort_values(
        "spearman_r", key=abs, ascending=False
    )

    disp = [
        "metric",
        "n",
        "pearson_r",
        "pearson_sig",
        "spearman_r",
        "spearman_sig",
        "distance_cor",
    ]

    print("\n" + "=" * 70)
    print("POPULATION-LEVEL: Twitter Sentiment vs Player Stats")
    print("=" * 70)
    for (align, pos), grp in pop_corr_df.groupby(["alignment", "pos_group"]):
        print(f"\n  {align}  |  {pos}")
        print("-" * 70)
        print(grp[disp].head(8).to_string(index=False))

    print("\n\n" + "=" * 70)
    print("PER-PLAYER: Twitter Sentiment vs Fantasy Points PPR (top 15 per alignment)")
    print("=" * 70)
    disp_p = [
        "player",
        "team",
        "position",
        "n",
        "pearson_r",
        "pearson_sig",
        "spearman_r",
        "spearman_sig",
        "distance_cor",
    ]
    for align, grp in per_player_df.groupby("alignment"):
        print(f"\n  {align}")
        print("-" * 70)
        print(grp[disp_p].head(15).to_string(index=False))

    out_dir = PROJECT_ROOT / "correlation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_corr_df.to_csv(out_dir / pop_filename, index=False, encoding="utf-8")
    per_player_df.to_csv(out_dir / per_player_filename, index=False, encoding="utf-8")
    print("\nSaved population + per-player correlation CSVs")


run(
    Path("~/Downloads/nfl_running_means.csv"),
    "twitter_player_population_correlations_running_means.csv",
    "twitter_player_individual_correlations_running_means.csv",
)
run(
    Path("~/Downloads/nfl_sentiment_2025_cleaned.csv"),
    "twitter_player_population_correlations_mean_stats.csv",
    "twitter_player_individual_correlations_mean_stats.csv",
)
