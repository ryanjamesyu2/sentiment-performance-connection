"""Per-player cumulative running means of game stats (for correlation scripts)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
default_in = PROJECT_ROOT / "data" / "nfl_sentiment_2025_cleaned.csv"
default_out = PROJECT_ROOT / "data" / "nfl_running_means.csv"

in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_in
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out

stats_df = pd.read_csv(in_path, encoding="utf-8")
cols = [
    "stats_snap_counts_offense_snaps",
    "stats_passing_pass_attempts",
    "stats_passing_pass_completions",
    "stats_passing_pass_yards",
    "stats_passing_pass_td",
    "stats_passing_pass_int",
    "stats_passing_qb_rating",
    "stats_rushing_rush_attempts",
    "stats_rushing_rush_yards",
    "stats_rushing_rush_td",
    "stats_receiving_targets",
    "stats_receiving_receptions",
    "stats_receiving_rec_yards",
    "stats_receiving_rec_td",
    "stats_tackles_tackle_total",
    "stats_tackles_sacks",
    "stats_interceptions_interceptions",
    "stats_interceptions_passes_defended",
    "stats_fumbles_fum_forced",
    "total_touches",
    "total_opportunities",
    "fantasy_points_ppr",
]
group_cols = ["player_id", "player_name"]

stats_df[cols] = stats_df.groupby(group_cols)[cols].cumsum()
stats_df["games_played"] = stats_df.groupby(group_cols).cumcount() + 1
for col in cols:
    stats_df[col] = np.round(stats_df[col] / stats_df["games_played"], 3)

stats_df.drop(columns=["games_played"], inplace=True)
stats_df.to_csv(out_path, index=False, encoding="utf-8")
