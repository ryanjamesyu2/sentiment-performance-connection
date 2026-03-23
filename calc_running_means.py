"""
A script to calculate NFL player's running means for each statistic. This
is a running mean per player per statistic, so that in correlation analysis,
we can compare each player's performance to their own past performance over
the course of this season.

The script takes two optional command line arguments:
- The path to the input CSV file containing the player statistics. If not
  provided, it defaults to "data/nfl_sentiment_2025_cleaned.csv".
- The path to the output CSV file where the running means will be saved. If
  not provided, it defaults to "data/running_means.csv".
"""

import pandas as pd
from sys import argv

in_path = argv[1] if len(argv) > 1 else "data/nfl_sentiment_2025_cleaned.csv"
out_path = argv[2] if len(argv) > 2 else "data/nfl_running_means.csv"

stats_df = pd.read_csv(in_path)
cols = [
    'stats_snap_counts_offense_snaps',
    'stats_passing_pass_attempts',
    'stats_passing_pass_completions',
    'stats_passing_pass_yards',
    'stats_passing_pass_td',
    'stats_passing_pass_int',
    'stats_passing_qb_rating',
    'stats_rushing_rush_attempts',
    'stats_rushing_rush_yards',
    'stats_rushing_rush_td',
    'stats_receiving_targets',
    'stats_receiving_receptions',
    'stats_receiving_rec_yards',
    'stats_receiving_rec_td',
    'stats_tackles_tackle_total',
    'stats_tackles_sacks',
    'stats_interceptions_interceptions',
    'stats_interceptions_passes_defended',
    'stats_fumbles_fum_forced',
    'total_touches',
    'total_opportunities',
    'fantasy_points_ppr'
]
group_cols = ['player_id', 'player_name']

stats_df[cols] = stats_df.groupby(group_cols)[cols].cumsum()
stats_df.to_csv(out_path, index=False)
