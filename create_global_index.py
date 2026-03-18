"""
A function to create a global, player-level sentiment index for each week
by aggregating local indices across data sources.

The file takes in a list of file paths for the files containing the local
indices, as well as a list of relative weights to use for each data source.

The output is a csv file containing the global index for each player and week
"""

from sys import argv
import pandas as pd
import numpy as np


player_list = pd.read_csv('sentiment_indices/player_list.csv')
weeks = pd.Series(
    ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',
        'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18']
)
teams = pd.Series(
    ['Philadelpia Eagles', 'Buffalo Bills', 'Cincinnati Bengals',
     'Indianapolis Colts', 'Kansas City Chiefs', 'Chicago Bears',
     'Tampa Bay Buccaneers', 'Seattle Seahawks', 'Dallas Cowboys',
     'New England Patriots']
)

df = pd.DataFrame({
    "player": pd.Series(np.tile(player_list['subject'], 18)),
    "team": teams.repeat(5).reset_index(drop=True),
    "week": weeks.repeat(len(player_list)).reset_index(drop=True),
    "global_index": pd.Series(np.zeros(len(player_list) * 18))
})

ind = 1
wt_sum = 0
while ind < len(argv) - 1:
    file_path = argv[ind]
    weight = float(argv[ind + 1])
    wt_sum += weight
    local_df = pd.read_csv(file_path)

    file_name = file_path.split("/")[-1]    # get just file name, not path
    data_source = file_name.split("_")[0].lower()

    if data_source == "twitter":
        df['global_index'] += local_df['local_index'] * weight
    elif data_source == "google" or data_source == "reddit":
        df = df.merge(local_df, left_on=['team', 'week'],
                      right_on=['subject', 'game_id'], how='left')
        df['global_index'] += df['local_index'] * weight
        df = df.drop(['subject', 'game_id', 'local_index'], axis=1)

    ind += 2

df.to_csv('sentiment_indices/global_index.csv', index=False)
