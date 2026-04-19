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


# Check that the correct number of command line arguments have been provided
if len(argv) < 3:
    raise Exception(
        "Please provide at least one file path and weight as arguments."
    )
elif (len(argv) - 1) % 2 != 0:
    raise Exception(
        f"Provide an even number of arguments: {len(argv) - 1} given."
    )

player_list = pd.read_csv('sentiment_indices/player_list.csv')
weeks = pd.Series(
    ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',
        'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18']
)
teams = pd.Series(
    ['Philadelphia Eagles', 'Buffalo Bills', 'Cincinnati Bengals',
     'Indianapolis Colts', 'Kansas City Chiefs', 'Chicago Bears',
     'Tampa Bay Buccaneers', 'Seattle Seahawks', 'Dallas Cowboys',
     'New England Patriots']
)

df = pd.DataFrame({
    "player": pd.Series(np.tile(player_list['subject'], 18)),
    "team": np.tile(teams.repeat(5).reset_index(drop=True), 18),
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

    group_var = 'player' if data_source == "twitter" else 'team'

    # Merge local index and fill nas with player's average sentiment
    # from weeks for which we have data
    df = df.merge(local_df, left_on=[group_var, 'week'],
                  right_on=['subject', 'game_id'], how='left')
    df['local_index'] = df['local_index'].fillna(
        df.groupby(group_var)['local_index'].transform('mean')
    )

    # Add weighted local index to global index
    df['global_index'] += df['local_index'] * weight

    # Drop unnecessary columns for next iteration
    df.drop(columns=['subject', 'game_id', 'local_index'], inplace=True)

    ind += 2

# Normalize global index by sum of weights
df['global_index'] = df['global_index'] / wt_sum

df.to_csv('sentiment_indices/global_index_test.csv', index=False)
