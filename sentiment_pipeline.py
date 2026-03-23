"""
A sentiment analysis piepline that uses pre-trained HuggingFace models
to predict the sentiment of given text from Twitter, Reddit, and Google.
The model used will change depending on which source the text is from.

Preprocessed data, stored in a local .csv file, is provided to the pipeline
via command line arguments. The pipeline will determine the source of the text
from the file name and use the appropriate model to predict the sentiment. We
utilize the predicted classes, as well as the probabilities of each class, to
calculate an expected-value based sentiment score for each text entry. The
results are then saved to a new .csv file.
"""

# Import necessary libraries
from sys import argv
# from pipeline_utils import predict_twitter
# from pipeline_utils import predict_reddit
# from pipeline_utils import predict_google
from pipeline_utils import predict_sentiment
from pipeline_utils import aggregate_local_index
from pipeline_utils import scale_local_index
from pipeline_utils import map_reddit_thread_to_week
from pipeline_utils import extract_sentences
from pipeline_utils import calc_google_weights
from pipeline_utils import get_game_start_times
import pandas as pd
import numpy as np

# Extract file name from command line arguments and determine data source
file_path = argv[1]
file_name = file_path.split("/")[-1]    # get just file name, not path
data_source = file_name.split("_")[0].lower()

# Read data into data frame
df = pd.read_csv(file_path)

# Load the appropriate model path and function based on the data source
# Also generate weights based on a metadata column
if data_source == "google":
    # Rename column so column names are consistent
    df = df.rename(columns={'team': 'subject', 'game_no': 'game_id'})

    bye_weeks = {
        'Philadelphia Eagles': 9,
        'Seattle Seahawks': 8,
        'Chicago Bears': 5,
        'Tampa Bay Buccaneers': 9,
        'Buffalo Bills': 7,
        'Cincinnati Bengals': 10,
        'Kansas City Chiefs': 10,
        'Indianapolis Colts': 11,
        'New England Patriots': 14,
        'Dallas Cowboys': 10
    }

    # Adjust game_id to account for bye weeks
    df['after_bye'] = df.apply(
        lambda x: x['game_id'] >= bye_weeks[x['subject']], axis=1
    )
    df['game_id'] = df.apply(
        lambda x: x['game_id'] + 1 if x['after_bye'] else x['game_id'], axis=1
    )

    df['game_id'] = 'W' + df['game_id'].astype(str)
    df = df[['subject', 'game_id', 'text_body', 'title']]

    # call function to split into sentences
    df['text'] = df['text_body'].apply(lambda x: extract_sentences(x))
    df = df.drop('text_body', axis=1)
    df = df.explode('text').reset_index(drop=True)

    # Calculate weights for Google data
    wts = calc_google_weights(df)
    df = df.drop('title', axis=1)
elif data_source == "twitter":
    # # SPRINT 1 - WEIGHT BY ENGAGEMENT
    # # add call to correct model and function for Twitter
    # df = df[["player", "team", "game_id", "text", "engagement_score"]]
    # # Rename column for consistency across data sources
    # df = df.rename(columns={"player": "subject"})

    # # Aggregate sentiment scores to player/team and week level
    # wts = df['engagement_score'].apply(lambda x: np.log(x + 1))

    # SPRINT 2 - WEIGHT BY TIME PROXIMITY TO GAME
    df = df[["team", "player", "game_id", "text", "created_at"]]
    df['created_at'] = pd.to_datetime(df['created_at'])

    z = zip(df['team'], df['game_id'])
    df['game_start_time'] = [get_game_start_times(x) for x in z]
    df['game_start_time'] = pd.to_datetime(df['game_start_time'])

    # Game start times are in ET, while tweets are in GMT/UTC
    # 1. Localize to Eastern Time (handles EST/EDT)
    df['game_start_time'] = df['game_start_time'].dt.tz_localize('US/Eastern')

    # 2. Convert to GMT (UTC)
    df['game_start_time'] = df['game_start_time'].dt.tz_convert('GMT')

    # Calculate hours of difference
    df['hours_diff'] = (df['created_at'] - df['game_start_time'])
    df['hours_diff'] = np.abs(df['hours_diff'].dt.total_seconds() / 3600)

    # Calculate weights, with rate parameter from ChatGPT
    lam = -np.log(0.25) / 24
    wts = np.exp(-lam * df['hours_diff'])

    # Drop team column and rename columns
    df = df.drop(['created_at', 'game_start_time'],
                 axis=1)
    df = df.rename(columns={'player': 'subject'})
else:
    df = df[['post_title', 'depth', 'body', 'score',
             'home_team', 'away_team', 'predicted_tag']]

    # filter out unsure comments (which is most of the data set currently)
    # also assign team using home and away team columns
    df = df[df['predicted_tag'] != 'unsure']
    df = df.assign(subject=np.where(
        df['predicted_tag'] == 'home_team',
        df['home_team'],
        df['away_team']
    ))

    # Rename column so column names are consistent
    df = df.rename(columns={'body': 'text'})

    # Map post title to week number
    df['game_id'] = df['post_title'].apply(map_reddit_thread_to_week)

    # Clip scores at 0 to avoid negative scores
    df['score'] = df['score'].clip(0)

    # Keep relevant columns
    df = df[['subject', 'game_id', 'text', 'score']]

    # create weights for local aggregation
    # use this formula so that weights fall in similar range as Twitter posts
    wts = np.log(df['score'] + 1) / 2

# Predict sentiment and create local index for data
df = predict_sentiment(df, data_source, "sentiment_scores_sprint2.csv")
local_index = aggregate_local_index(df, wts)
local_index = scale_local_index(local_index)

# Save to CSV
out_path = "sentiment_indices/" + data_source + "_local_index_sprint2.csv"
local_index.to_csv(out_path, index=False)
