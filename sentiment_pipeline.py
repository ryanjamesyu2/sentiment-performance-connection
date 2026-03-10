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
from pipeline_utils import predict_twitter
from pipeline_utils import predict_reddit
from pipeline_utils import predict_google
from pipeline_utils import aggregate_local_index
from pipeline_utils import scale_local_index
from pipeline_utils import map_reddit_thread_to_week
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
    df['game_id'] = 'W' + df['week'].astype(str)
    # [TODO] Add column for subject when we have it (thru LLM)
    df = df[['game_id', 'text_body']]
    df = df.rename(columns={'text_body': 'text'})

    # [TODO] If necessary, call function to split into sentences

    # add call to correct model and function for Google
    df = predict_google(df)
    # [TODO] Calculate weights for Google data
    wts = np.ones(len(df))    # placeholder for equal weights for now

    google_local = aggregate_local_index(df, wts)

    # Rescale local Google index
    google_local = scale_local_index(google_local)

    # Save to CSV
    out_path = "sentiment_indices/google_local_index.csv"
    google_local.to_csv(out_path, index=False)
elif data_source == "twitter":
    # add call to correct model and function for Twitter
    df = df[["player", "game_id", "text", "engagement_score"]]
    df = predict_twitter(df)

    # Aggregate sentiment scores to player/team and week level
    log_eng_scores = df['engagement_score'].apply(lambda x: np.log(x + 1))

    # Rename column for consistency across data sources
    df = df.rename(columns={"player": "subject"})
    twitter_local = aggregate_local_index(df, log_eng_scores)

    # Rescale local index to be between 0 and 100 on a weekly basis
    twitter_local = scale_local_index(twitter_local)

    # Save to CSV file
    out_path = "sentiment_indices/twitter_local_index.csv"
    twitter_local.to_csv(out_path, index=False)
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

    # predict sentiment for comments
    df = predict_reddit(df)

    # create weights for local aggregation
    # use this formula so that weights fall in similar range as Twitter posts
    log_scores = np.log(df['score'] + 1) / 2
    reddit_local = aggregate_local_index(df, log_scores)

    # Rescale local index to be between 0 and 100 on a weekly basis
    reddit_local = scale_local_index(reddit_local)

    # Save to CSV file
    out_path = "sentiment_indices/reddit_local_index.csv"
    reddit_local.to_csv(out_path, index=False)
