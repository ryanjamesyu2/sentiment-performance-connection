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
    # add call to correct model and function for Google
    predict_google(df, "google")
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
    df = df[['post_title', 'depth', 'body', 'score']]

    # Rename column so column names are consistent
    df.rename(columns={'body': 'text'})

    # Map post title to week number
    week_dict = {
        'Post Game Thread: Washington Commanders at Philadelphia Eagles': 18,
        'Post Game Thread: Dallas Cowboys at Philadelphia Eagles': 1,
        'Post Game Thread: Philadelphia Eagles at Kansas City Chiefs': 2,
        'Post Game Thread: Los Angeles Rams at Philadelphia Eagles': 3,
        'Post Game Thread: Philadelphia Eagles at Tampa Bay Buccaneers': 4,
        'Post Game Thread: Denver Broncos at Philadelphia Eagles': 5,
        'Post Game Thread: Philadelphia Eagles at New York Giants': 6,
        'Post Game Thread: Philadelphia Eagles at Minnesota Vikings': 7,
        'Post Game Thread: New York Giants at Philadelphia Eagles': 8,
        'Post Game Thread: Philadelphia Eagles at Green Bay Packers': 10,
        'Post Game Thread: Detroit Lions at Philadelphia Eagles': 11,
        'Post Game Thread: Philadelphia Eagles at Dallas Cowboys': 12,
        'Post Game Thread: Chicago Bears at Philadelphia Eagles': 13,
        'Post Game Thread: Philadelphia Eagles at Los Angeles Chargers': 14,
        'Post Game Thread: Las Vegas Raiders at Philadelphia Eagles': 15,
        'Post Game Thread: Philadelphia Eagles at Washington Commanders': 16,
        'Post Game Thread: Philadelphia Eagles at Buffalo Bills': 17
    }
    df['game_id'] = df['post_title'].map(week_dict)

    # predict sentiment for comments
    df = predict_reddit(df)

    # Run through LLM to assign comments to teams, with column name "subject"
    pass

    # create weights for local aggregation
    # use this formula so that weights fall in similar range as Twitter posts
    log_eng_scores = np.log(df['score'] + 1) / 2
    reddit_local = aggregate_local_index(df, log_eng_scores)

    # Save to CSV file
    out_path = "sentiment_indices/reddit_local_index.csv"
    reddit_local.to_csv(out_path, index=False)
