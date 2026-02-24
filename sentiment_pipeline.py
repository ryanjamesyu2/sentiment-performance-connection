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
    predict_google(df)
elif data_source == "twitter":
    # add call to correct model and function for Twitter
    df = df[["player", "game_id", "text", "engagement_score"]]
    df = predict_twitter(df)

    # Aggregate sentiment scores to player/team and week level
    log_eng_scores = df['engagement_score'].apply(lambda x: np.log(x + 1))
    twitter_local = aggregate_local_index(df, log_eng_scores, "twitter")

    # Rescale local index to be between 0 and 100 on a weekly basis
    twitter_local = scale_local_index(twitter_local)

    # Save to CSV file
    out_path = "sentiment_indices/twitter_local_index.csv"
    twitter_local.to_csv(out_path, index=False)
else:
    # add call to correct model and function for Reddit
    predict_reddit(df)
