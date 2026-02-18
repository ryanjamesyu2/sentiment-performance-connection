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
from sentiment_prediction import predict_twitter
from sentiment_prediction import predict_reddit
from sentiment_prediction import predict_google

# Extract file name from command line arguments and determine data source
file_name = argv[1]
data_source = file_name.split("_")[0].lower()

# Load the appropriate model path and function based on the data source
if data_source == "google":
    # add call to correct model and function for Google
    predict_google(file_name)
elif data_source == "twitter":
    # add call to correct model and function for Twitter
    predict_twitter(file_name)
else:
    # add call to correct model and function for Reddit
    predict_reddit(file_name)
