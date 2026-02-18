"""
A set of functions to predict the sentiment of text data from Twitter, Reddit,
and Google. These functions will be imported and used in the sentiment analysis
pipeline defined in sentiment_pipeline.py.
"""

# Import necessary libraries
# from transformers import pipeline
from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import pipeline_configs as pc
import pandas as pd
import numpy as np
from scipy.special import softmax


def calc_sentiment_score(scores, config):
    """
    A function to calculate an expected-value based sentiment score for a
    given set of class probabilities

    Parameters:
    -----------
    scores: NumPy array
        Contains class probabilities for a given text entry
    config: AutoConfig object
        Contains configs for the model used to generate the class
        probabilities, including a mapping from class ids to labels

    Returns:
    --------
    sent_score: float
        An expected-value based sentiment score for the given text entry
    """
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    sent_score = 0
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        if label == "Positive":
            val = 1
        elif label == "Neutral":
            val = 0
        else:
            val = -1
        s = scores[ranking[i]]
        sent_score += val * s

    return sent_score


def predict_twitter(file_name, out_file_name="twitter_sentiment_scores.csv"):
    """
    A function to predict the sentiment of Twitter text data using a
    pre-trained HuggingFace model

    Parameters:
    -----------
    file_name: str
        The name of the .csv file containing the preprocessed Twitter data
    out_file_name: str
        The name of the .csv file to save the results to (default is
        "twitter_sentiment_scores.csv")

    Returns:
    --------
    None
    """
    # Define necessary variables and read data
    df = pd.read_csv(file_name)
    df = df[["player", "game_id", "text", "engagement_score"]]
    model_path = pc.TWITTER_MODEL

    # Iterate through data frame and predict sentiment for each entry
    # Adapted from example code provided in HuggingFace model card
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.save_pretrained(model_path)

    sentiment_scores = []

    # Iterate through each text
    for t in df["text"]:
        # Generate output from model
        encoded_input = tokenizer(t, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sent_score = calc_sentiment_score(scores, config)

        # Append to list of sentiment scores
        sentiment_scores.append(sent_score)

    # Add sentiment scores to data frame and save to new .csv file
    df["sentiment_scores"] = sentiment_scores
    df.to_csv(out_file_name, index=False)


def predict_reddit(file_name):
    # add code to predict sentiment for Reddit data using the appropriate
    # model and function
    pass


def predict_google(file_name):
    # add code to predict sentiment for Google data using the appropriate
    # model and function
    pass
