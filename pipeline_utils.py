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
        if label == "positive":
            val = 1
        elif label == "neutral":
            val = 0
        else:
            val = -1
        s = scores[ranking[i]]
        sent_score += val * s

    return sent_score


def predict_twitter(df, out_file_name="twitter_sentiment_scores.csv"):
    """
    A function to predict the sentiment of Twitter text data using a
    pre-trained HuggingFace model

    Parameters:
    -----------
    df: pandas DataFrame
        The preprocessed Twitter data
    out_file_name: str
        The name of the .csv file to save the results to (default is
        "twitter_sentiment_scores.csv")

    Returns:
    --------
    df: pandas DataFrame
        The input data frame with an additional column containing the predicted
        sentiment scores for each text entry
    """
    # Temporary to test with first 100 entries - remove later
    # df = df.iloc[:100, :]

    model_path = pc.TWITTER_MODEL

    # Iterate through data frame and predict sentiment for each entry
    # Adapted from example code provided in HuggingFace model card
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    sentiment_scores = []

    # Iterate through each text
    for t in df["text"]:
        # Generate output from model
        encoded_input = tokenizer(
            t,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sent_score = calc_sentiment_score(scores, config)

        # Append to list of sentiment scores
        sentiment_scores.append(sent_score)

    # Add sentiment scores to data frame and save to new .csv file
    df["sentiment_scores"] = sentiment_scores
    df.to_csv(out_file_name, index=False)

    return df


def predict_reddit(df):
    # add code to predict sentiment for Reddit data using the appropriate
    # model and function
    pass


def predict_google(df):
    # add code to predict sentiment for Google data using the appropriate
    # model and function
    pass


def aggregate_local_index(in_df, weights, source):
    """
    A function to calculate a weighted average of sentiment scores for a given
    data source. They are aggregated to the player/team and week combination
    level.

    Parameters:
    -----------
    in_df: pandas DataFrame
        A DataFrame containing sentiment scores for a given player/team and
        week combination
    weights: list of floats
        A list of weights corresponding to the sentiment scores, generated
        from a metadata column such as engagement score
    source: str
        The source of the data, either "twitter", "reddit", or "google"

    Returns:
    --------
    out_df: DataFrame
        A DataFrame containing the weighted average sentiment score for the
        given
        player/team and week combination
    """

    if source == "twitter":
        # Define variables we need
        out_df = pd.DataFrame(columns=["player", "game_id", "local_index"])
        unique_levels = in_df[["player", "game_id"]].drop_duplicates()

        # Iterate through each possible player/team
        for _, row in unique_levels.iterrows():
            # Get indices for the given player/team and week combination
            player = row["player"]
            game_id = row["game_id"]
            inds = (in_df["player"] == player) & (in_df["game_id"] == game_id)

            # Get the correct rows in df and weights
            subset_df = in_df[inds]
            scores = subset_df["sentiment_scores"].tolist()
            w = weights[inds].tolist()

            # Calculate weighted average and add to output df
            local_index = np.average(scores, weights=w)
            out_df.loc[len(out_df), :] = [player, game_id, local_index]
    elif source == "reddit":
        # add code for Reddit data
        pass
    else:
        # add code for Google data
        pass

    return out_df


def scale_local_index(df):
    """
    A function to scale the local index values to be between 0 and 100
    via min-max scaling on a week-by-week basis.

    Parameters:
    -----------
    df: DataFrame
        A DataFrame containing the local index values to be scaled

    Returns:
    --------
    out_df: DataFrame
        A DataFrame containing scaled local index values, between 0 and 100
    """
    # Create output df
    out_df = pd.DataFrame(columns=df.columns)

    # Iterate through each week
    for week in df['game_id']:
        # Get index entries for that week
        week_inds = (df['game_id'] == week)
        week_df = df[week_inds]
        local_index = week_df['local_index']

        # Determine minimum and maximum observed sentiment in given week
        max_val = max(local_index)
        min_val = min(local_index)

        # Scale local index for that week
        num = (week_df['local_index'] - min_val)
        den = (max_val - min_val)
        week_df['local_index'] = num / den * 100

        # Concatenate weekly data frame to previous weeks
        out_df = pd.concat([out_df, week_df], ignore_index=True)

    return out_df
