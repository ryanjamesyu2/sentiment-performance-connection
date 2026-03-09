"""
A set of functions to predict the sentiment of text data from Twitter, Reddit,
and Google. These functions will be imported and used in the sentiment analysis
pipeline defined in sentiment_pipeline.py.
"""

# Import necessary libraries
from transformers import pipeline
import pipeline_configs as pc
import pandas as pd
import numpy as np
import torch


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

    model_path = pc.TWITTER_MODEL
    model_function = pc.TWITTER_FUNCTION

    # Run with pre-built HuggingFace pipeline, while batching inputs
    sentiment_scores = predict_sentiment(model_path, model_function, df)

    # Calculate EV sentiment scores using class probabilities
    final_scores = calc_sentiment_scores(sentiment_scores)

    # Add sentiment scores to data frame and save to new .csv file
    df["sentiment_scores"] = final_scores
    df.to_csv(out_file_name, index=False)

    return df


def predict_reddit(df, out_file_name="reddit_sentiment_scores.csv"):
    """
    A function to predict the sentiment of Reddit text data using a
    pre-trained HuggingFace model

    Parameters:
    -----------
    df: pandas DataFrame
        The preprocessed Reddit data
    out_file_name: str
        The name of the .csv file to save the results to (default is
        "reddit_sentiment_scores.csv")

    Returns:
    --------
    df: pandas DataFrame
        The input data frame with an additional column containing the predicted
        sentiment scores for each text entry
    """
    model_path = pc.REDDIT_MODEL
    model_function = pc.REDDIT_FUNCTION

    # Run with pre-built HuggingFace pipeline, while batching inputs
    sentiment_scores = predict_sentiment(model_path, model_function, df)

    # Calculate EV sentiment scores using class probabilities
    final_scores = calc_sentiment_scores(sentiment_scores)

    # Add sentiment scores to data frame and save to new .csv file
    df["sentiment_scores"] = final_scores
    df.to_csv(out_file_name, index=False)

    return df


def predict_google(df, out_file_name="google_sentiment_scores.csv"):
    # add code to predict sentiment for Google data using the appropriate
    # model and function
    model_path = pc.GOOGLE_MODEL
    model_function = pc.GOOGLE_FUNCTION

    # Run with pre-built HuggingFace pipeline, while batching inputs
    sentiment_scores = predict_sentiment(model_path, model_function, df)

    # Calculate EV sentiment scores using class probabilities
    # [TODO] adjust this function, since Google has 5 classes
    final_scores = calc_sentiment_scores(sentiment_scores)

    # Add sentiment scores to data frame and save to new .csv file
    df["sentiment_scores"] = final_scores
    df.to_csv(out_file_name, index=False)


def aggregate_local_index(in_df, weights):
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

    Returns:
    --------
    out_df: DataFrame
        A DataFrame containing the weighted average sentiment score for the
        given
        player/team and week combination
    """

    # Define variables we need
    out_df = pd.DataFrame(columns=["subject", "game_id", "local_index"])
    unique_levels = in_df[["subject", "game_id"]].drop_duplicates()

    # Iterate through each possible player/team
    for _, row in unique_levels.iterrows():
        # Get indices for the given player/team and week combination
        subject = row["subject"]
        game_id = row["game_id"]
        inds = (in_df["subject"] == subject) & (in_df["game_id"] == game_id)

        # Get the correct rows in df and weights
        subset_df = in_df[inds]
        scores = subset_df["sentiment_scores"].tolist()
        w = weights[inds].tolist()

        # Calculate weighted average and add to output df
        local_index = np.average(scores, weights=w)
        out_df.loc[len(out_df), :] = [subject, game_id, local_index]

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
    for week in df['game_id'].unique():
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


def predict_sentiment(model_path, model_function, df):
    """
    A function to load a pretrained HuggingFace model pipeline and predict
    the sentiment of a given list of texts

    Parameters
    ----------
    model_path: str
        A string containing the model path, as described on the HuggingFace
        model card
    model_function: str
        A string containing the model function, as described on the HuggingFace
        model card
    df: DataFrame
        A data frame containing the texts to be classified in a column named
        'text'

    Returns
    -------
    list
        list of json objects containing class labels and probabilities
    """
    classifier = pipeline(
        model_function,
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=64
    )

    sentiment_scores = classifier(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_all_scores=True
    )

    return sentiment_scores


def calc_sentiment_scores(model_output):
    final_scores = []
    for example in model_output:
        scores_dict = {d["label"].lower(): d["score"] for d in example}

        pos = scores_dict.get("positive", 0.0)
        neg = scores_dict.get("negative", 0.0)

        final_scores.append(pos - neg)

    return final_scores
