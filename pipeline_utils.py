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
        given player/team and week combination
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
        top_k=None
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


def map_reddit_thread_to_week(thread_title):
    week_dict = {
        # Eagles
        'Post Game Thread: Washington Commanders at Philadelphia Eagles':
        'W18',
        'Post Game Thread: Dallas Cowboys at Philadelphia Eagles': 'W1',
        'Post Game Thread: Philadelphia Eagles at Kansas City Chiefs': 'W2',
        'Post Game Thread: Los Angeles Rams at Philadelphia Eagles': 'W3',
        'Post Game Thread: Philadelphia Eagles at Tampa Bay Buccaneers': 'W4',
        'Post Game Thread: Denver Broncos at Philadelphia Eagles': 'W5',
        'Post Game Thread: Philadelphia Eagles at New York Giants': 'W6',
        'Post Game Thread: Philadelphia Eagles at Minnesota Vikings': 'W7',
        'Post Game Thread: New York Giants at Philadelphia Eagles': 'W8',
        'Post Game Thread: Philadelphia Eagles at Green Bay Packers': 'W10',
        'Post Game Thread: Detroit Lions at Philadelphia Eagles': 'W11',
        'Post Game Thread: Philadelphia Eagles at Dallas Cowboys': 'W12',
        'Post Game Thread: Chicago Bears at Philadelphia Eagles': 'W13',
        'Post Game Thread: Philadelphia Eagles at Los Angeles Chargers': 'W14',
        'Post Game Thread: Las Vegas Raiders at Philadelphia Eagles': 'W15',
        'Post Game Thread: Philadelphia Eagles at Washington Commanders':
        'W16',
        'Post Game Thread: Philadelphia Eagles at Buffalo Bills': 'W17',

        # Bills
        'Post Game Thread: Baltimore Ravens at Buffalo Bills': 'W1',
        'Post Game Thread: Buffalo Bills at New York Jets': 'W2',
        'Post Game Thread: Miami Dolphins at Buffalo Bills': 'W3',
        'Post Game Thread: New Orleans Saints at Buffalo Bills': 'W4',
        'Post Game Thread: New England Patriots at Buffalo Bills': 'W5',
        'Post Game Thread: Buffalo Bills at Atlanta Falcons': 'W6',
        'Post Game Thread: Buffalo Bills at Carolina Panthers': 'W8',
        'Post Game Thread: Kansas City Chiefs at Buffalo Bills': 'W9',
        'Post Game Thread: Buffalo Bills at Miami Dolphins': 'W10',
        'Post Game Thread: Tampa Bay Buccaneers at Buffalo Bills': 'W11',
        'Post Game Thread: Buffalo Bills at Houston Texans': 'W12',
        'Post Game Thread: Buffalo Bills at Pittsburgh Steelers': 'W13',
        'Post Game Thread: Cincinnati Bengals at Buffalo Bills': 'W14',
        'Post Game Thread: Buffalo Bills at New England Patriots': 'W15',
        'Post Game Thread: Buffalo Bills at Cleveland Browns': 'W16',
        'Post Game Thread: Philadelphia Eagles at Buffalo Bills': 'W17',
        'Post Game Thread: New York Jets at Buffalo Bills': 'W18',

        # Bears
        'Post Game Thread: Minnesota Vikings at Chicago Bears': 'W1',
        'Post Game Thread: Chicago Bears at Detroit Lions': 'W2',
        'Post Game Thread: Dallas Cowboys at Chicago Bears': 'W3',
        'Post Game Thread: Chicago Bears at Las Vegas Raiders': 'W4',
        'Post Game Thread: Chicago Bears at Washington Commanders': 'W6',
        'Post Game Thread: New Orleans Saints at Chicago Bears': 'W7',
        'Post Game Thread: Chicago Bears at Baltimore Ravens': 'W8',
        'Post Game Thread: Chicago Bears at Cincinnati Bengals': 'W9',
        'Post Game Thread: New York Giants at Chicago Bears': 'W10',
        'Post Game Thread: Chicago Bears at Minnesota Vikings': 'W11',
        'Post Game Thread: Pittsburgh Steelers at Chicago Bears': 'W12',
        'Post Game Thread: Chicago Bears at Philadelphia Eagles': 'W13',
        'Post Game Thread: Chicago Bears at Green Bay Packers': 'W14',
        'Post Game Thread: Cleveland Browns at Chicago Bears': 'W15',
        'Post Game Thread: Green Bay Packers at Chicago Bears': 'W16',
        'Post Game Thread: Chicago Bears at San Francisco 49ers': 'W17',
        'Post Game Thread: Detroit Lions at Chicago Bears': 'W18',

        # Bengals
        'Post Game Thread: Cincinnati Bengals at Cleveland Browns': 'W1',
        'Post Game Thread: Jacksonville Jaguars at Cincinnati Bengals': 'W2',
        'Post Game Thread: Cincinnati Bengals at Minnesota Vikings': 'W3',
        'Post Game Thread: Cincinnati Bengals at Denver Broncos': 'W4',
        'Post Game Thread: Detroit Lions at Cincinnati Bengals': 'W5',
        'Post Game Thread: Cincinnati Bengals at Green Bay Packers': 'W6',
        'Post Game Thread: Pittsburgh Steelers at Cincinnati Bengals': 'W7',
        'Post Game Thread: New York Jets at Cincinnati Bengals': 'W8',
        'Post Game Thread: Chicago Bears at Cincinnati Bengals': 'W9',
        'Post Game Thread: Cincinnati Bengals at Pittsburgh Steelers': 'W11',
        'Post Game Thread: New England Patriots at Cincinnati Bengals': 'W12',
        'Post Game Thread: Cincinnati Bengals at Baltimore Ravens': 'W13',
        'Post Game Thread: Cincinnati Bengals at Buffalo Bills': 'W14',
        'Post Game Thread: Baltimore Ravens at Cincinnati Bengals': 'W15',
        'Post Game Thread: Cincinnati Bengals at Miami Dolphins': 'W16',
        'Post Game Thread: Arizona Cardinals at Cincinnati Bengals': 'W17',
        'Post Game Thread: Cleveland Browns at Cincinnati Bengals': 'W18',

        # Colts
        'Post Game Thread: Miami Dolphins at Indianapolis Colts': 'W1',
        'Post Game Thread: Denver Broncos at Indianapolis Colts': 'W2',
        'Post Game Thread: Indianapolis Colts at Tennessee Titans': 'W3',
        'Post Game Thread: Indianapolis Colts at Los Angeles Rams': 'W4',
        'Post Game Thread: Las Vegas Raiders at Indianapolis Colts': 'W5',
        'Post Game Thread: Arizona Cardinals at Indianapolis Colts': 'W6',
        'Post Game Thread: Indianapolis Colts at Los Angeles Chargers': 'W7',
        'Post Game Thread: Tennessee Titans at Indianapolis Colts': 'W8',
        'Post Game Thread: Indianapolis Colts at Pittsburgh Steelers': 'W9',
        'Post Game Thread: Atlanta Falcons at Indianapolis Colts': 'W10',
        'Post Game Thread: Indianapolis Colts at Kansas City Chiefs': 'W12',
        'Post Game Thread: Houston Texans at Indianapolis Colts': 'W13',
        'Post Game Thread: Indianapolis Colts at Jacksonville Jaguars': 'W14',
        'Post Game Thread: Indianapolis Colts at Seattle Seahawks': 'W15',
        'Post Game Thread: San Francisco 49ers at Indianapolis Colts': 'W16',
        'Post Game Thread: Jacksonville Jaguars at Indianapolis Colts': 'W17',
        'Post Game Thread: Indianapolis Colts at Houston Texans': 'W18',

        # Chiefs
        'Post Game Thread: Kansas City Chiefs at Los Angeles Chargers': 'W1',
        'Post Game Thread: Philadelphia Eagles at Kansas City Chiefs': 'W2',
        'Post Game Thread: Kansas City Chiefs at New York Giants': 'W3',
        'Post Game Thread: Baltimore Ravens at Kansas City Chiefs': 'W4',
        'Post Game Thread: Kansas City Chiefs at Jacksonville Jaguars': 'W5',
        'Post Game Thread: Detroit Lions at Kansas City Chiefs': 'W6',
        'Post Game Thread: Las Vegas Raiders at Kansas City Chiefs': 'W7',
        'Post Game Thread: Washington Commanders at Kansas City Chiefs': 'W8',
        'Post Game Thread: Kansas City Chiefs at Buffalo Bills': 'W9',
        'Post Game Thread: Kansas City Chiefs at Denver Broncos': 'W11',
        'Post Game Thread: Indianapolis Colts at Kansas City Chiefs': 'W12',
        'Post Game Thread: Kansas City Chiefs at Dallas Cowboys': 'W13',
        'Post Game Thread: Houston Texans at Kansas City Chiefs': 'W14',
        'Post Game Thread: Los Angeles Chargers at Kansas City Chiefs': 'W15',
        'Post Game Thread: Kansas City Chiefs at Tennessee Titans': 'W16',
        'Post Game Thread: Denver Broncos at Kansas City Chiefs': 'W17',
        'Post Game Thread: Kansas City Chiefs at Las Vegas Raiders': 'W18',

        # Buccaneers
        'Post Game Thread: Tampa Bay Buccaneers at Atlanta Falcons': 'W1',
        'Post Game Thread: Tampa Bay Buccaneers at Houston Texans': 'W2',
        'Post Game Thread: New York Jets at Tampa Bay Buccaneers': 'W3',
        'Post Game Thread: Philadelphia Eagles at Tampa Bay Buccaneers': 'W4',
        'Post Game Thread: Tampa Bay Buccaneers at Seattle Seahawks': 'W5',
        'Post Game Thread: San Francisco 49ers at Tampa Bay Buccaneers': 'W6',
        'Post Game Thread: Tampa Bay Buccaneers at Detroit Lions': 'W7',
        'Post Game Thread: Tampa Bay Buccaneers at New Orleans Saints': 'W8',
        'Post Game Thread: New England Patriots at Tampa Bay Buccaneers':
        'W10',
        'Post Game Thread: Tampa Bay Buccaneers at Buffalo Bills': 'W11',
        'Post Game Thread: Tampa Bay Buccaneers at Los Angeles Rams': 'W12',
        'Post Game Thread: Arizona Cardinals at Tampa Bay Buccaneers': 'W13',
        'Post Game Thread: New Orleans Saints at Tampa Bay Buccaneers': 'W14',
        'Post Game Thread: Atlanta Falcons at Tampa Bay Buccaneers': 'W15',
        'Post Game Thread: Tampa Bay Buccaneers at Carolina Panthers': 'W16',
        'Post Game Thread: Tampa Bay Buccaneers at Miami Dolphins': 'W17',
        'Post Game Thread: Carolina Panthers at Tampa Bay Buccaneers': 'W18',

        # Seahawks
        'Post Game Thread: San Francisco 49ers at Seattle Seahawks': 'W1',
        'Post Game Thread: Seattle Seahawks at Pittsburgh Steelers': 'W2',
        'Post Game Thread: New Orleans Saints at Seattle Seahawks': 'W3',
        'Post Game Thread: Seattle Seahawks at Arizona Cardinals': 'W4',
        'Post Game Thread: Tampa Bay Buccaneers at Seattle Seahawks': 'W5',
        'Post Game Thread: Seattle Seahawks at Jacksonville Jaguars': 'W6',
        'Post Game Thread: Houston Texans at Seattle Seahawks': 'W7',
        'Post Game Thread: Seattle Seahawks at Washington Commanders': 'W9',
        'Post Game Thread: Arizona Cardinals at Seattle Seahawks': 'W10',
        'Post Game Thread: Seattle Seahawks at Los Angeles Rams': 'W11',
        'Post Game Thread: Seattle Seahawks at Tennessee Titans': 'W12',
        'Post Game Thread: Minnesota Vikings at Seattle Seahawks': 'W13',
        'Post Game Thread: Seattle Seahawks at Atlanta Falcons': 'W14',
        'Post Game Thread: Indianapolis Colts at Seattle Seahawks': 'W15',
        'Post Game Thread: Los Angeles Rams at Seattle Seahawks': 'W16',
        'Post Game Thread: Seattle Seahawks at Carolina Panthers': 'W17',
        'Post Game Thread: Seattle Seahawks at San Francisco 49ers': 'W18',

        # Patriots
        'Post Game Thread: Las Vegas Raiders at New England Patriots': 'W1',
        'Post Game Thread: New England Patriots at Miami Dolphins': 'W2',
        'Post Game Thread: Pittsburgh Steelers at New England Patriots': 'W3',
        'Post Game Thread: Carolina Panthers at New England Patriots': 'W4',
        'Post Game Thread: New England Patriots at Buffalo Bills': 'W5',
        'Post Game Thread: New England Patriots at New Orleans Saints': 'W6',
        'Post Game Thread: New England Patriots at Tennessee Titans': 'W7',
        'Post Game Thread: Cleveland Browns at New England Patriots': 'W8',
        'Post Game Thread: Atlanta Falcons at New England Patriots': 'W9',
        'Post Game Thread: New England Patriots at Tampa Bay Buccaneers':
        'W10',
        'Post Game Thread: New York Jets at New England Patriots': 'W11',
        'Post Game Thread: New England Patriots at Cincinnati Bengals': 'W12',
        'Post Game Thread: New York Giants at New England Patriots': 'W13',
        'Post Game Thread: Buffalo Bills at New England Patriots': 'W15',
        'Post Game Thread: New England Patriots at Baltimore Ravens': 'W16',
        'Post Game Thread: New England Patriots at New York Jets': 'W17',
        'Post Game Thread: Miami Dolphins at New England Patriots': 'W18',

        # Cowboys
        'Post Game Thread: Dallas Cowboys at Philadelphia Eagles': 'W1',
        'Post Game Thread: New York Giants at Dallas Cowboys': 'W2',
        'Post Game Thread: Dallas Cowboys at Chicago Bears': 'W3',
        'Post Game Thread: Green Bay Packers at Dallas Cowboys': 'W4',
        'Post Game Thread: Dallas Cowboys at New York Jets': 'W5',
        'Post Game Thread: Dallas Cowboys at Carolina Panthers': 'W6',
        'Post Game Thread: Washington Commanders at Dallas Cowboys': 'W7',
        'Post Game Thread: Dallas Cowboys at Denver Broncos': 'W8',
        'Post Game Thread: Arizona Cardinals at Dallas Cowboys': 'W9',
        'Post Game Thread: Dallas Cowboys at Las Vegas Raiders': 'W11',
        'Post Game Thread: Philadelphia Eagles at Dallas Cowboys': 'W12',
        'Post Game Thread: Kansas City Chiefs at Dallas Cowboys': 'W13',
        'Post Game Thread: Dallas Cowboys at Detroit Lions': 'W14',
        'Post Game Thread: Minnesota Vikings at Dallas Cowboys': 'W15',
        'Post Game Thread: Los Angeles Chargers at Dallas Cowboys': 'W16',
        'Post Game Thread: Dallas Cowboys at Washington Commanders': 'W17',
        'Post Game Thread: Dallas Cowboys at New York Giants': 'W18'
    }
    return week_dict.get(thread_title, "Unknown")
