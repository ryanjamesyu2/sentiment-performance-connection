import pandas as pd
import numpy as np
import re


def clean_tweet_text(text):
    """
    Minimal text normalization for transformer models.
    """
    if pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove zero-width and weird whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_tweet_df(df):
    """
    Light filtering and cleaning for relevance and quality
    """

    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=["text", "player", "team", "game_id"])

    # Ensure numeric columns are numeric
    num_cols = ["likes", "retweets", "replies", "engagement_score"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Normalize booleans
    if "is_post_game" in df.columns:
        df["is_post_game"] = df["is_post_game"].astype(bool)

    if "verified" in df.columns:
        df["verified"] = df["verified"].astype(bool)

    # Remove exact duplicate tweets
    df = df.drop_duplicates(
        subset=["text", "player", "team", "game_id", "created_at"]
    )

    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(clean_tweet_text)

    # Basic length features
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().apply(len)

    # Remove empty or near-empty tweets
    df = df[df["word_len"] >= 3]

    # Remove pure mentions / handles (e.g. "@JalenHurts")
    df = df[~df["text"].str.match(r"^(@\w+\s*)+$")]

    # Remove pure emojis / symbols
    df = df[df["text"].str.contains(r"[A-Za-z0-9]", regex=True)]

    # Remove obvious retweet boilerplate
    df = df[~df["text"].str.startswith("RT ")]

    # Remove leading/trailing quotes and stray characters
    df["text"] = df["text"].str.strip(" '\"\n\t")

    df = df.reset_index(drop=True)

    # Drop helper columns
    df = df.drop(columns=["char_len", "word_len"], errors="ignore")

    return df


def is_betting_line(text):
    """
    Returns True if the text follows structural patterns of a betting recap
    rather than a sentiment-driven fan post.
    """

    # Pattern 1: Point Spreads and Moneylines (e.g., PHI -7.5, DAL +3, -110)
    spread_pattern = r'\b[A-Z]{2,3}\s?[\+\-]\d+(\.\d)?'

    # Pattern 2: Over/Under (e.g., O/U 48.5, Total: 51.5)
    total_pattern = r'\b(O/U|Over/Under|Total:?)\s?\d{2,3}(\.\d)?'

    # Pattern 3: Moneyline odds (e.g., +250, -150) appearing at the end or in a list
    odds_pattern = r'[:\s][\+\-][1-9]\d{2}\b'

    # Pattern 4: Combined line (e.g., PHI -3 | 44.5 | -110)
    struct_pattern = r'\d+(\.\d)?\s?[\/|]\s?[\+\-]?\d+'

    # Check for matches
    if re.search(spread_pattern, text) or \
       re.search(total_pattern, text) or \
       re.search(odds_pattern, text) or \
       re.search(struct_pattern, text):
        return True
    return False


def filter_betting_lines(df):
    """
    Filters out rows where the text follows structural patterns of a betting recap
    rather than a sentiment-driven fan post.
    """
    return df[~df['text'].apply(is_betting_line)]
