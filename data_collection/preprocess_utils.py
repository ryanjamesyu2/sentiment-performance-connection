import re

import numpy as np
import pandas as pd


def clean_tweet_text(text):
    if pd.isna(text):
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_tweet_df(df):
    df = df.copy()

    df = df.dropna(subset=["text", "player", "team", "game_id"])

    num_cols = ["likes", "retweets", "replies", "engagement_score"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "is_post_game" in df.columns:
        df["is_post_game"] = df["is_post_game"].astype(bool)

    if "verified" in df.columns:
        df["verified"] = df["verified"].astype(bool)

    df = df.drop_duplicates(
        subset=["text", "player", "team", "game_id", "created_at"]
    )

    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(clean_tweet_text)

    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().apply(len)

    df = df[df["word_len"] >= 3]

    df = df[~df["text"].str.match(r"^(@\w+\s*)+$")]

    df = df[df["text"].str.contains(r"[A-Za-z0-9]", regex=True)]

    df = df[~df["text"].str.startswith("RT ")]

    df["text"] = df["text"].str.strip(" '\"\n\t")

    df = df.reset_index(drop=True)

    df = df.drop(columns=["char_len", "word_len"], errors="ignore")

    return df


def is_betting_line(text):
    spread_pattern = r"\b[A-Z]{2,3}\s?[\+\-]\d+(\.\d)?"
    total_pattern = r"\b(O/U|Over/Under|Total:?)\s?\d{2,3}(\.\d)?"
    odds_pattern = r"[:\s][\+\-][1-9]\d{2}\b"
    struct_pattern = r"\d+(\.\d)?\s?[\/|]\s?[\+\-]?\d+"

    if (
        re.search(spread_pattern, text)
        or re.search(total_pattern, text)
        or re.search(odds_pattern, text)
        or re.search(struct_pattern, text)
    ):
        return True
    return False


def filter_betting_lines(df):
    return df[~df["text"].apply(is_betting_line)]
