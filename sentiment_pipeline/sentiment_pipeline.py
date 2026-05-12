"""CLI: score CSV text by source (twitter|reddit|google from filename prefix)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from pipeline_utils import predict_sentiment
from pipeline_utils import aggregate_local_index
from pipeline_utils import scale_local_index
from pipeline_utils import map_reddit_thread_to_week
from pipeline_utils import extract_sentences
from pipeline_utils import calc_google_weights
from pipeline_utils import get_game_start_times
import pandas as pd
import numpy as np

file_path = sys.argv[1]
file_name = file_path.split("/")[-1]
data_source = file_name.split("_")[0].lower()

try:
    df = pd.read_csv(file_path, encoding="utf-8")
except FileNotFoundError:
    print(f"File {file_path} not found.", file=sys.stderr)
    sys.exit(1)

if data_source == "google":
    df = df.rename(columns={'team': 'subject', 'game_no': 'game_id'})

    bye_weeks = {
        'Philadelphia Eagles': 9,
        'Seattle Seahawks': 8,
        'Chicago Bears': 5,
        'Tampa Bay Buccaneers': 9,
        'Buffalo Bills': 7,
        'Cincinnati Bengals': 10,
        'Kansas City Chiefs': 10,
        'Indianapolis Colts': 11,
        'New England Patriots': 14,
        'Dallas Cowboys': 10
    }

    df['after_bye'] = df.apply(
        lambda x: x['game_id'] >= bye_weeks[x['subject']], axis=1
    )
    df['game_id'] = df.apply(
        lambda x: x['game_id'] + 1 if x['after_bye'] else x['game_id'], axis=1
    )

    df['game_id'] = 'W' + df['game_id'].astype(str)
    df = df[['subject', 'game_id', 'text_body', 'title']]

    df['text'] = df['text_body'].apply(lambda x: extract_sentences(x))
    df = df.drop('text_body', axis=1)
    df = df.explode('text').reset_index(drop=True)

    wts = calc_google_weights(df)
    df = df.drop('title', axis=1)
elif data_source == "twitter":
    df = df[df["player"] != "Keenan Allen"]

    df = df[["team", "player", "game_id", "text", "created_at"]]
    df['created_at'] = pd.to_datetime(df['created_at'])

    z = zip(df['team'], df['game_id'])
    df['game_start_time'] = [get_game_start_times(x) for x in z]
    df['game_start_time'] = pd.to_datetime(df['game_start_time'])

    # Kickoff times are stored naive US/Eastern; tweets are UTC.
    df['game_start_time'] = df['game_start_time'].dt.tz_localize('US/Eastern')
    df['game_start_time'] = df['game_start_time'].dt.tz_convert('GMT')

    df['hours_diff'] = (df['created_at'] - df['game_start_time'])
    df['hours_diff'] = np.abs(df['hours_diff'].dt.total_seconds() / 3600)

    lam = -np.log(0.25) / 24
    wts = np.exp(-lam * df['hours_diff'])

    df = df.drop(['created_at', 'game_start_time'],
                 axis=1)
    df = df.rename(columns={'player': 'subject'})
elif data_source == "reddit":
    df = df[['post_title', 'depth', 'body', 'score',
             'home_team', 'away_team', 'predicted_tag']]

    df = df[df['predicted_tag'] != 'unsure']
    df = df.assign(subject=np.where(
        df['predicted_tag'] == 'home_team',
        df['home_team'],
        df['away_team']
    ))

    df = df.rename(columns={'body': 'text'})

    df['game_id'] = df['post_title'].apply(map_reddit_thread_to_week)

    df['score'] = df['score'].clip(0)

    df = df[['subject', 'game_id', 'text', 'score']]

    wts = np.log(df['score'] + 1) / 2
else:
    raise ValueError("Data source not recognized. Please check file name.")

df = predict_sentiment(df, data_source, "sentiment_scores_sprint3.csv")
local_index = aggregate_local_index(df, wts)
local_index = scale_local_index(local_index)

out_path = (
    PROJECT_ROOT / "sentiment_indices" / f"{data_source}_local_index_sprint3.csv"
)
local_index.to_csv(out_path, index=False, encoding="utf-8")
