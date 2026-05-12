"""Combine per-source local sentiment indices into one player-week global index."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) < 3:
    sys.exit("usage: python create_global_index.py <path1> <w1> [<path2> <w2> ...]")
if (len(sys.argv) - 1) % 2 != 0:
    sys.exit(f"expected path weight pairs; got {len(sys.argv) - 1} trailing args")

player_list = pd.read_csv(
    PROJECT_ROOT / "sentiment_indices" / "player_list.csv", encoding="utf-8"
)
weeks = pd.Series(
    [
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
        "W7",
        "W8",
        "W9",
        "W10",
        "W11",
        "W12",
        "W13",
        "W14",
        "W15",
        "W16",
        "W17",
        "W18",
    ]
)
teams = pd.Series(
    [
        "Philadelphia Eagles",
        "Buffalo Bills",
        "Cincinnati Bengals",
        "Indianapolis Colts",
        "Kansas City Chiefs",
        "Chicago Bears",
        "Tampa Bay Buccaneers",
        "Seattle Seahawks",
        "Dallas Cowboys",
        "New England Patriots",
    ]
)

df = pd.DataFrame(
    {
        "player": pd.Series(np.tile(player_list["subject"], 18)),
        "team": np.tile(teams.repeat(5).reset_index(drop=True), 18),
        "week": weeks.repeat(len(player_list)).reset_index(drop=True),
        "global_index": pd.Series(np.zeros(len(player_list) * 18)),
    }
)

ind = 1
wt_sum = 0.0

while ind < len(sys.argv) - 1:
    file_path = sys.argv[ind]
    weight = float(sys.argv[ind + 1])
    wt_sum += weight
    local_df = pd.read_csv(file_path, encoding="utf-8")

    file_name = file_path.split("/")[-1]
    data_source = file_name.split("_")[0].lower()

    group_var = "player" if data_source == "twitter" else "team"

    df = df.merge(
        local_df,
        left_on=[group_var, "week"],
        right_on=["subject", "game_id"],
        how="left",
    )
    df["local_index"] = df["local_index"].fillna(
        df.groupby(group_var)["local_index"].transform("mean")
    )

    df["global_index"] += df["local_index"] * weight

    df.drop(columns=["subject", "game_id", "local_index"], inplace=True)

    ind += 2

df["global_index"] = df["global_index"] / wt_sum

df.to_csv(
    PROJECT_ROOT / "sentiment_indices" / "global_index_test.csv",
    index=False,
    encoding="utf-8",
)
