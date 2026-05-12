from pathlib import Path

import pandas as pd
from transformers import pipeline

pipe = pipeline("fill-mask", model="microsoft/SportsBERT")

players = [
    "Jalen Hurts",
    "Saquon Barkley",
    "Saquon",
    "AJ Brown",
    "A.J. Brown",
    "Quinyon Mitchell",
    "Zack Baun"
]

queries = [
    " is a [MASK]",
    " is an [MASK]",
    " is the [MASK]",
    " had a [MASK] performance",
    " is getting [MASK]",
    " is a [MASK] player",
    " is one of the most [MASK] players",
    " looks [MASK] on the field",
    " has been [MASK] this season"
]

cols = ["Player", "Query", "Keyword"]
keywords_df = pd.DataFrame(columns=cols)

for player in players:
    for query in queries:
        full_query = player + query
        results = pipe(full_query)
        for r in results:
            new_row = [player, full_query, r['token_str'].strip()]
            keywords_df.loc[len(keywords_df)] = new_row

out = Path(__file__).resolve().parent.parent / "keywords.csv"
keywords_df.to_csv(out, index=False, encoding="utf-8")
