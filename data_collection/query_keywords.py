# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd

# Define model to use
pipe = pipeline("fill-mask", model="microsoft/SportsBERT")

# Define players we are interested in
players = [
    "Jalen Hurts",
    "Saquon Barkley",
    "Saquon",
    "AJ Brown",
    "A.J. Brown",
    "Quinyon Mitchell",
    "Zack Baun"
]

# Define query templates
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

# Create empty DataFrame to store results
cols = ["Player", "Query", "Keyword"]
keywords_df = pd.DataFrame(columns=cols)

# Generate keywords for each player and query, store in df
for player in players:
    for query in queries:
        full_query = player + query
        results = pipe(full_query)
        for r in results:
            new_row = [player, full_query, r['token_str'].strip()]
            keywords_df.loc[len(keywords_df)] = new_row

# Save results to CSV
keywords_df.to_csv("keywords.csv", index=False)
