# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="microsoft/SportsBERT")
query = "Jalen Hurts is a [MASK]"

results = pipe(query)
print(results)
