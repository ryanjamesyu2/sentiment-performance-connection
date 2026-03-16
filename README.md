# The Sentiment Performance Connection
MADS '26 capstone project repository. All code developed by Anirudh Karanam, Steve Wong, and Ryan Yu. Project completed in partnership with BTA Sports, LLC. It was developed using the anaconda python distribution, and utilizes conda virtual environments to ensure consistency when running code. 

This pipeline takes input data collected from X, Reddit, or Google for the course of an entire NFL season and creates sentiment indices based on the textual data. Each tweet, Reddit comment, or sentence in a Google News article is individually given a predicted sentiment score by subtracting the probability of it having negative sentiment from the probability of it having positive sentiment, as given by HuggingFace transformer models. Then, each player/week combination for X and each team/week combination for Reddit and Google and aggregated by taking a weighted average and scaled to be on a 0-100 scale. 

## Virtual Environment
The virtual environment used for all the code and processing in this repository can be found in the `env.yml` file. A running list of all the packages included in the environment is:

* pandas
* matplotlib
* seaborn
* awswrangler
* ipykernel
* transformers
* scikit-learn
* torch
* desearch-py
* python-dotenv
* pysentiment2
* spaCy

To create the environment from the .yml file, run the following command:

```
conda env create -f env.yml
```

For spaCy, need to run the following command first, only once. This downloads the model used to parse Google News articles into individual sentences, which is necessary to abide by the 512 token limit for the HuggingFace transformer models.

```
python -m spacy download en_core_web_sm
```

## Data

Each data file must be stored in a .csv file. Before running the pipeline, each data file should also begin with "[data_source]\_" (i.e. either "twitter_", "reddit_", or "google_"). The pipeline uses this prefix to the data file to determine how to preprocess the data and to determine which HuggingFace model to utilize. 

**Note that each data source has different requirements for structure, based on how it was collected. Even if columns between data sources have similar names, it does not mean that the pipeline is expecting the same data type for each source of data.**

### X (Twitter)

Twitter data was collected using the Desearch API. The code used can be found in the `desearch_scrape.py` file in this repository. 

The .csv file must contain at least 4 columns:
* "player", which contains a string of the player name
* "game_id", which contains a string of the form "W#" denoting which game week the NFL game was a part of. Replace "#" with an integer ranging from 1 to 18.
* "text", which contains a string of the tweet contents
* "engagement_score", which contains a numerical measurement of tweet engagement and is used to calculate the weight for each tweet 

In the first sprint, Twitter data was weighted of an engagement score calculated during data collection. It is a function of likes, comments, and retweets.

### Reddit

The Reddit data was collected via a Selenium driven scraper. The code can be found in the `reddit.ipynb` notebook. The data was collected from r/NFL post game threads. 

The Reddit data must have at least 5 columns:
* "post_title", which contains a string of the form "Post Game Thread: Washington Commanders at Philadelphia Eagles" (mimics the official post title on Reddit)
* "home_team", which contains a string of the home team for the game
* "away team", which contains a string of the away team for the game
* "score", which contains a numerical measurement of comment engagement and is used to calculate the weight for each tweet 
* "predicted_tag", which is either "home_team", "away_team", or "unsure". This is a tag to attribute sentiment to the correct team, and is generated using the code in the `reddit_home_away.ipynb` notebook.

Reddit data is also initially weighted by an engagement score, which is a function of upvotes and downvotes of the comment.

### Google News

Google News data was also collected via a Selenium scraper. The input data file is expecting one row in the file per article, rather than 1 per sentence. The pipeline automatically splits the articles into sentences. The .csv file must contain at least 4 columns:
* "team", which contains a string of the team name
* "game_no", which contains a integer denoting which game week the NFL game was a part of (ranging from 1 to 18)
* "text_body", which contains a string of the entire article contents
* "title", which contains a string of the article title and is used to group sentences when calculating weights

In sprint 1, Google News data is weighted by the sentences' location within the article. Research shows that the majority of readers do not read the entire article, and many skim for just headlines. So, we weight sentences earlier in the article (which readers are more likely to see) with higher weights. Weights follow an exponential decaying pattern, with the first 25% of sentences comprising 75% of the total weight for the article.

## Using the Pipeline

The pipeline currently is limited to one data file at a time, with the data file containing data from only one data source. First, activate the `sentiment` virtual environment using the command:

```
conda activate sentiment
```

Then, in the terminal, run one simple command to kick off the entire pipeline.

```
python sentiment_pipeline data_file_path
```

Replace `data_file_path` with the file path to your data file of choice. Include the file name in this path, and remember that each data file must be prepended with the source of the data. The data file can also lie within a nested file structure, relative to the location of the `sentiment_pipeline.py` file. An example valid value for the command line argument is "data/reddit_combined_scored.csv".