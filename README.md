# The Sentiment Performance Connection
MADS '26 capstone project repository. All code developed by Anirudh Karanam, Steve Wong, and Ryan Yu. Project completed in partnership with BTA Sports, LLC. Developed using the anaconda python distribution, and utilizes conda virtual environments to ensure consistency when running code. 

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
* statsmodels

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

The Reddit data was collected via a Selenium driven scraper. The code can be found in the `reddit_home_away.ipynb` notebook. The data was collected from r/NFL post game threads. 

The Reddit data must have at least 5 columns:
* "post_title", which contains a string of the form "Post Game Thread: Washington Commanders at Philadelphia Eagles" (mimics the official post title on Reddit)
* "home_team", which contains a string of the home team for the game
* "away team", which contains a string of the away team for the game
* "score", which contains a numerical measurement of comment engagement and is used to calculate the weight for each tweet 
* "predicted_tag", which is either "home_team", "away_team", or "unsure". This is a tag to attribute sentiment to the correct team, and is generated using the code in the `nfl_team_reference_classifier.py` file.

Reddit data is also initially weighted by an engagement score, which is a function of upvotes and downvotes of the comment. The scraping of Reddit was originally developed and ran on a Mac. If you are using a different operating system, slight adjustments might need to be made to the code base. 

### Google News

Google News data was also collected via a Selenium scraper. The input data file is expecting one row in the file per article, rather than 1 per sentence. The pipeline automatically splits the articles into sentences. The .csv file must contain at least 4 columns:
* "team", which contains a string of the team name
* "game_no", which contains a integer denoting which game week the NFL game was a part of (ranging from 1 to 18)
* "text_body", which contains a string of the entire article contents
* "title", which contains a string of the article title and is used to group sentences when calculating weights

In sprint 1, Google News data is weighted by the sentences' location within the article. Research shows that the majority of readers do not read the entire article, and many skim for just headlines. So, we weight sentences earlier in the article (which readers are more likely to see) with higher weights. Weights follow an exponential decaying pattern, with the first 25% of sentences comprising 75% of the total weight for the article.

Similarly to Reddit, the code was developed and ran on a Mac. Slight adjustments may be necessary if using a Windows or Linux machine.

## Using the Pipeline to Create Sentiment Indices

The pipeline currently is limited to one data file at a time, with the data file containing data from only one data source. First, activate the `sentiment` virtual environment using the command:

```
conda activate sentiment
```

Make sure to create a folder called `sentiment_indices` if one does not already exist to store the resulting files, or change the file paths to direct the outputs to your desired locations. The rest of this document will assume you are using the default locations. Then, in the terminal, run one simple command to kick off the entire pipeline.

```
python sentiment_pipeline data_file_path
```

Replace `data_file_path` with the file path to your data file of choice. Include the file name in this path, and remember that each data file must be prepended with the source of the data. The data file can also lie within a nested file structure, relative to the location of the `sentiment_pipeline.py` file. An example valid value for the command line argument is "data/reddit_combined_scored.csv".

Following a successful run, two files will be created. First, in the same directory as the `sentiment_pipeline.py` file, a `[source]_sentiment_scores.csv` file will be created. This file contains the subject of the text, the game ID (referring to week in the season), the text itself, and the sentiment score assigned to that individual text. Sentiment scores are created using an expected value approach from the results of a pre-trained HuggingFace model. The information for the models used for each data source can be found in the `pipeline_configs.py` file. Any helper functions used are stored in the `pipeline_utils.py` module. For a given text, the model returns the probability of the text belonging to three sentiment classes: positive, neutral, and negative. In order to create a sentiment score that captures the strength of the sentiment, we create sentiment scores using the following formula:

$$
sentiment\_score = P(positive) - P(negative)  
$$

The second file created is stored as `sentiment_indices/[source]_local_index.csv` file, and contains the aggregated sentiment index for each subject/week combination. Each text is weighted according to some metadata variable, such as engagement or time. Then, we take the weighted average within a subject/week combination. Lastly, within a given week, we scale the weighted averages for players to be on a 0-100 scale for interpretability.

After you have run the pipeline for all 3 data sources and have all 3 local indices, you can aggregate them to a global index using the `create_global_index.py` script. An example call to this script is as follows:

```
python create_global_index file1 w1 file2 w2 file3 w3
```

In this example call,

* file# is the file path to the a local index CSV file produced by the aforementioned pipeline
* w# is the relative weight given to that local index in the weighted average

The weights are all relative, so $w1=2, w2=1, w3=1$ will produce the same results as $w1=.5,w2=.25,w3=.25$. Additionally, the script is not limited to only 3 data sources. In the future, if more data sources are added, simply append more filepath/weight pairs in the command line arguments.

For the global sentiment index stored in the `sentiment_indices/global_index.csv` file in this repository, we aggregated the local indices in the `sentiment_indices/twitter_local_index_sprint2.csv`, `sentiment_indices/reddit_local_index.csv`, and `sentiment_indices/google_local_index.csv` files using a 50%/25%/25% weighting scheme.

## Analysis

### Sentiment/Performance Correlation Analysis

The main goal of the project was to determine whether any correlation exists between our created sentiment indices and the observed athletic performance of the players. To do so, we performed a series of correlation analyses.

For the Twitter and Global indices, we started by performing correlation analyses at the player level. For each relevant combination of player and athletic performance metric, we calculated three types of correlation: 

* Pearson
* Spearman
* Distance

Pearson correlation measures the strength and direction of linear correlation between the two variables, Spearman correlation measures the strength and direction of the ranks of the two variables, and Distance Correlation measures the strength of any (not necessarily linear or monotonic) correlation between the two variables. These correlation analyses were performed at 3 different time offsets.

* Concurrent (Week X sentiment and Week X performance)
* Sentiment leads (Week X+1 sentiment and Week X performance)
* Performance leads (Week X sentiment and Week X+1 performance)

Of these, performance leading is of the most relevance and importance. In additional to performing correlation on an individual player level, we also performed the correlation on the position-group level (QBs, WRs, RBs, WRs, DEF).

For the Reddit and Google indices, we perform similar analyses, except we perform them on the team level. This is because by the nature of how data was collected and organized, Reddit and Google have sentiment index values for each team/week combination rather than player/week combination. Other than this difference, the correlation analysis was performed in the same way as the Twitter and Global indices.

### High Impact Player Deep Dives

One hypothesis is that superstar players have a different relationship between their sentiment index values and athletic performance when compared to the overarching population. To test this hypothesis, we focused on 6 specific players:

* Jalen Hurts
* Josh Allen
* Baker Mayfield
* AJ Brown
* JaMarr Chase
* George Pickens

For each player, we first look at concurrent correlation. After standardizing each time series to be on the same scale, we plot the sentiment index against all relevant athletic performance metrics to get a visual representation of the correlations. We also create a table of correlations (Pearson, Spearman, and Distance) between each statistic and both the Twitter and Global indices (since these are the indices stored at the player level).

Then, we repeat the process using performance leads correlation (that is, correlation Week X sentiment with Week X+1 performance). Our findings indicate that there is no overarching pattern between these superstars. There is a range of players with negative, close to 0, and positive correlations. However, we did notice that the two Eagles players, Jalen Hurts and AJ Brown, both had fairly strong negative correlations, motivating a different hypothesis that relationship between sentiment and performance may actually be a team level effect. 

### PPR Fantasy Point Regression Model

Not only do we want to investigate the correlation between players' sentiment and atheltic performance, but we also wish to determine if sentiment has any predictive value for academic performance. To test, we fit a linear regression model, predicting PPR fantasy points as a function of:

* Previous week's sentiment index
* Previous week's PPR fantasy points
* Opposing DVOA (a measure of opponent's defensive strength)
* Position group (QB, RB, WR, TE)

We use fantasy points as the response variable, as it is the only atheltic performance metric we have access to that is common across all position groups. While it is not a direct measure of athletic performance, PPR fantasy points is a deterministic function of other direct measures. Additionally, we filter our data set to only offensive players, as most fantasy football leagues do not have individual fantasy point values for defensive players (but rather 1 for the team's defensive unit as a whole).

We first fit a baseline model, which does not include any sentiment index as a predictor. After, we fit two different regression models, one adding the Twitter index and the other adding the Global index. We use the previous week's sentiment as the predictor so that we can test if the sentiment will be able to forecast/predict a player's performance in the next game. Then, using these 3 regression models, we perform 2 partial F-Tests, one comparing the baseline model to the Twitter model and one comparing the baseline model to the Global model. The results of these tests will inform us if sentiment provides any additional predictive power after accounting for the other predictors.

We find that the Twitter index alone provides no additional predictive power, but the Global index provides a small amount of additional predictive power. Holding all other variables constant, each additional point in the previous week's sentiment index on average leads to that player scoring 0.05 *fewer* fantasy points.

We also fit two more regression models and perform an additional partial F-Test to test the hypothesis that each team has a different relationship between fantasy points and sentiment. Our new baseline model for this test is has the following predictors:

* Previous week's global sentiment index
* Previous week's PPR fantasy points
* Opposing DVOA (a measure of opponent's defensive strength)
* Position group (QB, RB, WR, TE)
* Team

Then, for the full model, we add an interaction term between the team and previous week's global sentiment index. Then, the partial F-Test of these models informs us if any 2 teams have a statistically different coefficient for the sentiment term. The results of the test did not provide enough evidence to suggest that teams have different relationships.
