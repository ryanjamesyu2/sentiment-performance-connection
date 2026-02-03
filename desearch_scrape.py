import pandas as pd
import time
from datetime import datetime, timezone
from desearch_py import Desearch
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('DESEARCH_API_KEY')
desearch = Desearch(API_KEY)

sentiment_filters = {
    "status": "(MVP OR elite OR generational OR fraud OR washed OR cooked OR overpaid OR bust OR trash)",
    "clutch_anxiety": "(choke OR 'ice in his veins' OR clutch OR 'game on the line' OR shaky OR nervous)",
    "tactical": "(scheme OR 'play call' OR 'bad read' OR adjustment OR 'check down' OR 'tunnel vision')",
    "physicality": "(effort OR 'body language' OR limp OR 'looks slow' OR tough OR injury OR conditioning)"
}

# sample players to test queries
players = {
    "Jalen Hurts": '("Jalen Hurts" OR @JalenHurts OR "QB1")',
    "Saquon Barkley": '("Saquon Barkley" OR @saquon OR "Saquon")',
    "AJ Brown": '("AJ Brown" OR @11AJB)',
    "Quinyon Mitchell": '("Quinyon Mitchell" OR "Quinyon")',
    "Zack Baun": '("Zack Baun" OR "Baun")'
}

# TODO: Include emojis in search queries using inbuilt desearch functionality

# schedule for 2025-26 season
# TODO: Add all 17 regular season games and results
# TODO: Modify to make this more modular and reusable for all teams
game_schedule = [
    {"id": "W1_Cowboys", "date": "2025-09-04"},
    {"id": "W2_Chiefs", "date": "2025-09-14"},
    {"id": "W6_Giants", "date": "2025-10-09"},
    {"id": "W7_Vikings", "date": "2025-10-19"},
    # {"id": "W12_Cowboys", "date": "2025-11-23"},
    # {"id": "W13_Bears", "date": "2025-11-28"},
    # {"id": "WC_Niners", "date": "2026-01-11"}
]


def build_dataset():
    all_records = []

    combined_sentiment_query = f"({' OR '.join(sentiment_filters.values())})"

    for game in game_schedule:
        game_dt = pd.to_datetime(game['date']).tz_localize('UTC')
        start_str = (game_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end_str = (game_dt + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

        for p_name, p_query in players.items():
            # TODO: Accept team name as a parameter
            query = f"{p_query} (Eagles OR Philly) {combined_sentiment_query} -filter:links"

            print(f"Querying {p_name} for {game['id']}...")

            try:
                # getting 'Top' posts to ensure high engagement
                results = desearch.basic_twitter_search(
                    query=query,
                    sort='Top',
                    start_date=start_str,
                    end_date=end_str,
                    min_likes=2
                )

                if results:
                    for post in results:
                        post_time = pd.to_datetime(post.get('created_at'))
                        if post_time.tzinfo is None:
                            post_time = post_time.tz_localize('UTC')

                        text = post.get('text', '').lower()
                        tags = [k for k, v in sentiment_filters.items() if any(word.strip("'()") in text for word in v.split(' OR '))]

                        record = {
                            "player": p_name,
                            "game_id": game['id'],
                            "text": post.get('text'),
                            "created_at": post_time,
                            "is_post_game": post_time > game_dt,
                            "likes": post.get('like_count', 0),
                            "retweets": post.get('retweet_count', 0),
                            "replies": post.get('reply_count', 0),
                            "verified": post.get('verified', False),
                            "sentiment_categories": ",".join(tags) if tags else "general"
                        }
                        all_records.append(record)

                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing {p_name} on {game['id']}: {e}")

    df = pd.DataFrame(all_records)

    if not df.empty:
        # Created an engagement score to measure virality
        df['engagement_score'] = df['likes'] + (df['retweets'] * 2) + (df['replies'] * 1.5)
        df = df.sort_values(by=['player', 'created_at'])

    return df


eagles_df = build_dataset()

if not eagles_df.empty:
    print(f"Total Rows: {len(eagles_df)}")
    print(eagles_df[['player', 'game_id', 'sentiment_categories', 'engagement_score']].head(10))
    eagles_df.to_csv('eagles_sentiment_2025_26.csv', index=False)
else:
    print("No data collected")