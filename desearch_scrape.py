import pandas as pd
import time
from datetime import datetime, timezone
from desearch_py import Desearch
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('DESEARCH_API_KEY')
desearch = Desearch(API_KEY)

# using keywords from sportsBERT model
sentiment_filters = {
    # overall player quality / public perception
    "status": (
        "MVP OR mvp OR star OR superstar OR special OR legend OR elite OR generational OR icon OR "
        "franchise QB OR franchise player OR top tier OR top-5 OR top 5 OR best in the league OR "
        "future star OR future of the franchise OR breakout star OR rising star OR "
        "fraud OR overrated OR underrated OR bust OR trash OR washed OR cooked OR "
        "liability OR overpaid OR waste of money OR not worth the contract"
    ),
    # in-game pressure, performance in key moments, narrative language
    "clutch_anxiety": (
        "clutch OR choked OR choke OR sold OR sold the game OR threw the game OR "
        "ice in his veins OR game on the line OR when it mattered OR big moment OR "
        "came up short OR disappeared OR folded OR no show OR quiet night OR "
        "showed up OR stepped up OR carried the team"
    ),
    # tactical / decision making / execution
    "tactical": (
        "bad read OR missed read OR missed the read OR checkdown OR check down OR "
        "forced it OR forced the throw OR tunnel vision OR stared down OR "
        "progressions OR went through his reads OR pocket presence OR "
        "play call OR playcalling OR scheme OR schemed OR game plan OR "
        "adjustment OR adjustments OR coaching decision OR bad play call OR "
        "execution OR poor execution OR mental mistake OR football IQ"
    ),
    # physical condition, availability, effort, body-language
    "physicality": (
        "injured OR injury OR hurt OR banged up OR limited OR on a snap count OR "
        "questionable OR doubtful OR out for the season OR IR OR season ending OR "
        "came back too early OR not 100% OR looks slow OR lost a step OR "
        "conditioning OR stamina OR gassed OR tired OR heavy legs OR "
        "effort OR lack of effort OR gave up OR body language OR looks checked out OR "
        "tough OR played through injury"
    ),

    # performance quality
    "performance_eval": (
        "great performance OR good performance OR bad performance OR poor performance OR "
        "standout performance OR career game OR career night OR breakout game OR "
        "solid game OR complete game OR dominant performance OR quiet game OR "
        "rough night OR awful game OR terrible game OR masterclass OR clinic OR "
        "looked great OR looks great OR looked good OR looks good OR "
        "looked lost OR looks lost OR looked shaky OR looks shaky OR "
        "locked in OR out of sync"
    ),
    # contract / money / value
    "contract_value": (
        "getting paid OR got paid OR payday OR contract OR extension OR max deal OR "
        "worth the money OR not worth it OR overpaid OR underpaid OR bargain OR steal OR "
        "cap hit OR salary cap OR dead cap OR restructure OR pay the man"
    ),
    # availability / discipline / off-field issues
    "availability": (
        "suspended OR fined OR ejected OR benched OR healthy scratch OR inactive OR "
        "missed time OR missed games OR absent OR traded OR trade rumors OR "
        "holding out OR holdout OR reinstated OR activated OR cleared to play"
    )
}

players = {
    "Jalen Hurts": '"Jalen Hurts" OR @JalenHurts',
    "Saquon Barkley": '"Saquon Barkley" OR @saquon',
    "AJ Brown": '"AJ Brown" OR @1kalwaysopen_',
    "Quinyon Mitchell": '"Quinyon Mitchell" OR Quinyon',
    "Zack Baun": '"Zack Baun" OR Baun'
}

# schedule for Eagles 2025-26 season
game_schedule = [
    {"id": "W1_Cowboys", "date": "2025-09-04", "result": "W 24-20", "home_away": "Home"},
    {"id": "W2_Chiefs", "date": "2025-09-14", "result": "W 20-17", "home_away": "Away"},
    {"id": "W3_Rams", "date": "2025-09-21", "result": "W 33-26", "home_away": "Home"},
    {"id": "W4_Buccaneers", "date": "2025-09-28", "result": "W 31-25", "home_away": "Away"},
    {"id": "W5_Broncos", "date": "2025-10-05", "result": "L 17-21", "home_away": "Home"},
    {"id": "W6_Giants", "date": "2025-10-09", "result": "L 17-34", "home_away": "Away"},
    {"id": "W7_Vikings", "date": "2025-10-19", "result": "W 28-22", "home_away": "Away"},
    {"id": "W8_Giants2", "date": "2025-10-26", "result": "W 38-20", "home_away": "Home"},
    {"id": "W10_Packers", "date": "2025-11-10", "result": "W 10-7", "home_away": "Away"},
    {"id": "W11_Lions", "date": "2025-11-16", "result": "W 16-9", "home_away": "Home"},
    {"id": "W12_Cowboys2", "date": "2025-11-23", "result": "L 21-24", "home_away": "Away"},
    {"id": "W13_Bears", "date": "2025-11-28", "result": "L 15-24", "home_away": "Home"},
    {"id": "W14_Chargers", "date": "2025-12-08", "result": "L 19-22", "home_away": "Away"},
    {"id": "W15_Raiders", "date": "2025-12-14", "result": "W 31-0", "home_away": "Home"},
    {"id": "W16_Commanders", "date": "2025-12-20", "result": "W 29-18", "home_away": "Away"},
    {"id": "W17_Bills", "date": "2025-12-28", "result": "W 13-12", "home_away": "Away"},
    {"id": "W18_Commanders2", "date": "2026-01-04", "result": "L 17-24", "home_away": "Home"},
]

OUTPUT_CSV = 'eagles_sentiment_2025_26_expanded_all_games.csv'
CSV_COLUMNS = [
    "player", "game_id", "game_date", "text", "created_at",
    "is_post_game", "likes", "retweets", "replies", "verified",
    "sentiment_categories", "engagement_score"
]

# categorize sentiment using keywords from sportsBERT model
def categorize_sentiment(text):
    """Categorize tweet text into sentiment categories using word boundaries"""
    if pd.isna(text) or text == '':
        return []
    
    text_lower = text.lower()
    tags = []
    
    for category, terms_string in sentiment_filters.items():
        # Split by OR and clean up terms
        terms = [term.strip().lower() for term in terms_string.split(' OR ')]
        
        # Check if any term appears in the text using word boundaries
        matched = False
        for term in terms:
            # Escape special regex characters and replace spaces with \s+ for flexible matching
            pattern = r'\b' + re.escape(term).replace(r'\ ', r'\s+') + r'\b'
            if re.search(pattern, text_lower):
                matched = True
                break
        
        if matched:
            tags.append(category)
    
    return tags

# build dataset for Eagles 2025-26 season
def build_dataset():
    all_records = []
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(OUTPUT_CSV, index=False)
    print(f"Created {OUTPUT_CSV} (appending results per player)\n")

    for game in game_schedule:
        game_dt = pd.to_datetime(game['date']).tz_localize('UTC')
        # Expanded time window: 3 days before to 3 days after
        start_str = (game_dt - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        end_str = (game_dt + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        
        for p_name, p_query in players.items():
            # Simplified query: no sentiment filtering in query
            query = f"({p_query}) (Eagles OR Philly) -filter:links"
            
            print(f'\nQuerying {p_name} for {game["id"]}')
            print(f'Query: {query}')
            print(f'Date range: {start_str} to {end_str}')
            
            try:
                results = desearch.basic_twitter_search(
                    query=query,
                    sort='Top',
                    start_date=start_str,
                    end_date=end_str,
                    min_likes=2
                )
                
                if results:
                    print(f'Found {len(results)} posts')
                    player_records = []
                    for post in results:
                        post_time = pd.to_datetime(post.get('created_at'))
                        if post_time.tzinfo is None:
                            post_time = post_time.tz_localize('UTC')
                        
                        text = post.get('text', '')
                        
                        # Categorize sentiment in post-processing
                        tags = categorize_sentiment(text)
                        
                        record = {
                            "player": p_name,
                            "game_id": game['id'],
                            "game_date": game['date'],
                            "text": text,
                            "created_at": post_time,
                            "is_post_game": post_time > game_dt,
                            "likes": post.get('like_count', 0),
                            "retweets": post.get('retweet_count', 0),
                            "replies": post.get('reply_count', 0),
                            "verified": post.get('verified', False),
                            "sentiment_categories": ",".join(tags) if tags else "general"
                        }
                        player_records.append(record)
                        all_records.append(record)
                    # Append player's results to the CSV file
                    chunk_df = pd.DataFrame(player_records)
                    chunk_df['engagement_score'] = chunk_df['likes'] + (chunk_df['retweets'] * 2) + (chunk_df['replies'] * 1.5)
                    chunk_df[CSV_COLUMNS].to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                else:
                    print('No results found')

                # Needed for rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing {p_name} on {game['id']}: {e}")
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(all_records)

    if not df.empty:
        # Create engagement score - look into tweaking formula
        df['engagement_score'] = df['likes'] + (df['retweets'] * 2) + (df['replies'] * 1.5)
        df = df.sort_values(by=['player', 'created_at'])

    return df


print("Starting data collection...")
eagles_df = build_dataset()

if not eagles_df.empty:
    print(f"Total Rows Collected: {len(eagles_df)}")

    print("Posts per player:")
    print(eagles_df['player'].value_counts())
    print("\nSentiment category distribution:")
    print(eagles_df['sentiment_categories'].value_counts().head(10))

    print(f"\nData written to {OUTPUT_CSV} (appended per player)")
else:
    print("No data collected")
