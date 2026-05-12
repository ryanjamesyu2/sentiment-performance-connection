import pandas as pd
import numpy as np
import time
import re
import os
from datetime import datetime, timezone
from desearch_py import Desearch
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DESEARCH_API_KEY")
if not API_KEY:
    raise SystemExit("DESEARCH_API_KEY missing; add to .env")

desearch = Desearch(API_KEY)

OUTPUT_CSV = '../data/twitter_combined_season_results.csv'
CSV_COLUMNS = [
    "team", "player", "game_id", "game_date", "text", "created_at",
    "is_post_game", "likes", "retweets", "replies", "verified",
    "sentiment_categories", "engagement_score"
]

# Sentiment filters (SportsBERT keywords)
sentiment_filters = {
    "status": "MVP OR mvp OR star OR superstar OR special OR legend OR elite OR generational OR icon OR franchise QB OR franchise player OR top tier OR top-5 OR top 5 OR best in the league OR future star OR future of the franchise OR breakout star OR rising star OR fraud OR overrated OR underrated OR bust OR trash OR washed OR cooked OR liability OR overpaid OR waste of money OR not worth the contract",
    "clutch_anxiety": "clutch OR choked OR choke OR sold OR sold the game OR threw the game OR ice in his veins OR game on the line OR when it mattered OR big moment OR came up short OR disappeared OR folded OR no show OR quiet night OR showed up OR stepped up OR carried the team",
    "tactical": "bad read OR missed read OR missed the read OR checkdown OR check down OR forced it OR forced the throw OR tunnel vision OR stared down OR progressions OR went through his reads OR pocket presence OR play call OR playcalling OR scheme OR schemed OR game plan OR adjustment OR adjustments OR coaching decision OR bad play call OR execution OR poor execution OR mental mistake OR football IQ",
    "physicality": "injured OR injury OR hurt OR banged up OR limited OR on a snap count OR questionable OR doubtful OR out for the season OR IR OR season ending OR came back too early OR not 100% OR looks slow OR lost a step OR conditioning OR stamina OR gassed OR tired OR heavy legs OR effort OR lack of effort OR gave up OR body language OR looks checked out OR tough OR played through injury",
    "performance_eval": "great performance OR good performance OR bad performance OR poor performance OR standout performance OR career game OR career night OR breakout game OR solid game OR complete game OR dominant performance OR quiet game OR rough night OR awful game OR terrible game OR masterclass OR clinic OR looked great OR looks great OR looked good OR looks good OR looked lost OR looks lost OR looked shaky OR looks shaky OR locked in OR out of sync",
    "contract_value": "getting paid OR got paid OR payday OR contract OR extension OR max deal OR worth the money OR not worth it OR overpaid OR underpaid OR bargain OR steal OR cap hit OR salary cap OR dead cap OR restructure OR pay the man",
    "availability": "suspended OR fined OR ejected OR benched OR healthy scratch OR inactive OR missed time OR missed games OR absent OR traded OR trade rumors OR holding out OR holdout OR reinstated OR activated OR cleared to play"
}

# Team name aliases for search queries (keeps queries natural)
TEAM_ALIASES = {
    "Eagles":     "Eagles OR Philly",
    "Bills":      "Bills OR Buffalo",
    "Bengals":    "Bengals OR Cincinnati",
    "Colts":      "Colts OR Indianapolis",
    "Chiefs":     "Chiefs OR Kansas City",
    "Bears":      "Bears OR Chicago",
    "Buccaneers": "Buccaneers OR Bucs OR Tampa",
    "Seahawks":   "Seahawks OR Seattle",
    "Cowboys":    "Cowboys OR Dallas",
    "Patriots":   "Patriots OR New England"
}

# Full Configuration for 10 Teams
TEAMS_CONFIG = {
    "Eagles": {
        "players": {
            "Jalen Hurts": '"Jalen Hurts" OR @JalenHurts',
            "Saquon Barkley": '"Saquon Barkley" OR @saquon',
            "AJ Brown": '"AJ Brown" OR @1kalwaysopen_',
            "Quinyon Mitchell": '"Quinyon Mitchell" OR Quinyon',
            "Zack Baun": '"Zack Baun" OR Baun'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-04", "result": "W 24-20"}, {"id": "W2", "date": "2025-09-14", "result": "W 20-17"},
            {"id": "W3", "date": "2025-09-21", "result": "W 33-26"}, {"id": "W4", "date": "2025-09-28", "result": "W 31-25"},
            {"id": "W5", "date": "2025-10-05", "result": "L 17-21"}, {"id": "W6", "date": "2025-10-09", "result": "L 17-34"},
            {"id": "W7", "date": "2025-10-19", "result": "W 28-22"}, {"id": "W8", "date": "2025-10-26", "result": "W 38-20"},
            {"id": "W10", "date": "2025-11-10", "result": "W 10-7"}, {"id": "W11", "date": "2025-11-16", "result": "W 16-9"},
            {"id": "W12", "date": "2025-11-23", "result": "L 21-24"}, {"id": "W13", "date": "2025-11-28", "result": "L 15-24"},
            {"id": "W14", "date": "2025-12-08", "result": "L 19-22"}, {"id": "W15", "date": "2025-12-14", "result": "W 31-0"},
            {"id": "W16", "date": "2025-12-20", "result": "W 29-18"}, {"id": "W17", "date": "2025-12-28", "result": "W 13-12"},
            {"id": "W18", "date": "2026-01-04", "result": "L 17-24"}
        ]
    },
    "Bills": {
        "players": {
            "Josh Allen": '"Josh Allen" OR @JoshAllenQB',
            "James Cook": '"James Cook" OR @thegreat_4',
            "Dalton Kincaid": '"Dalton Kincaid" OR @_DaltonKincaid',
            "Matt Milano": '"Matt Milano" OR @MatthewMilanoo',
            "Greg Rousseau": '"Greg Rousseau" OR @gregrousseau'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "W 41-40"}, {"id": "W2", "date": "2025-09-14", "result": "W 30-10"},
            {"id": "W3", "date": "2025-09-18", "result": "W 31-21"}, {"id": "W4", "date": "2025-09-28", "result": "W 31-19"},
            {"id": "W5", "date": "2025-10-05", "result": "L 20-23"}, {"id": "W6", "date": "2025-10-13", "result": "L 14-24"},
            {"id": "W8", "date": "2025-10-26", "result": "W 40-9"},  {"id": "W9", "date": "2025-11-02", "result": "W 28-21"},
            {"id": "W10", "date": "2025-11-09", "result": "L 13-30"}, {"id": "W11", "date": "2025-11-16", "result": "W 44-32"},
            {"id": "W12", "date": "2025-11-20", "result": "L 19-23"}, {"id": "W13", "date": "2025-11-30", "result": "W 26-7"},
            {"id": "W14", "date": "2025-12-07", "result": "W 39-34"}, {"id": "W15", "date": "2025-12-14", "result": "W 35-31"},
            {"id": "W16", "date": "2025-12-21", "result": "W 23-20"}, {"id": "W17", "date": "2025-12-28", "result": "L 12-13"},
            {"id": "W18", "date": "2026-01-04", "result": "W 35-8"}
        ]
    },
    "Bengals": {
        "players": {
            "Joe Burrow": '"Joe Burrow" OR @JoeyB',
            "JaMarr Chase": '"JaMarr Chase" OR @Real10jayy__',
            "Tee Higgins": '"Tee Higgins" OR @teehiggins5',
            "Trey Hendrickson": '"Trey Hendrickson" OR Hendrickson',
            "Evan McPherson": '"Evan McPherson" OR @McPherson_Evan'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "W 17-16"}, {"id": "W2", "date": "2025-09-14", "result": "W 31-27"},
            {"id": "W3", "date": "2025-09-21", "result": "L 10-48"}, {"id": "W4", "date": "2025-09-29", "result": "L 3-28"},
            {"id": "W5", "date": "2025-10-05", "result": "L 24-37"}, {"id": "W6", "date": "2025-10-12", "result": "L 18-27"},
            {"id": "W7", "date": "2025-10-16", "result": "W 33-31"}, {"id": "W8", "date": "2025-10-26", "result": "L 38-39"},
            {"id": "W9", "date": "2025-11-02", "result": "L 42-47"}, {"id": "W11", "date": "2025-11-16", "result": "L 12-34"},
            {"id": "W12", "date": "2025-11-23", "result": "L 20-26"}, {"id": "W13", "date": "2025-11-27", "result": "W 32-14"},
            {"id": "W14", "date": "2025-12-07", "result": "L 34-39"}, {"id": "W15", "date": "2025-12-14", "result": "L 0-24"},
            {"id": "W16", "date": "2025-12-21", "result": "W 45-21"}, {"id": "W17", "date": "2025-12-28", "result": "W 37-14"},
            {"id": "W18", "date": "2026-01-04", "result": "L 18-20"}
        ]
    },
    "Colts": {
        "players": {
            "Daniel Jones": '"Daniel Jones" OR @Daniel_Jones10',
            "Jonathan Taylor": '"Jonathan Taylor" OR @JayT23',
            "Michael Pittman Jr": '"Michael Pittman Jr" OR @MikePitt_Jr',
            "Quenton Nelson": '"Quenton Nelson" OR @BigQ56',
            "Laiatu Latu": '"Laiatu Latu" OR @laiatu_latu'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "W 33-8"},  {"id": "W2", "date": "2025-09-14", "result": "W 29-28"},
            {"id": "W3", "date": "2025-09-21", "result": "W 41-20"}, {"id": "W4", "date": "2025-09-28", "result": "L 20-27"},
            {"id": "W5", "date": "2025-10-05", "result": "W 40-6"},  {"id": "W6", "date": "2025-10-12", "result": "W 31-27"},
            {"id": "W7", "date": "2025-10-19", "result": "W 38-24"}, {"id": "W8", "date": "2025-10-26", "result": "W 38-14"},
            {"id": "W9", "date": "2025-11-02", "result": "L 20-27"}, {"id": "W10", "date": "2025-11-09", "result": "W 31-25"},
            {"id": "W12", "date": "2025-11-23", "result": "L 20-23"}, {"id": "W13", "date": "2025-11-30", "result": "L 16-20"},
            {"id": "W14", "date": "2025-12-07", "result": "L 19-36"}, {"id": "W15", "date": "2025-12-14", "result": "L 16-18"},
            {"id": "W16", "date": "2025-12-22", "result": "L 27-48"}, {"id": "W17", "date": "2025-12-28", "result": "L 17-23"},
            {"id": "W18", "date": "2026-01-04", "result": "L 30-38"}
        ]
    },
    "Chiefs": {
        "players": {
            "Patrick Mahomes": '"Patrick Mahomes" OR @PatrickMahomes',
            "Travis Kelce": '"Travis Kelce" OR @tkelce',
            "Chris Jones": '"Chris Jones" OR @StoneColdJones',
            "Trent McDuffie": '"Trent McDuffie" OR trent_mcduffie',
            "Xavier Worthy": '"Xavier Worthy" OR @XavierWorthy'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-05", "result": "L 21-27"}, {"id": "W2", "date": "2025-09-14", "result": "L 17-20"},
            {"id": "W3", "date": "2025-09-21", "result": "W 22-9"},  {"id": "W4", "date": "2025-09-28", "result": "W 37-20"},
            {"id": "W5", "date": "2025-10-06", "result": "L 28-31"}, {"id": "W6", "date": "2025-10-12", "result": "W 30-17"},
            {"id": "W7", "date": "2025-10-19", "result": "W 31-0"},  {"id": "W8", "date": "2025-10-27", "result": "W 28-7"},
            {"id": "W9", "date": "2025-11-02", "result": "L 21-28"}, {"id": "W11", "date": "2025-11-16", "result": "L 19-22"},
            {"id": "W12", "date": "2025-11-23", "result": "W 23-20"}, {"id": "W13", "date": "2025-11-27", "result": "L 28-31"},
            {"id": "W14", "date": "2025-12-07", "result": "L 10-20"}, {"id": "W15", "date": "2025-12-14", "result": "L 13-16"},
            {"id": "W16", "date": "2025-12-21", "result": "L 9-26"},  {"id": "W17", "date": "2025-12-25", "result": "L 13-20"},
            {"id": "W18", "date": "2026-01-04", "result": "L 12-14"}
        ]
    },
    "Bears": {
        "players": {
            "Caleb Williams": '"Caleb Williams" OR @CALEBcsw',
            "DJ Moore": '"DJ Moore" OR @idjmoore',
            "Luther Burden": '"Luther Burden" OR @lutherburden3',
            "Rome Odunze": '"Rome Odunze" OR @RomeOdunze',
            "Jaylon Johnson": '"Jaylon Johnson" OR @NBAxJay1'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-08", "result": "L 24-27"}, {"id": "W2", "date": "2025-09-14", "result": "L 21-52"},
            {"id": "W3", "date": "2025-09-21", "result": "W 31-14"}, {"id": "W4", "date": "2025-09-28", "result": "W 25-24"},
            {"id": "W6", "date": "2025-10-13", "result": "W 25-24"}, {"id": "W7", "date": "2025-10-19", "result": "W 26-14"},
            {"id": "W8", "date": "2025-10-26", "result": "L 16-30"}, {"id": "W9", "date": "2025-11-02", "result": "W 47-42"},
            {"id": "W10", "date": "2025-11-09", "result": "W 24-20"}, {"id": "W11", "date": "2025-11-16", "result": "W 19-17"},
            {"id": "W12", "date": "2025-11-23", "result": "W 31-28"}, {"id": "W13", "date": "2025-11-28", "result": "W 24-15"},
            {"id": "W14", "date": "2025-12-07", "result": "L 21-28"}, {"id": "W15", "date": "2025-12-14", "result": "W 31-3"},
            {"id": "W16", "date": "2025-12-20", "result": "W 22-16"}, {"id": "W17", "date": "2025-12-28", "result": "L 38-42"},
            {"id": "W18", "date": "2026-01-04", "result": "L 16-19"}
        ]
    },
    "Buccaneers": {
        "players": {
            "Baker Mayfield": '"Baker Mayfield" OR @bakermayfield',
            "Mike Evans": '"Mike Evans" OR @MikeEvans13_',
            "Chris Godwin": '"Chris Godwin" OR @CGtwelve_',
            "Antoine Winfield Jr": '"Antoine Winfield Jr" OR @AntoineWJr11',
            "Vita Vea": '"Vita Vea" OR @VitaVea'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "W 23-20"}, {"id": "W2", "date": "2025-09-15", "result": "W 20-19"},
            {"id": "W3", "date": "2025-09-21", "result": "W 29-27"}, {"id": "W4", "date": "2025-09-28", "result": "L 25-31"},
            {"id": "W5", "date": "2025-10-05", "result": "W 38-35"}, {"id": "W6", "date": "2025-10-12", "result": "W 30-19"},
            {"id": "W7", "date": "2025-10-20", "result": "L 9-24"},  {"id": "W8", "date": "2025-10-26", "result": "W 23-3"},
            {"id": "W10", "date": "2025-11-09", "result": "L 23-28"}, {"id": "W11", "date": "2025-11-16", "result": "L 32-44"},
            {"id": "W12", "date": "2025-11-23", "result": "L 7-34"},  {"id": "W13", "date": "2025-11-30", "result": "W 20-17"},
            {"id": "W14", "date": "2025-12-07", "result": "L 20-24"}, {"id": "W15", "date": "2025-12-11", "result": "L 28-29"},
            {"id": "W16", "date": "2025-12-21", "result": "L 20-23"}, {"id": "W17", "date": "2025-12-28", "result": "L 17-20"},
            {"id": "W18", "date": "2026-01-03", "result": "W 16-14"}
        ]
    },
    "Seahawks": {
        "players": {
            "Sam Darnold": '"Sam Darnold" OR Darnold',
            "Jaxon Smith-Njigba": '"Jaxon Smith-Njigba" OR @jaxon_smith1',
            "Tariq Woolen": '"Tariq Woolen" OR @_Tariqwoolen',
            "Kenneth Walker III": '"Kenneth Walker III" OR @Kenneth_Walker9',
            "Devon Witherspoon": '"Devon Witherspoon" OR @DevonWitherspo1'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "L 13-17"}, {"id": "W2", "date": "2025-09-14", "result": "W 31-17"},
            {"id": "W3", "date": "2025-09-21", "result": "W 44-13"}, {"id": "W4", "date": "2025-09-25", "result": "W 23-20"},
            {"id": "W5", "date": "2025-10-05", "result": "L 35-38"}, {"id": "W6", "date": "2025-10-12", "result": "W 20-12"},
            {"id": "W7", "date": "2025-10-20", "result": "W 27-19"}, {"id": "W9", "date": "2025-11-02", "result": "W 38-14"},
            {"id": "W10", "date": "2025-11-09", "result": "W 44-22"}, {"id": "W11", "date": "2025-11-16", "result": "L 19-21"},
            {"id": "W12", "date": "2025-11-23", "result": "W 30-24"}, {"id": "W13", "date": "2025-11-30", "result": "W 26-0"},
            {"id": "W14", "date": "2025-12-07", "result": "W 37-9"},  {"id": "W15", "date": "2025-12-14", "result": "W 18-16"},
            {"id": "W16", "date": "2025-12-18", "result": "W 38-37"}, {"id": "W17", "date": "2025-12-28", "result": "W 27-10"},
            {"id": "W18", "date": "2026-01-03", "result": "W 13-3"}
        ]
    },
    "Cowboys": {
        "players": {
            "Dak Prescott": '"Dak Prescott" OR @dak',
            "CeeDee Lamb": '"CeeDee Lamb" OR @_CeeDeeThree',
            "Quinnen Williams": '"Quinnen Williams" OR @QuinnenW22',
            "Javonte Williams": '"Javonte Williams" OR @javontewill33',
            "George Pickens": '"George Pickens" OR @_GeorgePickens'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-04", "result": "L 20-24"}, {"id": "W2", "date": "2025-09-14", "result": "W 40-37"},
            {"id": "W3", "date": "2025-09-21", "result": "L 14-31"}, {"id": "W4", "date": "2025-09-28", "result": "T 40-40"},
            {"id": "W5", "date": "2025-10-05", "result": "W 37-22"}, {"id": "W6", "date": "2025-10-12", "result": "L 27-30"},
            {"id": "W7", "date": "2025-10-19", "result": "W 44-22"}, {"id": "W8", "date": "2025-10-26", "result": "L 24-44"},
            {"id": "W9", "date": "2025-11-03", "result": "L 17-27"}, {"id": "W11", "date": "2025-11-17", "result": "W 33-16"},
            {"id": "W12", "date": "2025-11-23", "result": "W 24-21"}, {"id": "W13", "date": "2025-11-27", "result": "W 31-28"},
            {"id": "W14", "date": "2025-12-04", "result": "L 30-44"}, {"id": "W15", "date": "2025-12-14", "result": "L 26-34"},
            {"id": "W16", "date": "2025-12-21", "result": "L 17-34"}, {"id": "W17", "date": "2025-12-25", "result": "W 30-23"},
            {"id": "W18", "date": "2026-01-04", "result": "L 17-34"}
        ]
    },
    "Patriots": {
        "players": {
            "Drake Maye": '"Drake Maye" OR @DrakeMaye2',
            "TreVeyon Henderson": '"TreVeyon Henderson" OR Henderson',
            "Stefon Diggs": '"Stefon Diggs" OR Diggs',
            "Christian Gonzalez": '"Christian Gonzalez" OR @chrisgonzo28',
            "Kayshon Boutte": '"Kayshon Boutte" OR @KayshonBoutte1'
        },
        "schedule": [
            {"id": "W1", "date": "2025-09-07", "result": "L 13-20"}, {"id": "W2", "date": "2025-09-14", "result": "W 33-27"},
            {"id": "W3", "date": "2025-09-21", "result": "L 14-21"}, {"id": "W4", "date": "2025-09-28", "result": "W 42-13"},
            {"id": "W5", "date": "2025-10-05", "result": "W 23-20"}, {"id": "W6", "date": "2025-10-12", "result": "W 25-19"},
            {"id": "W7", "date": "2025-10-19", "result": "W 31-13"}, {"id": "W8", "date": "2025-10-26", "result": "W 32-13"},
            {"id": "W9", "date": "2025-11-02", "result": "W 24-23"}, {"id": "W10", "date": "2025-11-09", "result": "W 28-23"},
            {"id": "W11", "date": "2025-11-13", "result": "W 27-14"}, {"id": "W12", "date": "2025-11-23", "result": "W 26-20"},
            {"id": "W13", "date": "2025-12-01", "result": "W 33-15"}, {"id": "W15", "date": "2025-12-14", "result": "L 31-35"},
            {"id": "W16", "date": "2025-12-21", "result": "W 28-24"}, {"id": "W17", "date": "2025-12-28", "result": "W 42-10"},
            {"id": "W18", "date": "2026-01-04", "result": "W 38-10"}
        ]
    }
}

def categorize_sentiment(text):
    if pd.isna(text) or text == '':
        return []
    text_lower = text.lower()
    tags = []
    for category, terms_string in sentiment_filters.items():
        terms = [term.strip().lower() for term in terms_string.split(' OR ')]
        matched = False
        for term in terms:
            pattern = r'\b' + re.escape(term).replace(r'\ ', r'\s+') + r'\b'
            if re.search(pattern, text_lower):
                matched = True
                break
        if matched:
            tags.append(category)
    return tags

def build_dataset():
    all_records = []
    # Initialize file with header
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Created {OUTPUT_CSV} (appending results per player)\n")

    for team_name, config in TEAMS_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"--- PROCESSING TEAM: {team_name} ---")
        print(f"{'='*50}")

        team_alias = TEAM_ALIASES[team_name]

        for game in config['schedule']:
            game_dt = pd.to_datetime(game['date']).tz_localize('UTC')
            start_str = (game_dt - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
            end_str   = (game_dt + pd.Timedelta(days=3)).strftime('%Y-%m-%d')

            for p_name, p_query in config['players'].items():
                query = f"({p_query}) ({team_alias}) -filter:links"

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
                        current_batch = []
                        for post in results:
                            post_time = pd.to_datetime(post.get('created_at'))
                            if post_time.tzinfo is None:
                                post_time = post_time.tz_localize('UTC')

                            text = post.get('text', '')
                            tags = categorize_sentiment(text)

                            record = {
                                "team": team_name,
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
                            current_batch.append(record)
                            all_records.append(record)

                        # Save batch to CSV
                        if current_batch:
                            batch_df = pd.DataFrame(current_batch)
                            batch_df['engagement_score'] = (
                                np.log2(batch_df['likes'] + 1) +
                                (2 * np.log2(batch_df['retweets'] + 1)) +
                                (0.5 * np.log2(batch_df['replies'] + 1))
                            ).round(2)
                            batch_df[CSV_COLUMNS].to_csv(
                                OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8"
                            )
                    else:
                        print('No results found')

                    time.sleep(0.2)  # Respect rate limits

                except Exception as e:
                    print(f"Error processing {p_name} on {game['id']}: {e}")
                    import traceback
                    traceback.print_exc()

    return pd.DataFrame(all_records)


# Execute
print("Starting Season 2025-26 Data Expansion...")
master_df = build_dataset()

if not master_df.empty:
    print(f"\nSUCCESS: Collected {len(master_df)} total posts.")
    print("\nTop 5 Teams by Volume:")
    print(master_df['team'].value_counts().head())
    print("\nSentiment category distribution:")
    print(master_df['sentiment_categories'].value_counts().head(10))
else:
    print("No data was collected. Check your API key and query windows.")