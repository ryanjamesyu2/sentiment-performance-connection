#!/usr/bin/env python3
"""Rule-based home vs away team labels for Reddit game-thread comments."""

import argparse
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


# ----------------------------
# 1) Team alias dictionary
# ----------------------------
TEAM_ALIASES: Dict[str, List[str]] = {
    "Arizona Cardinals": ["arizona", "cardinals", "cards", "az"],
    "Atlanta Falcons": ["atlanta", "falcons", "dirty birds", "atl"],
    "Baltimore Ravens": ["baltimore", "ravens", "ravens flock", "bal"],
    "Buffalo Bills": ["buffalo", "bills", "bills mafia", "buf"],
    "Carolina Panthers": ["carolina", "panthers", "cats", "car"],
    "Chicago Bears": ["chicago", "bears", "da bears", "chi"],
    "Cincinnati Bengals": ["cincinnati", "bengals", "who dey", "cin"],
    "Cleveland Browns": ["cleveland", "browns", "dawg pound", "cle"],
    "Dallas Cowboys": ["dallas", "cowboys", "boys", "america's team", "americas team", "dal"],
    "Denver Broncos": ["denver", "broncos", "broncos country", "den"],
    "Detroit Lions": ["detroit", "lions", "one pride", "det"],
    "Green Bay Packers": ["green bay", "packers", "pack", "go pack go", "gb", "gpg"],
    "Houston Texans": ["houston", "texans", "htown", "hou"],
    "Indianapolis Colts": ["indianapolis", "indy", "colts", "ind"],
    "Jacksonville Jaguars": ["jacksonville", "jaguars", "jags", "jax", "duval"],
    "Kansas City Chiefs": ["kansas city", "chiefs", "kc", "chiefs kingdom"],
    "Las Vegas Raiders": ["las vegas", "raiders", "raiders nation", "lv"],
    "Los Angeles Chargers": ["los angeles chargers", "chargers", "bolts", "lac"],
    "Los Angeles Rams": ["los angeles rams", "rams", "la rams", "lar"],
    "Miami Dolphins": ["miami", "dolphins", "fins", "phins", "mia"],
    "Minnesota Vikings": ["minnesota", "vikings", "vikes", "skol", "min"],
    "New England Patriots": ["new england", "patriots", "pats", "ne"],
    "New Orleans Saints": ["new orleans", "saints", "who dat", "no"],
    "New York Giants": ["new york giants", "giants", "gmen", "nyg"],
    "New York Jets": ["new york jets", "jets", "gang green", "nyj"],
    "Philadelphia Eagles": ["philadelphia", "philly", "eagles", "birds", "go birds", "phi"],
    "Pittsburgh Steelers": ["pittsburgh", "steelers", "steel curtain", "pit"],
    "San Francisco 49ers": ["san francisco", "49ers", "niners", "9ers", "sf"],
    "Seattle Seahawks": ["seattle", "seahawks", "hawks", "12s", "sea"],
    "Tampa Bay Buccaneers": ["tampa bay", "buccaneers", "bucs", "tb"],
    "Tennessee Titans": ["tennessee", "titans", "titan up", "ten"],
    "Washington Commanders": ["washington", "commanders", "skins", "wash", "wsh"],
}


# ----------------------------
# 2) Player alias starter set
#    Edit / expand as desired
# ----------------------------
TEAM_PLAYER_ALIASES: Dict[str, List[str]] = {
    "Arizona Cardinals": ["kyler", "murray", "trey mcbride", "mcbride", "budda", "baker", "marvin harrison", "mhj"],
    "Atlanta Falcons": ["bijan", "robinson", "drake london", "london", "pitts", "kyle pitts", "penix", "michael penix", "jessie bates"],
    "Baltimore Ravens": ["lamar", "lamar jackson", "derrick henry", "henry", "zay flowers", "flowers", "roquan", "bateman", "mark andrews", "andrews"],
    "Buffalo Bills": ["josh allen", "allen", "james cook", "cook", "kincaid", "dalton kincaid", "shakir", "keon coleman", "milano"],
    "Carolina Panthers": ["bryce young", "bryce", "chuba", "hubbard", "thielen", "xavier legette", "legette", "derrick brown"],
    "Chicago Bears": ["caleb", "caleb williams", "dj moore", "moore", "rome odunze", "odunze", "swift", "jaylon johnson", "montez sweat"],
    "Cincinnati Bengals": ["burrow", "joe burrow", "chase", "ja'marr", "jamarr", "higgins", "tee higgins", "hendrickson", "trey hendrickson"],
    "Cleveland Browns": ["chubb", "nick chubb", "ward", "denzel ward", "jeremiah owusu-koramoah", "jok", "garrett", "myles garrett"],
    "Dallas Cowboys": ["dak", "dak prescott", "ceedee", "cd lamb", "lamb", "micah", "micah parsons", "parsons", "diggs", "trevon diggs", "ferguson"],
    "Denver Broncos": ["bo nix", "nix", "courtland sutton", "sutton", "ps2", "pat surtain", "surtain", "javonte", "mims"],
    "Detroit Lions": ["goff", "jared goff", "gibbs", "jahmyr", "montgomery", "amon-ra", "st brown", "laporta", "hutch", "hutchinson"],
    "Green Bay Packers": ["love", "jordan love", "jacobs", "josh jacobs", "reed", "jayden reed", "watson", "doubs", "mckinney"],
    "Houston Texans": ["cj stroud", "stroud", "nico", "nico collins", "tank dell", "dell", "mixon", "will anderson", "anderson"],
    "Indianapolis Colts": ["anthony richardson", "arich", "richardson", "jt", "jonathan taylor", "taylor", "pitman", "michael pittman", "buckner"],
    "Jacksonville Jaguars": ["trevor", "trevor lawrence", "lawrence", "etienne", "travis etienne", "btj", "brian thomas", "hines-allen", "josh hines-allen"],
    "Kansas City Chiefs": ["mahomes", "patrick mahomes", "kelce", "travis kelce", "pacheco", "isiah pacheco", "chris jones", "jones", "worthy", "rashee rice"],
    "Las Vegas Raiders": ["maxx", "maxx crosby", "crosby", "bowers", "brock bowers", "jakobi", "mayer", "wilkins"],
    "Los Angeles Chargers": ["herbert", "justin herbert", "lad mcconkey", "mcconkey", "joey bosa", "bosa", "slater", "derwin", "derwin james"],
    "Los Angeles Rams": ["stafford", "matthew stafford", "puka", "puka nacua", "kyren", "kyren williams", "kupp", "aaron donald"],
    "Miami Dolphins": ["tua", "tagovailoa", "tyreek", "reek", "waddle", "achane", "mostert", "ramsey"],
    "Minnesota Vikings": ["jefferson", "jjettas", "jjetas", "justin jefferson", "addison", "hockenson", "hock", "darnold"],
    "New England Patriots": ["maye", "drake maye", "gonzalez", "christian gonzalez", "stevenson", "rhamondre", "judon", "pop douglas"],
    "New Orleans Saints": ["kamara", "alvin kamara", "olave", "carr", "derrick carr", "shaheed", "demario", "demario davis"],
    "New York Giants": ["nabers", "malik nabers", "dexter lawrence", "dex", "burns", "brian burns", "thibodeaux", "kayvon"],
    "New York Jets": ["rodgers", "aaron rodgers", "garrett wilson", "wilson", "breece", "breece hall", "sauce", "sauce gardner", "quinnen"],
    "Philadelphia Eagles": ["hurts", "jalen hurts", "aj brown", "brown", "saquon", "saquon barkley", "devonta", "smitty", "lane johnson", "carter", "jalen carter"],
    "Pittsburgh Steelers": ["watt", "tj watt", "pickens", "george pickens", "najee", "naji", "najee harris", "minkah", "cam heyward"],
    "San Francisco 49ers": ["purdy", "brock purdy", "cmc", "mccaffrey", "deebo", "aiyuk", "kittle", "trent williams", "bosa", "nick bosa"],
    "Seattle Seahawks": ["geno", "geno smith", "dk", "metcalf", "jsn", "jaxon smith-njigba", "walker", "kenneth walker", "woolen", "love"],
    "Tampa Bay Buccaneers": ["baker", "baker mayfield", "evans", "mike evans", "godwin", "winfield", "lavonte", "vita vea"],
    "Tennessee Titans": ["levis", "will levis", "pollard", "tony pollard", "ridley", "calvin ridley", "hopkins", "sneed", "jeffery simmons"],
    "Washington Commanders": ["jayden", "jayden daniels", "daniels", "mclaurin", "terry", "brian robinson", "brob", "allen", "payne", "luvu"],
}


# ----------------------------
# 3) General regex / helpers
# ----------------------------
WORD_RE = re.compile(r"[a-z0-9']+")

FAN_WORDS = {"we", "us", "our", "ours", "ourselves"}
NEGATION_WORDS = {"not", "isnt", "isn't", "aint", "ain't", "never", "no"}

COMMON_AMBIGUOUS_ALIASES = {
    "allen", "brown", "jones", "love", "cook", "wilson", "moore", "baker", "williams",
    "johnson", "ward", "taylor", "james", "smith"
}


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("&amp;", " and ")
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return WORD_RE.findall(normalize_text(s))


def unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def get_team_aliases(team_name: str) -> List[str]:
    aliases = TEAM_ALIASES.get(team_name, []).copy()
    aliases.append(team_name.lower())
    return unique_in_order([a.lower() for a in aliases if a])


def get_team_players(team_name: str) -> List[str]:
    return unique_in_order([a.lower() for a in TEAM_PLAYER_ALIASES.get(team_name, [])])


def contains_phrase(text: str, phrase: str) -> bool:
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def score_side(
    body_text: str,
    title_text: str,
    subreddit_text: str,
    team_name: str,
    other_team_name: str,
) -> Tuple[float, List[str], List[str]]:
    """
    Returns:
      score,
      matched_terms,
      reasons
    """
    team_aliases = get_team_aliases(team_name)
    team_players = get_team_players(team_name)

    other_aliases = set(get_team_aliases(other_team_name))
    other_players = set(get_team_players(other_team_name))

    score = 0.0
    matched_terms = []
    reasons = []

    # Team aliases / nicknames / abbreviations
    for alias in team_aliases:
        if contains_phrase(body_text, alias):
            w = 3.0
            if len(alias) <= 3:
                # Light penalty for short abbreviations like KC / DAL / NE
                w = 1.6
            score += w
            matched_terms.append(alias)
            reasons.append(f"team_alias:{alias}")

    # Players
    for player in team_players:
        if contains_phrase(body_text, player):
            w = 2.5
            if player in COMMON_AMBIGUOUS_ALIASES:
                w = 1.3
            score += w
            matched_terms.append(player)
            reasons.append(f"player:{player}")

    # Strong bonus if body contains team term and fan language
    body_tokens = set(tokenize(body_text))
    if FAN_WORDS & body_tokens and matched_terms:
        score += 0.8
        reasons.append("fan_language_with_team_clue")

    # Title bonus: if post title is specifically about this matchup
    # and body mentions city/team/player, reinforce slightly
    team_mentioned_in_title = any(contains_phrase(title_text, a) for a in team_aliases)
    if team_mentioned_in_title and matched_terms:
        score += 0.6
        reasons.append("post_title_alignment")

    # Subreddit hint
    # e.g. subreddit = eagles, cowboys, patriots
    if subreddit_text:
        for alias in team_aliases:
            if alias.replace(" ", "") == subreddit_text.replace("_", "").replace("-", ""):
                score += 1.2
                reasons.append(f"subreddit_match:{alias}")
                break

    # Penalty if most matches are actually also aliases for the opponent
    overlap_hits = [m for m in matched_terms if m in other_aliases or m in other_players]
    if overlap_hits:
        score -= 0.8 * len(overlap_hits)
        reasons.append("opponent_overlap_penalty")

    return score, unique_in_order(matched_terms), reasons


def scores_to_probs(home_score: float, away_score: float) -> Tuple[float, float, float]:
    """
    Convert scores into three-class probabilities:
      pct_home, pct_away, pct_unsure

    Behavior:
    - If both sides have weak evidence, unsure gets a lot of mass.
    - If one side clearly dominates, unsure shrinks.
    """
    # Ensure non-negative for calibration step
    hs = max(home_score, 0.0)
    aw = max(away_score, 0.0)
    evidence = hs + aw

    # Uncertainty calibration
    if evidence == 0:
        return 0.10, 0.10, 0.80

    # When evidence is small, keep high unsure.
    unsure = max(0.05, min(0.85, math.exp(-0.55 * evidence)))

    # Softmax between home and away
    ex_h = math.exp(hs)
    ex_a = math.exp(aw)
    denom = ex_h + ex_a
    home_share = ex_h / denom
    away_share = ex_a / denom

    pct_home = (1 - unsure) * home_share
    pct_away = (1 - unsure) * away_share
    pct_unsure = unsure

    # round-safe normalization
    total = pct_home + pct_away + pct_unsure
    pct_home /= total
    pct_away /= total
    pct_unsure /= total

    return pct_home, pct_away, pct_unsure


def classify_row(row: pd.Series) -> pd.Series:
    body = normalize_text(row.get("body", ""))
    title = normalize_text(row.get("post_title", ""))
    subreddit = normalize_text(row.get("subreddit", ""))
    home_team = str(row.get("home_team", "")).strip()
    away_team = str(row.get("away_team", "")).strip()

    if not home_team or not away_team:
        return pd.Series({
            "pct_home": 0.0,
            "pct_away": 0.0,
            "pct_unsure": 1.0,
            "predicted_tag": "unsure",
            "matched_home_terms": "",
            "matched_away_terms": "",
            "classifier_reason": "missing_home_or_away_team"
        })

    home_score, home_terms, home_reasons = score_side(body, title, subreddit, home_team, away_team)
    away_score, away_terms, away_reasons = score_side(body, title, subreddit, away_team, home_team)

    pct_home, pct_away, pct_unsure = scores_to_probs(home_score, away_score)

    best = max(
        [("home_team", pct_home), ("away_team", pct_away), ("unsure", pct_unsure)],
        key=lambda x: x[1]
    )[0]

    # Margin guard: if home and away are too close and both not much above unsure, call unsure
    if abs(pct_home - pct_away) < 0.12 and max(pct_home, pct_away) < 0.60:
        best = "unsure"

    reason_parts = []
    if home_terms:
        reason_parts.append("home_hits=" + "|".join(home_terms))
    if away_terms:
        reason_parts.append("away_hits=" + "|".join(away_terms))
    if home_reasons:
        reason_parts.append("home_reasons=" + "|".join(unique_in_order(home_reasons)))
    if away_reasons:
        reason_parts.append("away_reasons=" + "|".join(unique_in_order(away_reasons)))
    if not reason_parts:
        reason_parts.append("no_team_specific_hits")

    return pd.Series({
        "pct_home": round(pct_home, 4),
        "pct_away": round(pct_away, 4),
        "pct_unsure": round(pct_unsure, 4),
        "predicted_tag": best,
        "matched_home_terms": "; ".join(home_terms),
        "matched_away_terms": "; ".join(away_terms),
        "classifier_reason": " || ".join(reason_parts)
    })


def main():
    parser = argparse.ArgumentParser(description="Classify Reddit comments as home team / away team / unsure.")
    parser.add_argument("--input", required=True, help="Path to input CSV, e.g. combined.csv")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")

    required = {"body", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    preds = df.apply(classify_row, axis=1)
    out = pd.concat([df, preds], axis=1)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Done. Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
