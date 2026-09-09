"""
02_prepare_data.py - Data Preparation & Cleaning Script
FIFA World Cup 2026: Pass Completion vs Match Outcome Analysis
HIT140 Foundations of Data Science

Transforms match-level data to team-match level (2 rows per match),
assigns outcome perspectives ('Win', 'Loss', 'Draw'),
filters out draws, and saves the cleaned dataset.
"""


import os
import pandas as pd

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fifa2026_raw_matches.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, "fifa2026_cleaned.csv")


def prepare_data():
    print("=" * 70)
    print("STEP 2: DATA PREPARATION & CLEANING")
    print("=" * 70)

    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Raw data file not found at: {RAW_CSV_PATH}. Please run src/01_extract_data.py first.")

    raw_df = pd.read_csv(RAW_CSV_PATH)
    print(f"Loaded raw match-level data: {len(raw_df)} matches from {RAW_CSV_PATH}")

    # Convert match-level to team-match level (2 rows per match)
    team_matches = []
    for _, row in raw_df.iterrows():
        mid = row["match_id"]
        date = row["date"]
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])
        h_pass = float(row["home_pass_completion"])
        a_pass = float(row["away_pass_completion"])

        # Determine outcome for home team
        if h_score > a_score:
            h_result = "Win"
            a_result = "Loss"
        elif h_score < a_score:
            h_result = "Loss"
            a_result = "Win"
        else:
            h_result = "Draw"
            a_result = "Draw"

        # Home team perspective
        team_matches.append({
            "match_id": mid,
            "date": date,
            "team": h_team,
            "opponent": a_team,
            "venue": "Home",
            "goals_for": h_score,
            "goals_against": a_score,
            "pass_completion": h_pass,
            "result": h_result,
        })

        # Away team perspective
        team_matches.append({
            "match_id": mid,
            "date": date,
            "team": a_team,
            "opponent": h_team,
            "venue": "Away",
            "goals_for": a_score,
            "goals_against": h_score,
            "pass_completion": a_pass,
            "result": a_result,
        })

    team_df = pd.DataFrame(team_matches)
    total_obs = len(team_df)

    # Count breakdown
    wins_count = (team_df["result"] == "Win").sum()
    losses_count = (team_df["result"] == "Loss").sum()
    draws_excluded = (team_df["result"] == "Draw").sum()

    # Filter to only Wins and Losses (exclude Draws for two-group independent-samples comparison)
    cleaned_df = team_df[team_df["result"].isin(["Win", "Loss"])].copy().reset_index(drop=True)

    # Save processed CSV
    cleaned_df.to_csv(PROCESSED_CSV_PATH, index=False)

    print("\n--- DATA PREPARATION SUMMARY ---")
    print(f"Total raw matches processed      : {len(raw_df)}")
    print(f"Total team-match observations    : {total_obs}")
    print(f"Draw observations excluded       : {draws_excluded} ({draws_excluded // 2} match draws)")
    print(f"Final clean observations         : {len(cleaned_df)}")
    print(f"  - Wins count                   : {wins_count}")
    print(f"  - Losses count                 : {losses_count}")
    print(f"Output saved to                  : {PROCESSED_CSV_PATH}")

    # Verify group size requirement
    if wins_count < 30 or losses_count < 30:
        print(f"\n[WARNING] Each group must have at least 30 observations. Current: Wins={wins_count}, Losses={losses_count}")
    else:
        print(f"\n[VALIDATION PASSED] Both groups have >= 30 observations (Wins={wins_count}, Losses={losses_count}).")
    print("--------------------------------\n")

    return cleaned_df


if __name__ == "__main__":
    prepare_data()
