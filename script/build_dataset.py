"""Build the analysis-ready match-level corners dataset.

Input : data/raw/wc2026_corners__1_.csv   (206 rows, extracted from the fixtures
                                            table and verified row-by-row against
                                            the live site)
Output: data/wc2026_match_corners.csv     (cleaned, with an outcome label)

The raw file is the manual extraction from "The Stats Don't Lie" fixtures table.
It is not committed here; this script documents exactly how the cleaned dataset
in data/ was produced and re-runs the integrity checks. Run the analysis with
script/task_corners_winners_vs_losers.py once data/wc2026_match_corners.csv exists.
"""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_corners__1_.csv"
OUT_PATH = PROJECT_ROOT / "data" / "wc2026_match_corners.csv"

m = pd.read_csv(RAW_PATH)
print(f"Rows loaded: {len(m)}")

# --- standardise team naming (the source uses two spellings for one team) ---
FIX = {'Bosnia & Herzegovina': 'Bosnia and Herzegovina'}
m['team'] = m.team.replace(FIX)
m['opponent'] = m.opponent.replace(FIX)
print(f"Distinct teams: {m.team.nunique()}")

# --- integrity checks -------------------------------------------------------
pairs = m.match_number.value_counts()
assert (pairs == 2).all(), "every match must have exactly 2 rows"

opp = m[['match_number', 'team', 'goals_for']].rename(
    columns={'team': 'opponent', 'goals_for': 'opp_goals'})
chk = m.merge(opp, on=['match_number', 'opponent'])
assert (chk.goals_against == chk.opp_goals).all(), \
    "goals_against must equal the opponent's goals_for"
print("Verified: every match has 2 rows and goals reconcile between them")

# --- derive the outcome label ----------------------------------------------
m['outcome'] = np.select(
    [m.goals_for > m.goals_against, m.goals_for < m.goals_against],
    ['Win', 'Loss'], default='Draw')

counts = m.outcome.value_counts()
assert counts['Win'] == counts['Loss'], "wins must equal losses"
assert counts['Draw'] % 2 == 0, "draws must come in pairs"
print(f"Outcomes: {counts.to_dict()}")

m['goal_diff'] = m.goals_for - m.goals_against
m['is_knockout'] = (m.stage != 'Group Stage').map({True: 'Yes', False: 'No'})

m = m[['match_number', 'date', 'stage', 'is_knockout', 'team', 'opponent',
       'corners', 'goals_for', 'goals_against', 'goal_diff', 'outcome']]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(m)} rows covering {m.match_number.nunique()} matches")
print(f"Usable for Win/Loss comparison: {(m.outcome != 'Draw').sum()} rows "
      f"({counts['Win']} wins, {counts['Loss']} losses)")
print("\nKNOWN LIMITATION: the third-place play-off is absent from the source")
print("fixtures table, so this covers 103 of 104 matches.")
