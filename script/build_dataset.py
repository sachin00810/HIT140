"""Build the analysis-ready match-level corners dataset.

Input : data/raw/wc2026_match_corners_raw.csv   one row per team per match, with
                                            the fields taken straight from the
                                            source fixtures table (no derived
                                            columns): match_number, date, stage,
                                            team, opponent, corners, goals_for,
                                            goals_against.
Output: data/wc2026_match_corners.csv        the same rows, cleaned, with the
                                            outcome label and helper columns the
                                            analysis needs.

Also read (reference only, never used to change a value):
        data/raw_data.csv                   the source website's SEPARATE
                                            per-team aggregate view. It is a
                                            different granularity (one row per
                                            team) and is known to disagree with
                                            the fixtures view for a handful of
                                            teams. This script reports those
                                            differences so they stay visible; it
                                            does not act on them.

Run this first, then script/analysis.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_match_corners_raw.csv"
AGG_PATH = PROJECT_ROOT / "data" / "raw_data.csv"
OUT_PATH = PROJECT_ROOT / "data" / "wc2026_match_corners.csv"

RAW_COLUMNS = ["match_number", "date", "stage", "team", "opponent",
               "corners", "goals_for", "goals_against"]
OUT_COLUMNS = ["match_number", "date", "stage", "is_knockout", "team", "opponent",
               "corners", "goals_for", "goals_against", "goal_diff", "outcome"]

# The source uses two spellings for one team; standardise to one.
NAME_FIXES = {"Bosnia & Herzegovina": "Bosnia and Herzegovina"}


def load_raw(path):
    m = pd.read_csv(path)
    print(f"Rows loaded: {len(m)}")

    missing = set(RAW_COLUMNS) - set(m.columns)
    if missing:
        raise KeyError(f"Raw file is missing expected columns: {missing}")

    m["team"] = m.team.str.strip().replace(NAME_FIXES)
    m["opponent"] = m.opponent.str.strip().replace(NAME_FIXES)

    for col in ("match_number", "corners", "goals_for", "goals_against"):
        m[col] = pd.to_numeric(m[col], errors="coerce")
    if m[["match_number", "corners", "goals_for", "goals_against"]].isna().any().any():
        raise ValueError("non-numeric value in a numeric column of the raw file")

    before = len(m)
    m = m.drop_duplicates(subset=["match_number", "team"])
    if len(m) != before:
        print(f"Dropped {before - len(m)} duplicate (match_number, team) rows")

    print(f"Distinct teams: {m.team.nunique()}")
    return m


def check_integrity(m):
    """Every check here must pass; a failure means the raw file is wrong."""
    pairs = m.match_number.value_counts()
    assert (pairs == 2).all(), \
        f"every match needs exactly 2 rows; offenders: {pairs[pairs != 2].to_dict()}"

    opp = m[["match_number", "team", "goals_for"]].rename(
        columns={"team": "opponent", "goals_for": "opp_goals"})
    chk = m.merge(opp, on=["match_number", "opponent"], how="left")
    assert chk.opp_goals.notna().all(), "a row's opponent is not in the same match"
    assert (chk.goals_against == chk.opp_goals).all(), \
        "goals_against must equal the opponent's goals_for"

    assert (m.corners >= 0).all(), "negative corner count"
    assert (m.corners == m.corners.round()).all(), "non-integer corner count"

    print(f"Integrity OK: {m.match_number.nunique()} matches, 2 rows each, "
          f"goals reconcile between the pair, corners are non-negative integers")


def add_derived(m):
    m["outcome"] = np.select(
        [m.goals_for > m.goals_against, m.goals_for < m.goals_against],
        ["Win", "Loss"], default="Draw")
    m["goal_diff"] = m.goals_for - m.goals_against
    m["is_knockout"] = np.where(m.stage != "Group Stage", "Yes", "No")

    counts = m.outcome.value_counts()
    assert counts.get("Win", 0) == counts.get("Loss", 0), "wins must equal losses"
    assert counts.get("Draw", 0) % 2 == 0, "draws must come in pairs"
    print(f"Outcomes: {counts.to_dict()}")
    return m


def crosscheck_aggregate(m, path):
    """Report, do not fix. The aggregate view is a second read of the same
    tournament from the source site; where it disagrees with the fixtures view
    we keep the fixtures value (it is the correct unit and was verified twice)
    and just print the gap so it is never silently forgotten."""
    if not path.exists():
        print(f"\nCross-check skipped: {path.name} not found")
        return

    agg = pd.read_csv(path)
    agg["team"] = agg.team.str.strip().replace(NAME_FIXES)

    fix = (m.groupby("team")
             .agg(games_fixtures=("match_number", "size"),
                  corners_fixtures=("corners", "sum"))
             .reset_index())
    cmp = fix.merge(agg[["team", "games_played", "ft_corners_for"]],
                    on="team", how="outer", indicator=True)

    only = cmp[cmp._merge != "both"]
    if len(only):
        print(f"\nTeams not in both files: {only.team.tolist()}")

    both = cmp[cmp._merge == "both"].copy()
    both["games_diff"] = both.games_fixtures - both.games_played
    both["corners_diff"] = both.corners_fixtures - both.ft_corners_for
    disagree = both[(both.games_diff != 0) | (both.corners_diff != 0)]

    print(f"\nCross-check vs {path.name}: "
          f"{len(both) - len(disagree)}/{len(both)} teams match exactly")
    if len(disagree):
        print("  Known source inconsistency — fixtures value kept. Differences "
              "(fixtures minus aggregate):")
        for _, r in disagree.sort_values("team").iterrows():
            print(f"    {r.team:<24} games {int(r.games_diff):+d}, "
                  f"corners {int(r.corners_diff):+d}")


def main():
    m = load_raw(RAW_PATH)
    check_integrity(m)
    m = add_derived(m)
    crosscheck_aggregate(m, AGG_PATH)

    # Keep the raw file's row order (source fixtures order: home team, then away).
    m = m[OUT_COLUMNS].reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_PATH, index=False)

    decided = (m.outcome != "Draw").sum()
    wins = int((m.outcome == "Win").sum())
    print(f"\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)} — "
          f"{len(m)} rows, {m.match_number.nunique()} matches")
    print(f"Usable for the Win/Loss comparison: {decided} rows "
          f"({wins} wins, {wins} losses); {len(m) - decided} draw rows excluded there")
    print("\nKNOWN LIMITATION: the third-place play-off is absent from the source "
          "fixtures table, so this covers 103 of 104 matches.")


if __name__ == "__main__":
    main()
