"""
Linear Regression 2.2 dataset: ONE ROW PER TEAM PER MATCH (208 rows).

Target   goals = goals scored by the team in the match (after extra time; shoot-out
         penalties are not counted)

Eight explanatory variables, every one known BEFORE kick-off. History variables use only
matches played on EARLIER calendar days:

  1 prior_goals_avg           team's average goals scored in its previous matches
  2 prior_sot_avg             team's average shots on target in its previous matches
  3 prior_shots_avg           team's average total shots in its previous matches
  4 prior_corners_avg         team's average corners won in its previous matches
  5 opp_prior_conceded_avg    opponent's average goals conceded in its previous matches
  6 team_fifa_pts             team's FIFA ranking points, pre-tournament release (11 June 2026)
  7 opp_fifa_pts              opponent's FIFA ranking points, same release
  8 rest_days                 days since the team's previous match

Teams in their opening match have no tournament history: variables 1-5 and 8 are left EMPTY
(NaN) for those rows - nothing is invented. The analysis fills them inside each training
fold only (mean of the training rows), and also reports results without those rows.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
FEATURES = ["prior_goals_avg", "prior_sot_avg", "prior_shots_avg", "prior_corners_avg",
            "opp_prior_conceded_avg", "team_fifa_pts", "opp_fifa_pts", "rest_days"]
HISTORY = ["prior_goals_avg", "prior_sot_avg", "prior_shots_avg", "prior_corners_avg",
           "opp_prior_conceded_avg", "rest_days"]


def num(x):           # "1p" -> 1, "p0" -> 0, "2" -> 2
    return int(re.sub(r"\D", "", str(x)))


def load_matches():
    """The verified 104-match table (same raw files as Linear Regression 1)."""
    t = pd.read_csv(RAW / "tsdl_match_table_raw.csv", dtype=str)
    f = pd.read_csv(RAW / "fifa_third_place_playoff.csv", dtype=str)
    final_pos = t.index[t.stage == "FINAL"][0]          # third-place play-off sits just before the Final
    m = pd.concat([t.iloc[:final_pos], f[t.columns.intersection(f.columns)], t.iloc[final_pos:]], ignore_index=True)
    m["match_no"] = np.arange(1, len(m) + 1)
    m["date"] = pd.to_datetime(m["date"], format="%d/%m/%Y")
    for c in ["score1", "score2"]:
        m[c] = m[c].map(num)
    for c in m.columns:
        if c not in ["stage", "date", "team1", "team2", "match_no", "score1", "score2"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


def load_ranking():
    r = pd.read_csv(RAW / "fifa_ranking_pre_tournament.csv")
    assert len(r) == 48 and r.team.is_unique
    return r.set_index("team").points


def build():
    m = load_matches()
    pts = load_ranking()
    rows = []
    for k, (a, b) in enumerate((("1", "2"), ("2", "1"))):
        rows.append(pd.DataFrame({
            "match_no": m.match_no, "date": m.date, "stage": m.stage, "side": k,
            "team": m[f"team{a}"], "opponent": m[f"team{b}"],
            "gf": m[f"score{a}"], "ga": m[f"score{b}"],
            "sot": m[f"sot_t{a}"], "shots": m[f"shots_t{a}"], "corners": m[f"corners_t{a}"]}))
    L = pd.concat(rows, ignore_index=True).sort_values(["match_no", "side"]).reset_index(drop=True)

    out = []
    for _, r in L.iterrows():
        h = L[(L.team == r.team) & (L.date < r.date)]
        ho = L[(L.team == r.opponent) & (L.date < r.date)]
        has = len(h) > 0
        out.append({
            "match_no": r.match_no, "date": r.date.strftime("%Y-%m-%d"), "stage": r.stage,
            "team": r.team, "opponent": r.opponent,
            "prior_goals_avg": h.gf.mean() if has else np.nan,
            "prior_sot_avg": h.sot.mean() if has else np.nan,
            "prior_shots_avg": h.shots.mean() if has else np.nan,
            "prior_corners_avg": h.corners.mean() if has else np.nan,
            "opp_prior_conceded_avg": ho.ga.mean() if len(ho) else np.nan,
            "team_fifa_pts": float(pts[r.team]), "opp_fifa_pts": float(pts[r.opponent]),
            "rest_days": (r.date - h.date.max()).days if has else np.nan,
            "goals": r.gf})
    return pd.DataFrame(out)


def checks(D):
    assert len(D) == 208 and D.groupby("match_no").size().eq(2).all(), "need 208 rows, 2 per match"
    assert len(FEATURES) == 8
    assert D.date.is_monotonic_increasing
    assert not D[["team_fifa_pts", "opp_fifa_pts", "goals"]].isna().any().any()
    # leakage guard: every history value must be computable from earlier days only
    first = D.groupby("team").date.transform("min") == D.date
    assert D.loc[first, "prior_goals_avg"].isna().all(), "opening-match rows must have no history"
    assert D.loc[~first, "prior_goals_avg"].notna().all()


if __name__ == "__main__":
    D = build(); checks(D)
    PROC.mkdir(parents=True, exist_ok=True)
    D.to_csv(PROC / "wc2026_lr22_dataset.csv", index=False)
    print(D.shape); print(D[FEATURES + ["goals"]].describe().T.round(3))
    print("rows without tournament history (opening matches):", int(D.prior_goals_avg.isna().sum()))
