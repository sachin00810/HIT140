"""
Linear Regression 2.1 dataset: ONE ROW PER MATCH (104 rows).

Target   goal_diff = score(team1) - score(team2)   (score after extra time; a penalty
         shoot-out is not counted, so shoot-out matches have goal_diff = 0)

Eight explanatory variables, every one known BEFORE kick-off. Each is built only from
matches played on EARLIER calendar days, as a difference "team1 minus team2":

  1 xg_diff        previous average expected goals (xG) created
  2 xga_diff       previous average xG conceded (= opponents' xG)
  3 fifa_pts_diff  FIFA world-ranking points from the PRE-TOURNAMENT release (11 June 2026,
                   published before the first kick-off); team1 points minus team2 points
  4 ppg_diff       tournament points per match so far (win 3, draw 1, loss 0)
  5 fouls_diff     previous average fouls committed
  6 rest_diff      days since own previous match
  7 is_knockout    1 = knockout stage, 0 = group stage
  8 host_adv       +1 team1 is a host nation, -1 team2 is, 0 neither (USA, Canada, Mexico)

Opening matches: no team has tournament history yet, so the five history differences are
0 there. fifa_pts_diff is a real pre-tournament strength signal available for every match,
so the 24 opening matches still carry information about which team is stronger.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
FEATURES = ["xg_diff", "xga_diff", "fifa_pts_diff", "ppg_diff", "fouls_diff", "rest_diff", "is_knockout", "host_adv"]
HISTORY = ["xg_diff", "xga_diff", "ppg_diff", "fouls_diff", "rest_diff"]   # need earlier tournament matches
HOSTS = {"USA", "Canada", "Mexico"}


def num(x):           # "1p" -> 1, "p0" -> 0, "2" -> 2
    return int(re.sub(r"\D", "", str(x)))


def load_matches():
    t = pd.read_csv(RAW / "tsdl_match_table_raw.csv", dtype=str)
    f = pd.read_csv(RAW / "fifa_third_place_playoff.csv", dtype=str)
    # third-place play-off sits just before the Final
    final_pos = t.index[t.stage == "FINAL"][0]
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


def long_format(m):
    rows = []
    for k, (a, b) in enumerate((("1", "2"), ("2", "1"))):
        d = pd.DataFrame({
            "match_no": m.match_no, "date": m.date, "side": k, "team": m[f"team{a}"],
            "gf": m[f"score{a}"], "ga": m[f"score{b}"],
            "xg": m[f"xg_t{a}"], "xga": m[f"xg_t{b}"], "fouls": m[f"fouls_t{a}"]})
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["gd"] = L.gf - L.ga
    L["points"] = np.select([L.gf > L.ga, L.gf == L.ga], [3, 1], 0)
    return L.sort_values(["match_no", "side"]).reset_index(drop=True)


def team_history(L, team, date):
    return L[(L.team == team) & (L.date < date)]


def build():
    m = load_matches()
    L = long_format(m)
    pts = load_ranking()
    out = []
    for _, r in m.iterrows():
        feats = {}
        per = {}
        for side, team in (("t1", r.team1), ("t2", r.team2)):
            h = team_history(L, team, r.date)
            if len(h) == 0:
                per[side] = None
            else:
                per[side] = {"xg": h.xg.mean(), "xga": h.xga.mean(),
                             "ppg": h.points.mean(), "fouls": h.fouls.mean(),
                             "rest": (r.date - h.date.max()).days}
        for name, key in [("xg_diff", "xg"), ("xga_diff", "xga"),
                          ("ppg_diff", "ppg"), ("fouls_diff", "fouls"), ("rest_diff", "rest")]:
            if per["t1"] is None and per["t2"] is None:
                feats[name] = 0.0                    # both teams opening their tournament
            else:
                assert per["t1"] is not None and per["t2"] is not None, "one team with history, one without"
                feats[name] = per["t1"][key] - per["t2"][key]
        feats["fifa_pts_diff"] = float(pts[r.team1] - pts[r.team2])
        feats["is_knockout"] = int(r.stage != "GROUP STAGE")
        feats["host_adv"] = int(r.team1 in HOSTS) - int(r.team2 in HOSTS)
        out.append({"match_no": r.match_no, "date": r.date.strftime("%Y-%m-%d"), "stage": r.stage,
                    "team1": r.team1, "team2": r.team2, **feats, "goal_diff": r.score1 - r.score2})
    return pd.DataFrame(out)


def checks(D):
    assert len(D) == 104 and D.match_no.is_unique, "need 104 unique matches"
    assert len(FEATURES) == 8
    assert not D[FEATURES + ["goal_diff"]].isna().any().any(), "missing values"
    assert D.date.is_monotonic_increasing, "matches must be in date order"


if __name__ == "__main__":
    D = build(); checks(D)
    PROC.mkdir(parents=True, exist_ok=True)
    D.to_csv(PROC / "wc2026_lr21_dataset.csv", index=False)
    print(D.shape); print(D[FEATURES + ["goal_diff"]].describe().T.round(3))
    print("stages:", D.stage.value_counts().to_dict())
    print("rows with all-zero history features (opening round):", int((D[HISTORY].abs().sum(axis=1) == 0).sum()))
