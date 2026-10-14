"""
Build the Linear Regression 2.1 dataset: ONE ROW PER MATCH (104 rows).

Target  : goal_diff = home_score - away_score   (score after extra time; a penalty
          shoot-out is NOT counted, so a shoot-out match has goal_diff = 0)

Every explanatory variable is a (home - away) difference of information that was
known BEFORE kick-off, i.e. built only from matches that had already finished.

  1 gf_diff       previous average goals scored          (home - away)
  2 ga_diff       previous average goals conceded        (home - away)
  3 rest_diff     rest days since own previous match     (home - away)
  4 xg_diff       previous average expected goals (xG)   (home - away)
  5 ppg_diff      tournament points per game so far      (home - away)
  6 fouls_diff    previous average fouls committed       (home - away)
  7 is_knockout   1 = knockout stage, 0 = group stage
  8 host_adv      +1 home side is a host nation, -1 away side is, 0 neither

Teams with no earlier match (their opener) have no history. Default handling
("tournament_mean"): use the average of ALL teams over the matches that had
finished before kick-off (leak-free). When nothing has finished yet (first
matchday) both sides get the same value, so the difference is 0.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

FEATURES = ["gf_diff", "ga_diff", "rest_diff", "xg_diff",
            "ppg_diff", "fouls_diff", "is_knockout", "host_adv"]
FINISH_LAG = pd.Timedelta(hours=3)   # a match counts as "known" 3h after kick-off
SHRINK_K = 2.0                       # pseudo-matches used by fill="shrunk"


def load_long():
    """Team-match table (208 rows): one row per team per match."""
    m = pd.read_csv(RAW / "matches.csv")
    s = pd.read_csv(RAW / "match_team_stats.csv")
    t = pd.read_csv(RAW / "team_lookup.csv")
    m["kickoff"] = pd.to_datetime(m["date"] + " " + m["kickoff_time_utc"])
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        d = m[["match_id", "kickoff", "stage_id", f"{side}_team_id", f"{other}_team_id",
               f"{side}_score", f"{other}_score", f"{side}_xg"]].copy()
        d.columns = ["match_id", "kickoff", "stage_id", "team_id", "opp_id", "gf", "ga", "xg"]
        d["side"] = side
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L = L.merge(s[["match_id", "team_id", "fouls"]], on=["match_id", "team_id"], how="left")
    L = L.merge(t[["team_id", "team"]], on="team_id")
    L["points"] = np.select([L.gf > L.ga, L.gf == L.ga], [3, 1], 0)
    return m, L.sort_values(["kickoff", "match_id", "side"]).reset_index(drop=True)


def build(fill="tournament_mean"):
    """fill: 'tournament_mean' (default) | 'zero' | 'shrunk'"""
    m, L = load_long()
    stats = {"gf": "gf", "ga": "ga", "xg": "xg", "fouls": "fouls", "ppg": "points"}
    out = []
    for _, r in L.iterrows():
        # this team's own earlier matches
        h = L[(L.team_id == r.team_id) & (L.kickoff < r.kickoff)]
        # every team's matches that had finished before this kick-off
        world = L[L.kickoff + FINISH_LAG <= r.kickoff]
        n = len(h)
        rec = {"match_id": r.match_id, "side": r.side, "n_prior": n}
        for name, col in stats.items():
            mu = world[col].mean() if len(world) else 0.0
            if fill == "zero":
                v = h[col].mean() if n else 0.0
            elif fill == "shrunk":
                v = (h[col].sum() + SHRINK_K * mu) / (n + SHRINK_K)
            else:
                v = h[col].mean() if n else mu
            rec[name] = v
        if n:
            rec["rest"] = (r.kickoff.normalize() - h.kickoff.max().normalize()).days
        else:
            rec["rest"] = np.nan   # opener: filled below
        out.append(rec)
    P = pd.DataFrame(out)
    # opener rest days: 0 if 'zero', otherwise the average rest of teams that have played again
    if fill == "zero":
        P["rest"] = P["rest"].fillna(0.0)
    else:
        known = P.dropna(subset=["rest"]).merge(L[["match_id", "side", "kickoff"]], on=["match_id", "side"])
        fills = []
        for _, r in P[P.rest.isna()].merge(L[["match_id", "side", "kickoff"]], on=["match_id", "side"]).iterrows():
            prior = known[known.kickoff + FINISH_LAG <= r.kickoff]["rest"]
            fills.append(prior.mean() if len(prior) else 0.0)
        P.loc[P.rest.isna(), "rest"] = fills

    H = P[P.side == "home"].set_index("match_id")
    A = P[P.side == "away"].set_index("match_id")
    M = m.set_index("match_id")
    lk = pd.read_csv(RAW / "team_lookup.csv").set_index("team_id")
    D = pd.DataFrame(index=M.index)
    D["date"] = M["date"]
    D["home_team"] = M["home_team_id"].map(lk["team"])
    D["away_team"] = M["away_team_id"].map(lk["team"])
    D["gf_diff"] = H["gf"] - A["gf"]
    D["ga_diff"] = H["ga"] - A["ga"]
    D["rest_diff"] = H["rest"] - A["rest"]
    D["xg_diff"] = H["xg"] - A["xg"]
    D["ppg_diff"] = H["ppg"] - A["ppg"]
    D["fouls_diff"] = H["fouls"] - A["fouls"]
    D["is_knockout"] = (M["stage_id"] >= 2).astype(int)
    D["host_adv"] = (M["home_team_id"].map(lk["is_host"]) - M["away_team_id"].map(lk["is_host"]))
    D["goal_diff"] = M["home_score"] - M["away_score"]
    D = D.reset_index().sort_values("match_id").reset_index(drop=True)
    return D


def checks(D):
    assert len(D) == 104 and D.match_id.is_unique, "need 104 unique matches"
    assert D[FEATURES].shape[1] == 8
    assert not D[FEATURES + ["goal_diff"]].isna().any().any(), "missing values"
    # leakage guard: nothing about the match itself may enter its own features
    assert not any(c in FEATURES for c in ["home_score", "away_score", "goal_diff"])


if __name__ == "__main__":
    D = build()
    checks(D)
    PROC.mkdir(parents=True, exist_ok=True)
    D.to_csv(PROC / "wc2026_lr21_dataset.csv", index=False)
    print(D.shape); print(D.describe().T.round(3)); print(D.head(8).to_string())
