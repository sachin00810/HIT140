# HIT140 Objective 2 - Linear Regression 2.1 (goal difference, 104 matches)

Predict `goal_diff = home_score - away_score` for each of the 104 World Cup 2026 matches
from 8 variables known BEFORE kick-off.

## Run
```
pip install -r requirements.txt
python script/build_dataset.py   # -> data/processed/wc2026_lr21_dataset.csv (104 rows)
python script/analysis.py        # -> output/ (tables, figures, results_log.txt)
```

## Data
`data/raw/matches.csv`, `match_team_stats.csv` are copied from the group repo (`Pratik task/data/`).
`data/raw/team_lookup.csv` maps team_id -> name (+ host flag); checked against every match
in `Dorjee/data/raw/Rawdata.csv` (103/103 scores and team pairs agree; the third-place
play-off is only in Pratik's files).

## The 8 explanatory variables (all "home minus away" unless stated)
| # | Variable | Meaning | Overlap with teammate's 2.2 list |
|---|----------|---------|----------------------------------|
| 1 | gf_diff | previous avg goals scored | shares idea with 2.2 #1 |
| 2 | ga_diff | previous avg goals conceded | shares idea with 2.2 #2 / #8 |
| 3 | rest_diff | rest days since own last match | shares idea with 2.2 #7 |
| 4 | xg_diff | previous avg expected goals (xG) | none |
| 5 | ppg_diff | tournament points per game so far | none |
| 6 | fouls_diff | previous avg fouls committed | none |
| 7 | is_knockout | knockout stage = 1 | none |
| 8 | host_adv | +1 home side is host (MEX/USA/CAN), -1 away side is, else 0 | none |

At most 3 ideas are shared with 2.2 (limit is 4). Shots, shots on target, possession and
corners are deliberately NOT used here.

## Leakage control
Features use only matches that kicked off earlier. Tested: changing a match's score changes
later matches' features but never its own or earlier ones. A team's opener (no history) gets
the tournament-wide average of finished matches (fill="tournament_mean").
