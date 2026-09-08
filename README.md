# HIT140 Assessment 2 — Objective 1

**Analytic task:** Is there a significant difference in the average number of
corners won by teams that **win** their match compared with teams that **lose**?
(2026 FIFA World Cup — Group 10)

## Layout

| Folder / file | What it is |
|---|---|
| `data/raw/wc2026_fixtures_raw.csv` | Raw input — one row per team per match, fields straight from the source fixtures table (no derived columns). 206 rows, 103 matches. |
| `data/raw_data.csv` | The source site's separate per-team aggregate view. Reference only — used to cross-check corner totals, never to change a value. |
| `data/wc2026_match_corners.csv` | Analysis dataset — the raw rows cleaned, with `outcome`, `goal_diff`, `is_knockout` added. Built by `build_dataset.py`. |
| `script/build_dataset.py` | `data/raw/` → `data/wc2026_match_corners.csv`: name fixes, integrity checks, derived columns, cross-check against the aggregate view. |
| `script/analysis.py` | The analysis: wrangling, stratified sampling, descriptives, 95% CIs, Welch t-test, figure. |
| `output/` | Everything `analysis.py` produces (regenerated on each run). |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python script/build_dataset.py     # rebuild the analysis dataset from data/raw/
python script/analysis.py          # run the analysis, write output/
```

Paths are resolved from the project root, so the working directory does not matter.
`build_dataset.py` reproduces the committed `data/wc2026_match_corners.csv` byte for byte.

### Outputs written to `output/`

| File | Contents |
|---|---|
| `corners_winners_vs_losers.png` | Boxplot + histogram of corners by outcome. |
| `descriptive_statistics.txt` / `.csv` | n, mean, median, sd, quartiles, skewness per group. |
| `confidence_intervals.txt` | 95% t-CIs for each group mean and for the difference (Welch). |
| `hypothesis_test_welch.txt` / `.csv` | Welch two-sample t-test: hypotheses, group summaries, mean difference, 95% CI, t, df, p, Cohen's d, decision. |

## Method (summary)

- **Unit of observation:** one team in one match. **Variable:** corners won (count).
- **Exclusion rule:** drawn matches dropped — in a draw neither team won nor lost.
- **Sample:** stratified simple random sample, 60 per group, seed `140`.
- **Test:** Welch two-sample t-test (`equal_var=False`), two-sided, alpha = 0.05.
- H0: mu_win = mu_loss  Ha: mu_win != mu_loss

## Result

Winning teams won on average **5.33** corners vs **3.85** for losing teams — a
difference of **1.48** corners per match. Welch t = 2.68, p = 0.0084, 95% CI for
the difference [0.39, 2.59] (excludes zero), Cohen's d = 0.49. **Reject H0.**
The effect is real but small-to-moderate: corners alone do not decide matches.

### Limitations

- Association, not causation — winning and winning corners both reflect control of play.
- Describes decided matches only (draws excluded).
- The third-place play-off is absent from the source fixtures table (103 of 104 matches).
- The source site's aggregate view disagrees on corner totals for 12 teams (by 1–9);
  the twice-verified fixtures figures are used. `build_dataset.py` prints the gaps.
