"""HIT140 Assessment 2 — Objective 1  |  Group 10

ANALYTIC QUESTION
  Is there a significant difference in the average number of corners won by
  teams that WIN their match compared with teams that LOSE?

  Population           : all team-match performances at the 2026 FIFA World Cup
  Unit of observation  : one team in one match
  Variable of interest : corners won by that team in that match (count)
  Grouping             : outcome, restricted to Win vs Loss
  Exclusion rule       : drawn matches excluded — in a draw neither team won
                         nor lost, so those rows belong to neither group
  H0: mu_win  = mu_loss
  Ha: mu_win != mu_loss        (two-sided)
  Test: Welch two-sample t-test (does not assume equal variances)

WHY TWO-SIDED
  The direction is genuinely uncertain. Winning teams may control play and win
  more corners; but losing teams chase the game and attack more in the closing
  stages. A one-tailed test would make a counter-intuitive result unreportable.

DATA PROVENANCE — put this on your slide
  Source: The Stats Don't Lie, FIFA World Cup 2026 fixtures table.
  Coverage: 103 of 104 matches (206 team-match rows). The third-place play-off
  is absent from the source fixtures table; the site's own "Total Games Played:
  104" does not reconcile with the 103 matches it actually lists.
  Cross-validation: corner totals were checked against the site's SEPARATE
  per-team aggregate view (data/raw/wc2026_team_corner_totals.csv). Twelve teams
  disagree; build_dataset.py prints the exact gaps on every run. The affected
  fixtures rows were re-read from the live table and confirmed unchanged, so the
  discrepancy is an inconsistency within the source website, not a transcription
  error. Match-level figures are used here because they are the correct unit of
  observation and were verified twice.

HOW TO RUN
  python script/build_dataset.py     # data/raw/ -> data/processed/wc2026_match_corners.csv
  python script/analysis.py          # this script

  Reads : data/processed/wc2026_match_corners.csv
  Writes: output/corners_winners_vs_losers.png
          output/descriptive_statistics.txt   and .csv
          output/confidence_intervals.txt
          output/hypothesis_test_welch.txt    and .csv

Requires: pandas, numpy, scipy, matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# CONFIG
# =============================================================================
# Paths are resolved from the project root (the folder that holds data/, script/
# and output/) so the script runs the same no matter what the working directory
# is.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "wc2026_match_corners.csv"
FIGURE_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR = PROJECT_ROOT / "output"      # descriptive stats, CIs and test results
TASK_NAME = "corners_winners_vs_losers"

VARIABLE = "corners"
GROUP_COLUMN = "outcome"
GROUP_A, GROUP_B = "Win", "Loss"
LABEL_A, LABEL_B = "Winning teams", "Losing teams"

CONFIDENCE = 0.95
ALPHA = 0.05
RANDOM_SEED = 140                  # agreed group-wide

# Sampling. The 158 non-draw rows are the full population of decided
# performances at this tournament.
#   "census" : analyse all of them.
#   "sample" : stratified simple random sample of SAMPLE_N rows per group.
#              This is what the brief asks for, so it is the default.
SAMPLING_MODE = "sample"
SAMPLE_N = 60                      # per group


# =============================================================================
# 1. DATA WRANGLING
# =============================================================================
def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"Rows loaded: {len(df)}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    required = {"match_number", "team", "opponent", VARIABLE,
                "goals_for", "goals_against", GROUP_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    df[VARIABLE] = pd.to_numeric(df[VARIABLE], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset=["match_number", "team"])
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate team-match rows")

    before = len(df)
    df = df.dropna(subset=[VARIABLE, GROUP_COLUMN])
    print(f"Dropped {before - len(df)} rows with missing values")

    pairs = df.match_number.value_counts()
    print(f"Matches covered: {df.match_number.nunique()} "
          f"(every match has 2 rows: {bool((pairs == 2).all())})")
    print(f"Distinct teams: {df.team.nunique()}")
    print(f"Corners: min = {df[VARIABLE].min()}, max = {df[VARIABLE].max()}")
    if (df[VARIABLE] < 0).any():
        print("  WARNING: negative corner counts found")

    return df


def apply_exclusion_rules(df):
    """Drop drawn matches — the key design decision for this task."""
    before = len(df)
    df = df[df[GROUP_COLUMN] != "Draw"]
    removed = before - len(df)
    print(f"\nExcluded {removed} rows from drawn matches ({removed // 2} matches); "
          f"{len(df)} rows remain")
    counts = df[GROUP_COLUMN].value_counts().to_dict()
    print(f"  Group sizes: {counts}")
    if counts.get("Win") != counts.get("Loss"):
        print("  WARNING: wins should equal losses")
    return df


# =============================================================================
# 2. DATA PREPARATION AND SAMPLING
# =============================================================================
def prepare_sample(df):
    if SAMPLING_MODE == "census":
        print(f"\nSampling: CENSUS — all {len(df)} decided team-match rows used.")
        return df.copy()

    if SAMPLING_MODE == "sample":
        parts = []
        for g in (GROUP_A, GROUP_B):
            sub = df[df[GROUP_COLUMN] == g]
            n = min(SAMPLE_N, len(sub))
            parts.append(sub.sample(n=n, random_state=RANDOM_SEED))
            print(f"  {g}: sampled {n} of {len(sub)}")
        out = pd.concat(parts)
        print(f"\nSampling: STRATIFIED SIMPLE RANDOM, seed = {RANDOM_SEED}, "
              f"total n = {len(out)}")
        print("  Stratified so both groups are equally represented, which keeps")
        print("  the comparison balanced for a given sample size.")
        return out

    raise ValueError("SAMPLING_MODE must be 'census' or 'sample'")


# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
def describe(series, label):
    s = series.dropna()
    out = {
        "n": len(s), "mean": s.mean(), "median": s.median(),
        "std_dev": s.std(ddof=1),          # ddof=1 = SAMPLE standard deviation
        "min": s.min(), "q1": s.quantile(0.25), "q3": s.quantile(0.75),
        "max": s.max(), "skewness": s.skew(),
    }
    print(f"\n--- Descriptive statistics: {label} ---")
    for k, v in out.items():
        print(f"  {k:>10}: {v:.4f}" if isinstance(v, float) else f"  {k:>10}: {v}")
    return out


def plot_groups(a, b, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].boxplot([a, b])
    axes[0].set_xticklabels([LABEL_A, LABEL_B])
    axes[0].set_ylabel("Corners won in the match")
    axes[0].set_title("Corners won by match outcome")

    bins = np.arange(-0.5, max(a.max(), b.max()) + 1.5, 1)
    axes[1].hist(a, bins=bins, alpha=0.6, label=f"{LABEL_A} (n={len(a)})",
                 edgecolor="black")
    axes[1].hist(b, bins=bins, alpha=0.6, label=f"{LABEL_B} (n={len(b)})",
                 edgecolor="black")
    axes[1].set_xlabel("Corners won in the match")
    axes[1].set_ylabel("Number of team-matches")
    axes[1].set_title("Distribution by outcome")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {filename}")


# =============================================================================
# 4. CONFIDENCE INTERVALS
# =============================================================================
def ci_mean(series, label, confidence=CONFIDENCE):
    """t-based CI for one group mean. We use t rather than z because the
    population standard deviation is unknown and estimated from the sample."""
    s = series.dropna()
    n, xbar, sd = len(s), s.mean(), s.std(ddof=1)
    se = sd / np.sqrt(n)
    t_star = stats.t.ppf((1 + confidence) / 2, n - 1)
    lo, hi = xbar - t_star * se, xbar + t_star * se
    print(f"\n--- {int(confidence*100)}% CI for mean: {label} ---")
    print(f"  mean = {xbar:.4f}, se = {se:.4f}, t* = {t_star:.4f} (df = {n-1})")
    print(f"  CI   = [{lo:.4f}, {hi:.4f}]")
    return {"mean": xbar, "se": se, "t_star": t_star, "df": n - 1,
            "margin": t_star * se, "lower": lo, "upper": hi}


def ci_difference(a, b, confidence=CONFIDENCE):
    """Welch CI for the difference of means (mean_a - mean_b).

    Welch-Satterthwaite degrees of freedom, because the two groups may have
    different variances. If this interval excludes zero, it agrees with a
    significant two-sided t-test at the same alpha.
    """
    a, b = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = a.mean() - b.mean()
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / \
         ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_star = stats.t.ppf((1 + confidence) / 2, df)
    lo, hi = diff - t_star * se, diff + t_star * se
    print(f"\n--- {int(confidence*100)}% CI for the DIFFERENCE ({LABEL_A} - {LABEL_B}) ---")
    print(f"  difference = {diff:.4f}, se = {se:.4f}, df = {df:.2f}")
    print(f"  CI         = [{lo:.4f}, {hi:.4f}]")
    print(f"  Interval {'EXCLUDES' if lo > 0 or hi < 0 else 'INCLUDES'} zero")
    return {"diff": diff, "se": se, "df": df, "lower": lo, "upper": hi}


# =============================================================================
# 5. HYPOTHESIS TEST
# =============================================================================
def welch_t_test(a, b, alpha=ALPHA):
    a, b = a.dropna(), b.dropna()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled_sd

    print(f"\n--- Welch two-sample t-test ---")
    print(f"  H0: mu_win = mu_loss")
    print(f"  Ha: mu_win != mu_loss")
    print(f"  {LABEL_A:<16} n = {len(a):>3}, mean = {a.mean():.4f}, sd = {a.std(ddof=1):.4f}")
    print(f"  {LABEL_B:<16} n = {len(b):>3}, mean = {b.mean():.4f}, sd = {b.std(ddof=1):.4f}")
    print(f"  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"  Cohen's d = {d:.4f}")
    print(f"  Decision at alpha = {alpha}: "
          f"{'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0'}")
    return {"t": t_stat, "p": p_value, "cohens_d": d}


def interpret(test, ci):
    print("\n--- Interpretation ---")
    if test["p"] < ALPHA:
        more = "more" if ci["diff"] > 0 else "fewer"
        print(f"  Winning teams won {more} corners than losing teams, by an average of")
        print(f"  {abs(ci['diff']):.2f} corners per match. The difference is statistically")
        print(f"  significant (p = {test['p']:.4f}) and the 95% CI for the difference")
        print(f"  excludes zero, so we reject H0.")
        print(f"  Cohen's d = {test['cohens_d']:.2f} — a small-to-moderate effect. The")
        print(f"  difference is real but modest; corners alone do not decide matches.")
    else:
        print(f"  Not enough evidence of a difference (p = {test['p']:.4f}).")
        print(f"  'Not significant' means no difference was DETECTED, not that the")
        print(f"  two groups were proven identical.")
    print("\n  Limitations to state on the slide:")
    print("   - Association, not causation. Winning does not cause corners, nor the")
    print("     reverse; both reflect a team's control of the match.")
    print("   - Drawn matches were excluded, so this describes decided matches only.")
    print("   - The third-place play-off is missing from the source data.")


# =============================================================================
# 6. SAVE RESULTS TO FILES
# =============================================================================
def save_descriptives(desc_a, desc_b):
    """Descriptive statistics as .txt (readable) and .csv (importable)."""
    rows = []
    for label, d in ((LABEL_A, desc_a), (LABEL_B, desc_b)):
        r = {"group": label}
        r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)

    with open(OUTPUT_DIR / "descriptive_statistics.txt", "w") as f:
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("Task: corners won by winning teams vs losing teams\n")
        f.write(f"Variable: {VARIABLE} — corners won by a team in a single match\n")
        f.write("=" * 62 + "\n\n")
        for label, d in ((LABEL_A, desc_a), (LABEL_B, desc_b)):
            f.write(f"{label}\n{'-' * len(label)}\n")
            for k, v in d.items():
                f.write(f"  {k:>10} : {v:.4f}\n" if isinstance(v, float)
                        else f"  {k:>10} : {v}\n")
            f.write("\n")
        f.write("Note: the median is reported next to the mean because corner counts\n")
        f.write("are right-skewed. The positive skewness values confirm this.\n")
    print(f"  wrote {OUTPUT_DIR / 'descriptive_statistics.txt'} and .csv")


def save_confidence_intervals(ci_a, ci_b, ci_diff):
    with open(OUTPUT_DIR / "confidence_intervals.txt", "w") as f:
        f.write("95% CONFIDENCE INTERVALS\n")
        f.write(f"Variable: {VARIABLE}\n")
        f.write("Method: t-based interval. The population standard deviation is\n")
        f.write("unknown and estimated from the sample, so t* is used, not z.\n")
        f.write("Formula: xbar +/- t* * (s / sqrt(n))\n")
        f.write("=" * 62 + "\n\n")
        for label, ci in ((LABEL_A, ci_a), (LABEL_B, ci_b)):
            f.write(f"{label}\n")
            f.write(f"  sample mean     : {ci['mean']:.4f}\n")
            f.write(f"  standard error  : {ci['se']:.4f}\n")
            f.write(f"  t*              : {ci['t_star']:.4f}   (df = {ci['df']})\n")
            f.write(f"  margin of error : {ci['margin']:.4f}\n")
            f.write(f"  95% CI          : [{ci['lower']:.4f}, {ci['upper']:.4f}]\n\n")
        f.write(f"DIFFERENCE ({LABEL_A} - {LABEL_B})\n")
        f.write("Method: Welch interval using Welch-Satterthwaite degrees of\n")
        f.write("freedom, because the two groups may have unequal variances.\n")
        f.write(f"  difference      : {ci_diff['diff']:.4f}\n")
        f.write(f"  standard error  : {ci_diff['se']:.4f}\n")
        f.write(f"  df              : {ci_diff['df']:.2f}\n")
        f.write(f"  95% CI          : [{ci_diff['lower']:.4f}, {ci_diff['upper']:.4f}]\n")
        excl = ci_diff["lower"] > 0 or ci_diff["upper"] < 0
        f.write(f"  Zero is {'EXCLUDED' if excl else 'INCLUDED'}, which is consistent with a\n")
        f.write(f"  {'significant' if excl else 'non-significant'} two-sided test at alpha = {ALPHA}.\n")
    print(f"  wrote {OUTPUT_DIR / 'confidence_intervals.txt'}")


def save_hypothesis_test(test, desc_a, desc_b, ci_diff):
    row = {
        "test": "Welch independent samples t-test",
        "variable": VARIABLE,
        "group_a": LABEL_A, "n_a": desc_a["n"],
        "mean_a": round(desc_a["mean"], 4), "sd_a": round(desc_a["std_dev"], 4),
        "group_b": LABEL_B, "n_b": desc_b["n"],
        "mean_b": round(desc_b["mean"], 4), "sd_b": round(desc_b["std_dev"], 4),
        "mean_difference": round(ci_diff["diff"], 4),
        "ci_lower": round(ci_diff["lower"], 4), "ci_upper": round(ci_diff["upper"], 4),
        "t_statistic": round(test["t"], 4), "df": round(ci_diff["df"], 2),
        "p_value": round(test["p"], 6), "alpha": ALPHA,
        "cohens_d": round(test["cohens_d"], 4),
        "decision": "Reject H0" if test["p"] < ALPHA else "Fail to reject H0",
    }
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "hypothesis_test_welch.csv", index=False)

    with open(OUTPUT_DIR / "hypothesis_test_welch.txt", "w") as f:
        f.write("HYPOTHESIS TEST — WELCH INDEPENDENT SAMPLES t-TEST\n")
        f.write("=" * 62 + "\n\n")
        f.write("Research question\n-----------------\n")
        f.write("  Is there a significant difference in the average number of corners\n")
        f.write("  won by teams that win their match compared with teams that lose?\n\n")
        f.write("Hypotheses\n----------\n")
        f.write("  H0 : mu_win  = mu_loss\n")
        f.write("  Ha : mu_win != mu_loss    (two-sided)\n\n")
        f.write("Why Welch\n---------\n")
        f.write("  equal_var=False, so equal population variances are not assumed.\n")
        f.write("  This is the safer default and costs almost nothing in power.\n\n")
        f.write("Group summaries\n---------------\n")
        f.write(f"  {LABEL_A:<16} n = {desc_a['n']:>3}, mean = {desc_a['mean']:.4f}, "
                f"sd = {desc_a['std_dev']:.4f}\n")
        f.write(f"  {LABEL_B:<16} n = {desc_b['n']:>3}, mean = {desc_b['mean']:.4f}, "
                f"sd = {desc_b['std_dev']:.4f}\n\n")
        f.write("Test results\n------------\n")
        f.write(f"  mean difference    : {ci_diff['diff']:.4f}\n")
        f.write(f"  95% CI for diff    : [{ci_diff['lower']:.4f}, {ci_diff['upper']:.4f}]\n")
        f.write(f"  t statistic        : {test['t']:.4f}\n")
        f.write(f"  degrees of freedom : {ci_diff['df']:.2f}\n")
        f.write(f"  p value            : {test['p']:.6f}\n")
        f.write(f"  Cohen's d          : {test['cohens_d']:.4f}\n")
        f.write(f"  alpha              : {ALPHA}\n")
        f.write(f"  DECISION           : "
                f"{'REJECT H0' if test['p'] < ALPHA else 'FAIL TO REJECT H0'}\n")
    print(f"  wrote {OUTPUT_DIR / 'hypothesis_test_welch.txt'} and .csv")


# =============================================================================
# 7. RUN
# =============================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"HIT140 Group 10 — {TASK_NAME}")
    print("=" * 70)

    df = load_and_clean(DATA_PATH)
    decided = apply_exclusion_rules(df)
    sample = prepare_sample(decided)

    a = sample.loc[sample[GROUP_COLUMN] == GROUP_A, VARIABLE]
    b = sample.loc[sample[GROUP_COLUMN] == GROUP_B, VARIABLE]

    desc_a = describe(a, LABEL_A)
    desc_b = describe(b, LABEL_B)
    plot_groups(a.values, b.values, FIGURE_DIR / f"{TASK_NAME}.png")

    ci_a = ci_mean(a, LABEL_A)
    ci_b = ci_mean(b, LABEL_B)
    ci = ci_difference(a, b)

    test = welch_t_test(a, b)
    interpret(test, ci)

    print(f"\n--- Saving result files to {OUTPUT_DIR}/ ---")
    save_descriptives(desc_a, desc_b)
    save_confidence_intervals(ci_a, ci_b, ci)
    save_hypothesis_test(test, desc_a, desc_b, ci)

    print("\nDone.")


if __name__ == "__main__":
    main()
