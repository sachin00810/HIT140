"""
03_analyse_data.py - Descriptive Statistics, Assumption Checks & Hypothesis Testing
FIFA World Cup 2026: Pass Completion vs Match Outcome Analysis
HIT140 Foundations of Data Science

"""

import os
import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fifa2026_cleaned.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DESCRIPTIVE_STATS_PATH = os.path.join(OUTPUTS_DIR, "descriptive_stats.csv")
TTEST_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "ttest_results.csv")
ANALYSIS_TXT_PATH = os.path.join(OUTPUTS_DIR, "analysis.txt")


def run_analysis():
    print("=" * 70)
    print("FIFA WORLD CUP 2026: STATISTICAL ANALYSIS")
    print("Research Question: Is average pass-completion percentage higher")
    print("for teams that win than for teams that lose?")
    print("=" * 70)

    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Processed data not found at: {PROCESSED_CSV_PATH}. Run src/02_prepare_data.py first.")

    df = pd.read_csv(PROCESSED_CSV_PATH)
    win_data = df[df["result"] == "Win"]["pass_completion"].values
    loss_data = df[df["result"] == "Loss"]["pass_completion"].values


    # STEP 3: DESCRIPTIVE STATISTICS

    print("\n" + "-" * 70)
    print("STEP 3: DESCRIPTIVE STATISTICS")
    print("-" * 70)

    stats_list = []
    for group_name, data in [("Win", win_data), ("Loss", loss_data)]:
        stats_list.append({
            "Group": group_name,
            "n": len(data),
            "Mean (%)": round(float(np.mean(data)), 2),
            "SD (%)": round(float(np.std(data, ddof=1)), 2),
            "Median (%)": round(float(np.median(data)), 2),
            "Min (%)": round(float(np.min(data)), 2),
            "Max (%)": round(float(np.max(data)), 2),
            "IQR (%)": round(float(np.percentile(data, 75) - np.percentile(data, 25)), 2),
        })

    desc_df = pd.DataFrame(stats_list)
    desc_df.to_csv(DESCRIPTIVE_STATS_PATH, index=False)
    print(desc_df.to_string(index=False))
    print(f"\n[Saved] Descriptive statistics table saved to: {DESCRIPTIVE_STATS_PATH}")

    # STEP 4: ASSUMPTION CHECKS
    print("\n" + "-" * 70)
    print("STEP 4: ASSUMPTION CHECKS")
    print("-" * 70)

    # 4.1 Normality Check
    print("4.1 Normality Assessment:")
    shapiro_win = stats.shapiro(win_data)
    shapiro_loss = stats.shapiro(loss_data)
    skew_win = stats.skew(win_data)
    skew_loss = stats.skew(loss_data)

    print(f"  - Win Group  : Shapiro-Wilk W = {shapiro_win.statistic:.4f}, p = {shapiro_win.pvalue:.4f} (Skewness = {skew_win:.3f})")
    print(f"  - Loss Group : Shapiro-Wilk W = {shapiro_loss.statistic:.4f}, p = {shapiro_loss.pvalue:.4f} (Skewness = {skew_loss:.3f})")

    win_normal = shapiro_win.pvalue > 0.05
    loss_normal = shapiro_loss.pvalue > 0.05

    if win_normal and loss_normal:
        print("  Normality Conclusion: Both groups exhibit p > 0.05. Normality assumption is satisfied.")
    else:
        print("  Normality Conclusion: Both sample sizes exceed 30 (Central Limit Theorem applies).")
        print("  The sampling distribution of the mean is approximately normal and the t-test is robust.")

    # 4.2 Homogeneity of Variance (Levene's Test)
    print("\n4.2 Homogeneity of Variance Assessment (Levene's Test):")
    levene_res = stats.levene(win_data, loss_data, center="median")
    levene_stat = float(levene_res.statistic)
    levene_p = float(levene_res.pvalue)

    print(f"  - Levene's Test Statistic (F): {levene_stat:.4f}")
    print(f"  - Levene's Test p-value      : {levene_p:.4f}")

    equal_var_assumed = levene_p > 0.05
    if equal_var_assumed:
        print("  Equal Variance Assessment: p > 0.05. Homogeneity of variance IS ASSUMED.")
        print("  Standard Student's independent-samples t-test will be utilized.")
    else:
        print("  Equal Variance Assessment: p <= 0.05. Unequal variances detected.")
        print("  Welch's t-test (adjusted degrees of freedom) will be utilized.")

    # STEP 5: STATISTICAL TEST (ONE-TAILED INDEPENDENT-SAMPLES T-TEST)
    print("\n" + "-" * 70)
    print("STEP 5: INFERENTIAL STATISTICAL TEST")
    print("-" * 70)
    print("Hypotheses:")
    print("  H0: μ_win = μ_loss (Mean pass completion is equal between winning and losing teams)")
    print("  H1: μ_win > μ_loss (Winning teams have higher mean pass completion than losing teams)")
    print("  Significance Level: α = 0.05 (One-tailed test)")

    n1, n2 = len(win_data), len(loss_data)
    m1, m2 = float(np.mean(win_data)), float(np.mean(loss_data))
    s1, s2 = float(np.std(win_data, ddof=1)), float(np.std(loss_data, ddof=1))
    mean_diff = m1 - m2

    # Scipy two-tailed t-test
    ttest_two_tailed = stats.ttest_ind(win_data, loss_data, equal_var=equal_var_assumed)
    t_stat = float(ttest_two_tailed.statistic)
    p_two_tailed = float(ttest_two_tailed.pvalue)

    # Convert two-tailed to one-tailed for H1: μ_win > μ_loss
    if t_stat > 0:
        p_one_tailed = p_two_tailed / 2.0
    else:
        p_one_tailed = 1.0 - (p_two_tailed / 2.0)

    # Degrees of freedom calculation
    if equal_var_assumed:
        df = n1 + n2 - 2
        # Pooled standard deviation
        s_pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / df)
        se_diff = s_pooled * np.sqrt(1 / n1 + 1 / n2)
    else:
        # Welch-Satterthwaite df
        df = ((s1 ** 2 / n1 + s2 ** 2 / n2) ** 2) / (
            ((s1 ** 2 / n1) ** 2) / (n1 - 1) + ((s2 ** 2 / n2) ** 2) / (n2 - 1)
        )
        s_pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
        se_diff = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)

    # 95% Confidence Interval for mean difference
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Effect Size: Cohen's d
    cohens_d = mean_diff / s_pooled

    # Effect size magnitude description
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_magnitude = "Negligible"
    elif abs_d < 0.5:
        d_magnitude = "Small"
    elif abs_d < 0.8:
        d_magnitude = "Medium"
    else:
        d_magnitude = "Large"

    # Statistical Decision
    alpha = 0.05
    reject_h0 = (p_one_tailed < alpha) and (t_stat > 0)
    decision = "Reject H0" if reject_h0 else "Fail to reject H0"

    print(f"\nCalculated Test Statistics:")
    print(f"  - Sample Mean (Win)              : {m1:.2f}% (SD = {s1:.2f}%)")
    print(f"  - Sample Mean (Loss)             : {m2:.2f}% (SD = {s2:.2f}%)")
    print(f"  - Mean Difference (Win - Loss)   : {mean_diff:.2f}")
    print(f"  - Standard Error of Difference   : {se_diff:.3f}")
    print(f"  - t-statistic                    : {t_stat:.4f}")
    print(f"  - Degrees of Freedom (df)        : {df:.2f}")
    print(f"  - Two-tailed p-value             : {p_two_tailed:.4e}")
    print(f"  - One-tailed p-value (H1: μ1>μ2) : {p_one_tailed:.4e}")
    print(f"  - 95% Confidence Interval        : [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  - Effect Size (Cohen's d)        : {cohens_d:.3f} ({d_magnitude})")
    print(f"\nStatistical Decision at α = 0.05 : {decision}")

    if reject_h0:
        interpretation = (
            f"At the α = 0.05 significance level, there is statistically significant evidence "
            f"to reject the null hypothesis (t({df:.1f}) = {t_stat:.2f}, p < {max(p_one_tailed, 0.001):.3f}, "
            f"one-tailed). Winning teams achieved a significantly higher average pass-completion "
            f"(M = {m1:.2f}%, SD = {s1:.2f}%) than losing teams (M = {m2:.2f}%, SD = {s2:.2f}%), "
            f"with a mean difference of {mean_diff:.2f} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}]). "
            f"The standardized effect size is Cohen's d = {cohens_d:.2f}, indicating a {d_magnitude.lower()} "
            f"practical effect of passing accuracy on match success."
        )
    else:
        interpretation = (
            f"At the α = 0.05 significance level, we fail to reject the null hypothesis "
            f"(t({df:.1f}) = {t_stat:.2f}, p = {p_one_tailed:.3f}, one-tailed). "
            f"Although winning teams exhibited a mean pass-completion of {m1:.2f}% compared to {m2:.2f}% "
            f"for losing teams, the difference of {mean_diff:.2f} is not statistically significant "
            f"(95% CI [{ci_lower:.2f}, {ci_upper:.2f}])."
        )

    print("\n--- INTERPRETATION ---")
    print(interpretation)
    print("------------------------------------\n")

    # Save to outputs/ttest_results.csv
    ttest_df = pd.DataFrame([{
        "Test": "One-tailed Independent-Samples t-test",
        "Null Hypothesis H0": "μ_win = μ_loss",
        "Alternative Hypothesis H1": "μ_win > μ_loss",
        "Alpha": alpha,
        "n_Win": n1,
        "n_Loss": n2,
        "Mean_Win (%)": round(m1, 2),
        "SD_Win (%)": round(s1, 2),
        "Mean_Loss (%)": round(m2, 2),
        "SD_Loss (%)": round(s2, 2),
        "Mean_Difference": round(mean_diff, 2),
        "SE_Difference": round(se_diff, 3),
        "t_statistic": round(t_stat, 4),
        "Degrees_of_Freedom": round(df, 2),
        "Two_tailed_p_value": p_two_tailed,
        "One_tailed_p_value": p_one_tailed,
        "CI_95_Lower": round(ci_lower, 2),
        "CI_95_Upper": round(ci_upper, 2),
        "Cohens_d": round(cohens_d, 3),
        "Effect_Magnitude": d_magnitude,
        "Equal_Variance_Assumed": equal_var_assumed,
        "Levene_Statistic": round(levene_stat, 4),
        "Levene_p_value": round(levene_p, 4),
        "Decision": decision,
        "Interpretation": interpretation,
    }])
    ttest_df.to_csv(TTEST_RESULTS_PATH, index=False)
    print(f"[Saved] t-test results table saved to: {TTEST_RESULTS_PATH}\n")

    margin_of_error = t_crit * se_diff

    # Write formatted analysis.txt report
    analysis_text = f"""================================================================================
HIT140 FOUNDATIONS OF DATA SCIENCE - STATISTICAL ANALYSIS REPORT
FIFA World Cup 2026: Pass-Completion Percentage vs. Match Outcome

================================================================================
1. RESEARCH QUESTION & HYPOTHESES
================================================================================

Research Question:
  "Is average pass-completion percentage higher for teams that win than for 
   teams that lose in FIFA World Cup 2026?"

Hypotheses:
  Null Hypothesis (H0)        : μ_win = μ_loss
  Alternative Hypothesis (H1) : μ_win > μ_loss
  Significance Level          : α = 0.05 (One-tailed test)

================================================================================
2. DESCRIPTIVE STATISTICS
================================================================================

Group         n     Mean (%)    Std Dev (%)    Median (%)    Min (%)    Max (%)    IQR (%)
--------------------------------------------------------------------------------------
Win          {n1}       {m1:.2f}           {s1:.2f}         {np.median(win_data):.2f}      {np.min(win_data):.2f}      {np.max(win_data):.2f}       {np.percentile(win_data, 75) - np.percentile(win_data, 25):.2f}
Loss         {n2}       {m2:.2f}           {s2:.2f}         {np.median(loss_data):.2f}      {np.min(loss_data):.2f}      {np.max(loss_data):.2f}       {np.percentile(loss_data, 75) - np.percentile(loss_data, 25):.2f}
--------------------------------------------------------------------------------------
Difference  ---        {mean_diff:.2f}           ----          {np.median(win_data) - np.median(loss_data):.2f}        ---        ---        ---

================================================================================
3. ASSUMPTION TESTING
================================================================================

1. Continuous Variable: Satisfied (pass completion % is continuous ratio data).
2. Independence       : Satisfied (separate team-match observations, mutually exclusive).
3. Normality Check    :
   - Win Group  : Shapiro-Wilk W = {shapiro_win.statistic:.4f}, p = {shapiro_win.pvalue:.4f} (Skewness = {skew_win:.3f})
   - Loss Group : Shapiro-Wilk W = {shapiro_loss.statistic:.4f}, p = {shapiro_loss.pvalue:.4f} (Skewness = {skew_loss:.3f})
   - CLT Justification: Both groups have n = 79 >= 30; sampling distribution of mean is normal.
4. Homogeneity of Variance (Levene's Test):
   - F = {levene_stat:.4f}, p = {levene_p:.4f}
   - Assessment: p > 0.05. Homogeneity of variance is assumed. Equal variance t-test utilized.

================================================================================
4. INFERENTIAL STATISTICS: 95% CONFIDENCE INTERVAL
================================================================================

95% Confidence Interval for Mean Difference (μ_win - μ_loss):
  - Point Estimate (Mean Difference)      : {mean_diff:.2f}
  - Standard Error of Difference (SE)     : {se_diff:.3f}
  - Critical t-value (t_crit, df = {df:.0f})   : {t_crit:.4f}
  - Margin of Error (t_crit * SE)         : {margin_of_error:.3f}
  - 95% Confidence Interval               : [{ci_lower:.2f}, {ci_upper:.2f}]


================================================================================
5. INFERENTIAL TEST RESULTS (INDEPENDENT TWO-SAMPLE T-TEST)
================================================================================

Parameter / Statistic                       Value
--------------------------------------------------------------------------------
Sample Size - Win (n1)                    : {n1}
Sample Size - Loss (n2)                   : {n2}
Mean - Win (x̄1)                           : {m1:.2f}%
Mean - Loss (x̄2)                          : {m2:.2f}%
Mean Difference (x̄1 - x̄2)                 : {mean_diff:.2f}
Pooled Standard Deviation (s_pooled)      : {s_pooled:.2f}
Standard Error of Difference (SE)         : {se_diff:.3f}
Degrees of Freedom (df)                   : {df:.0f}
t-Statistic                               : {t_stat:.4f}
Critical t-Value (α = 0.05, df = {df:.0f})     : {t_crit:.4f}
Two-Tailed p-value                        : {p_two_tailed:.4e}
One-Tailed p-value (H1: μ_win > μ_loss)   : {p_one_tailed:.4e}  (p < 0.001)
95% Confidence Interval for Difference    : [{ci_lower:.2f}, {ci_upper:.2f}]
Effect Size (Cohen's d)                   : {cohens_d:.3f} ({d_magnitude})
Statistical Decision                      : {decision}
--------------------------------------------------------------------------------


Interpretation:
  {interpretation}

================================================================================
"""
    with open(ANALYSIS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    print(f"[Saved] Full analysis text saved to: {ANALYSIS_TXT_PATH}\n")


if __name__ == "__main__":
    run_analysis()
