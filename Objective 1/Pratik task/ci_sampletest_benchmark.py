import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# FIFA WORLD CUP 2026 - TASK 2
# One-Sample T-Test Against Benchmark = 3
# ==========================================

# Load datasets
team_stats = pd.read_csv("data/match_team_stats.csv")
matches = pd.read_csv("data/matches.csv")

# ==========================================
# Create team-level goal data
# ==========================================

home_goals = matches[
    ["match_id", "home_team_id", "home_score"]
].copy()

home_goals = home_goals.rename(columns={
    "home_team_id": "team_id",
    "home_score": "goals"
})

away_goals = matches[
    ["match_id", "away_team_id", "away_score"]
].copy()

away_goals = away_goals.rename(columns={
    "away_team_id": "team_id",
    "away_score": "goals"
})

# Combine home and away goals
goals = pd.concat(
    [home_goals, away_goals],
    ignore_index=True
)

# Make sure IDs have the same data type
team_stats["match_id"] = team_stats["match_id"].astype(int)
team_stats["team_id"] = team_stats["team_id"].astype(int)

goals["match_id"] = goals["match_id"].astype(int)
goals["team_id"] = goals["team_id"].astype(int)

# ==========================================
# Merge team statistics and goals
# ==========================================

data = pd.merge(
    team_stats,
    goals,
    on=["match_id", "team_id"],
    how="inner"
)

# ==========================================
# Calculate shots on target per goal
# ==========================================

# Remove matches where team scored zero goals
scoring_data = data[data["goals"] > 0].copy()

scoring_data["shots_per_goal"] = (
    scoring_data["shots_on_target"] /
    scoring_data["goals"]
)

# ==========================================
# One-sample t-test
# Benchmark = 3
# ==========================================

benchmark = 3

sample = scoring_data["shots_per_goal"].dropna()

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(
    sample,
    benchmark
)

# ==========================================
# 95% Confidence Interval
# ==========================================

mean_value = sample.mean()
standard_error = stats.sem(sample)

confidence_level = 0.95
degrees_of_freedom = len(sample) - 1

t_critical = stats.t.ppf(
    (1 + confidence_level) / 2,
    degrees_of_freedom
)

margin_of_error = t_critical * standard_error

ci_lower = mean_value - margin_of_error
ci_upper = mean_value + margin_of_error

# ==========================================
# Print Results
# ==========================================

print("\n==========================================")
print("WORLD CUP 2026 - TASK 2 STATISTICAL TEST")
print("==========================================")

print(f"Number of observations: {len(sample)}")
print(f"Sample mean: {mean_value:.2f}")
print(f"Benchmark: {benchmark:.2f}")
print(f"Standard deviation: {sample.std():.2f}")

print("\n--- One-Sample T-Test ---")
print("Null hypothesis (H0): Mean = 3")
print("Alternative hypothesis (H1): Mean != 3")

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

print("\n--- 95% Confidence Interval ---")
print(f"Lower bound: {ci_lower:.2f}")
print(f"Upper bound: {ci_upper:.2f}")

# ==========================================
# Statistical Decision
# ==========================================

alpha = 0.05

print("\n--- Statistical Decision ---")

if p_value < alpha:
    print("Reject H0.")
    print(
        "There is statistically significant evidence "
        "that the mean differs from 3."
    )
else:
    print("Fail to reject H0.")
    print(
        "There is not enough statistical evidence "
        "to conclude that the mean differs from 3."
    )

# ==========================================
# Save statistical results
# ==========================================

results = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Sample mean",
        "Benchmark",
        "T-statistic",
        "P-value",
        "95% CI lower",
        "95% CI upper"
    ],
    "Value": [
        len(sample),
        mean_value,
        benchmark,
        t_statistic,
        p_value,
        ci_lower,
        ci_upper
    ]
})

results.to_csv(
    "task2_statistical_test_results.csv",
    index=False
)

print("\nResults saved to:")
print("task2_statistical_test_results.csv")