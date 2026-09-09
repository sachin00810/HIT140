import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# FIFA WORLD CUP 2026 - TASK 2
# Team Attacking Efficiency
# ==========================================

# Load datasets
stats = pd.read_csv("data/match_team_stats.csv")
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

# Combine home and away teams
goals = pd.concat([home_goals, away_goals], ignore_index=True)

# Make sure IDs have the same data type
stats["match_id"] = stats["match_id"].astype(int)
stats["team_id"] = stats["team_id"].astype(int)

goals["match_id"] = goals["match_id"].astype(int)
goals["team_id"] = goals["team_id"].astype(int)

# ==========================================
# Merge shots on target with goals
# ==========================================

team_data = pd.merge(
    stats,
    goals,
    on=["match_id", "team_id"],
    how="inner"
)

# ==========================================
# Calculate overall attacking efficiency
# ==========================================

total_shots_on_target = team_data["shots_on_target"].sum()
total_goals = team_data["goals"].sum()

shots_on_target_per_goal = total_shots_on_target / total_goals

print("\n==========================================")
print("WORLD CUP 2026 - TASK 2 RESULTS")
print("==========================================")

print(f"Total shots on target: {total_shots_on_target}")
print(f"Total goals scored: {total_goals}")
print(
    f"Average shots on target required for one goal: "
    f"{shots_on_target_per_goal:.2f}"
)

# ==========================================
# Team-level summary
# ==========================================

team_summary = team_data.groupby("team_id").agg(
    total_shots_on_target=("shots_on_target", "sum"),
    total_goals=("goals", "sum")
).reset_index()

team_summary["shots_on_target_per_goal"] = (
    team_summary["total_shots_on_target"] /
    team_summary["total_goals"]
)

# Only show teams that scored at least one goal
team_summary = team_summary[
    team_summary["total_goals"] > 0
]

team_summary = team_summary.sort_values(
    "shots_on_target_per_goal"
)

print("\n==========================================")
print("TEAM-LEVEL ATTACKING EFFICIENCY")
print("==========================================")

print(team_summary.to_string(index=False))

# ==========================================
# Save results
# ==========================================

team_summary.to_csv(
    "task2_team_efficiency_results.csv",
    index=False
)

print("\nResults saved to:")
print("task2_team_efficiency_results.csv")

# ==========================================
# Create graph
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    team_summary["team_id"].astype(str),
    team_summary["shots_on_target_per_goal"]
)

plt.xlabel("Team ID")
plt.ylabel("Shots on Target per Goal")
plt.title("World Cup 2026 - Shots on Target Required per Goal")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "task2_shots_on_target_per_goal.png",
    dpi=300
)

plt.show()

print("\nGraph saved to:")
print("task2_shots_on_target_per_goal.png")