
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------------
# 1. Set paths
# -----------------------------------------

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

matches = pd.read_csv(DATA / "matches.csv")
stats = pd.read_csv(DATA / "match_team_stats.csv")

# Change this if your dataset uses another date column
date_candidates = ["match_date", "date", "kickoff_date"]

date_col = next(
    (c for c in date_candidates if c in matches.columns),
    None
)

if date_col is None:
    raise ValueError(
        "No match date column found. "
        "Check matches.csv and update date_candidates."
    )

required_matches = [
    "match_id", "home_team_id", "away_team_id",
    "home_score", "away_score"
]

required_stats = [
    "match_id", "team_id", "shots_on_target",
    "total_shots", "possession_pct", "corners"
]

for name, df, required in [
    ("matches", matches, required_matches),
    ("stats", stats, required_stats)
]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

matches["match_date"] = pd.to_datetime(
    matches[date_col], errors="raise", utc=True
)

if matches["match_id"].duplicated().any():
    raise ValueError("Duplicate match IDs found.")

if stats.duplicated(["match_id", "team_id"]).any():
    raise ValueError("Duplicate team-match records found.")

# -----------------------------------------
# 2. Create one row per team per match
# -----------------------------------------

home = matches[[
    "match_id", "match_date", "home_team_id",
    "away_team_id", "home_score", "away_score"
]].rename(columns={
    "home_team_id": "team_id",
    "away_team_id": "opponent_id",
    "home_score": "goals",
    "away_score": "goals_conceded"
})

away = matches[[
    "match_id", "match_date", "away_team_id",
    "home_team_id", "away_score", "home_score"
]].rename(columns={
    "away_team_id": "team_id",
    "home_team_id": "opponent_id",
    "away_score": "goals",
    "home_score": "goals_conceded"
})

teams = pd.concat([home, away], ignore_index=True)

data = teams.merge(
    stats[required_stats],
    on=["match_id", "team_id"],
    how="left",
    validate="one_to_one"
)

if len(data) != 208:
    raise ValueError(
        f"Expected 208 observations, found {len(data)}."
    )

if data[required_stats[2:]].isna().any().any():
    raise ValueError("Missing team statistics after merging.")

# -----------------------------------------
# 3. Sort matches chronologically
# -----------------------------------------

data = data.sort_values(
    ["match_date", "match_id", "team_id"]
).reset_index(drop=True)

# -----------------------------------------
# 4. Create historical predictors
# -----------------------------------------

history_cols = {
    "goals": "prior_goals_avg",
    "goals_conceded": "prior_conceded_avg",
    "shots_on_target": "prior_sot_avg",
    "total_shots": "prior_shots_avg",
    "possession_pct": "prior_possession_avg",
    "corners": "prior_corners_avg"
}

for source, target in history_cols.items():
    data[target] = data.groupby("team_id")[source].transform(
        lambda s: s.shift(1).expanding().mean()
    )

data["previous_date"] = (
    data.groupby("team_id")["match_date"].shift(1)
)

data["rest_days"] = (
    (data["match_date"] - data["previous_date"])
    .dt.total_seconds() / 86400
)

# Pre-match opponent defensive record
opponent_history = data[[
    "match_id", "team_id", "prior_conceded_avg"
]].rename(columns={
    "team_id": "opponent_id",
    "prior_conceded_avg": "opponent_prior_conceded"
})

data = data.merge(
    opponent_history,
    on=["match_id", "opponent_id"],
    how="left",
    validate="one_to_one"
)

# Zero-history encoding for opening matches.
# This is a modelling limitation, not a measured
# pre-tournament performance value.
features = [
    "prior_goals_avg",
    "prior_conceded_avg",
    "prior_sot_avg",
    "prior_shots_avg",
    "prior_possession_avg",
    "prior_corners_avg",
    "rest_days",
    "opponent_prior_conceded"
]

data[features] = data[features].fillna(0)

assert len(features) == 8
assert len(data) == 208
assert not data[features].isna().any().any()

data.to_csv(
    OUT / "regression_dataset_208.csv",
    index=False
)

# -----------------------------------------
# 5. Chronological train/test split
# Keep both teams from a match together
# -----------------------------------------

match_order = (
    matches.sort_values(["match_date", "match_id"])
    ["match_id"].tolist()
)

split = int(len(match_order) * 0.8)

train_ids = set(match_order[:split])
test_ids = set(match_order[split:])

train = data[data["match_id"].isin(train_ids)]
test = data[data["match_id"].isin(test_ids)]

X_train = train[features]
X_test = test[features]

y_train = train["goals"]
y_test = test["goals"]

# -----------------------------------------
# 6. Train multiple linear regression
# -----------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Baseline: predict training mean for all
baseline_pred = np.full(
    len(y_test), y_train.mean()
)

# -----------------------------------------
# 7. Evaluate models
# -----------------------------------------

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

results = []

for label, actual, predicted in [
    ("Linear regression - train", y_train, train_pred),
    ("Linear regression - test", y_test, test_pred),
    ("Mean baseline - test", y_test, baseline_pred)
]:
    metrics = evaluate(actual, predicted)
    results.append({"Model": label, **metrics})

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(
    OUT / "model_metrics.csv", index=False
)

print("\nMODEL PERFORMANCE")
print(metrics_df.to_string(index=False))

# -----------------------------------------
# 8. Regression coefficients
# -----------------------------------------

coefficients = pd.DataFrame({
    "Variable": features,
    "Coefficient": model.coef_
})

coefficients.to_csv(
    OUT / "regression_coefficients.csv",
    index=False
)

print("\nINTERCEPT:", model.intercept_)
print("\nCOEFFICIENTS")
print(coefficients.to_string(index=False))

# -----------------------------------------
# 9. Save test predictions
# -----------------------------------------

predictions = test[[
    "match_id", "team_id", "goals"
]].copy()

predictions["predicted_goals"] = test_pred
predictions["residual"] = (
    predictions["goals"] -
    predictions["predicted_goals"]
)

predictions.to_csv(
    OUT / "test_predictions.csv", index=False
)

# -----------------------------------------
# 10. Actual vs predicted plot
# -----------------------------------------

plt.figure(figsize=(7, 6))
plt.scatter(y_test, test_pred, alpha=0.7)

limits = [
    min(y_test.min(), test_pred.min()),
    max(y_test.max(), test_pred.max())
]

plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual goals")
plt.ylabel("Predicted goals")
plt.title("Regression 2.2: Actual vs Predicted Goals")
plt.tight_layout()
plt.savefig(
    OUT / "actual_vs_predicted.png",
    dpi=300
)
plt.close()

# -----------------------------------------
# 11. Residual plot
# -----------------------------------------

residuals = y_test.to_numpy() - test_pred

plt.figure(figsize=(7, 5))
plt.scatter(test_pred, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted goals")
plt.ylabel("Residual (actual - predicted)")
plt.title("Regression 2.2: Residual Analysis")
plt.tight_layout()
plt.savefig(
    OUT / "residual_plot.png",
    dpi=300
)
plt.close()

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))
print("Total observations:", len(data))
print("Explanatory variables:", len(features))
print("\nOutput files saved in:", OUT.resolve())

