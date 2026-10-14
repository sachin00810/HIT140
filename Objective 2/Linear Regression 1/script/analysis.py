"""
Linear Regression 2.1 - predict the goal difference (team1 - team2) of a World Cup 2026 match
from 8 pre-match variables.  Run build_dataset.py first (or just run this file).

Outputs go to ../output/.  Random seed = 140 everywhere so results are reproducible.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import (RepeatedKFold, KFold, GridSearchCV, cross_val_predict, TimeSeriesSplit)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd

warnings.filterwarnings("ignore")
SEED, ROOT = 140, Path(__file__).resolve().parents[1]
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
EDA   = OUT / "1_exploratory_data_analysis"      # descriptives, correlations, multicollinearity
MODEL = OUT / "2_regression_model"               # fitted OLS, coefficients, assumption checks
EVAL  = OUT / "3_model_evaluation"               # cross-validation, hold-out, algorithm comparison
FEAT  = OUT / "4_feature_experiments"            # ablation, subsets, sensitivity
for _d in (EDA, MODEL, EVAL, FEAT):
    _d.mkdir(parents=True, exist_ok=True)
F, Y = bd.FEATURES, "goal_diff"
LABEL = {"xg_diff": "Prev. avg xG created (diff)", "xga_diff": "Prev. avg xG conceded (diff)",
         "fifa_pts_diff": "Pre-tournament FIFA points (diff)", "ppg_diff": "Points per game so far (diff)",
         "fouls_diff": "Prev. avg fouls (diff)", "rest_diff": "Rest days (diff)",
         "is_knockout": "Knockout stage (0/1)", "host_adv": "Host-nation edge (-1/0/+1)"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titleweight": "bold"})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ------------------------------------------------------------------ 1. data
D = bd.build(); bd.checks(D)
D.to_csv(bd.PROC / "wc2026_lr21_dataset.csv", index=False)
X, y = D[F], D[Y]
say(f"Dataset: {D.shape[0]} matches, {len(F)} explanatory variables, target = {Y}")

# ------------------------------------------------------------------ 2. EDA
desc = D[F + [Y]].describe().T
desc["skew"] = D[F + [Y]].skew()
desc.round(3).to_csv(EDA / "descriptive_statistics.csv")
say("\n== Descriptive statistics =="); say(desc.round(3).to_string())
say(f"\nTarget: mean {y.mean():.2f}, sd {y.std():.2f}; team1 wins {np.mean(y>0):.0%}, "
    f"draws {np.mean(y==0):.0%}, loses {np.mean(y<0):.0%}")

corr = D[F + [Y]].corr()
corr.round(3).to_csv(EDA / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=.8); ax.set_title("Correlation matrix"); fig.tight_layout()
fig.savefig(EDA / "fig1_correlation_heatmap.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.hist(y, bins=np.arange(y.min() - .5, y.max() + 1.5, 1), color=BLUE, edgecolor="white")
ax.set_xlabel("Goal difference (team1 - team2)"); ax.set_ylabel("Matches"); ax.set_title("Target distribution")
fig.tight_layout(); fig.savefig(EDA / "fig2_target_distribution.png"); plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
for a, c in zip(axs.ravel(), F):
    a.scatter(D[c], y, s=14, alpha=.6, color=BLUE)
    b = np.polyfit(D[c], y, 1); xs = np.linspace(D[c].min(), D[c].max(), 20)
    a.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.5)
    a.set_title(f"{c}  (r={corr.loc[c, Y]:.2f})", fontsize=8); a.set_xlabel("")
axs[0, 0].set_ylabel("goal_diff"); axs[1, 0].set_ylabel("goal_diff")
fig.suptitle("Each predictor vs goal difference", fontweight="bold"); fig.tight_layout()
fig.savefig(EDA / "fig3_scatter_vs_target.png"); plt.close(fig)

Xc = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=F, name="VIF")
vif.round(2).to_csv(EDA / "vif.csv"); say("\n== Variance inflation factors (multicollinearity; >5 = concern) =="); say(vif.round(2).to_string())

# ------------------------------------------------------------------ 3. OLS on all matches (inference)
ols = sm.OLS(y, Xc).fit()
(MODEL / "ols_summary.txt").write_text(str(ols.summary()))
ci = ols.conf_int(); coef = pd.DataFrame({"coef": ols.params, "std_err": ols.bse, "t": ols.tvalues,
    "p_value": ols.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
std_beta = (ols.params[F] * X.std() / y.std()); coef["std_beta"] = std_beta
coef.round(4).to_csv(MODEL / "ols_coefficients.csv")
say(f"\n== OLS on all 104 matches ==  R2={ols.rsquared:.3f}  adj R2={ols.rsquared_adj:.3f}  "
    f"F-test p={ols.f_pvalue:.3f}  AIC={ols.aic:.1f}")
say(coef.round(3).to_string())

fig, ax = plt.subplots(figsize=(6.4, 3.8))
c = coef.loc[F][::-1]; pos = np.arange(len(c))
ax.errorbar(c.coef, pos, xerr=[c.coef - c.ci_low, c.ci_high - c.coef], fmt="o", color=BLUE, capsize=3)
ax.axvline(0, color=GREY, ls="--"); ax.set_yticks(pos); ax.set_yticklabels([LABEL[i] for i in c.index])
ax.set_xlabel("OLS coefficient (goals), 95% CI"); ax.set_title("Which variables matter?"); fig.tight_layout()
fig.savefig(MODEL / "fig4_coefficients.png"); plt.close(fig)

# diagnostics
res, fit = ols.resid, ols.fittedvalues
bp = het_breuschpagan(res, Xc); sh = stats.shapiro(res); dw = durbin_watson(res)
cook = ols.get_influence().cooks_distance[0]
say(f"\n== Diagnostics ==  Breusch-Pagan p={bp[1]:.3f} | Shapiro-Wilk p={sh.pvalue:.3f} | "
    f"Durbin-Watson={dw:.2f} | max Cook's D={cook.max():.3f} (match {int(D.match_no[cook.argmax()])})")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].scatter(fit, res, s=14, alpha=.7, color=BLUE); axs[0].axhline(0, color=ORANGE)
axs[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
sm.qqplot(res, line="s", ax=axs[1], markersize=3); axs[1].set_title("Normal Q-Q")
axs[2].stem(D.match_no, cook, markerfmt=".", basefmt=" "); axs[2].set(xlabel="match_no", title="Cook's distance")
fig.tight_layout(); fig.savefig(MODEL / "fig5_diagnostics.png"); plt.close(fig)

# ------------------------------------------------------------------ 4. out-of-sample evaluation
def models():
    return {
        "Baseline (mean)": DummyRegressor(),
        "OLS (8 vars)": LinearRegression(),
        "Ridge (alpha by CV)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso (alpha by CV)": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=20000)),
        "Huber (robust)": make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)),
        "OLS + best-k selection": GridSearchCV(
            make_pipeline(SelectKBest(f_regression), LinearRegression()),
            {"selectkbest__k": [1, 2, 3, 4, 5, 6, 8]}, cv=5, scoring="neg_mean_squared_error"),
        # non-linear / ensemble competitors (fixed, modest settings: only 104 rows)
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                       subsample=0.8, random_state=SEED),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.5)),
        "k-NN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    }
def metrics(yt, yp):
    return {"R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp), "RMSE": mean_squared_error(yt, yp) ** .5}

def repeated_cv(model, Xd, yd, reps=20):
    rows = []
    for r in range(reps):
        p = cross_val_predict(model, Xd, yd, cv=KFold(5, shuffle=True, random_state=SEED + r))
        rows.append(metrics(yd, p))
    return pd.DataFrame(rows)

def cv_table(Xd, yd, mods):
    out = []
    for name, mdl in mods.items():
        d = repeated_cv(mdl, Xd, yd)
        out.append({"model": name, **{f"{k}": d[k].mean() for k in d}, **{f"{k}_sd": d[k].std() for k in d}})
    return pd.DataFrame(out).set_index("model")

say("\n== Repeated 5-fold CV (20 repeats, out-of-fold predictions) ==")
cvt = cv_table(X, y, models()); cvt.round(3).to_csv(EVAL / "model_comparison_cv.csv"); say(cvt.round(3).to_string())

# chronological checks (predict later matches from earlier ones)
split = int(round(len(D) * .8)); tr, te = D.index[:split], D.index[split:]
chrono = []
for name, mdl in models().items():
    mdl.fit(X.loc[tr], y.loc[tr]); chrono.append({"model": name, **metrics(y.loc[te], mdl.predict(X.loc[te]))})
chrono = pd.DataFrame(chrono).set_index("model")
chrono.round(3).to_csv(EVAL / "model_comparison_chronological_holdout.csv")
say(f"\n== Chronological hold-out: train first {split} matches, test last {len(te)} =="); say(chrono.round(3).to_string())

tsr = []
for name, mdl in models().items():
    folds = []
    for a, b in TimeSeriesSplit(n_splits=5, test_size=15).split(X):
        mdl.fit(X.iloc[a], y.iloc[a]); p = mdl.predict(X.iloc[b])
        folds.append(metrics(y.iloc[b], p))
    f = pd.DataFrame(folds); tsr.append({"model": name, "R2": f.R2.mean(), "MAE": f.MAE.mean(), "RMSE": f.RMSE.mean()})
tsr = pd.DataFrame(tsr).set_index("model"); tsr.round(3).to_csv(EVAL / "model_comparison_rolling_origin.csv")
say("\n== Rolling-origin CV (5 expanding windows, 15 test matches each) =="); say(tsr.round(3).to_string())

# train vs CV gap for plain OLS
lr = LinearRegression().fit(X, y)
say(f"\nOLS in-sample R2 = {r2_score(y, lr.predict(X)):.3f}  vs  cross-validated R2 = {cvt.loc['OLS (8 vars)','R2']:.3f}")

# ------------------------------------------------------------------ 5. feature performance
abl = []
base = cvt.loc["OLS (8 vars)"]
for c in F:
    keep = [k for k in F if k != c]
    d = repeated_cv(LinearRegression(), X[keep], y)
    s = repeated_cv(LinearRegression(), X[[c]], y)
    abl.append({"feature": c, "alone_cv_R2": s.R2.mean(), "drop_one_cv_R2": d.R2.mean(),
                "delta_R2_when_dropped": d.R2.mean() - base.R2, "delta_RMSE_when_dropped": d.RMSE.mean() - base.RMSE})
abl = pd.DataFrame(abl).set_index("feature"); abl.round(4).to_csv(FEAT / "feature_ablation.csv")
say("\n== Feature performance (cross-validated) =="); say(abl.round(3).to_string())

# best small subsets, chosen by repeated CV (exploratory - optimistic because chosen on the same data)
from itertools import combinations
subs = []
for k in range(1, 6):
    for comb in combinations(F, k):
        subs.append({"vars": ", ".join(comb), "k": k, "cv_R2": repeated_cv(LinearRegression(), X[list(comb)], y, reps=5).R2.mean()})
subs = pd.DataFrame(subs).sort_values("cv_R2", ascending=False).head(8)
subs.round(4).to_csv(FEAT / "best_subsets_exploratory.csv", index=False)
say("\n== Top subsets by CV R2 (exploratory, optimistic) =="); say(subs.round(3).to_string(index=False))

# ------------------------------------------------------------------ 6. opening-round sensitivity
# 24 opening-round matches have no history (all history differences = 0). Do they hurt the model?
has_hist = D[bd.HISTORY].abs().sum(axis=1) > 0   # the 5 tournament-history variables only
Xh, yh = X[has_hist], y[has_hist]
d_all = repeated_cv(LinearRegression(), X, y); d_hist = repeated_cv(LinearRegression(), Xh, yh)
d_base_h = repeated_cv(DummyRegressor(), Xh, yh)
nh = int(has_hist.sum())
sens = pd.DataFrame({f"all {len(D)} matches (OLS)": {"n": len(D), "cv_R2": d_all.R2.mean(), "cv_RMSE": d_all.RMSE.mean()},
                     f"{nh} matches with history (OLS)": {"n": nh, "cv_R2": d_hist.R2.mean(), "cv_RMSE": d_hist.RMSE.mean()},
                     f"{nh} matches with history (mean baseline)": {"n": nh, "cv_R2": d_base_h.R2.mean(), "cv_RMSE": d_base_h.RMSE.mean()}}).T
sens.round(4).to_csv(FEAT / "opening_round_sensitivity.csv")
say("\n== Sensitivity: matches with vs without any history =="); say(sens.round(3).to_string())

# ------------------------------------------------------------------ 7. predictions figure
best_name = "OLS (8 vars)"
oof = np.mean([cross_val_predict(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=SEED + r))
               for r in range(20)], axis=0)
pd.DataFrame({"match_no": D.match_no, "team1": D.team1, "team2": D.team2, "actual": y,
              "cv_predicted": oof}).round(3).to_csv(EVAL / "cv_predictions.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(oof, y, s=16, alpha=.7, color=BLUE); lim = [-4, 6]
axs[0].plot(lim, lim, "--", color=GREY); axs[0].set(xlabel="Predicted (cross-validated)", ylabel="Actual goal difference",
    title=f"OLS, 5-fold CV  (R2={cvt.loc[best_name,'R2']:.2f})")
names = list(cvt.index)
axs[1].barh(names[::-1], cvt["RMSE"].values[::-1], xerr=cvt["RMSE_sd"].values[::-1], color=[GREY if n.startswith("Base") else BLUE for n in names[::-1]])
axs[1].set(xlabel="CV RMSE (goals) - lower is better", title="Model comparison")
fig.tight_layout(); fig.savefig(EVAL / "fig6_prediction_and_model_comparison.png"); plt.close(fig)

(OUT / "results_log.txt").write_text("\n".join(log))
print("\nDone. Files in", OUT)
