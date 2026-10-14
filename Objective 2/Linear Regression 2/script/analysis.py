"""
Linear Regression 2.2 - predict the goals scored by a team in a World Cup 2026 match.
Run:  python script/analysis.py        (writes to output/)

EDA -> VIF -> OLS (cluster-robust by match) -> diagnostics -> repeated CV (both teams of a match
always in the same fold) -> chronological hold-out -> rolling-origin -> feature ablation ->
sensitivity without opening matches.  Opening-match gaps are filled INSIDE each training fold.
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
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor, PoissonRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd

.filterwarnings("ignore")
SEED, ROOT = 140, Path(__file__).resolve().parents[1]
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
EDA   = OUT / "1_exploratory_data_analysis"      # descriptives, correlations, multicollinearity
MODEL = OUT / "2_regression_model"               # fitted OLS, coefficients, assumption checks
EVAL  = OUT / "3_model_evaluation"               # cross-validation, hold-out, algorithm comparison
FEAT  = OUT / "4_feature_experiments"            # ablation, subsets, sensitivity
for _d in (EDA, MODEL, EVAL, FEAT):
    _d.mkdir(parents=True, exist_ok=True)
F, Y = bd.FEATURES, "goals"
LABEL = {"prior_goals_avg": "Prev. avg goals scored", "prior_sot_avg": "Prev. avg shots on target",
         "prior_shots_avg": "Prev. avg shots", "prior_corners_avg": "Prev. avg corners",
         "opp_prior_conceded_avg": "Opponent prev. avg goals conceded", "team_fifa_pts": "Team FIFA points (pre-tournament)",
         "opp_fifa_pts": "Opponent FIFA points (pre-tournament)", "rest_days": "Rest days"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titleweight": "bold"})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ------------------------------------------------------------------ 1. data
D = bd.build(); bd.checks(D)
D.to_csv(bd.PROC / "wc2026_lr22_dataset.csv", index=False)
X, y, grp = D[F], D[Y], D.match_no.values
Xi = X.fillna(X.mean())                       # in-sample fill, used for descriptive/inference only
has_hist = D.prior_goals_avg.notna().values
say(f"Dataset: {len(D)} team-matches ({D.match_no.nunique()} matches), {len(F)} explanatory variables, target = {Y}")
say(f"Rows with no tournament history (team's opening match): {int((~has_hist).sum())}")

# ------------------------------------------------------------------ 2. EDA
desc = D[F + [Y]].describe().T; desc["skew"] = D[F + [Y]].skew()
desc.round(3).to_csv(EDA / "descriptive_statistics.csv")
say("\n== Descriptive statistics =="); say(desc.round(3).to_string())
say(f"\nTarget: mean {y.mean():.2f}, variance {y.var():.2f} (Poisson-like count); "
    f"0 goals in {np.mean(y == 0):.0%} of team-matches, 3+ goals in {np.mean(y >= 3):.0%}")

corr = D[F + [Y]].corr()
corr.round(3).to_csv(EDA / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(7.4, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=.8); ax.set_title("Correlation matrix (rows with history)"); fig.tight_layout()
fig.savefig(EDA / "fig1_correlation_heatmap.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.hist(y, bins=np.arange(-.5, y.max() + 1.5, 1), color=BLUE, edgecolor="white")
ax.set_xlabel("Goals scored by a team in a match"); ax.set_ylabel("Team-matches"); ax.set_title("Target distribution")
fig.tight_layout(); fig.savefig(EDA / "fig2_target_distribution.png"); plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
for a, c in zip(axs.ravel(), F):
    ok = D[c].notna()
    a.scatter(D.loc[ok, c], y[ok], s=12, alpha=.5, color=BLUE)
    b = np.polyfit(D.loc[ok, c], y[ok], 1); xs = np.linspace(D.loc[ok, c].min(), D.loc[ok, c].max(), 20)
    a.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.5)
    a.set_title(f"{c}  (r={corr.loc[c, Y]:.2f})", fontsize=8)
axs[0, 0].set_ylabel("goals"); axs[1, 0].set_ylabel("goals")
fig.suptitle("Each predictor vs goals scored", fontweight="bold"); fig.tight_layout()
fig.savefig(EDA / "fig3_scatter_vs_target.png"); plt.close(fig)

Xc = sm.add_constant(Xi)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=F, name="VIF")
vif.round(2).to_csv(EDA / "vif.csv"); say("\n== Variance inflation factors (>5 = concern) =="); say(vif.round(2).to_string())

# ------------------------------------------------------------------ 3. OLS (inference)
ols = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": grp})    # two rows per match -> cluster by match
(MODEL / "ols_summary.txt").write_text(str(ols.summary()))
ci = ols.conf_int(); coef = pd.DataFrame({"coef": ols.params, "std_err_clustered": ols.bse, "t": ols.tvalues,
                                          "p_value": ols.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
coef["std_beta"] = ols.params[F] * Xi.std() / y.std()
coef.round(4).to_csv(MODEL / "ols_coefficients.csv")
plain = sm.OLS(y, Xc).fit()
say(f"\n== OLS on all 208 rows (standard errors clustered by match) ==  R2={plain.rsquared:.3f}  "
    f"adj R2={plain.rsquared_adj:.3f}  AIC={plain.aic:.1f}"); say(coef.round(3).to_string())

fig, ax = plt.subplots(figsize=(6.6, 3.8))
c = coef.loc[F][::-1]; pos = np.arange(len(c))
ax.errorbar(c.coef, pos, xerr=[c.coef - c.ci_low, c.ci_high - c.coef], fmt="o", color=BLUE, capsize=3)
ax.axvline(0, color=GREY, ls="--"); ax.set_yticks(pos); ax.set_yticklabels([LABEL[i] for i in c.index])
ax.set_xscale("symlog", linthresh=.01); ax.set_xlabel("OLS coefficient (goals per unit), 95% CI"); ax.set_title("Which variables matter?")
fig.tight_layout(); fig.savefig(MODEL / "fig4_coefficients.png"); plt.close(fig)

res, fit = plain.resid, plain.fittedvalues
bp = het_breuschpagan(res, Xc); sh = stats.shapiro(res); cook = plain.get_influence().cooks_distance[0]
say(f"\n== Diagnostics ==  Breusch-Pagan p={bp[1]:.3f} | Shapiro-Wilk p={sh.pvalue:.4f} (goals are counts, so "
    f"residuals are not normal) | max Cook's D={cook.max():.3f}")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].scatter(fit, res, s=12, alpha=.6, color=BLUE); axs[0].axhline(0, color=ORANGE); axs[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
stats.probplot(res, plot=axs[1]); axs[1].set_title("Normal Q-Q")
axs[2].stem(np.arange(len(cook)), cook, markerfmt=".", basefmt=" "); axs[2].set(xlabel="row", title="Cook's distance")
fig.tight_layout(); fig.savefig(MODEL / "fig5_diagnostics.png"); plt.close(fig)

# ------------------------------------------------------------------ 4. out-of-sample evaluation
imp = lambda m: make_pipeline(SimpleImputer(strategy="mean"), m)             # fitted on training rows only
def models():
    sc = lambda m: make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), m)
    return {
        "Baseline (mean)": DummyRegressor(),
        "OLS (8 vars)": imp(LinearRegression()),
        "Ridge (alpha by CV)": sc(RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso (alpha by CV)": sc(LassoCV(cv=5, random_state=SEED, max_iter=20000)),
        "Huber (robust)": sc(HuberRegressor(max_iter=1000)),
        "Poisson regression": sc(PoissonRegressor(alpha=1e-3, max_iter=1000)),
        "Random forest": imp(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "Gradient boosting": imp(GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.05, subsample=.8, random_state=SEED)),
        "SVR (RBF)": sc(SVR(C=1.0, epsilon=.5)),
        "k-NN (k=10)": sc(KNeighborsRegressor(n_neighbors=10)),
    }
def metrics(yt, yp):
    return {"R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp), "RMSE": mean_squared_error(yt, yp) ** .5}

def match_folds(rep, mask=None, k=5):
    """Fold id per row; both teams of a match always share a fold."""
    rng = np.random.RandomState(SEED + rep)
    ms = np.unique(grp if mask is None else grp[mask]); rng.shuffle(ms)
    fold_of = {m: i % k for i, m in enumerate(ms)}
    return np.array([fold_of.get(m, -1) for m in grp])

def repeated_cv(model, Xd, yd, mask=None, reps=20):
    rows = []
    idx = np.arange(len(Xd)) if mask is None else np.where(mask)[0]
    for r in range(reps):
        f = match_folds(r, mask); p = np.full(len(Xd), np.nan)
        for k in range(5):
            te = idx[f[idx] == k]; tr = idx[f[idx] != k]
            m = clone(model).fit(Xd.iloc[tr], yd.iloc[tr]); p[te] = m.predict(Xd.iloc[te])
        rows.append(metrics(yd.iloc[idx], p[idx]))
    return pd.DataFrame(rows)

def cv_table(Xd, yd, mods, mask=None):
    out = []
    for n, mdl in mods.items():
        d = repeated_cv(mdl, Xd, yd, mask)
        out.append({"model": n, **{k: d[k].mean() for k in d}, **{f"{k}_sd": d[k].std() for k in d}})
    return pd.DataFrame(out).set_index("model")

say("\n== Repeated 5-fold CV by match (20 repeats) ==")
cvt = cv_table(X, y, models()); cvt.round(3).to_csv(EVAL / "model_comparison_cv.csv"); say(cvt.round(3).to_string())

order = D.match_no.values; cut = int(round(D.match_no.nunique() * .8))
tr_m = np.isin(order, np.unique(order)[:cut]); te_m = ~tr_m
chrono = []
for n, mdl in models().items():
    mdl.fit(X[tr_m], y[tr_m]); chrono.append({"model": n, **metrics(y[te_m], mdl.predict(X[te_m]))})
chrono = pd.DataFrame(chrono).set_index("model"); chrono.round(3).to_csv(EVAL / "model_comparison_chronological_holdout.csv")
say(f"\n== Chronological hold-out: train first {cut} matches ({tr_m.sum()} rows), test the last {te_m.sum()//2} =="); say(chrono.round(3).to_string())

ms_all = np.unique(order); roll = []
for n, mdl in models().items():
    folds = []
    for s in range(5):
        a = ms_all[: 60 + 8 * s]; b = ms_all[60 + 8 * s: 68 + 8 * s]
        ia, ib = np.isin(order, a), np.isin(order, b)
        folds.append(metrics(y[ib], clone(mdl).fit(X[ia], y[ia]).predict(X[ib])))
    f = pd.DataFrame(folds); roll.append({"model": n, "R2": f.R2.mean(), "MAE": f.MAE.mean(), "RMSE": f.RMSE.mean()})
roll = pd.DataFrame(roll).set_index("model"); roll.round(3).to_csv(EVAL / "model_comparison_rolling_origin.csv")
say("\n== Rolling-origin CV (5 expanding windows, 8 test matches each) =="); say(roll.round(3).to_string())

# ------------------------------------------------------------------ 5. feature experiments
base = repeated_cv(models()["OLS (8 vars)"], X, y)
abl = []
for c in F:
    keep = [k for k in F if k != c]
    drop = repeated_cv(models()["OLS (8 vars)"], X[keep], y); alone = repeated_cv(models()["OLS (8 vars)"], X[[c]], y)
    abl.append({"feature": c, "alone_cv_R2": alone.R2.mean(), "drop_one_cv_R2": drop.R2.mean(),
                "delta_R2_when_dropped": drop.R2.mean() - base.R2.mean(), "delta_RMSE_when_dropped": drop.RMSE.mean() - base.RMSE.mean()})
abl = pd.DataFrame(abl).set_index("feature"); abl.round(4).to_csv(FEAT / "feature_ablation.csv")
say(f"\n== Feature ablation (OLS CV R2 with all 8 = {base.R2.mean():.3f}) =="); say(abl.round(3).to_string())

sens = []
for lab, mask in [("all 208 rows (opening rows filled in-fold)", None), ("only rows with tournament history (160)", has_hist)]:
    for n in ["Baseline (mean)", "OLS (8 vars)", "Ridge (alpha by CV)"]:
        d = repeated_cv(models()[n], X, y, mask)
        sens.append({"rows": lab, "model": n, "R2": d.R2.mean(), "RMSE": d.RMSE.mean()})
sens = pd.DataFrame(sens); sens.round(3).to_csv(FEAT / "opening_round_sensitivity.csv", index=False)
say("\n== Sensitivity: do the opening-match rows matter? =="); say(sens.round(3).to_string(index=False))

# ------------------------------------------------------------------ 6. figures
p = np.zeros(len(D)); cnt = np.zeros(len(D))
for r in range(20):
    f = match_folds(r)
    for k in range(5):
        te, trn = f == k, f != k
        p[te] += clone(models()["OLS (8 vars)"]).fit(X[trn], y[trn]).predict(X[te]); cnt[te] += 1
p /= cnt
pd.DataFrame({"match_no": D.match_no, "team": D.team, "actual": y, "cv_pred": p}).round(3).to_csv(EVAL / "cv_predictions.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
axs[0].scatter(p, y + np.random.RandomState(1).uniform(-.12, .12, len(y)), s=12, alpha=.5, color=BLUE)
axs[0].plot([0, 4], [0, 4], "--", color=GREY); axs[0].set(xlabel="Predicted goals (cross-validated)", ylabel="Actual goals",
                                                            title=f"OLS, 5-fold CV (R2={cvt.loc['OLS (8 vars)', 'R2']:.2f})")
m_ = cvt.RMSE[::-1]; axs[1].barh(m_.index, m_, xerr=cvt.RMSE_sd[::-1], color=[GREY if i == "Baseline (mean)" else BLUE for i in m_.index])
axs[1].set(xlabel="CV RMSE (goals) - lower is better", title="Model comparison")
fig.tight_layout(); fig.savefig(EVAL / "fig6_prediction_and_model_comparison.png"); plt.close(fig)

(OUT / "results_log.txt").write_text("\n".join(log))
print("\nDone. Files in", OUT)
