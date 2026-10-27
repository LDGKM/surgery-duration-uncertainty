"""Point predictions and prediction intervals for case duration.

    python src/models.py      # fits everything, writes results/ and figures/

Point predictions
  B0  global median of the training set
  B1  median duration of the same procedure in the training set (the usual
      "historical average" used by schedulers), falling back to the operation
      type, then to the global median when a procedure is rare
  GBM gradient-boosted trees predicting the conditional median

Prediction intervals, nominal coverage 80% (10th to 90th percentile)
  I1  empirical 10th-90th percentiles of the same procedure (historical)
  I2  quantile gradient boosting (10% and 90%), uncalibrated
  I3  split conformal around the GBM median, on the log scale
      (Vovk et al. 2005; Angelopoulos & Bates 2021)
  I4  conformalized quantile regression, CQR (Romano, Patterson & Candès 2019)
  I5  group-conditional ("Mondrian") CQR: the CQR adjustment is calibrated
      separately for each department, so that the guarantee holds within each
      department and not only on average (Vovk et al. 2005)

The calibration set is used only by I3 and I4; all methods are compared on
the same held-out test set.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data import FEATURES, ROOT, load, split

FIG, RES = ROOT / "figures", ROOT / "results"
ALPHA = 0.20                          # 1 - nominal coverage
QUANTILES = (0.1, 0.5, 0.6, 0.7, 0.8, 0.9)
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
COLORS = {"I1 procedure history": "#9ca3af", "I2 quantile GBM": "#ea8c2c", "I3 conformal": "#2563eb", "I4 CQR": "#dc2626",
          "I5 CQR by department": "#16a34a"}
METHODS = (("I1", "I1 procedure history"), ("I2", "I2 quantile GBM"), ("I3", "I3 conformal"), ("I4", "I4 CQR"),
           ("I5", "I5 CQR by department"))
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False})


def gbm(quantile: float, seed: int = 0) -> HistGradientBoostingRegressor:
    """One model per quantile; categorical columns are handled natively."""
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                                         categorical_features="from_dtype", random_state=seed)


def fit_quantile_models(train: pd.DataFrame) -> dict[float, HistGradientBoostingRegressor]:
    return {q: gbm(q).fit(train[FEATURES], train["duration"]) for q in QUANTILES}


def procedure_history(train: pd.DataFrame, rows: pd.DataFrame, q: float) -> np.ndarray:
    """q-th percentile of past durations for the same procedure; falls back to
    the operation type (when the procedure has < 10 past cases), then global."""
    by_proc = train.groupby("opname", observed=True)["duration"].agg(["count", lambda s: s.quantile(q)])
    by_type = train.groupby("optype", observed=True)["duration"].agg(["count", lambda s: s.quantile(q)])
    by_proc.columns = by_type.columns = ["n", "v"]
    glob = train["duration"].quantile(q)
    out = []
    for p, t in zip(rows["opname"], rows["optype"]):
        if p in by_proc.index and by_proc.at[p, "n"] >= 10:
            out.append(by_proc.at[p, "v"])
        elif t in by_type.index and by_type.at[t, "n"] >= 10:
            out.append(by_type.at[t, "v"])
        else:
            out.append(glob)
    return np.asarray(out, dtype=float)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of conformity scores."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def predict_all(train, cal, test, models) -> pd.DataFrame:
    """Every point prediction and interval for the test set, in one table."""
    P = lambda q, d: models[q].predict(d[FEATURES])  # noqa: E731
    out = test[["caseid", "department", "approach", "opname", "emop", "duration"]].copy()
    out["B0"] = train["duration"].median()
    out["B1"] = procedure_history(train, test, 0.5)
    out["GBM"] = P(0.5, test)
    for q in QUANTILES:
        out[f"q{int(q * 100)}"] = P(q, test)
    # I1 historical percentiles
    out["I1_lo"], out["I1_hi"] = procedure_history(train, test, 0.1), procedure_history(train, test, 0.9)
    # I2 raw quantile GBM
    out["I2_lo"], out["I2_hi"] = out["q10"], out["q90"]
    # I3 split conformal on the log scale: multiplicative band around the median
    s3 = np.abs(np.log(cal["duration"]) - np.log(P(0.5, cal)))
    k3 = conformal_quantile(s3, ALPHA)
    out["I3_lo"], out["I3_hi"] = out["GBM"] * np.exp(-k3), out["GBM"] * np.exp(k3)
    # I4 CQR: widen (or shrink) the quantile band by a calibrated amount
    lo_c, hi_c = P(0.1, cal), P(0.9, cal)
    s4 = np.maximum(lo_c - cal["duration"], cal["duration"] - hi_c)
    k4 = conformal_quantile(s4.to_numpy(), ALPHA)
    out["I4_lo"], out["I4_hi"] = out["q10"] - k4, out["q90"] + k4
    # I5 Mondrian CQR: one adjustment per department (needs enough calibration cases)
    k5 = {}
    for dep, idx in cal.groupby("department", observed=True).groups.items():
        sc = s4.loc[idx].to_numpy()
        k5[dep] = conformal_quantile(sc, ALPHA) if len(sc) >= 20 else k4
    adj = test["department"].map(k5).astype(float).fillna(k4).to_numpy()
    out["I5_lo"], out["I5_hi"] = out["q10"] - adj, out["q90"] + adj
    out.attrs.update({"k3_log": k3, "k4_min": k4, "k5_min": {str(k): round(v, 1) for k, v in k5.items()},
                      "cal_per_department": {str(k): int(v) for k, v in cal["department"].value_counts().items()}})
    return out


def point_metrics(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, name in (("B0", "B0 global median"), ("B1", "B1 procedure history"), ("GBM", "GBM median")):
        e = p[m] - p["duration"]
        rows.append({"model": name, "MAE_min": e.abs().mean(), "RMSE_min": np.sqrt((e ** 2).mean()),
                     "within_30min": (e.abs() <= 30).mean(),
                     "R2": 1 - (e ** 2).sum() / ((p["duration"] - p["duration"].mean()) ** 2).sum()})
    return pd.DataFrame(rows)


def interval_score(y, lo, hi, alpha=ALPHA):
    """Winkler / interval score: width plus a penalty for each miss (lower is better)."""
    return (hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)


def interval_metrics(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, name in METHODS:
        lo, hi, y = p[f"{k}_lo"], p[f"{k}_hi"], p["duration"]
        cov = ((y >= lo) & (y <= hi))
        by_dep = cov.groupby(p["department"], observed=True).mean()
        rows.append({"method": name, "coverage": cov.mean(), "mean_width_min": (hi - lo).mean(),
                     "interval_score": interval_score(y, lo, hi).mean(),
                     "worst_department_coverage": by_dep.min()})
    return pd.DataFrame(rows)


def conditional_coverage(p: pd.DataFrame) -> pd.DataFrame:
    """Coverage by surgical context: department, approach, and predicted-duration tercile."""
    p = p.copy()
    p["size"] = pd.qcut(p["GBM"], 3, labels=["short", "medium", "long"])
    rows = []
    for col, label in (("department", "Department"), ("approach", "Approach"), ("size", "Predicted length")):
        for g, d in p.groupby(col, observed=True):
            if len(d) < 30:
                continue
            for k, name in METHODS:
                cov = ((d["duration"] >= d[f"{k}_lo"]) & (d["duration"] <= d[f"{k}_hi"])).mean()
                rows.append({"group": f"{label}: {g}", "n": len(d), "method": name, "coverage": cov,
                             "width": (d[f"{k}_hi"] - d[f"{k}_lo"]).mean()})
    return pd.DataFrame(rows)


def fig_examples(p: pd.DataFrame) -> None:
    """40 random test cases sorted by prediction, with their CQR interval."""
    d = p.sample(40, random_state=1).sort_values("GBM").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(d))
    ax.vlines(x, d["I4_lo"], d["I4_hi"], color="#fca5a5", lw=6, label="80% interval (CQR)")
    ax.scatter(x, d["GBM"], color=INK, s=14, zorder=3, label="predicted median")
    inside = (d["duration"] >= d["I4_lo"]) & (d["duration"] <= d["I4_hi"])
    ax.scatter(x[inside], d["duration"][inside], color="#2563eb", s=26, marker="D", zorder=4, label="actual (inside)")
    ax.scatter(x[~inside], d["duration"][~inside], color="#dc2626", s=34, marker="X", zorder=4, label="actual (outside)")
    ax.set_xticks([]); ax.set_ylabel("minutes in the operating room")
    ax.set_title("Forty test cases: prediction, interval and what really happened"); ax.legend(ncol=4, fontsize=8, loc="upper left")
    FIG.mkdir(exist_ok=True); fig.savefig(FIG / "01_intervals_examples.png"); plt.close(fig)


def fig_coverage(cc: pd.DataFrame, im: pd.DataFrame) -> None:
    fig, a1 = plt.subplots(figsize=(9, 4.8))
    groups = cc["group"].drop_duplicates().tolist()
    y = np.arange(len(groups))
    for i, (m, c) in enumerate(COLORS.items()):
        d = cc[cc.method == m].set_index("group").reindex(groups)
        a1.scatter(d["coverage"], y + (i - 2) * 0.14, color=c, s=22, label=f"{m} (overall {im.set_index('method').at[m, 'coverage']:.2f})")
    a1.axvline(1 - ALPHA, color=INK, ls="--", lw=1)
    a1.set_yticks(y, [f"{g} (n={int(cc[cc.group == g].n.iloc[0])})" for g in groups], fontsize=8)
    a1.invert_yaxis(); a1.set_xlabel("coverage on the test set (target 0.80)")
    a1.set_title("Does the 80% interval hold in every context?"); a1.legend(fontsize=7.5, loc="lower left")
    fig.savefig(FIG / "02_coverage.png"); plt.close(fig)


def fig_repeats(r: pd.DataFrame) -> None:
    """Interval score (lower is better) and coverage over 10 random splits."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    names = [m for _, m in METHODS]
    r = r.set_index("method").loc[names]
    cols = [COLORS[m] for m in names]
    a1.barh(names, r["score_mean"], xerr=r["score_sd"], color=cols, capsize=3)
    a1.set_xlim(r["score_mean"].min() - 25, r["score_mean"].max() + 15); a1.invert_yaxis()
    a1.set_title("Interval score, 10 splits (lower is better)"); a1.set_xlabel("minutes")
    a2.barh(names, r["coverage_mean"], xerr=r["coverage_sd"], color=cols, capsize=3)
    a2.axvline(1 - ALPHA, color=INK, ls="--", lw=1); a2.set_xlim(0.6, 0.85); a2.invert_yaxis(); a2.set_yticks([])
    a2.set_title("Coverage, 10 splits (target 0.80)")
    fig.savefig(FIG / "02b_repeated_splits.png"); plt.close(fig)


if __name__ == "__main__":
    df = load()
    train, cal, test = split(df)
    models = fit_quantile_models(train)
    p = predict_all(train, cal, test, models)
    RES.mkdir(exist_ok=True)
    p.to_csv(RES / "test_predictions.csv", index=False)
    pm, im, cc = point_metrics(p), interval_metrics(p), conditional_coverage(p)
    pm.round(3).to_csv(RES / "point_metrics.csv", index=False)
    im.round(3).to_csv(RES / "interval_metrics.csv", index=False)
    cc.round(3).to_csv(RES / "conditional_coverage.csv", index=False)
    (RES / "calibration.json").write_text(json.dumps({"conformal_log_halfwidth": p.attrs["k3_log"],
                                                      "cqr_adjustment_min": p.attrs["k4_min"],
                                                      "cqr_by_department_min": p.attrs["k5_min"],
                                                      "calibration_cases_per_department": p.attrs["cal_per_department"],
                                                      "n_train": len(train), "n_cal": len(cal), "n_test": len(test)}, indent=2))
    fig_examples(p); fig_coverage(cc, im)
    pd.set_option("display.width", 160)
    print(pm.round(3)); print(im.round(3)); print(json.loads((RES / "calibration.json").read_text()))


def repeated_evaluation(n_seeds: int = 10) -> pd.DataFrame:
    """Same pipeline on n different random splits, to see how much the
    results depend on the split (mean and standard deviation per method)."""
    df = load()
    rows = []
    for seed in range(n_seeds):
        tr, ca, te = split(df, seed=seed)
        p = predict_all(tr, ca, te, fit_quantile_models(tr))
        im = interval_metrics(p).assign(seed=seed)
        pm = point_metrics(p)
        im["gbm_mae"] = pm.loc[pm.model == "GBM median", "MAE_min"].iloc[0]
        im["hist_mae"] = pm.loc[pm.model == "B1 procedure history", "MAE_min"].iloc[0]
        rows.append(im)
    allr = pd.concat(rows)
    allr.to_csv(RES / "repeated_splits_raw.csv", index=False)
    return allr.groupby("method").agg(coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
                                      width_mean=("mean_width_min", "mean"), score_mean=("interval_score", "mean"),
                                      score_sd=("interval_score", "std"),
                                      worst_dept_cov_mean=("worst_department_coverage", "mean")).reset_index()
