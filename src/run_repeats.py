"""Repeat the prediction experiment on 10 random splits:  python src/run_repeats.py"""
import pandas as pd

from models import RES, fig_repeats, repeated_evaluation

if __name__ == "__main__":
    r = repeated_evaluation(10)
    r.round(3).to_csv(RES / "repeated_splits.csv", index=False)
    fig_repeats(r)
    raw = pd.read_csv(RES / "repeated_splits_raw.csv")
    pd.set_option("display.width", 160)
    print(r.round(3))
    print("MAE GBM %.1f +- %.1f | procedure history %.1f +- %.1f" % (raw.gbm_mae.mean(), raw.gbm_mae.std(), raw.hist_mae.mean(), raw.hist_mae.std()))
    # paired comparison: same split, two methods (removes the split-to-split noise)
    w = raw.pivot(index="seed", columns="method", values="interval_score")
    pairs = [("I4 CQR", "I1 procedure history"), ("I4 CQR", "I2 quantile GBM"), ("I4 CQR", "I3 conformal"),
             ("I5 CQR by department", "I4 CQR")]
    paired = pd.DataFrame([{"method": a, "versus": b, "mean_difference": (w[a] - w[b]).mean(), "sd": (w[a] - w[b]).std(),
                            "splits_where_better": int(((w[a] - w[b]) < 0).sum())} for a, b in pairs])
    paired.round(2).to_csv(RES / "repeated_splits_paired.csv", index=False)
    print(paired.round(2))
