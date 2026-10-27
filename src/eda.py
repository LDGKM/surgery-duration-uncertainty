"""Exploratory look at case durations:  python src/eda.py  -> figures/00_durations.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load
from models import FIG, INK

if __name__ == "__main__":
    df = load()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.hist(df["duration"], bins=np.arange(0, 840, 20), color="#93c5fd", edgecolor="white")
    a1.axvline(df["duration"].median(), color=INK, ls="--", lw=1)
    a1.set_title("Minutes in the operating room"); a1.set_xlabel("minutes"); a1.set_ylabel("cases")
    a1.annotate(f"median {df['duration'].median():.0f} min\nmean {df['duration'].mean():.0f} min", (df["duration"].median(), 0),
                xytext=(250, 380), fontsize=9, color=INK)
    top = df["opname"].value_counts().index[:12]
    d = df[df["opname"].isin(top)]
    order = d.groupby("opname", observed=True)["duration"].median().sort_values().index
    a2.boxplot([d.loc[d["opname"] == o, "duration"] for o in order], orientation="horizontal", widths=0.6, showfliers=False,
               medianprops={"color": "#dc2626"})
    a2.set_yticks(range(1, len(order) + 1), list(order), fontsize=8)
    a2.set_title("Same procedure, very different durations (12 most frequent)"); a2.set_xlabel("minutes")
    FIG.mkdir(exist_ok=True); fig.savefig(FIG / "00_durations.png"); plt.close(fig)
    cv = d.groupby("opname", observed=True)["duration"].agg(lambda s: s.std() / s.mean())
    print("coefficient of variation within the 12 main procedures: %.2f to %.2f" % (cv.min(), cv.max()))
