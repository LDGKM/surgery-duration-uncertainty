"""Does better uncertainty make better operating-room schedules?

    python src/scheduling.py      (needs results/test_predictions.csv from models.py)

Simulation on the held-out test cases only:
  * a "day" = 4 operating rooms of 8 hours (480 minutes) and a waiting list of
    24 elective cases drawn at random from the test set;
  * a mixed-integer program chooses which cases to operate and in which room,
    maximising the planned workload (sum of predicted median durations);
  * the schedule is then "played" with the real durations, and we measure
    overtime (minutes beyond 480), idle time and the workload actually done.

The planning rules differ only in the capacity constraint of each room r:

  historical     sum of procedure medians                    <= 480
  ML median      sum of GBM medians                          <= 480
  fixed buffer   (1 + b) x sum of GBM medians                <= 480,   b in {5%, ..., 30%}
  quantile       sum of GBM tau-quantiles                    <= 480,   tau in {0.6, 0.7, 0.8, 0.9}
  robust         sum of medians + the Gamma largest deviations d_i <= 480
                 (budget of uncertainty, Bertsimas & Sim 2004), with
                 d_i = upper bound of the conformal interval - median,
                 so the uncertainty set is learned from data (Bertsimas, Gupta & Kallus 2018).

Model: rooms are identical, so we enumerate every set of cases that fits in
one room under the rule, then solve a set-packing problem (choose at most 4
disjoint sets maximising the planned workload). This is exact and avoids the
weak big-M / dual reformulation of the robust constraint, which left a 20%
optimality gap in a first version. Solver: HiGHS, through scipy.optimize.milp.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from models import FIG, GRID, INK, MUTED, RES

ROOMS, CAPACITY, POOL, DAYS = 4, 480.0, 24, 200


def room_load(plan, dev, gamma, idx):
    """Planned load of a room holding cases idx. Robust version: nominal load plus
    the Gamma largest deviations (a fractional Gamma takes that share of the next one)."""
    load = plan[idx].sum()
    if dev is not None and gamma > 0:
        d = np.sort(dev[idx])[::-1]
        k = int(np.floor(gamma))
        load += d[:k].sum() + (gamma - k) * (d[k] if k < len(d) else 0.0)
    return load


def feasible_rooms(plan, dev, gamma):
    """Every set of cases that fits in one room under the planning rule
    (depth-first search; adding a case never lowers the load, so we prune)."""
    n = len(plan)
    order = np.argsort(-plan)
    out = []

    def dfs(start, chosen):
        for j in range(start, n):
            idx = chosen + [order[j]]
            if room_load(plan, dev, gamma, np.array(idx)) <= CAPACITY:
                out.append(idx)
                dfs(j + 1, idx)
    dfs(0, [])
    return out


def solve_day(median, plan, dev=None, gamma=0.0):
    """Pattern formulation: list every feasible room content, then choose at
    most ROOMS disjoint ones maximising the planned workload (sum of medians).
    This set-packing model is exact for identical rooms and works for every
    planning rule, including the robust one, without big-M constraints.
    Solved with HiGHS through scipy.optimize.milp. Returns the room of each case."""
    n = len(median)
    pats = feasible_rooms(plan, dev, gamma)
    m = len(pats)
    A = np.zeros((n + 1, m))
    for k, idx in enumerate(pats):
        A[idx, k] = 1
    A[n, :] = 1                                           # at most ROOMS rooms
    value = np.array([median[idx].sum() for idx in pats])
    res = milp(-value, constraints=LinearConstraint(A, -np.inf, np.r_[np.ones(n), ROOMS]),
               integrality=np.ones(m), bounds=Bounds(0, 1), options={"time_limit": 60, "disp": False})
    room = np.full(n, -1)
    if res.x is not None:
        chosen = [pats[k] for k in np.flatnonzero(res.x > 0.5)]
        for r, idx in enumerate(chosen):
            room[idx] = r
    return room


def play(room, actual, median):
    """Realised outcome of one day: per-room totals with the true durations.
    Also returns, for the scheduled cases, how much longer they really took
    than their predicted median (positive = the plan was optimistic)."""
    loads = np.array([actual[room == r].sum() for r in range(ROOMS)])
    sel = room >= 0
    over = np.maximum(0, loads - CAPACITY)
    return {"overtime_min": over.mean(), "rooms_in_overtime": (over > 0).mean(), "overtime_30": (over > 30).mean(),
            "idle_min": np.maximum(0, CAPACITY - loads).mean(), "utilisation": np.minimum(loads, CAPACITY).mean() / CAPACITY,
            "cases": int(sel.sum()), "excess_vs_median": float((actual[sel] - median[sel]).mean()) if sel.any() else 0.0}


def policies(d: pd.DataFrame):
    m = d["GBM"].to_numpy()
    yield "historical", 0, dict(plan=d["B1"].to_numpy())
    yield "ML median", 0, dict(plan=m)
    for b in (0.05, 0.10, 0.15, 0.20, 0.30):
        yield "fixed buffer", b, dict(plan=m * (1 + b))
    for tau in (60, 70, 80, 90):
        yield "quantile", tau / 100, dict(plan=d[f"q{tau}"].to_numpy())
    dev = np.maximum(0, d["I3_hi"].to_numpy() - m)
    for g in (0.5, 1.0, 1.5, 2.0, 3.0):
        yield "robust (learned set)", g, dict(plan=m, dev=dev, gamma=g)


def simulate() -> pd.DataFrame:
    pred = pd.read_csv(RES / "test_predictions.csv")
    pool = pred[pred["emop"] == 0].reset_index(drop=True)            # elective cases only
    rng = np.random.default_rng(42)
    rows = []
    for day in range(DAYS):
        d = pool.iloc[rng.choice(len(pool), POOL, replace=False)].reset_index(drop=True)
        actual = d["duration"].to_numpy()
        for name, param, kw in policies(d):
            room = solve_day(d["GBM"].to_numpy(), **kw)
            rows.append({"day": day, "policy": name, "param": param, **play(room, actual, d["GBM"].to_numpy())})
    return pd.DataFrame(rows)


SHOW = {"historical": ("#9ca3af", "s"), "ML median": (INK, "o"), "fixed buffer": ("#ea8c2c", "o"),
        "quantile": ("#2563eb", "o"), "robust (learned set)": ("#dc2626", "o")}


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby(["policy", "param"])
    s = g[["overtime_min", "rooms_in_overtime", "overtime_30", "idle_min", "utilisation", "cases", "excess_vs_median"]].mean()
    s["overtime_se"] = g["overtime_min"].std() / np.sqrt(sim["day"].nunique())
    return s.reset_index()


def fig_frontier(s: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, (c, mk) in SHOW.items():
        d = s[s.policy == name].sort_values("utilisation")
        ax.plot(d["overtime_min"], 100 * d["utilisation"], marker=mk, color=c, lw=1.8, ms=7, label=name)
        for _, r in d.iterrows():
            if name in ("fixed buffer", "quantile", "robust (learned set)"):
                lab = {"fixed buffer": f"+{int(r.param * 100)}%", "quantile": f"q{int(r.param * 100)}",
                       "robust (learned set)": f"Γ={r.param:g}"}[name]
                ax.annotate(lab, (r.overtime_min, 100 * r.utilisation), xytext=(5, -10), textcoords="offset points",
                            fontsize=7.5, color=c)
    ax.set_xlabel("mean overtime per room and day (minutes)  ← better")
    ax.set_ylabel("room utilisation with real durations (%)  ↑ better")
    ax.set_title("Planning rules: utilisation vs overtime (200 simulated days)")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(FIG / "03_schedule_frontier.png"); plt.close(fig)


if __name__ == "__main__":
    pred = pd.read_csv(RES / "test_predictions.csv")
    el = pred[pred["emop"] == 0]
    print("all elective test cases: mean (actual - predicted median) = %.1f min" % (el["duration"] - el["GBM"]).mean())
    sim = simulate()
    sim.to_csv(RES / "schedule_simulation.csv", index=False)
    s = summarise(sim)
    s.round(4).to_csv(RES / "schedule_summary.csv", index=False)
    fig_frontier(s)
    pd.set_option("display.width", 180)
    print(s.round(3).to_string())
