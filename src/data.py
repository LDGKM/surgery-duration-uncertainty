"""Download, clean and split the VitalDB clinical table.

    python src/data.py      # downloads data/raw/vitaldb_cases.csv if needed and prints a summary

VitalDB (Lee et al., Scientific Data 2022; PhysioNet, CC BY 4.0) describes
6,388 non-cardiac surgeries performed at Seoul National University Hospital.
Times are given in seconds relative to the start of the case, so no calendar
date is available.

Target: case duration in minutes = caseend - casestart, i.e. how long the
operating room is occupied by the patient (anaesthesia + surgery). This is
the quantity an operating-room schedule has to reserve.

Only information available when the schedule is made is used as input:
patient characteristics, pre-operative tests and the planned procedure.
Everything measured during surgery (blood loss, drugs, fluids, lines placed)
is excluded to avoid leakage.
"""
from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "vitaldb_cases.csv"
URL = "https://api.vitaldb.net/cases"

CATEGORICAL = ["sex", "department", "optype", "opname", "dx", "approach", "position", "ane_type", "preop_ecg", "preop_pft"]
NUMERIC = ["age", "height", "weight", "bmi", "asa", "emop", "preop_htn", "preop_dm", "preop_hb", "preop_plt", "preop_pt",
           "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]
FEATURES = CATEGORICAL + NUMERIC
MAX_LEVELS = 150      # rare categories are grouped into "Other" (gradient boosting handles at most 255 levels)


def download() -> None:
    """The API returns a gzip-compressed CSV."""
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        payload = r.read()
    try:
        payload = gzip.decompress(payload)
    except OSError:
        pass                                   # already plain text
    RAW.write_bytes(payload)


def load() -> pd.DataFrame:
    """Clean table: one row per case, target in minutes, typed features."""
    download()
    df = pd.read_csv(RAW)
    df["duration"] = (df["caseend"] - df["casestart"]) / 60.0          # minutes in the operating room
    df["surgery_minutes"] = (df["opend"] - df["opstart"]) / 60.0       # incision to closure (descriptive only)
    # Outliers: keep cases between 15 minutes and 16 hours (removes 3 cases of 16-17 h out of 6,388)
    df = df[(df["duration"] >= 15) & (df["duration"] <= 16 * 60)].copy()
    # Implausible biological values become missing (they are typing errors, not patients)
    df.loc[~df["bmi"].between(10, 70), "bmi"] = np.nan
    df.loc[~df["height"].between(100, 230), "height"] = np.nan
    df.loc[~df["weight"].between(25, 250), "weight"] = np.nan
    # Categorical encoding: missing -> "Unknown", rare levels -> "Other"
    for c in CATEGORICAL:
        s = df[c].fillna("Unknown").astype(str).str.strip()
        keep = s.value_counts().index[:MAX_LEVELS]
        df[c] = pd.Categorical(s.where(s.isin(keep), "Other"))
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train 60% / calibration 20% / test 20%, grouped by patient so that a
    patient's operations never appear on both sides of a split."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    tr_idx, rest_idx = next(g1.split(df, groups=df["subjectid"]))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    cal_idx, te_idx = next(g2.split(rest, groups=rest["subjectid"]))
    return df.iloc[tr_idx].copy(), rest.iloc[cal_idx].copy(), rest.iloc[te_idx].copy()


if __name__ == "__main__":
    d = load()
    tr, ca, te = split(d)
    print(f"{len(d):,} cases, {d['subjectid'].nunique():,} patients; train {len(tr):,}, calibration {len(ca):,}, test {len(te):,}")
    print(d["duration"].describe().round(1))
    print("missing share of the features used:")
    print(d[NUMERIC].isna().mean().sort_values(ascending=False).head(8).round(3))
