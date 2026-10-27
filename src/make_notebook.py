"""Build and execute notebooks/surgery_duration_uncertainty.ipynb."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# From data to robust surgery schedules\n"
       "Learning the uncertainty of surgery duration from open data (VitalDB, 6,385 operations) and measuring what it changes "
       "in operating-room schedules. Code in `src/`; this notebook shows the results."),
    code("import sys; sys.path.insert(0, '../src')\nimport json, pandas as pd\nfrom IPython.display import Image\n"
         "pd.set_option('display.precision', 3); pd.set_option('display.width', 160)\nfrom data import load, split, FEATURES"),
    md("## 1. Data"),
    code("df = load(); tr, ca, te = split(df)\nprint(len(df), 'cases;', len(tr), 'train,', len(ca), 'calibration,', len(te), 'test (grouped by patient)')\n"
         "df['duration'].describe()"),
    code("Image('../figures/00_durations.png')"),
    md("Durations are right-skewed (mean above median) and vary a lot even within one procedure: that variability is what a schedule must absorb."),
    md("## 2. Point predictions"),
    code("pd.read_csv('../results/point_metrics.csv')"),
    md("## 3. Prediction intervals (nominal 80%)"),
    code("pd.read_csv('../results/interval_metrics.csv')"),
    code("Image('../figures/01_intervals_examples.png')"),
    code("Image('../figures/02_coverage.png')"),
    md("Over 10 random splits, and paired on the same splits:"),
    code("pd.read_csv('../results/repeated_splits.csv')"),
    code("pd.read_csv('../results/repeated_splits_paired.csv')"),
    code("Image('../figures/02b_repeated_splits.png')"),
    md("## 4. Scheduling under uncertainty\n200 simulated days, 4 rooms of 480 minutes, 24 elective cases on the waiting list; "
       "exact set-packing model solved with HiGHS, then played with the real durations."),
    code("s = pd.read_csv('../results/schedule_summary.csv'); s"),
    code("Image('../figures/03_schedule_frontier.png')"),
    md("## 5. A first step towards FHIR\nThe first cases exported as HL7 FHIR R4 Patient, Encounter and Procedure resources:"),
    code("b = json.load(open('../data/fhir_bundle_sample.json')); print(len(b['entry']), 'resources')\n"
         "print(json.dumps(b['entry'][2]['resource'], indent=1)[:700])"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
path = ROOT / "notebooks" / "surgery_duration_uncertainty.ipynb"
NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(path.parent)}}).execute()
nbf.write(nb, path)
print("written", path)
