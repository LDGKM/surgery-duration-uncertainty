"""Run the whole study:  python run_all.py   (about 10 minutes)"""
import subprocess
import sys
from pathlib import Path

for step in ("data.py", "eda.py", "models.py", "run_repeats.py", "scheduling.py", "fhir.py", "make_notebook.py"):
    print(f"\n=== {step}")
    subprocess.run([sys.executable, step], check=True, cwd=Path(__file__).parent / "src")
