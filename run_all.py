# Rebuild every result with one command:
#     python run_all.py
#
# Step 1: download the data (backfill.py)
# Step 2: run the whole notebook from top to bottom
#         (the finished copy is saved as karachi_air_quality_full_run.ipynb)
# Then start the dashboard with:  python -m streamlit run app.py
#
# This can take several minutes because of the SARIMA steps.

import subprocess
import sys

steps = [
    [sys.executable, "backfill.py"],
    [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", "karachi_air_quality_full.ipynb",
        "--output", "karachi_air_quality_full_run.ipynb",
        "--ExecutePreprocessor.timeout=3600",
    ],
]

for step in steps:
    print("Running:", " ".join(step))
    result = subprocess.run(step)
    if result.returncode != 0:
        print("This step failed, so I stopped. Fix the error above and run again.")
        sys.exit(1)

print("")
print("Done. Now start the dashboard with:  python -m streamlit run app.py")
