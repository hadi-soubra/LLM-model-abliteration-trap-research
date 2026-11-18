"""Copy LEARNING_LOG.md and the update time into the notebook, so the journal page shows the current log.

    .venv/bin/python tools/build_notebook.py
"""
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = {"built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "journal": (ROOT / "LEARNING_LOG.md").read_text()}
(ROOT / "report" / "data.js").write_text("window.REPORT = " + json.dumps(report) + ";\n")
print("report/data.js updated")
