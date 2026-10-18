"""Print this box's state for scripts/sync_results_site.py as one JSON object (run from ~/pi):
30-seed score lines, progress of every x_* training run, filmed showcase clips and active filming watchers."""
import glob
import json
import re
import subprocess
from pathlib import Path

out = {"c30": [], "runs": {}, "clips": [], "watching": []}
summary = Path("outputs/c30_summary.txt")
if summary.exists():
    out["c30"] = summary.read_text().splitlines()
procs = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
for log in sorted(glob.glob("logs/train_x_*.log")):
    run = Path(log).stem[len("train_"):]
    try:
        total = int(json.loads(Path(f"runs/{run}/run.json").read_text())["args"]["steps"])
    except (OSError, ValueError, KeyError):
        total = None
    step, saved, crashed = 0, [], False
    for line in open(log, errors="ignore"):
        if line.startswith("step="):
            step = int(line[5:line.index(" ")])
        elif line.startswith("saved runs/"):
            saved.append(int(re.search(r"step_(\d+)", line).group(1)))
        elif line.startswith("Traceback"):
            crashed = True
    alive = any("pi_model.train" in p and f"runs/{run} " in p + " " for p in procs)
    state = "done" if total in saved else "running" if alive else "stopped" if crashed or step else "starting"
    out["runs"][run] = {"step": max([step] + saved), "of": total, "saved": saved, "state": state}
for path in glob.glob("outputs/showcase/*.json"):
    card = json.loads(Path(path).read_text())
    card["stem"] = Path(path).stem
    out["clips"].append(card)
for p in procs:
    m = re.search(r"showcase_when_ready\.sh (\S+)", p)
    if m and not p.startswith("grep"):
        out["watching"].append(m.group(1))
print(json.dumps(out))
