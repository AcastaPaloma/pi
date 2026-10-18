"""Print this box's state for scripts/sync_results_site.py as one JSON object (run from ~/pi):
30-seed score lines, progress of every x_* training run, filmed showcase clips, active filming watchers, and the
training-loss history of every run on this box (runs/*/log.jsonl, thinned to about 300 points per run)."""
import glob
import json
import re
import subprocess
from pathlib import Path

out = {"c30": [], "runs": {}, "clips": [], "watching": [], "loss": {}}
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
POINTS = 300
for path in sorted(glob.glob("runs/*/log.jsonl")):
    rows = []
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
            rows.append((int(r["step"]), float(r.get("flow_loss", r.get("loss"))), float(r.get("ce_loss") or 0.0)))
        except (ValueError, KeyError, TypeError):
            continue
    rows = sorted({r[0]: r for r in rows}.values())  # one row per step (a resumed log can repeat steps)
    if len(rows) < 3:
        continue
    keep = rows[:: max(1, -(-len(rows) // POINTS))]
    if keep[-1] != rows[-1]:
        keep.append(rows[-1])
    entry = {"s": [r[0] for r in keep], "f": [float(f"{r[1]:.4g}") for r in keep]}
    if any(r[2] > 0 for r in keep):
        entry["c"] = [float(f"{r[2]:.4g}") for r in keep]
    out["loss"][Path(path).parent.name] = entry
print(json.dumps(out))
