"""Collect every closed-loop evaluation from all machines into one leaderboard (autoresearch-style).

    python3 scripts/leaderboard.py                # prints the table, writes docs/RESULTS.md
    python3 scripts/leaderboard.py --hosts ubuntu@1.2.3.4,...

Each machine keeps `~/pi/outputs/eval_<run>_step<N>[_<task>]/results.json` (written by
pi_model.eval_sim). Success rates get Wilson 95% intervals so small samples are read with care.
Stop rule used in docs/RESEARCH_LOG.md: a run is only stopped after >= 4 evaluated checkpoints when
the upper bound of its best checkpoint is below the lower bound of the leading run on the same task.
"""
import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HOSTS = ["ubuntu@192.222.52.191", "ubuntu@129.213.124.160", "ubuntu@209.20.159.198", "ubuntu@209.20.157.232",
         "ubuntu@209.20.158.194", "ubuntu@132.145.199.243"]
REMOTE = r'''python3 - <<'EOF'
import json, glob, os, statistics as s
for f in glob.glob(os.path.expanduser("~/pi/outputs/eval_*/results.json")):
    try:
        r = json.load(open(f))
        res = r.get("results", [])
        if not res:
            continue
        print(json.dumps({"dir": os.path.basename(os.path.dirname(f)), "n": len(res),
            "success": sum(bool(x["success"]) for x in res), "lift": sum(bool(x.get("lifted")) for x in res),
            "strict": sum(bool(x["success"]) and bool(x.get("lifted")) for x in res),
            "closest_mm": round(1000 * s.median(x.get("closest_hand_to_cube_m", float("nan")) for x in res)),
            "both": sum(all((x.get("touched") or {"a": False}).values()) for x in res),
            "task": res[0].get("task_id", "pick_place"), "mtime": os.path.getmtime(f)}))
    except Exception:
        continue
EOF'''


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse(name):
    m = re.match(r"eval_(.+?)_step(\d+)(?:_(.+))?$", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def collect(hosts):
    rows = []
    for h in hosts:
        try:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", h, REMOTE],
                                 capture_output=True, text=True, timeout=90).stdout
        except subprocess.TimeoutExpired:
            continue
        for line in out.splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = parse(r["dir"])
            if not p:
                continue
            run, step, task = p
            r.update(run=run, step=step, task=task or r["task"], host=h.split("@")[1])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", default=",".join(HOSTS))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "RESULTS.md")
    args = ap.parse_args()
    rows = collect(args.hosts.split(","))
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["run"], r["task"])].append(r)
    lines = [f"# Closed-loop results (generated {datetime.now():%Y-%m-%d %H:%M})", "",
             "Success with Wilson 95% interval; strict = success with the object lifted (rules out pushing it into place); "
             "lift = object raised; closest = median hand-to-object distance.", "",
             "| Run | Task | Step | Success | Strict | 95% CI | Lift | Closest | Both arms | Host |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = []
    for (run, task), rs in sorted(by_run.items()):
        rs.sort(key=lambda r: r["step"])
        for r in rs:
            lo, hi = wilson(r["success"], r["n"])
            lines.append(f"| {run} | {task} | {r['step']} | {r['success']}/{r['n']} | {r.get('strict', r['success'])}/{r['n']} | {lo:.2f}–{hi:.2f} | "
                         f"{r['lift']}/{r['n']} | {r['closest_mm']} mm | {r['both']}/{r['n']} | {r['host']} |")
        best = max(rs, key=lambda r: (r["success"] / r["n"], r["lift"] / r["n"]))
        lo, hi = wilson(best["success"], best["n"])
        trend = " → ".join(f"{r['success']}/{r['n']}" for r in rs[-4:])
        summary.append((task, best["success"] / best["n"], f"| {run} | {task} | {len(rs)} | step {best['step']}: "
                        f"{best['success']}/{best['n']} ({lo:.2f}–{hi:.2f}) | {trend} |"))
    lines += ["", "## Best per run", "", "| Run | Task | Checkpoints | Best | Last four |", "| --- | --- | --- | --- | --- |"]
    lines += [s for _, _, s in sorted(summary, key=lambda x: (x[0], -x[1]))]
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
