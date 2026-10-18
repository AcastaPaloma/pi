"""Keep the write-up's Results tab in step with training (run on the Mac, e.g. in every 20-minute check).

    python3 scripts/sync_results_site.py             # pull, regenerate, test, commit and push if anything changed
    python3 scripts/sync_results_site.py --dry-run   # pull and regenerate only
    python3 scripts/sync_results_site.py --no-publish   # also start filming watchers, but do not commit or push

From each H100 box it reads the 30-seed scores (outputs/c30_summary.txt), the progress of every x_* training run
and the clips filmed by scripts/box/showcase_when_ready.sh, copies new clips into writeup/assets/results/, and
starts a filming watcher for any run that has none, so every model is on video before it is discarded. It then
regenerates writeup/results-runs.js; the hand-written tree (writeup/results-content.js) names its runs by id, so
score bars, per-checkpoint tables and running state always come from the boxes. Only the generated file and new
clips are committed.
"""
import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "writeup"
CLIPS = SITE / "assets" / "results"
OUT = SITE / "results-runs.js"
HOSTS = ["192.222.52.191", "192.222.55.167"]
TASKS = ["pick_place_any", "bin", "handover", "two_bins"]
NO_WATCH = {"x_control_s1"}  # filmed at step 5000 by hand (the best checkpoint); its clips are in results-content.js
CLIP_RE = re.compile(r"^(pick_place|pick_place_any|bin|handover|two_bins)_(x_.+)_(\d+)_s(\d+)(_goal)?$")
LINE_RE = re.compile(r"^C30 (\S+) delay=(\d+) (.*) total=(\d+)$")


def ssh(host, cmd, timeout=180):
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", f"ubuntu@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{host}: {r.stderr.strip()[-300:]}")
    return r.stdout


def parse_name(name):
    """'x_goal_s0_3000_goal' -> ('x_goal_s0', 3000, True); 'base_dagger2_7000' -> ('base_dagger2', 7000, False)."""
    m = re.match(r"^(.+)_(\d+)(_goal)?$", name)
    return (m.group(1), int(m.group(2)), bool(m.group(3))) if m else (name, None, False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    runs, notes, new_clips, started = {}, [], 0, []
    for host in HOSTS:
        subprocess.run(["rsync", "-a", str(ROOT / "scripts/box/showcase_when_ready.sh"), str(ROOT / "scripts/box/results_status.py"),
                        f"ubuntu@{host}:~/pi/scripts/box/"], check=True)
        subprocess.run(["rsync", "-a", str(ROOT / "scripts/record_showcase.py"), f"ubuntu@{host}:~/pi/scripts/"], check=True)
        state = json.loads(ssh(host, "cd ~/pi && python3 scripts/box/results_status.py"))
        for run, info in state["runs"].items():
            runs.setdefault(run, {"scores": {}, "clips": []}).update(
                {"step": info["step"], "of": info["of"], "state": info["state"], "host": host})
        for line in state["c30"]:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            name, delay, body, total = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            run, step, goal = parse_name(name)
            per = dict(re.findall(r"(\w+)=(\d+)/30", body))
            if any(t not in per for t in TASKS):
                continue
            entry = runs.setdefault(run, {"scores": {}, "clips": []})
            entry["scores"][(step, delay, goal)] = {"step": step, "delay": delay, "goal": goal, "total": total,
                                                    "tasks": [int(per[t]) for t in TASKS]}
        for card in state["clips"]:
            m = CLIP_RE.match(card["stem"])
            if not m:
                continue  # hand-filmed clips (e.g. handover_c1_5000_s4002) are listed in results-content.js
            task, run, step, seed, goal = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), bool(m.group(5))
            for ext in ("mp4", "jpg"):
                dest = CLIPS / f"{card['stem']}.{ext}"
                if not dest.exists():
                    subprocess.run(["scp", "-q", f"ubuntu@{host}:~/pi/outputs/showcase/{card['stem']}.{ext}", str(dest)], check=True)
                    new_clips += ext == "mp4"
            runs.setdefault(run, {"scores": {}, "clips": []})["clips"].append({
                "task": task, "step": step, "seed": seed, "layout": seed - 3999, "goal": goal,
                "success": bool(card["success"]), "seconds": float(card["seconds"]),
                "video": f"assets/results/{card['stem']}.mp4", "poster": f"assets/results/{card['stem']}.jpg"})
        filmed = {(m.group(2), int(m.group(3))) for c in state["clips"] if (m := CLIP_RE.match(c["stem"]))}
        for run, info in state["runs"].items():
            final = info["of"]
            if run in NO_WATCH or run in state["watching"] or final is None or (run, final) in filmed:
                continue
            if not args.dry_run:
                ssh(host, f"cd ~/pi && (setsid nohup bash scripts/box/showcase_when_ready.sh {run} {final} "
                          f"> logs/showcase_{run}.log 2>&1 < /dev/null &) ; true", timeout=60)
            started.append(f"{run}@{final}")
    for entry in runs.values():
        entry["scores"] = sorted(entry["scores"].values(), key=lambda s: (s["goal"], s["delay"], s["step"]))
        entry["clips"].sort(key=lambda c: (c["step"], c["goal"], c["seed"]))
    body = "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in sorted(runs.items())) + "\n}"
    header = "// Generated by scripts/sync_results_site.py from the training boxes; do not edit by hand.\n"
    old = OUT.read_text() if OUT.exists() else ""
    changed = f"export const runs = {body};" not in old
    if changed:
        stamp = datetime.now().strftime("%-d %b %Y, %H:%M")
        OUT.write_text(f"{header}export const updated = '{stamp}';\nexport const runs = {body};\n")
    if started:
        notes.append("filming watchers started: " + ", ".join(started))
    summary = f"{len(runs)} runs, {sum(len(r['scores']) for r in runs.values())} scores, {new_clips} new clips"
    if args.dry_run or args.no_publish or not changed:
        mode = "dry run" if args.dry_run else "not published" if args.no_publish else "no change"
        print(f"sync: {summary}; {mode}" + (f"; {notes[0]}" if notes else ""))
        return
    subprocess.run(["npm", "--prefix", str(SITE), "test", "--silent"], check=True, capture_output=True)
    git = lambda *a, **k: subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True, text=True, **k)
    git("add", str(OUT), str(CLIPS))
    staged = git("diff", "--cached", "--name-only").stdout.split()
    if not staged:
        print(f"sync: {summary}; nothing to commit")
        return
    git("commit", "-q", "-m", f"Results tab: sync scores and clips from the training boxes ({summary})",
        "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", str(OUT), str(CLIPS))  # only these paths
    if subprocess.run(["git", "-C", str(ROOT), "push", "-q"], capture_output=True).returncode:
        git("pull", "-q", "--rebase", "--autostash")
        git("push", "-q")
    print(f"sync: {summary}; pushed {git('rev-parse', '--short', 'HEAD').stdout.strip()}" + (f"; {notes[0]}" if notes else ""))


if __name__ == "__main__":
    main()
