"""Render one recorded episode for inspection: three cameras, captions and joint traces.

    python scripts/inspect_episode.py data/sim_ladder_v1/episode_200003 --out episode.mp4 [--sheet sheet.png]

Top: overhead | left wrist | right wrist frames as recorded (448 px, every frame_stride steps).
Captions: instruction, current subtask label, step/time, quality, mistake flag.
Bottom: commanded joint targets of both arms over the whole episode (left solid colors, right
lighter), grippers 0-100, with a cursor at the current step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

JOINT_COLORS = [(230, 80, 60), (240, 170, 40), (90, 180, 80), (60, 140, 230), (160, 90, 220), (40, 40, 40)]
NAMES = ["pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper"]


def plot_strip(actions, width, height, step):
    img = Image.new("RGB", (width, height), (250, 250, 248))
    d = ImageDraw.Draw(img)
    T = len(actions)
    half = height // 2
    for arm in range(2):
        top = arm * half
        d.text((6, top + 4), "left arm" if arm == 0 else "right arm", fill=(30, 30, 30))
        block = actions[:, arm * 6:(arm + 1) * 6]
        lo, hi = -120.0, 120.0
        for j in range(6):
            y = block[:, j]
            if j == 5:
                y = y * 2.4 - 120  # 0..100 gripper onto the same axis
            pts = [(int(i / max(T - 1, 1) * (width - 1)), int(top + half - 8 - (v - lo) / (hi - lo) * (half - 24)))
                   for i, v in zip(range(0, T, max(1, T // width)), y[::max(1, T // width)])]
            if len(pts) > 1:
                d.line(pts, fill=JOINT_COLORS[j], width=2)
        d.line([(0, top + half - 1), (width, top + half - 1)], fill=(200, 200, 200))
    x = int(step / max(T - 1, 1) * (width - 1))
    d.line([(x, 0), (x, height)], fill=(0, 0, 0), width=1)
    for j, name in enumerate(NAMES):
        d.text((width - 520 + j * 86, 4), name, fill=JOINT_COLORS[j])
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, help="Also write a contact sheet PNG of 8 moments")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    meta = json.loads((args.episode / "meta.json").read_text())
    steps = np.load(args.episode / "steps.npz", allow_pickle=False)
    actions, sub_idx, mistake = steps["action"], steps["subtask_index"], steps["mistake"]
    subtasks = meta["subtasks"]
    cams = meta["cameras"]
    frame_steps = sorted(int(p.stem) for p in (args.episode / "frames" / cams[0]).glob("*.jpg"))
    S = 448
    W = S * len(cams)
    frames = []
    for t in frame_steps:
        row = np.concatenate([np.asarray(Image.open(args.episode / "frames" / c / f"{t:05d}.jpg").convert("RGB")) for c in cams], 1)
        canvas = Image.new("RGB", (W, S + 70 + 220), (255, 255, 255))
        canvas.paste(Image.fromarray(row), (0, 0))
        d = ImageDraw.Draw(canvas)
        k = min(t, len(sub_idx) - 1)
        d.text((8, S + 6), f"{meta['task']}   [{meta['task_id']}, style {meta['style']}, quality {meta['quality']}, "
                           f"success {meta['success']}]", fill=(20, 20, 20))
        d.text((8, S + 26), f"subtask: {subtasks[sub_idx[k]]}", fill=(10, 70, 160))
        d.text((8, S + 46), f"step {t} / {meta['steps']}  ({t / meta['control_hz']:.1f} s)"
                            + ("   MISTAKE SEGMENT (retrying grasp)" if mistake[k] else ""), fill=(160, 40, 30) if mistake[k] else (60, 60, 60))
        canvas.paste(plot_strip(actions, W, 220, t), (0, S + 70))
        for i, c in enumerate(cams):
            d.text((i * S + 8, 8), c, fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    import av
    container = av.open(str(args.out), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    for f in frames:
        for p in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(p)
    for p in stream.encode():
        container.mux(p)
    container.close()
    if args.sheet:
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        tiles = [Image.fromarray(frames[i]).resize((frames[i].shape[1] // 3, frames[i].shape[0] // 3)) for i in idx]
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * 2, h * 4), (255, 255, 255))
        for n, tile in enumerate(tiles):
            sheet.paste(tile, ((n % 2) * w, (n // 2) * h))
        sheet.save(args.sheet)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
