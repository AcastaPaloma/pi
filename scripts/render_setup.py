"""Render the simulated SO-101 rig: scene view plus the three Arducam sensor views.

    python scripts/render_setup.py --out outputs/mujoco_setup.png
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pi_sim import RobotEnv
from pi_sim.demo import ScriptedDemo


def scene_view(env, width, height, *, azimuth, elevation, distance, lookat):
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    renderer.update_scene(env.data, camera=cam)
    rgb = renderer.render().copy()
    renderer.close()
    return rgb


def label(image, text):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([8, 8, 8 + box[2] - box[0] + 16, 8 + box[3] - box[1] + 14], fill=(20, 24, 30))
    draw.text((16, 12), text, fill="white", font=font)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/mujoco_setup.png"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--phase", default="transfer", help="Scripted demo phase to freeze at")
    parser.add_argument("--color", default="red")
    args = parser.parse_args()
    with RobotEnv(task="pick_place", object_color=args.color) as env:
        env.reset(seed=args.seed, options={"placement_range": "wide"})
        demo = ScriptedDemo(env)
        for _ in range(env.max_episode_steps):
            if demo.stage == args.phase and demo.ticks > 5:
                break
            env.step(demo.action())
        wide = scene_view(env, 1920, 900, azimuth=135, elevation=-24, distance=1.75, lookat=[0.0, -0.05, 0.22])
        closeup = scene_view(env, 960, 720, azimuth=160, elevation=-20, distance=0.75, lookat=[-0.15, -0.05, 0.08])
        views = {cam: env.render_camera(cam, native=True) for cam in env.camera_names}
    sheet = Image.new("RGB", (1920, 900 + 720 + 16 * 3), (12, 14, 18))
    sheet.paste(label(wide, "MuJoCo rig: two SO-101 followers, overhead gantry camera, wrist cameras (Arducam IMX708 USB)"), (0, 0))
    y = 900 + 16
    sheet.paste(label(closeup, "Close-up: left follower lifting the printed cube"), (0, y))
    x = 960 + 16
    tile_w, tile_h = 944, 352
    names = {"overhead": "overhead camera (B030401, 61 deg H, 0.70 m above table)",
             "left_wrist": "left wrist camera", "right_wrist": "right wrist camera"}
    for i, cam in enumerate(env.camera_names):
        tile = Image.fromarray(views[cam]).resize((tile_w if i == 0 else (tile_w - 16) // 2, tile_h), Image.Resampling.BILINEAR)
        tile = label(np.asarray(tile), names[cam])
        if i == 0:
            sheet.paste(tile, (x, y))
        else:
            sheet.paste(tile, (x + (i - 1) * ((tile_w - 16) // 2 + 16), y + tile_h + 16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"Saved {args.out} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
