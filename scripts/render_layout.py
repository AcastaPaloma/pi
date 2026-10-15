"""Dimensioned top view of the rig (table, arm bases, workspace, overhead camera).

    python scripts/render_layout.py --out docs/images/rig_layout.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/images/rig_layout.png"))
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.52, 0.5)
    ax.set_facecolor("#f7f4ee")
    # Table 100 x 80 cm, top at z = 0
    ax.add_patch(Rectangle((-0.5, -0.4), 1.0, 0.8, facecolor="#e6d3b4", edgecolor="#6b5a43", lw=1.5))
    # Arm bases: SO-101 base footprints at x = +-0.18, y = -0.22, pointing +y
    for x, name in ((-0.18, "left follower"), (0.18, "right follower")):
        ax.add_patch(Rectangle((x - 0.05, -0.27), 0.10, 0.10, facecolor="#f2c230", edgecolor="#5a4a10", lw=1.2))
        ax.annotate("", xy=(x, -0.05), xytext=(x, -0.17), arrowprops=dict(arrowstyle="->", lw=1.5, color="#5a4a10"))
        ax.text(x, -0.33, f"{name}\nbase ({x:+.2f}, -0.22)", ha="center", va="top", fontsize=9)
    # Left-arm workspace used by the tasks
    ax.add_patch(Rectangle((-0.30, -0.08), 0.25, 0.21, fill=False, ls="--", lw=1.2, edgecolor="#2d6a4f"))
    ax.text(-0.175, 0.205, "task workspaces (dashed)\nleft: x -0.30..-0.05, y -0.08..0.13", ha="center", va="bottom", fontsize=8, color="#2d6a4f")
    ax.add_patch(Rectangle((0.05, -0.08), 0.25, 0.21, fill=False, ls="--", lw=1.2, edgecolor="#2d6a4f"))
    # Object and goal nominal positions
    ax.add_patch(Rectangle((-0.215, -0.04), 0.03, 0.03, facecolor="#d9382a", edgecolor="k"))
    ax.text(-0.31, -0.03, "cube 30 mm\n(-0.20, -0.025)", ha="right", va="center", fontsize=8)
    ax.add_patch(Circle((-0.14, -0.02), 0.04, facecolor="#4fb37f", alpha=0.6, edgecolor="#2d6a4f"))
    ax.text(-0.14, 0.03, "goal disk 80 mm\n(-0.14, -0.02)", ha="center", va="bottom", fontsize=8)
    # Overhead camera
    ax.add_patch(Rectangle((-0.016, -0.03 - 0.016), 0.032, 0.032, facecolor="#1b3a4b", edgecolor="k"))
    ax.text(0.03, -0.085, "overhead camera B030401 at (0, -0.03)\nlens 0.70 m above the table, looking straight down\nimage top = +y", ha="left", va="top", fontsize=8, color="#1b3a4b")
    # Overhead footprint at table level: 61 x 47.8 degrees at 0.70 m -> 0.82 x 0.62 m
    ax.add_patch(Rectangle((-0.41, -0.03 - 0.31), 0.82, 0.62, fill=False, lw=1.0, edgecolor="#1b3a4b", ls=":"))
    ax.text(-0.40, 0.29, "overhead field of view at the table: 0.82 x 0.62 m", ha="left", va="bottom", fontsize=8, color="#1b3a4b")
    # Gantry
    ax.plot([-0.56, 0.56], [0.15, 0.15], color="#444", lw=4, solid_capstyle="round")
    ax.plot([0, 0], [-0.03, 0.15], color="#444", lw=2)
    ax.text(0.30, 0.165, "gantry crossbar, underside 0.755 m above the table", ha="center", va="bottom", fontsize=8, color="#444")
    # Dimensions
    ax.annotate("", xy=(0.5, -0.46), xytext=(-0.5, -0.46), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(0, -0.47, "table 1.00 m", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(0.55, 0.4), xytext=(0.55, -0.4), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(0.56, 0, "0.80 m", va="center", fontsize=9, rotation=90)
    ax.annotate("", xy=(0.18, -0.44), xytext=(-0.18, -0.44), arrowprops=dict(arrowstyle="<->", lw=1, color="#5a4a10"))
    ax.text(0, -0.43, "bases 0.36 m apart", ha="center", va="bottom", fontsize=8, color="#5a4a10")
    ax.set_xlabel("x (m), origin at table center; +y points away from the arms")
    ax.set_ylabel("y (m)")
    ax.set_title("SO-101 rig layout used by pi_sim (top view, tabletop z = 0)")
    ax.grid(True, alpha=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
