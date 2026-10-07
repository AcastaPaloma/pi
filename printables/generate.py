"""Generate watertight ASCII STL objects in millimeters, with no dependencies.

Run: python printables/generate.py. Color comes from filament, not STL files.
"""
from math import cos, sin, pi, sqrt
from pathlib import Path


def box(x, y, z):
    v = [(a, b, c) for c in (0., z) for b in (-y / 2, y / 2) for a in (-x / 2, x / 2)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return [(v[a], v[b], v[c]) for a, b, c, d in faces] + [(v[a], v[c], v[d]) for a, b, c, d in faces]


def cylinder(diameter, height, segments=96):
    rings = [[(diameter / 2 * cos(i * 2 * pi / segments), diameter / 2 * sin(i * 2 * pi / segments), z)
              for i in range(segments)] for z in (0., height)]
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        triangles.extend([((0., 0., 0.), rings[0][j], rings[0][i]),
                          ((0., 0., height), rings[1][i], rings[1][j]),
                          (rings[0][i], rings[0][j], rings[1][j]),
                          (rings[0][i], rings[1][j], rings[1][i])])
    return triangles


def write_stl(path, triangles):
    lines = [f"solid {path.stem}"]
    for a, b, c in triangles:
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        length = sqrt(sum(t * t for t in n))
        n = [t / length for t in n]
        lines += ["  facet normal " + " ".join(f"{t:.9g}" for t in n), "    outer loop"]
        lines += ["      vertex " + " ".join(f"{t:.9g}" for t in vertex) for vertex in (a, b, c)]
        lines += ["    endloop", "  endfacet"]
    path.write_text("\n".join(lines + [f"endsolid {path.stem}"]) + "\n")


if __name__ == "__main__":
    root = Path(__file__).parent
    for size in (24, 30, 36):
        write_stl(root / f"cube_{size}mm.stl", box(size, size, size))
        write_stl(root / f"cylinder_{size}mm.stl", cylinder(size, size))
    write_stl(root / "goal_disk_80mm.stl", cylinder(80, 2))
    print(f"Wrote seven STL files to {root}")
