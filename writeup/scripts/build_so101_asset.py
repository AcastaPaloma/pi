"""Build the compact SO-101 asset used by the inference page.

Reads the vendored MuJoCo Menagerie model (pi_sim/assets/vendor/robotstudio_so101,
Apache-2.0), keeps the visual meshes, remeshes and decimates them, splits normals at sharp
edges, and writes:

  writeup/assets/so101/so101.bin   quantized vertices, normals and indices
  writeup/assets/so101/so101.json  body tree, joints, geoms and mesh offsets

Run once after changing the vendored model:
  python writeup/scripts/build_so101_asset.py
Requires numpy, trimesh, fast-simplification, scikit-image, scipy and networkx.
"""
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "pi_sim/assets/vendor/robotstudio_so101"
OUT = ROOT / "writeup/assets/so101"
KEEP = 0.1          # fraction of the original triangle count kept per mesh
PITCH = 0.0008       # voxel size for remeshing, meters
CREASE = math.radians(48)


def numbers(text, default):
    return [float(v) for v in text.split()] if text else default


def quat(text):
    w, x, y, z = numbers(text, [1, 0, 0, 0])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / n, x / n, y / n, z / n]


def build():
    xml = ET.parse(SOURCE / "so101.xml").getroot()
    colors = {m.get("name"): numbers(m.get("rgba"), [1, 1, 1, 1]) for m in xml.iter("material")}
    meshes, blob = {}, bytearray()

    def mesh_entry(name):
        if name in meshes:
            return name
        source = trimesh.load(SOURCE / "assets" / f"{name}.stl")
        # The printed parts are perforated with hex holes, which stop quadric
        # decimation early. A ~1 mm voxel remesh closes them and keeps the silhouette.
        voxels = source.voxelized(pitch=PITCH).fill()
        shell = voxels.marching_cubes
        shell.apply_transform(voxels.transform)
        shell.apply_translation(source.bounds.mean(0) - shell.bounds.mean(0))
        trimesh.smoothing.filter_taubin(shell, iterations=16)
        target = int(np.clip(len(source.faces) * KEEP, 500, 4200))
        points, faces = fast_simplification.simplify(
            shell.vertices.astype(np.float32), shell.faces.astype(np.int32), target_count=target)
        simple = trimesh.Trimesh(points, faces, process=True)
        smooth = trimesh.graph.smooth_shade(simple, angle=CREASE)
        v = smooth.vertices.astype(np.float64)
        n = smooth.vertex_normals.astype(np.float64)
        f = smooth.faces.astype(np.uint32)
        low, high = v.min(0), v.max(0)
        scale = np.maximum(high - low, 1e-9) / 65535
        q = np.round((v - low) / scale - 32768).astype(np.int16)
        nq = np.round(np.clip(n, -1, 1) * 127).astype(np.int8)
        wide = len(v) > 65535
        entry = {"vertices": len(v), "triangles": len(f), "min": low.tolist(), "scale": scale.tolist()}
        for key, data in (("position", q.tobytes()), ("normal", nq.tobytes()),
                          ("index", f.astype(np.uint32 if wide else np.uint16).tobytes())):
            while len(blob) % 4:
                blob.append(0)
            entry[key] = [len(blob), len(data)]
            blob.extend(data)
        entry["wideIndex"] = wide
        meshes[name] = entry
        return name

    def body(node):
        joint = node.find("joint")
        out = {
            "name": node.get("name"),
            "pos": numbers(node.get("pos"), [0, 0, 0]),
            "quat": quat(node.get("quat")),
            "joint": None if joint is None else {
                "name": joint.get("name"), "axis": numbers(joint.get("axis"), [0, 0, 1]),
                "range": numbers(joint.get("range"), [-math.pi, math.pi])},
            "geoms": [], "sites": {}, "cameras": {}, "children": [],
        }
        for geom in node.findall("geom"):
            if geom.get("class") != "visual":
                continue
            out["geoms"].append({"mesh": mesh_entry(geom.get("mesh")), "pos": numbers(geom.get("pos"), [0, 0, 0]),
                                 "quat": quat(geom.get("quat")), "rgba": colors[geom.get("material")]})
        for site in node.findall("site"):
            out["sites"][site.get("name")] = {"pos": numbers(site.get("pos"), [0, 0, 0]), "quat": quat(site.get("quat"))}
        for child in node.findall("body"):
            if child.get("name") == "camera_mount":
                # Fold the camera mount into its parent; it has no joint.
                for geom in child.findall("geom"):
                    if geom.get("class") == "visual":
                        out["geoms"].append({"mesh": mesh_entry(geom.get("mesh")), "pos": numbers(geom.get("pos"), [0, 0, 0]),
                                             "quat": quat(geom.get("quat")), "rgba": colors[geom.get("material")]})
                cam = child.find("camera")
                out["cameras"]["wrist"] = {"pos": numbers(cam.get("pos"), [0, 0, 0]), "quat": quat(cam.get("quat")),
                                           "fovy": float(cam.get("fovy"))}
                continue
            out["children"].append(body(child))
        return out

    tree = body(xml.find("worldbody/body"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "so101.bin").write_bytes(bytes(blob))
    (OUT / "so101.json").write_text(json.dumps({
        "source": "MuJoCo Menagerie robotstudio_so101 (Apache-2.0), visual meshes decimated for the web",
        "units": "meters, z up, MuJoCo quaternions as [w, x, y, z]",
        "meshes": meshes, "root": tree}, separators=(",", ":")))
    triangles = sum(m["triangles"] for m in meshes.values())
    print(f"{len(meshes)} meshes, {triangles} triangles, {len(blob) / 1024:.0f} KiB")


if __name__ == "__main__":
    build()
