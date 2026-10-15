"""Compose pinned SO-101 Menagerie models into a colored-object tabletop scene."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .cameras import CAMERAS, FIXED_CAMERAS, load_camera_rig

ASSETS = Path(__file__).parent / "assets" / "vendor"
ARMS = ("left", "right")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
COLORS = {"red": (0.92, 0.24, 0.14, 1), "green": (0.2, 0.72, 0.3, 1),
          "blue": (0.16, 0.4, 0.92, 1), "yellow": (0.95, 0.8, 0.15, 1)}


def _prefixed(path: Path, prefix: str) -> ET.Element:
    root = ET.parse(path).getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.attrib["file"]).stem))
        mesh.set("file", str(path.parent / "assets" / mesh.attrib["file"]))
    refs = {"name", "class", "childclass", "mesh", "material", "joint", "joint1",
            "joint2", "body1", "body2", "tendon", "target"}
    for elem in root.iter():
        for key in refs & elem.attrib.keys():
            elem.set(key, prefix + elem.attrib[key])
    return root


def _numbers(values):
    return " ".join(map(str, values))


def _configure_camera(camera, spec):
    # MuJoCo principalpixel is a frustum offset, not OpenCV's top-left cx/cy.
    fx, fy, cx, cy = spec.intrinsics
    camera.attrib.clear()
    camera.attrib.update(mode="fixed", pos=_numbers(spec.pos), quat=_numbers(spec.quat),
                         resolution=f"{spec.width} {spec.height}", sensorsize="1 1",
                         focalpixel=f"{fx} {fy}",
                         principalpixel=f"{spec.width / 2 - cx} {spec.height / 2 - cy}", ipd="0")


def _fixed_camera(world, name, spec):
    mount = ET.SubElement(world, "body", name=f"{name}_camera_mount",
                          pos=_numbers(spec.pos), quat=_numbers(spec.quat))
    camera = ET.SubElement(mount, "camera")
    _configure_camera(camera, spec)
    camera.set("name", name)
    camera.set("pos", "0 0 0")
    camera.set("quat", "1 0 0 0")
    # 32x32 mm U20CAM board proxies, with the optical center in front
    # of the lens. They never obstruct their own feed or alter robot contacts.
    common = {"contype": "0", "conaffinity": "0", "group": "2"}
    ET.SubElement(mount, "geom", name=f"{name}_camera_housing", type="box",
                  size="0.016 0.016 0.0008", pos="0 0 0.017", rgba="0.08 0.3 0.16 1", **common)
    ET.SubElement(mount, "geom", name=f"{name}_camera_lens", type="cylinder",
                  size="0.006 0.007", pos="0 0 0.009", rgba="0.08 0.09 0.11 1", **common)
    x, y, z = spec.pos
    if name == "overhead":
        # A gantry outside the table edges; its exact dimensions are nominal.
        bar_y, bar_z = y + 0.18, z + 0.055
        for side, bar_x in (("left", x - 0.56), ("right", x + 0.56)):
            ET.SubElement(world, "geom", name=f"overhead_{side}_support", type="capsule", size="0.01",
                          fromto=_numbers([bar_x, bar_y, -0.74, bar_x, bar_y, bar_z]),
                          rgba="0.22 0.24 0.27 1", **common)
        ET.SubElement(world, "geom", name="overhead_crossbar", type="capsule", size="0.01",
                      fromto=_numbers([x - 0.56, bar_y, bar_z, x + 0.56, bar_y, bar_z]),
                      rgba="0.22 0.24 0.27 1", **common)
        ET.SubElement(world, "geom", name="overhead_camera_bracket", type="capsule", size="0.008",
                      fromto=_numbers([x, y, z + 0.025, x, bar_y, bar_z]),
                      rgba="0.22 0.24 0.27 1", **common)
    else:
        ET.SubElement(world, "geom", name=f"{name}_camera_stand", type="capsule", size="0.008",
                      fromto=_numbers([x, y, -0.74, x, y, z + 0.035]), rgba="0.22 0.24 0.27 1", **common)


def build_model(object_shape="cube", object_color="red", camera_rig=None) -> mujoco.MjModel:
    if object_shape not in ("cube", "cylinder") or object_color not in COLORS:
        raise ValueError("Choose cube/cylinder and red/green/blue/yellow")
    cameras = load_camera_rig(camera_rig)
    root = ET.fromstring('''<mujoco model="pi_sim_bimanual_so101">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
      <visual><global offwidth="640" offheight="640"/><quality shadowsize="2048"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
      <statistic center="0 0 0.15" extent="0.9"/>
      <default><mesh maxhullvert="128"/></default><asset>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.12 0.15 0.19" rgb2="0.17 0.2 0.24" width="256" height="256"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="3 3"/>
      </asset>
      <worldbody>
        <light name="key_light" pos="0 -0.5 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.76" material="floor_mat"/>
        <geom name="table" type="box" size="0.5 0.4 0.025" pos="0 0 -0.025" rgba="0.72 0.64 0.51 1"/>
        <body name="cube" pos="-0.18 0 0.016">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.015 0.015 0.015" mass="0.015" friction="1 0.01 0.001" rgba="0.92 0.24 0.14 1"/>
        </body>
        <body name="goal" mocap="true" pos="-0.12 0.06 0.001">
          <geom name="goal_disk_geom" type="cylinder" size="0.04 0.001" contype="0" conaffinity="0" rgba="0.18 0.65 0.40 0.7"/>
          <site name="reach_target" pos="0 0 0.07" size="0.012" rgba="0.18 0.9 0.4 0.55"/>
        </body>
      </worldbody>
      <contact/><tendon/><equality/><actuator/>
    </mujoco>''')
    obj = root.find(".//geom[@name='cube_geom']")
    obj.set("type", "box" if object_shape == "cube" else "cylinder")
    obj.set("size", "0.015 0.015 0.015" if object_shape == "cube" else "0.015 0.015")
    obj.set("rgba", " ".join(map(str, COLORS[object_color])))
    for name in FIXED_CAMERAS:
        _fixed_camera(root.find("worldbody"), name, cameras[name])
    # Both bases point into the tabletop (+Y); one wrist camera per follower.
    for side, x in (("left", -0.18), ("right", 0.18)):
        arm = _prefixed(ASSETS / "robotstudio_so101/so101.xml", f"{side}_")
        camera = arm.find(f".//camera[@name='{side}_wrist_cam']")
        _configure_camera(camera, cameras[f"{side}_wrist"])
        camera.set("name", f"{side}_wrist")
        wrapper = ET.SubElement(root.find("worldbody"), "body", name=f"{side}_pedestal",
                                pos=f"{x} -0.22 0", euler=f"0 0 {np.pi / 2}")
        wrapper.append(arm.find("worldbody/body"))
        for section in ("default", "asset", "contact", "tendon", "equality", "actuator"):
            contents = arm.find(section)
            if contents is not None:
                root.find(section).extend([el for el in contents if section != "default" or el.tag == "default"])
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
