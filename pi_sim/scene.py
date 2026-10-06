"""Compose unmodified, vendored Menagerie models into a tabletop scene."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ASSETS = Path(__file__).parent / "assets" / "vendor"
ARMS = ("left", "right")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def _prefixed(path: Path, prefix: str) -> ET.Element:
    root = ET.parse(path).getroot()
    # Menagerie meshes can get implicit names from filenames; make them explicit.
    for mesh in root.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.attrib["file"]).stem))
        mesh.set("file", str(path.parent / "assets" / mesh.attrib["file"]))
    refs = {"name", "class", "childclass", "mesh", "material", "joint", "joint1",
            "joint2", "body1", "body2", "tendon", "target"}
    for elem in root.iter():
        for key in refs & elem.attrib.keys():
            elem.set(key, prefix + elem.attrib[key])
    return root


def build_model() -> mujoco.MjModel:
    root = ET.fromstring('''<mujoco model="pi_sim_bimanual_ur5e">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
      <visual><global offwidth="640" offheight="640"/><quality shadowsize="2048"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
      <statistic center="0 0 0.2" extent="1.6"/>
      <default/><asset>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.12 0.15 0.19" rgb2="0.17 0.2 0.24" width="256" height="256"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="3 3"/>
      </asset>
      <worldbody>
        <light pos="0 -1 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.76" material="floor_mat"/>
        <geom name="table" type="box" size="0.9 0.6 0.035" pos="0 0 -0.035" rgba="0.72 0.64 0.51 1"/>
        <geom name="leg1" type="box" size="0.04 0.04 0.345" pos="-0.8 -0.5 -0.415" rgba="0.18 0.2 0.23 1"/>
        <geom name="leg2" type="box" size="0.04 0.04 0.345" pos="0.8 -0.5 -0.415" rgba="0.18 0.2 0.23 1"/>
        <geom name="leg3" type="box" size="0.04 0.04 0.345" pos="-0.8 0.5 -0.415" rgba="0.18 0.2 0.23 1"/>
        <geom name="leg4" type="box" size="0.04 0.04 0.345" pos="0.8 0.5 -0.415" rgba="0.18 0.2 0.23 1"/>
        <camera name="front" pos="1.3 -1.7 1.25" xyaxes="0.794 0.607 0 -0.261 0.342 0.903" fovy="48"/>
        <body name="cube" pos="-0.2 0 0.021">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.04" friction="1 0.01 0.001" rgba="0.92 0.24 0.14 1"/>
        </body>
        <body name="goal" mocap="true" pos="-0.15 0.18 0.001">
          <geom type="cylinder" size="0.065 0.001" contype="0" conaffinity="0" rgba="0.18 0.65 0.40 0.7"/>
          <site name="reach_target" pos="0 0 0.14" size="0.018" rgba="0.18 0.9 0.4 0.55"/>
        </body>
      </worldbody>
      <contact/><tendon/><equality/><actuator/>
    </mujoco>''')
    for side, x, yaw in (("left", -0.65, -1.57079632679), ("right", 0.65, 1.57079632679)):
        arm = _prefixed(ASSETS / "universal_robots_ur5e/ur5e.xml", f"{side}_")
        hand = _prefixed(ASSETS / "robotiq_2f85/2f85.xml", f"{side}_gripper_")
        wrist = arm.find(f".//body[@name='{side}_wrist_3_link']")
        mount = ET.SubElement(wrist, "body", name=f"{side}_tool_mount", pos="0 0.1 0", quat="-1 1 0 0")
        mount.append(hand.find("worldbody/body"))
        hand_base = mount.find(f".//body[@name='{side}_gripper_base']")
        ET.SubElement(hand_base, "camera", name=f"{side}_wrist", pos="0.055 0 0.04", quat="0 1 0 0", fovy="75")
        wrapper = ET.SubElement(root.find("worldbody"), "body", name=f"{side}_pedestal", pos=f"{x} 0 0", euler=f"0 0 {yaw}")
        wrapper.append(arm.find("worldbody/body"))
        for source in (arm, hand):
            for section in ("default", "asset", "contact", "tendon", "equality", "actuator"):
                contents = source.find(section)
                if contents is not None:
                    root.find(section).extend(list(contents))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
