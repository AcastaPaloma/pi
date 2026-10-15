"""Camera optics and poses shared by simulation and physical capture.

Intrinsics use OpenCV pixels (origin at top left, x right, y down).
Mount poses use MuJoCo axes (x right, y up, looking along -z).
"""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

CAMERAS = ("overhead", "left_wrist", "right_wrist")
FIXED_CAMERAS = ("overhead",)
DEFAULT_RIG = Path(__file__).parent / "assets" / "so101_camera_rig.json"


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    fps: float
    rotation: int
    intrinsics: tuple[float, float, float, float]
    distortion: tuple[float, ...]
    parent_body: str
    pos: tuple[float, ...]
    quat: tuple[float, ...]
    hardware_model: str | None
    calibration_status: str
    optics_note: str
    reference_url: str | None

    @property
    def K(self):
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])

    def metadata(self):
        return {"hardware_model": self.hardware_model,
                "calibration_status": self.calibration_status,
                "optics_note": self.optics_note, "reference_url": self.reference_url,
                "resolution_wh": [self.width, self.height], "requested_fps": self.fps,
                "rotation_ccw_degrees": self.rotation,
                "intrinsic_matrix_before_rotation": self.K.tolist(),
                "distortion_opencv_k1_k2_p1_p2_k3": list(self.distortion),
                "parent_body": self.parent_body,
                "mount_position_m": list(self.pos), "mount_quaternion_wxyz": list(self.quat)}

    def render_plan(self):
        """Overscan distorted lenses so the output has no artificial black rim.

        OpenCV is needed only for a nonzero Brown-Conrady distortion profile.
        The map samples an ideal pinhole image to produce the raw lens image.
        """
        if not any(self.distortion):
            return self.width, self.height, self.K, None
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Nonzero camera distortion needs opencv-python; install pi-sim[capture]") from exc
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        pixels = np.stack([u, v], axis=-1).astype(np.float64)
        ideal = cv2.undistortPointsIter(
            pixels.reshape(-1, 1, 2), self.K, np.asarray(self.distortion), None, self.K,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
        ).reshape(self.height, self.width, 2)
        if not np.isfinite(ideal).all():
            raise ValueError("Camera distortion produces invalid rays")
        lower = np.floor(ideal.min(axis=(0, 1))).astype(int)
        upper = np.ceil(ideal.max(axis=(0, 1))).astype(int)
        left, top = np.maximum(-lower, 0)
        right, bottom = np.maximum(upper - [self.width - 1, self.height - 1], 0)
        width, height = int(self.width + left + right), int(self.height + top + bottom)
        if max(width, height) > 4096:
            raise ValueError("Camera distortion needs an overscan larger than 4096 pixels; check calibration")
        K = self.K
        K[:2, 2] += [left, top]
        mapping = (ideal + [left, top]).astype(np.float32)
        return width, height, K, mapping


def _vector(value, count, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {count} finite numbers")
    return result


def load_camera_rig(source=None):
    """Load a JSON path or mapping; require the same three named views."""
    if source is None:
        source = DEFAULT_RIG
    raw = json.loads(Path(source).expanduser().read_text()) if isinstance(source, (str, Path)) else source
    if raw.get("schema_version") != 1 or set(raw.get("cameras", {})) != set(CAMERAS):
        raise ValueError(f"Camera rig schema_version=1 requires exactly {CAMERAS}")
    cameras = {}
    for name in CAMERAS:
        item = raw["cameras"][name]
        width, height = item["width"], item["height"]
        if any(type(n) is not int or not 16 <= n <= 4096 for n in (width, height)):
            raise ValueError(f"{name}: width/height must be integers in 16..4096")
        fps, rotation = item.get("fps", 30), item.get("rotation", 0)
        if not np.isfinite(fps) or not 0 < fps <= 120 or rotation not in (0, 90, 180, 270):
            raise ValueError(f"{name}: invalid FPS or rotation")
        optics = item["intrinsics"]
        intrinsics = _vector([optics[k] for k in ("fx", "fy", "cx", "cy")], 4, f"{name} intrinsics")
        if np.any(intrinsics[:2] <= 0) or not 0 <= intrinsics[2] < width or not 0 <= intrinsics[3] < height:
            raise ValueError(f"{name}: positive focal lengths and a principal point inside the image are required")
        distortion = _vector(item.get("distortion", [0] * 5), 5, f"{name} distortion")
        pose = item["pose"]
        parent = pose["parent_body"]
        expected = f"{name.removesuffix('_wrist')}_camera_mount" if name.endswith("_wrist") else "world"
        if parent != expected:
            raise ValueError(f"{name}: parent_body must be {expected}")
        pos = _vector(pose["pos"], 3, f"{name} pos")
        if ("quat_wxyz" in pose) == ("xyaxes" in pose):
            raise ValueError(f"{name}: supply exactly one of quat_wxyz or xyaxes")
        if "quat_wxyz" in pose:
            quat = _vector(pose["quat_wxyz"], 4, f"{name} quat_wxyz")
            if np.linalg.norm(quat) < 1e-8:
                raise ValueError(f"{name}: quaternion must be nonzero")
            quat /= np.linalg.norm(quat)
        else:
            x, y = _vector(pose["xyaxes"], 6, f"{name} xyaxes").reshape(2, 3)
            if not np.isclose(np.linalg.norm(x), 1) or not np.isclose(np.linalg.norm(y), 1) or abs(x @ y) > 1e-5:
                raise ValueError(f"{name}: xyaxes must be orthonormal")
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.column_stack([x, y, np.cross(x, y)]).ravel())
        model = item.get("hardware_model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError(f"{name}: hardware_model must be text or null")
        cameras[name] = CameraSpec(width, height, float(fps), rotation, tuple(intrinsics),
                                   tuple(distortion), parent, tuple(pos), tuple(quat), model,
                                   item.get("calibration_status", "nominal"),
                                   item.get("optics_note", ""), item.get("reference_url"))
    return cameras


def camera_metadata(spec, model, data, name):
    """Measured simulator pose in OpenCV axes, before optional image rotation."""
    cid = model.camera(name).id
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
    transform[:3, 3] = data.cam_xpos[cid]
    return {**spec.metadata(), "world_from_camera_opencv_before_rotation": transform.tolist(),
            "pose_source": "simulation_kinematics", "fps_basis": "profile only; reads render the current state"}
