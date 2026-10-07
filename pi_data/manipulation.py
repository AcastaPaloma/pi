"""Pinned SO-101 manipulation datasets; retain every source's camera identities.

Public single-arm recordings are six-dimensional. They are not padded into
bimanual action labels, nor silently converted from unknown motor units.
"""
from .dataset import FoldingDataset

DATASETS = {
    "multitask": {
        "repo": "Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1",
        "revision": "bf6f568e9ec218fff838fb266e161724ec4f7e2c",
        "cameras": ("wrist_left", "desk_view"),
    },
    "cubes": {
        "repo": "BrutalCaesar/phi_so101_cubes_cylinder_v1",
        "revision": "014a212f6a30d68b39015546a9590c0927b8b9fa",
        "cameras": ("wrist", "front", "top"),
        # Video interval is 14 frames longer than the numeric episode at this revision.
        "excluded_episodes": (62,),
    },
}


class ManipulationDataset(FoldingDataset):
    """Same NumPy/sample/chunk API, with 6D state and dataset-native cameras.

    root is an offline LeRobot v3 root, containing meta/, data/, and videos/.
    Native joint units require a recorder/calibration audit before robot replay.
    """
    def __init__(self, root=None, *, dataset="multitask"):
        if dataset not in DATASETS:
            raise ValueError(f"Choose dataset from {tuple(DATASETS)}")
        source = DATASETS[dataset]
        super().__init__(root, **source, subset="", dimension=6,
                         units=["source_native_unverified"] * 6,
                         units_source="https://huggingface.co/datasets/" + source["repo"])
        expected = [f"{joint}.pos" for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                               "wrist_flex", "wrist_roll", "gripper")]
        if self.names != expected:
            raise ValueError("Unexpected SO-101 joint order; an explicit adapter is required")
