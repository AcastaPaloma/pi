"""Inspect exactly what your encoder receives. No torch or model required.

    conda run -n mini-vla python examples/encoder_inputs.py
"""
from pi_data import FoldingDataset

ds = FoldingDataset()
episode = ds.episode_ids("train")[0]
batch_item = ds.arrays(episode, 100, history=1, goal_seconds=2)
for key, value in batch_item.items():
    if key != "metadata_json":
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")

# Encoder inputs: images, proprioception, task, control_mode, optional
# subgoal_images and history. The camera order is stable and explicit.
# Supervision: actions [50,12], with action_valid [50] masking padding.
# Do not concatenate action labels or goal timestamps into the encoder inputs.
# Fit normalization on ds.train_statistics(), not all dataset episodes.
