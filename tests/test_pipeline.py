"""Record -> prepare -> dataset -> train step -> policy in the simulator (tiny model, CPU)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from pi_model import Pi07, Pi07Config
from pi_model.data import Pi07Dataset, collate, prepare, speed_prompt_value, to_device
from pi_model.eval_sim import run_episode
from pi_model.policy import HistoryBuffer, Pi07Policy
from pi_model.prompt import PromptContext
from pi_model.tokenizer import ByteTokenizer
from pi_model.weights import save_checkpoint
from pi_sim import RobotEnv
from pi_sim.record import record_episode


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "episodes"
        with RobotEnv(task="pick_place", control_hz=50, episode_seconds=12) as env:
            record_episode(env, root / "episode_00000", seed=0, style="clean", image_size=64)
            record_episode(env, root / "episode_00001", seed=1, style="sloppy", image_size=64)
        cls.root = root
        cls.cfg = Pi07Config.tiny()
        cls.stats = prepare(root, root / "pi07_meta", cls.cfg, chunk_stride=10)
        cls.tokenizer = ByteTokenizer(cls.cfg.text.vocab)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prepare_outputs(self):
        self.assertEqual(self.stats["control_hz"], 50)
        self.assertEqual(self.stats["cameras"], ["overhead", "left_wrist", "right_wrist"])
        self.assertTrue((self.root / "pi07_meta" / "fast" / "fast_bpe.json").exists())
        self.assertLess(self.stats["fast_tokens_per_chunk_mean"], 300)
        self.assertGreater(speed_prompt_value(self.stats, "unknown task"), 0)

    def test_dataset_sample_and_batch_layout(self):
        ds = Pi07Dataset(self.root, self.root / "pi07_meta", self.cfg, self.tokenizer, train=True)
        self.assertEqual(len(ds), int(self.stats["samples"]))
        sample = ds[len(ds) - 5]
        S = self.cfg.vision.image_size
        self.assertEqual(sample["images"].shape, (3, 6, S, S, 3))
        self.assertEqual(sample["actions"].shape, (50, self.cfg.action_dim))
        self.assertFalse(sample["action_step_valid"].all())        # chunk runs past the episode end
        self.assertTrue(sample["frame_valid"][-1])
        self.assertTrue(sample["fast_ids"][-1] == self.tokenizer.eos_id)
        batch = collate([ds[10], ds[60], sample], self.cfg)
        T = self.cfg.vision.tokens_per_image
        N = 3 * T + 3 * T + batch["text_ids"].shape[1] + 6 + batch["fast_ids"].shape[1]
        self.assertEqual(batch["lm_target_mask"].shape, (3, N))
        last_state = 6 * T + batch["text_ids"].shape[1] + 5
        self.assertTrue(batch["lm_target_mask"][0, last_state])
        self.assertEqual(int(batch["lm_targets"][0, last_state]), int(batch["fast_ids"][0, 0]))
        self.assertEqual(int(batch["lm_target_mask"][0].sum()), int(batch["fast_valid"][0].sum()))
        moved = to_device(batch, torch.device("cpu"))
        self.assertEqual(moved["images"].shape, (3, 3, 6, 3, S, S))
        self.assertLessEqual(float(moved["images"].max()), 1.0)
        self.assertGreaterEqual(float(moved["images"].min()), -1.0)

    def test_subtask_objective_batches(self):
        ds = Pi07Dataset(self.root, self.root / "pi07_meta", self.cfg, self.tokenizer, train=False, objective="subtask")
        batch = collate([ds[100], ds[200]], self.cfg)
        self.assertNotIn("fast_ids", batch)
        self.assertEqual(batch["state"].shape[1], 0)
        self.assertGreater(int(batch["lm_target_mask"].sum()), 4)
        model = Pi07(self.cfg)
        out = model(to_device(batch, torch.device("cpu")))
        self.assertGreater(float(out["ce_loss"]), 0)
        self.assertEqual(float(out["flow_loss"]), 0)

    def test_train_steps_reduce_loss_and_policy_runs_in_sim(self):
        torch.manual_seed(0)
        ds = Pi07Dataset(self.root, self.root / "pi07_meta", self.cfg, self.tokenizer, train=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=lambda s: collate(s, self.cfg))
        batch = to_device(next(iter(loader)), torch.device("cpu"))
        model = Pi07(self.cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for _ in range(12):
            out = model(batch)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            losses.append(float(out["loss"]))
        self.assertLess(losses[-1], losses[0])
        ckpt = Path(self.tmp.name) / "ckpt"
        save_checkpoint(ckpt, model, config=self.cfg, extra={"tokenizer": None, "step": 12})
        (ckpt / "stats.json").write_text(json.dumps(self.stats))
        policy = Pi07Policy(ckpt, device="cpu", denoise_steps=2)
        with RobotEnv(task="pick_place", control_mode="joint", control_hz=50, episode_seconds=2) as env:
            result = run_episode(env, policy, seed=5, execute=25, delay=4, cfg_beta=1.5, subtask=None,
                                 image_size=self.cfg.vision.image_size, placement="wide", visual_randomization=False)
        self.assertEqual(result["steps"], 100)
        self.assertGreaterEqual(result["inference_calls"], 4)
        history = HistoryBuffer(policy.cameras, hz=50)
        S = self.cfg.vision.image_size
        for step in (0, 50, 100):
            history.append(step, {c: np.zeros((S, S, 3), np.uint8) for c in policy.cameras}, np.zeros(12))
        frames, valid, states, seconds = history.sample(6, 50)
        self.assertEqual(valid.tolist(), [False, False, False, True, True, True])
        self.assertEqual(seconds.tolist(), [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        chunk = policy.act(history, PromptContext(task="Place the red cube on the green disk.", speed_steps=236, quality=5, mistake=False))
        self.assertEqual(chunk.shape, (50, 12))
        self.assertTrue(np.isfinite(chunk).all())


if __name__ == "__main__":
    unittest.main()
