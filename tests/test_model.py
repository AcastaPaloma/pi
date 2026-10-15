"""π0.7 model invariants on the tiny configuration (CPU, seconds)."""
import unittest

import numpy as np
import torch

from pi_model import Pi07, Pi07Config
from pi_model.fast import FastTokenizer, fast_to_vocab, vocab_to_fast
from pi_model.model import ACTION, FAST, OBS, STATE, SUBGOAL, TEXT, attention_mask
from pi_model.vision import MemVisionEncoder
from pi_model.weights import load_gemma3, map_name


def tiny_batch(cfg, B=2, V=3, K=6, G=3, Lt=10, Lf=8):
    S = cfg.vision.image_size
    T = cfg.vision.tokens_per_image
    torch.manual_seed(0)
    batch = {
        "images": torch.randn(B, V, K, 3, S, S),
        "frame_valid": torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)[:B],
        "seconds_ago": torch.tensor([[5., 4, 3, 2, 1, 0]] * B), "camera_valid": torch.ones(B, V, dtype=torch.bool),
        "subgoals": torch.randn(B, G, 3, S, S), "subgoal_valid": torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)[:B],
        "text_ids": torch.randint(3, 300, (B, Lt)), "text_valid": torch.ones(B, Lt, dtype=torch.bool),
        "state": torch.randn(B, K, cfg.state_dim), "state_valid": torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)[:B],
        "fast_ids": torch.randint(3000, cfg.text.vocab, (B, Lf)), "fast_valid": torch.ones(B, Lf, dtype=torch.bool),
        "actions": torch.randn(B, 50, cfg.action_dim),
        "action_valid": torch.cat([torch.ones(B, 12), torch.zeros(B, cfg.action_dim - 12)], 1).bool(),
        "rtc_delay": torch.tensor([0, 7])[:B],
    }
    N = V * T + G * T + Lt + K + Lf
    start = V * T + G * T + Lt + K
    lm_mask = torch.zeros(B, N, dtype=torch.bool)
    lm_targets = torch.zeros(B, N, dtype=torch.long)
    lm_mask[:, start - 1:N - 1] = True
    lm_targets[:, start - 1:N - 1] = batch["fast_ids"]
    batch["lm_target_mask"], batch["lm_targets"] = lm_mask, lm_targets
    return batch


def inference_batch(batch):
    return {k: v for k, v in batch.items() if k not in ("fast_ids", "fast_valid", "actions", "action_valid",
                                                        "rtc_delay", "lm_target_mask", "lm_targets")}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Pi07Config.tiny()
        torch.manual_seed(0)
        cls.model = Pi07(cls.cfg)

    def test_losses_and_knowledge_insulation(self):
        batch = tiny_batch(self.cfg)
        out = self.model(batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(float(out["ce_loss"]), 0)
        self.assertGreater(float(out["flow_loss"]), 0)
        self.model.zero_grad()
        self.model(batch)["flow_loss"].backward()
        vlm = [p for n, p in self.model.named_parameters() if n.startswith(("vision.", "backbone.", "state_proj."))]
        self.assertTrue(all(p.grad is None or p.grad.abs().sum() == 0 for p in vlm))
        expert = [p for n, p in self.model.named_parameters() if n.startswith("expert.")]
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert))
        self.model.zero_grad()
        self.model(batch)["ce_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in self.model.named_parameters() if n.startswith("backbone.layers.0")))
        self.assertTrue(all(p.grad is None or p.grad.abs().sum() == 0 for p in expert))
        self.model.zero_grad()

    def test_rtc_prefix_is_clean_and_unsupervised(self):
        batch = tiny_batch(self.cfg)
        batch["rtc_delay"] = torch.full((2,), 50)
        self.assertEqual(float(self.model(batch)["flow_loss"]), 0.0)
        inf = inference_batch(batch)
        prefix = torch.randn(2, 6, self.cfg.action_dim)
        chunk = self.model.sample_actions(inf, prefix_actions=prefix, delay=6, steps=2)
        self.assertEqual(tuple(chunk.shape), (2, 50, self.cfg.action_dim))
        torch.testing.assert_close(chunk[:, :6], prefix)
        guided = self.model.sample_actions(inf, steps=2, cfg_beta=1.3, uncond_batch=inf)
        self.assertEqual(tuple(guided.shape), (2, 50, self.cfg.action_dim))

    def test_attention_rules(self):
        seg = torch.tensor([[OBS, OBS, SUBGOAL, TEXT, TEXT, STATE, FAST, FAST]])
        pos = torch.arange(8)[None]
        valid = torch.ones(1, 8, dtype=torch.bool)
        m = attention_mask(seg, pos, seg, pos, valid)[0, 0]
        self.assertTrue(m[0, 1] and m[1, 0])            # observations bidirectional
        self.assertTrue(m[2, 0] and not m[0, 2])        # subgoal sees obs, not vice versa
        self.assertTrue(m[3, 2] and m[4, 3] and not m[3, 4])   # text causal, sees subgoal
        self.assertTrue(m[5, 4] and not m[4, 5])        # state after text
        self.assertTrue(m[7, 6] and not m[6, 7] and not m[5, 6])   # FAST causal, invisible to earlier tokens
        seg_a = torch.tensor([[ACTION, ACTION]])
        pos_a = torch.tensor([[8, 9]])
        ma = attention_mask(seg_a, pos_a, torch.cat([seg, seg_a], 1), torch.cat([pos, pos_a], 1),
                            torch.ones(1, 10, dtype=torch.bool))[0, 0]
        self.assertTrue(ma[0, 0] and ma[0, 5] and ma[0, 9] and ma[1, 8])   # actions see prefix and each other
        self.assertFalse(ma[0, 6] or ma[0, 7])                              # but never the FAST tokens
        windowed = attention_mask(seg, pos, seg, pos, valid, window=2)[0, 0]
        self.assertFalse(windowed[4, 0])
        self.assertTrue(windowed[4, 3])
        dropped = valid.clone()
        dropped[0, 1] = False
        self.assertFalse(attention_mask(seg, pos, seg, pos, dropped)[0, 0][0, 1])

    def test_mem_single_frame_equivalence(self):
        enc = MemVisionEncoder(self.cfg.vision, self.cfg.text.hidden).eval()
        S = self.cfg.vision.image_size
        frames = torch.randn(2, 6, 3, S, S)
        single = enc(frames[:, -1:])
        masked = enc(frames, torch.tensor([[5., 4, 3, 2, 1, 0]] * 2), torch.tensor([[0, 0, 0, 0, 0, 1]] * 2, dtype=torch.bool))
        torch.testing.assert_close(single, masked, atol=1e-5, rtol=1e-5)
        history = enc(frames, torch.tensor([[5., 4, 3, 2, 1, 0]] * 2), torch.ones(2, 6, dtype=torch.bool))
        self.assertFalse(torch.allclose(single, history, atol=1e-4))
        self.assertEqual(float(enc.time_code(torch.zeros(1, 1)).abs().max()), 0.0)
        self.assertEqual(tuple(single.shape), (2, self.cfg.vision.tokens_per_image, self.cfg.text.hidden))

    def test_generate_text(self):
        batch = inference_batch(tiny_batch(self.cfg))
        batch.pop("state"), batch.pop("state_valid")
        out = self.model.generate_text(batch, max_new_tokens=3)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(1 <= len(o) <= 3 for o in out))

    def test_full_configuration_parameter_budget(self):
        with torch.device("meta"):
            report = Pi07(Pi07Config.full()).parameter_report()
        self.assertAlmostEqual(report["vision"] / 1e9, 0.42, delta=0.03)
        self.assertAlmostEqual(report["backbone"] / 1e9, 3.88, delta=0.1)
        self.assertAlmostEqual(report["expert"] / 1e9, 0.86, delta=0.03)
        cfg = Pi07Config.full()
        self.assertEqual(cfg.vision.tokens_per_image, 256)
        self.assertEqual(cfg.vision.temporal_layers, (3, 7, 11, 15, 19, 23))
        self.assertEqual(cfg.text.vocab, 262208)
        self.assertEqual([cfg.text.is_sliding(i) for i in range(6)], [True] * 5 + [False])


class FastTests(unittest.TestCase):
    def test_round_trip_compression_and_vocab_mapping(self):
        rng = np.random.default_rng(0)
        H, A = 50, 12
        tt = np.linspace(0, 1, H)[:, None]
        chunks = np.stack([30 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0, A) * tt + rng.uniform(0, 6, A))
                           + rng.uniform(-40, 40, A) for _ in range(200)])
        tok = FastTokenizer(H, A, vocab_size=1024).fit(chunks)
        ids = tok.encode(chunks[0])
        self.assertLess(len(ids), H * A / 3)
        self.assertLess(np.abs(tok.decode(ids) - chunks[0]).max(), 8)
        vocab = fast_to_vocab(ids, 262208, 128)
        self.assertEqual(vocab_to_fast(vocab, 262208, 128, 1024), ids)
        self.assertTrue(all(262208 - 128 - 1024 <= v < 262208 - 128 for v in vocab))


class WeightTests(unittest.TestCase):
    def test_gemma_name_mapping(self):
        self.assertEqual(map_name("language_model.model.layers.3.mlp.gate_proj.weight"), ("backbone.layers.3.mlp.gate_proj.weight", None))
        self.assertEqual(map_name("model.language_model.layers.3.self_attn.q_norm.weight"), ("backbone.layers.3.self_attn.q_norm.weight", None))
        self.assertEqual(map_name("model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias"), ("vision.layers.0.mlp.fc1.bias", None))
        self.assertEqual(map_name("multi_modal_projector.mm_input_projection_weight"), ("vision.projector.mm_input_projection.weight", "transpose"))
        self.assertIsNone(map_name("lm_head.weight"))

    def test_synthetic_gemma_checkpoint_loads_every_vlm_tensor(self):
        model = Pi07(Pi07Config.tiny())
        state = model.state_dict()
        tensors = []
        for ours, t in state.items():
            if ours == "vision.projector.mm_input_projection.weight":
                tensors.append(("multi_modal_projector.mm_input_projection_weight", torch.randn_like(t).t()))
            elif ours.startswith("vision.projector.mm_soft_emb_norm."):
                tensors.append(("multi_modal_projector.mm_soft_emb_norm.weight", torch.randn_like(t)))
            elif ours.startswith("vision.layers."):
                tensors.append(("model.vision_tower.vision_model.encoder.layers." + ours[len("vision.layers."):], torch.randn_like(t)))
            elif ours.startswith("vision.post_layernorm."):
                tensors.append(("model.vision_tower.vision_model.post_layernorm." + ours.split(".")[-1], torch.randn_like(t)))
            elif ours == "vision.position_embedding.weight":
                tensors.append(("model.vision_tower.vision_model.embeddings.position_embedding.weight", torch.randn(4096, t.shape[1])))
            elif ours.startswith("vision."):
                tensors.append(("model.vision_tower.vision_model.embeddings." + ours[len("vision."):], torch.randn_like(t)))
            elif ours.startswith("backbone.layers."):
                tensors.append(("model.language_model.layers." + ours[len("backbone.layers."):], torch.randn_like(t)))
            elif ours == "backbone.embed_tokens.weight":
                tensors.append(("model.language_model.embed_tokens.weight", torch.randn_like(t)))
            elif ours == "backbone.norm.weight":
                tensors.append(("model.language_model.norm.weight", torch.randn_like(t)))
        report = load_gemma3(model, tensors)
        self.assertEqual([k for k in report["untouched"] if k.startswith(("vision.", "backbone."))], [])
        self.assertTrue(all(k.startswith(("expert.", "state_proj.")) for k in report["untouched"]))
        with self.assertRaises(ValueError):
            load_gemma3(model, [("model.language_model.norm.weight", torch.zeros(3))])


if __name__ == "__main__":
    unittest.main()
