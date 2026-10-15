"""Load Gemma 3 4B into the paper-size model on CPU and prove it runs: python scripts/box/check_gemma.py weights/gemma-3-4b-pt"""
import sys
import time

import torch

from pi_model import Pi07, Pi07Config
from pi_model.fast import fast_to_vocab
from pi_model.prompt import PromptContext, no_dropout, prompt_ids
from pi_model.tokenizer import load_tokenizer
from pi_model.weights import gemma_loaded_fraction, load_gemma3


def main(path):
    t0 = time.time()
    cfg = Pi07Config.full()
    model = Pi07(cfg)
    report = load_gemma3(model, path)
    untouched = sorted(set(k.split(".")[0] for k in report["untouched"]))
    print(f"loaded={len(report['loaded'])} skipped={len(report['skipped'])} "
          f"vlm_fraction={gemma_loaded_fraction(report, model):.4f} untouched_modules={untouched} ({time.time() - t0:.0f}s)")
    print("skipped:", report["skipped"][:5])
    tok = load_tokenizer(f"{path}/tokenizer.model", cfg.text.vocab)
    context = PromptContext(task="Place the red cube on the green disk.", subtask="Lift the object off the table.",
                            speed_steps=236, quality=5, mistake=False)
    ids = prompt_ids(tok, context, no_dropout())
    print("prompt tokens:", len(ids), "->", repr(tok.decode(ids[1:])))
    fast_ids = fast_to_vocab([0, 1, 500, 1023], cfg.text.vocab, cfg.fast_skip)
    print("FAST vocab ids", fast_ids, "pieces:", [tok.sp.id_to_piece(i) for i in fast_ids])
    model.eval()
    S = cfg.vision.image_size
    batch = {"images": torch.randn(1, 3, 1, 3, S, S), "frame_valid": torch.ones(1, 1, dtype=torch.bool),
             "seconds_ago": torch.zeros(1, 1), "camera_valid": torch.ones(1, 3, dtype=torch.bool),
             "subgoals": torch.zeros(1, 0, 3, S, S), "subgoal_valid": torch.zeros(1, 0, dtype=torch.bool),
             "text_ids": torch.tensor([ids]), "text_valid": torch.ones(1, len(ids), dtype=torch.bool),
             "state": torch.zeros(1, 1, cfg.state_dim), "state_valid": torch.ones(1, 1, dtype=torch.bool)}
    with torch.no_grad():
        t1 = time.time()
        prefix = model.encode_prefix(batch)
        hidden, kvs = model.run_prefix(prefix)
        logits = model.backbone.logits(hidden[0, -1])
        print(f"prefix {tuple(hidden.shape)} finite={bool(torch.isfinite(hidden).all())} "
              f"top next-token piece={tok.sp.id_to_piece(int(logits.argmax()))!r} ({time.time() - t1:.0f}s on CPU)")
        x = model.sample_actions(batch, steps=1)
        print("sample_actions ok", tuple(x.shape), "finite", bool(torch.isfinite(x).all()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights/gemma-3-4b-pt")
