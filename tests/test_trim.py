import numpy as np
from pathlib import Path
from pi_model.data import Episode, still_tail_steps, trim_still_tail


def _episode(success, parked_steps, moving_steps=100):
    ee = np.zeros((moving_steps + parked_steps, 14), np.float32)
    ee[:moving_steps, 0] = 0.0                                   # left hand starts 20 cm away
    ee[moving_steps:, 0] = 0.2                                   # then both hands park
    meta = {"steps": len(ee), "control_hz": 50, "frame_stride": 5, "success": success}
    return Episode(Path("."), meta, {"ee_pose": ee})


def test_still_tail_counts_parked_steps():
    assert still_tail_steps(_episode(False, 500).steps) == 500


def test_trim_keeps_one_second_of_a_failed_episode():
    e = _episode(False, 500)
    assert trim_still_tail(e, 1.0)
    assert e.sample_steps == 100 + 50
    assert e.samples == 30


def test_successful_and_short_tails_are_untouched():
    ok, short = _episode(True, 500), _episode(False, 20)
    assert not trim_still_tail(ok, 1.0) and ok.samples == 120
    assert not trim_still_tail(short, 1.0) and short.sample_steps is None
