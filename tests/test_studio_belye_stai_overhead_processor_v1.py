from __future__ import annotations

import numpy as np
import pytest

from audio_workbench.belye_stai_overhead import (
    BelyeStaiOverheadRecipe,
    render_overhead,
)
from audio_workbench.belye_stai_vocal import _bell, _filter, _gain, _level
from audio_workbench.mixing.compression import CompressorConfig


def _raw_pair(sr=44100, seconds=.6):
    n = int(sr * seconds)
    rng = np.random.default_rng(441)
    l = (rng.normal(0, .02, n) + .006*np.sin(2*np.pi*390*np.arange(n)/sr)).astype(np.float32)
    r = (rng.normal(0, .018, n) + .005*np.sin(2*np.pi*510*np.arange(n)/sr)).astype(np.float32)
    return l, r


def test_no_change_matches_independent_frozen_recipe_sample_exactly():
    sr = 44100
    l, r = _raw_pair(sr)
    recipe = BelyeStaiOverheadRecipe(sample_rate=sr, frames=len(l))
    got = render_overhead(l, r, recipe=recipe)

    ref = np.column_stack([l, _gain(r, 3.0)]).astype(np.float32)
    ref = _filter(ref, sr, 190, 13500)
    ref = _bell(ref, sr, 6000, -1.2, .7)
    ref, _ = _level(ref, sr, -34.0, 65)

    assert np.max(np.abs(got.processed - ref)) == 0.0
    assert got.report["compressor"]["mode"] == "no_compressor_in_delivered_recipe"
    assert got.report["precompression"]["right_calibration_db"] == 3.0


def test_candidate_uses_source_bound_frozen_level_and_never_relevels():
    sr = 44100
    l, r = _raw_pair(sr)
    recipe = BelyeStaiOverheadRecipe(sample_rate=sr, frames=len(l))
    base = render_overhead(l, r, recipe=recipe)
    cfg = CompressorConfig(threshold_dbfs=-28, ratio=2, attack_ms=14, release_ms=180,
                           knee_db=5, max_gr_db=2.2)
    cand = render_overhead(l, r, compression_config=cfg,
                           frozen_controls=base.controls, recipe=recipe)
    assert cand.processed.shape == (len(l), 2)
    assert cand.report["level"]["level_reestimated"] is False
    assert cand.report["level"]["gain_db"] == base.controls.final_gain_db
    assert cand.report["compressor"]["mode"] == "overhead_director_linked_replacement"
    assert cand.report["requires_full_session_rerender"] is True


def test_candidate_rejects_missing_or_foreign_controls():
    sr = 44100
    l, r = _raw_pair(sr)
    recipe = BelyeStaiOverheadRecipe(sample_rate=sr, frames=len(l))
    cfg = CompressorConfig(threshold_dbfs=-28, ratio=2, attack_ms=14, release_ms=180,
                           knee_db=5, max_gr_db=2.2)
    with pytest.raises(ValueError, match="source-bound"):
        render_overhead(l, r, compression_config=cfg, recipe=recipe)
    base = render_overhead(l, r, recipe=recipe)
    other = l.copy(); other[10] += .001
    with pytest.raises(ValueError, match="different source"):
        render_overhead(other, r, compression_config=cfg,
                        frozen_controls=base.controls, recipe=recipe)
