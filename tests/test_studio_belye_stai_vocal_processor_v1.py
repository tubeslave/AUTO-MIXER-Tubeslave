from dataclasses import replace

import numpy as np
import pytest

from audio_workbench.belye_stai_vocal import (
    BelyeStaiLeadVocalRecipe,
    prepare_lead_vocal,
    render_lead_vocal,
)
from audio_workbench.mixing.compression import CompressorConfig


def _source(seconds: int = 3) -> np.ndarray:
    sr = 44_100
    t = np.arange(sr * seconds) / sr
    x = .09 * np.sin(2 * np.pi * 220 * t) + .035 * np.sin(2 * np.pi * 1800 * t)
    x *= np.where((t % 1.0) > .15, 1.0, .05)
    return x.astype("float32")


def _recipe(x: np.ndarray) -> BelyeStaiLeadVocalRecipe:
    return replace(BelyeStaiLeadVocalRecipe(), frames=len(x))


def test_no_override_is_deterministic_non_mutating_and_not_auto_promoted():
    x = _source()
    original = x.copy()
    first = render_lead_vocal(x, recipe=_recipe(x))
    second = render_lead_vocal(x, recipe=_recipe(x))

    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(first.processed, second.processed)
    assert first.report["first_compressor_mode"] == "frozen_delivery_recipe"
    assert first.report["requires_human_review"] is False
    assert first.report["baseline_eligible"] is False


def test_prepare_output_is_the_exact_first_compressor_insert_boundary():
    x = _source()
    prepared, _ = prepare_lead_vocal(x, recipe=_recipe(x))
    rendered = render_lead_vocal(x, recipe=_recipe(x))
    np.testing.assert_array_equal(prepared, rendered.pre_compression)


def test_override_replaces_first_compressor_in_place_and_requires_human_review():
    x = _source()
    recipe = _recipe(x)
    baseline = render_lead_vocal(x, recipe=recipe)
    config = CompressorConfig(
        threshold_dbfs=-32,
        ratio=2.5,
        attack_ms=40,
        release_ms=180,
        knee_db=6,
        max_gr_db=4,
    )
    candidate = render_lead_vocal(x, recipe=recipe, first_stage_config=config)

    assert candidate.report["first_compressor_mode"] == "replacement"
    assert candidate.report["first_compressor"]["mode"] == "compression_director_replacement"
    assert candidate.report["requires_human_review"] is True
    assert candidate.report["requires_human_listening"] is True
    assert candidate.report["baseline_eligible"] is False
    assert candidate.processed.shape == x.shape
    assert not np.array_equal(candidate.processed, baseline.processed)


def test_wrong_shape_or_length_fail_closed():
    recipe = replace(BelyeStaiLeadVocalRecipe(), frames=1000)
    with pytest.raises(ValueError):
        render_lead_vocal(np.zeros((1000, 2), "float32"), recipe=recipe)
    with pytest.raises(ValueError):
        render_lead_vocal(np.zeros(999, "float32"), recipe=recipe)
