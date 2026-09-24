import numpy as np
import pytest

from audio_workbench.belye_stai_session import (
    BelyeStaiSessionRenderer, RECIPE_ID, SOURCE_IDS,
)
from audio_workbench.routed_contribution import SessionRender


def _sources(sr=44100, seconds=.8):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(411)
    result = {}
    for i, name in enumerate(SOURCE_IDS):
        # Broadband deterministic fixture keeps all detector/percentile paths active.
        x = (.012 * np.sin(2 * np.pi * (95 + 31 * i) * t)
             + .004 * np.sin(2 * np.pi * (650 + 43 * i) * t)
             + .0015 * rng.standard_normal(n))
        result[name] = x.astype("float32")
    return result


def test_belye_session_renderer_is_deterministic_and_returns_required_anchors():
    sources = _sources()
    originals = {k: v.copy() for k, v in sources.items()}
    renderer = BelyeStaiSessionRenderer(sources)
    first = renderer({})
    second = renderer({})
    assert isinstance(first, SessionRender)
    assert first.mix.shape == (len(next(iter(sources.values()))), 2)
    np.testing.assert_array_equal(first.mix, second.mix)
    np.testing.assert_array_equal(first.vocal_bus, second.vocal_bus)
    np.testing.assert_array_equal(first.drums_bus, second.drums_bus)
    np.testing.assert_array_equal(first.early_room, second.early_room)
    assert first.metadata["recipe_id"] == RECIPE_ID
    assert first.metadata["accent_refinement"]["center_seconds"] == 159.056
    assert first.metadata["source_dependent_routing"]
    for key in SOURCE_IDS:
        np.testing.assert_array_equal(sources[key], originals[key])


def test_source_override_runs_through_full_graph_and_does_not_mutate_inputs():
    sources = _sources()
    renderer = BelyeStaiSessionRenderer(sources)
    baseline = renderer({})
    candidate = (sources["VALERA_VOX"] * np.float32(.72)).astype("float32")
    candidate_before = candidate.copy()
    changed = renderer({"VALERA_VOX": candidate})
    np.testing.assert_array_equal(candidate, candidate_before)
    assert changed.mix.shape == baseline.mix.shape
    assert float(np.max(np.abs(changed.mix - baseline.mix))) > 1e-5
    # Lead vocal changes its own bus and also source-dependent masking/routing.
    assert float(np.max(np.abs(changed.vocal_bus - baseline.vocal_bus))) > 1e-5


def test_session_renderer_fails_closed_on_incomplete_or_mismatched_sources():
    sources = _sources()
    incomplete = dict(sources)
    incomplete.pop("BASS")
    with pytest.raises(ValueError):
        BelyeStaiSessionRenderer(incomplete)
    renderer = BelyeStaiSessionRenderer(sources)
    with pytest.raises(ValueError):
        renderer({"UNKNOWN": sources["BASS"]})
    with pytest.raises(ValueError):
        renderer({"BASS": sources["BASS"][:-1]})


def test_session_renderer_rejects_non_mono_source_layout():
    sources = _sources()
    sources["BASS"] = np.column_stack([sources["BASS"], sources["BASS"]])
    with pytest.raises(ValueError):
        BelyeStaiSessionRenderer(sources)
