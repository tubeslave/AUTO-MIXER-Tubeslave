"""Regression reproductions from the EQ/masking/dynamics safety audit."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def nodes_module():
    # Test the pure NumPy graph without requiring its optional ML package imports.
    path = Path(__file__).parents[1] / "backend/ml/processing_graph.py"
    spec = importlib.util.spec_from_file_location("shadow_dsp_nodes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hpf_stereo_does_not_leak_left_state_into_silent_right():
    node = nodes_module().HPFNode(100)
    x = np.zeros((48000, 2))
    x[-1, 0] = .5
    output = node.process(x)
    assert np.max(np.abs(output[:, 1])) == 0
    tail = node.process(np.zeros((1024, 2)))
    assert np.max(np.abs(tail[:, 0])) > 0
    assert np.max(np.abs(tail[:, 1])) == 0


def test_hpf_stereo_chunk_invariance():
    HPF = nodes_module().HPFNode
    x = np.random.default_rng(2).normal(0, .01, (48000, 2))
    full = HPF(100).process(x)
    streamed_node = HPF(100)
    chunks = [streamed_node.process(block) for block in np.array_split(x, 12)]
    np.testing.assert_allclose(np.concatenate(chunks), full, atol=1e-12)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_auto_eq_limiter_nonfinite_does_not_poison_future_updates(bad):
    from auto_eq_processing import EQLimiter
    limiter = EQLimiter()
    original = limiter.process(-1)
    assert limiter.process(bad) == original
    assert np.isfinite(limiter.process(-2))


def test_live_planner_does_not_mutate_readback_or_audio():
    from copy import deepcopy
    from live_shared_mix import LiveSharedMixChannel, build_live_shared_mix_plan
    t = np.arange(48000) / 48000
    channels = [LiveSharedMixChannel(
        channel_id=1, name="bass", role="bass", stems=("BASS",), priority=.8,
        audio=.3 * np.sin(2 * np.pi * 80 * t), sample_rate=48000,
        fader_db=-6, auto_corrections_enabled=True, current_eq_gain={1: 0},
    )]
    before = deepcopy(channels)
    build_live_shared_mix_plan(channels, 48000)
    assert channels[0].fader_db == before[0].fader_db
    assert channels[0].current_eq_gain == before[0].current_eq_gain
    np.testing.assert_array_equal(channels[0].audio, before[0].audio)


def test_observation_intercepts_raw_write_path():
    from observation_mixer import ObservationMixerClient
    class RealMixer:
        def _send_raw(self, payload):
            raise AssertionError("Raw writes forbidden")
        def send_packet(self, payload):
            raise AssertionError("Packet writes forbidden")
    proxy = ObservationMixerClient(RealMixer())
    assert proxy._send_raw(b"midi") is True
    assert proxy.send_packet(b"osc") is True
    assert len(proxy.get_operations()) == 2
