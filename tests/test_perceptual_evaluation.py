"""Tests for perceptual shadow evaluation."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo
from autofoh_models import RuntimeState
from autofoh_safety import AutoFOHSafetyController, ChannelEQMove
from perceptual import PerceptualConfig, PerceptualEvaluator
from perceptual.metrics import cosine_distance, embedding_mse
from perceptual.reward import RewardSignal


def _sine(freq_hz=440.0, amplitude=0.3, sample_rate=48000, duration_sec=0.25):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_evaluator_extracts_lightweight_embedding():
    evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "backend": "lightweight",
            "log_scores": False,
            "async_evaluation": False,
            "window_seconds": 1,
        }
    )
    audio = _sine()

    embedding = evaluator.extract_embedding(audio, 48000, channel_name="Vox", instrument_type="leadVocal")
    result = evaluator.score_change(audio, audio * 0.8, 48000, context={"channel": "Vox"})

    assert embedding.ndim == 1
    assert embedding.size > 16
    assert result.backend == "lightweight"
    assert result.verdict in {"improved", "worse", "neutral"}


def test_config_accepts_standalone_perceptual_yaml_shape():
    config = PerceptualConfig.from_mapping(
        {
            "perceptual": {
                "enabled": True,
                "backend": "lightweight",
                "window_seconds": 1,
                "log_scores": False,
            }
        }
    )

    assert config.enabled is True
    assert config.backend == "lightweight"
    assert config.window_seconds == 1


def test_mert_backend_absence_falls_back_to_lightweight():
    evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "backend": "mert",
            "model_name": "missing-local/mert-model",
            "local_files_only": True,
            "log_scores": False,
        }
    )

    assert evaluator.backend.name == "lightweight"
    assert evaluator.extract_embedding(_sine(), 48000).size > 0


def test_reference_embedding_is_cached_for_candidate_scoring():
    evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "backend": "lightweight",
            "log_scores": False,
            "reference_embedding_cache_size": 4,
        }
    )

    class CountingBackend:
        name = "counting"

        def __init__(self):
            self.calls = 0

        def extract(self, audio_buffer, sample_rate, channel_name=None, instrument_type=None):
            self.calls += 1
            audio = np.asarray(audio_buffer, dtype=np.float32)
            return np.array([audio.size, float(sample_rate), 1.0], dtype=np.float32)

    backend = CountingBackend()
    evaluator.backend = backend
    before = _sine(freq_hz=220.0)
    after = _sine(freq_hz=330.0)
    reference = _sine(freq_hz=440.0)
    context = {
        "channel": "mix_bus",
        "instrument": "full_mix",
        "reference_audio": reference,
        "reference_sample_rate": 48000,
        "reference_cache_key": "depeche-reference-window",
    }

    evaluator.score_change(before, after, 48000, context=context)
    evaluator.score_change(before, after * 0.9, 48000, context=context)

    assert backend.calls == 5


def test_candidate_batch_scores_only_top_prefiltered_candidates():
    evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "backend": "lightweight",
            "log_scores": False,
            "max_candidate_scores": 2,
            "window_seconds": 1,
        }
    )
    before = _sine(freq_hz=220.0)
    candidates = [
        _sine(freq_hz=230.0),
        _sine(freq_hz=330.0),
        _sine(freq_hz=440.0),
    ]

    results = evaluator.score_candidate_batch(
        before,
        candidates,
        48000,
        contexts=[{"channel": f"cand-{idx}"} for idx in range(3)],
        prefilter_scores=[0.1, 0.9, 0.4],
    )

    assert [item["index"] for item in results] == [1, 2]


def test_embedding_metrics_are_stable():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    assert embedding_mse(a, a) == 0.0
    assert cosine_distance(a, a) == 0.0
    assert embedding_mse(a, b) == 1.0
    assert cosine_distance(a, b) == 1.0


def test_jsonl_shadow_logging(tmp_path):
    log_path = tmp_path / "perceptual_decisions.jsonl"
    evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "backend": "lightweight",
            "log_scores": True,
            "async_evaluation": False,
            "log_path": str(log_path),
            "window_seconds": 1,
        }
    )

    result = evaluator.record_shadow_decision(
        _sine(amplitude=0.6),
        _sine(amplitude=0.3),
        48000,
        context={
            "channel": "Lead Vox",
            "instrument": "leadVocal",
            "action": {"type": "ChannelFaderMove", "target_db": -6.0},
        },
        osc_sent=True,
    )

    assert result is not None
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["channel"] == "Lead Vox"
    assert rows[0]["osc_sent"] is True
    assert "features_before" in rows[0]
    assert "reward_signal" in rows[0]


class _Mixer:
    is_connected = True
    state = {}

    def __init__(self):
        self.calls = []
        self.eq_gain = {}
        self.eq_freq = {}

    def get_eq_band_gain(self, channel, band):
        band_num = int(str(band).replace("g", ""))
        return self.eq_gain.get((channel, band_num), 0.0)

    def get_eq_band_frequency(self, channel, band):
        band_num = int(str(band).replace("f", ""))
        return self.eq_freq.get((channel, band_num), 350.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        self.eq_gain[(channel, band)] = gain
        self.eq_freq[(channel, band)] = freq
        return True


class _Audio:
    def __init__(self, before, after):
        self.buffers = [before, after]

    def get_buffer(self, channel, size):
        if len(self.buffers) > 1:
            return self.buffers.pop(0)[:size]
        return self.buffers[0][:size]


def test_shadow_mode_does_not_change_osc_behavior(tmp_path):
    mixer = _Mixer()
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.PRE_SHOW_CHECK
    engine.soundcheck_profile_use_phase_target_action_guards = False
    engine.perceptual_config = {
        "enabled": True,
        "mode": "shadow",
        "evaluate_channels": True,
        "window_seconds": 1,
    }
    engine.perceptual_evaluator = PerceptualEvaluator(
        {
            "enabled": True,
            "mode": "shadow",
            "backend": "lightweight",
            "async_evaluation": False,
            "log_path": str(tmp_path / "perceptual_decisions.jsonl"),
            "window_seconds": 1,
        }
    )
    engine.audio_capture = _Audio(_sine(amplitude=0.5), _sine(amplitude=0.25))
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Gtr 1",
            preset="electricGuitar",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
        )
    }

    decision = engine._execute_action(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=350.0,
            gain_db=-1.0,
            q=1.3,
            reason="Mud cleanup",
        )
    )
    outcomes = engine._evaluate_pending_actions(force=True)

    assert decision is not None and decision.sent is True
    assert len(outcomes) == 1
    assert mixer.calls == [("set_eq_band", 1, 2, 350.0, -1.0, 1.3)]


def test_reward_signal_combines_stably():
    reward = RewardSignal.combine(
        engineering_score=0.5,
        perceptual_score=-0.25,
        safety_score=1.0,
    )

    assert -1.0 <= reward.combined_score <= 1.0
    assert reward.to_dict()["combined_score"] == reward.combined_score
