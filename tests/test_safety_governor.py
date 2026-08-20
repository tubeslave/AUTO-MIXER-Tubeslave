import numpy as np
import pytest
import soundfile as sf

from ai_mixing_pipeline.models import CandidateAction, MixCandidate
from ai_mixing_pipeline.safety_governor import SafetyGovernor
from ai_mixing_pipeline.decision_layer.action_schema import CandidateActionSet, EQAction, GainAction
from ai_mixing_pipeline.decision_layer.safety_governor import DecisionSafetyGovernor


def _write(path, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.2), dtype=np.float32) / sr
    audio = (amp * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(path, audio, sr)


def test_safety_governor_rejects_dangerous_gain_action(tmp_path):
    wav = tmp_path / "safe.wav"
    _write(wav, amp=0.1)
    candidate = MixCandidate(
        candidate_id="001",
        label="danger",
        actions=[
            CandidateAction(
                action_type="gain_change",
                target="vocal",
                parameters={"gain_db": 3.0},
            )
        ],
    )
    governor = SafetyGovernor(
        {"safety": {"max_gain_change_db_per_step": 1.0, "max_true_peak_dbfs": -1.0}}
    )

    result = governor.evaluate(candidate, str(wav))

    assert result.passed is False
    assert any(reason.startswith("excessive_gain_change") for reason in result.reasons)


def test_safety_governor_rejects_clipped_render(tmp_path):
    wav = tmp_path / "clipped.wav"
    _write(wav, amp=1.2)
    candidate = MixCandidate(candidate_id="001", label="clipped")
    governor = SafetyGovernor({"safety": {"forbid_clipping": True, "max_true_peak_dbfs": -1.0}})

    result = governor.evaluate(candidate, str(wav))

    assert result.passed is False
    assert any("true_peak" in reason or "clipping" in reason for reason in result.reasons)


def test_decision_layer_safety_rejects_unsafe_actions():
    governor = DecisionSafetyGovernor({"safety": {"max_gain_change_db_per_step": 1.0, "max_eq_change_db_per_step": 1.5}})

    gain_result = governor.check_actions(CandidateActionSet("too_loud", [GainAction("vocal", 2.0)]))
    eq_result = governor.check_actions(CandidateActionSet("too_much_eq", [EQAction("vocal", "b1", 250.0, -3.0, 1.0)]))

    assert gain_result["passed"] is False
    assert any("gain_change" in reason for reason in gain_result["reasons"])
    assert eq_result["passed"] is False
    assert any("eq_change" in reason for reason in eq_result["reasons"])


def test_decision_layer_safety_rejects_clipped_render(tmp_path):
    wav = tmp_path / "clipped.wav"
    _write(wav, amp=1.2)
    governor = DecisionSafetyGovernor({"safety": {"forbid_clipping": True, "max_true_peak_dbfs": -1.0}})

    result = governor.check_render(CandidateActionSet("clipped"), wav)

    assert result["passed"] is False
    assert any("true_peak" in reason or "clipping" in reason for reason in result["reasons"])


def test_decision_layer_safety_scores_margin_for_safe_render(tmp_path):
    wav = tmp_path / "near_limit.wav"
    _write(wav, amp=0.75)
    governor = DecisionSafetyGovernor({"safety": {"forbid_clipping": True, "max_true_peak_dbfs": -1.0, "min_headroom_db": 1.0}})

    result = governor.check_render(CandidateActionSet("near_limit"), wav)

    assert result["passed"] is True
    assert 0.0 < result["safety_score"] < 1.0
    assert result["penalties"]


@pytest.mark.parametrize("malformed", [float("nan"), float("inf"), float("-inf")])
def test_safety_governors_reject_non_finite_action_values(tmp_path, malformed):
    wav = tmp_path / "safe.wav"
    _write(wav, amp=0.1)

    final_result = SafetyGovernor().evaluate(
        MixCandidate(
            candidate_id="malformed",
            label="malformed",
            actions=[
                CandidateAction(
                    action_type="gain_change",
                    target="vocal",
                    parameters={"gain_db": malformed},
                )
            ],
        ),
        str(wav),
    )
    decision_result = DecisionSafetyGovernor().check_actions(
        CandidateActionSet("malformed", [GainAction("vocal", malformed)])
    )

    assert final_result.passed is False
    assert any("invalid_numeric:action:gain_change:gain_db" in reason for reason in final_result.reasons)
    assert decision_result["passed"] is False
    assert any("invalid_numeric:action:gain:gain_db" in reason for reason in decision_result["reasons"])


def test_safety_governors_reject_non_finite_render_metrics(monkeypatch):
    metrics = {
        "level": {
            "true_peak_dbtp": float("nan"),
            "headroom_db": float("inf"),
            "clip_count": 0,
        },
        "stereo": {
            "phase_cancellation_risk": False,
            "inter_channel_correlation": float("nan"),
        },
        "spectral": {"harshness_proxy": 0.0, "muddiness_proxy": 0.0},
    }
    monkeypatch.setattr(
        "ai_mixing_pipeline.safety_governor.governor.measure_audio_file",
        lambda _path: metrics,
    )
    monkeypatch.setattr(
        "ai_mixing_pipeline.decision_layer.safety_governor.measure_audio_file",
        lambda _path: metrics,
    )

    final_result = SafetyGovernor().evaluate(
        MixCandidate(candidate_id="malformed", label="malformed"),
        "unused.wav",
    )
    decision_result = DecisionSafetyGovernor().check_render(
        CandidateActionSet("malformed"),
        "unused.wav",
    )

    assert final_result.passed is False
    assert decision_result["passed"] is False
    assert any("invalid_numeric:render:true_peak_dbtp" in reason for reason in final_result.reasons)
    assert any("invalid_numeric:render:true_peak_dbtp" in reason for reason in decision_result["reasons"])


def test_safety_governors_reject_non_finite_critic_values(tmp_path):
    wav = tmp_path / "safe.wav"
    _write(wav, amp=0.1)
    critics = {
        "mert": {"delta": {"vocal_clarity": float("nan")}},
        "panns_or_beats": {"scores": {"bleed_score": float("inf")}},
    }

    final_result = SafetyGovernor().evaluate(
        MixCandidate(candidate_id="malformed", label="malformed"),
        str(wav),
        critic_results=critics,
    )
    decision_result = DecisionSafetyGovernor().check_critics(
        CandidateActionSet("malformed"),
        critics,
    )

    assert final_result.passed is False
    assert decision_result["passed"] is False
    assert any("invalid_numeric:critic:vocal_clarity" in reason for reason in final_result.reasons)
    assert any("invalid_numeric:critic:vocal_clarity" in reason for reason in decision_result["reasons"])
