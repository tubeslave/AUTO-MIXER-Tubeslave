import numpy as np
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
