import pytest

from automixer.decision.models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_PAN,
    ActionDecision,
    ActionPlan,
)
from automixer.safety import SafetyGate, SafetyGateConfig


def test_safety_gate_blocks_dangerous_gain_change():
    plan = ActionPlan(
        plan_id="test",
        decisions=[
            ActionDecision(
                id="gain.too_much",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"channel_id": 1, "gain_db": 4.0},
                reason="test large lift",
                confidence=0.9,
                risk_level="low",
                source_modules=["test"],
                expected_audio_effect="too much gain",
                safe_to_apply=True,
            )
        ],
    )
    gate = SafetyGate(SafetyGateConfig(max_gain_change_db=1.0, dry_run=False))

    result = gate.evaluate_plan(
        plan,
        current_state={"channel:1": {"true_peak_dbtp": -12.0}},
        live_mode=True,
    )

    assert result.allowed_count == 0
    assert result.blocked_count == 1
    assert result.blocked[0]["reason"] == "gain_step_exceeds_limit"


def test_safety_gate_rate_limits_repeated_actions():
    now = [100.0]
    gate = SafetyGate(
        SafetyGateConfig(max_gain_change_db=1.0, max_live_gain_increase_db=1.0),
        time_provider=lambda: now[0],
    )
    plan = ActionPlan(
        plan_id="test",
        decisions=[
            ActionDecision(
                id="gain.small",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"channel_id": 1, "gain_db": 0.5},
                reason="test move",
                confidence=0.9,
                risk_level="low",
                source_modules=["test"],
                expected_audio_effect="small gain",
                safe_to_apply=True,
            )
        ],
    )

    first = gate.evaluate_plan(plan, current_state={"channel:1": {"true_peak_dbtp": -12.0}})
    second = gate.evaluate_plan(plan, current_state={"channel:1": {"true_peak_dbtp": -12.0}})

    assert first.allowed_count == 1
    assert second.blocked[0]["reason"] == "rate_limited"


def test_safety_gate_blocks_positive_gain_when_true_peak_is_unknown():
    gate = SafetyGate(
        SafetyGateConfig(max_gain_change_db=1.0, max_live_gain_increase_db=1.0)
    )
    plan = ActionPlan(
        plan_id="missing-headroom",
        decisions=[
            ActionDecision(
                id="gain.unknown_headroom",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"channel_id": 1, "gain_db": 0.5},
                reason="test missing evidence",
                confidence=0.9,
                risk_level="low",
                source_modules=["test"],
                expected_audio_effect="small gain",
                safe_to_apply=True,
            )
        ],
    )

    result = gate.evaluate_plan(plan, current_state={"channel:1": {}}, live_mode=True)

    assert result.allowed_count == 0
    assert result.blocked[0]["reason"] == "true_peak_headroom_unknown"


@pytest.mark.parametrize(
    ("action_type", "parameters"),
    [
        (ACTION_GAIN, {"gain_db": float("nan")}),
        (ACTION_EQ, {"gain_db": float("inf"), "q": 1.0, "frequency_hz": 1000.0}),
        (ACTION_COMPRESSION, {"ratio": "nan", "threshold_db": -18.0}),
        (ACTION_PAN, {"pan": float("-inf")}),
    ],
)
def test_safety_gate_rejects_non_finite_parameters_before_dispatch(action_type, parameters):
    plan = ActionPlan(
        plan_id="non-finite",
        decisions=[
            ActionDecision(
                id="non-finite.action",
                action_type=action_type,
                target="channel:1",
                parameters=parameters,
                reason="security regression",
                confidence=0.9,
                risk_level="low",
                source_modules=["test"],
                expected_audio_effect="none",
                safe_to_apply=True,
            )
        ],
    )

    result = SafetyGate(SafetyGateConfig(dry_run=False)).evaluate_plan(
        plan,
        current_state={"channel:1": {"true_peak_dbtp": -12.0, "pan": 0.0}},
        live_mode=True,
    )

    assert result.allowed_count == 0
    assert result.blocked[0]["reason"] == "non_finite_parameter"
