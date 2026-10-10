from automixer.decision.models import ACTION_GAIN, ActionDecision, ActionPlan
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
