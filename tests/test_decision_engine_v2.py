from automixer.decision import DecisionEngine
from automixer.decision.models import ACTION_EQ, ACTION_GAIN


def test_decision_engine_creates_explainable_action_plan():
    engine = DecisionEngine()

    plan = engine.create_action_plan(
        {
            "source_module": "test_analyzer",
            "channels": [
                {
                    "channel_id": 1,
                    "name": "Lead",
                    "role": "lead_vocal",
                    "metrics": {
                        "lufs": -25.0,
                        "target_lufs": -20.0,
                        "true_peak_dbtp": -8.0,
                        "SibilanceIndex": 3.2,
                    },
                    "confidence": 0.9,
                }
            ],
        },
        critic_evaluations={"channels": [{"channel_id": 1, "confidence": 0.9}]},
        mode="live",
    )

    action_types = {decision.action_type for decision in plan.decisions}

    assert ACTION_GAIN in action_types
    assert ACTION_EQ in action_types
    first_gain = next(decision for decision in plan.decisions if decision.action_type == ACTION_GAIN)
    assert first_gain.reason
    assert first_gain.expected_audio_effect
    assert first_gain.source_modules
    assert first_gain.safe_to_apply is True
    assert 0.0 <= first_gain.confidence <= 1.0
