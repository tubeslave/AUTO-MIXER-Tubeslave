from automixer.decision.models import ACTION_GAIN, ActionDecision, ActionPlan
from automixer.executor import ActionPlanExecutor
from automixer.safety import SafetyGate, SafetyGateConfig


class FakeMixer:
    def __init__(self):
        self.calls = []
        self.fader = {1: -12.0}

    def get_fader(self, channel):
        return self.fader[channel]

    def set_fader(self, channel, value_db):
        self.calls.append(("set_fader", channel, value_db))
        self.fader[channel] = value_db
        return True


def test_executor_dry_run_does_not_send_mixer_writes():
    mixer = FakeMixer()
    gate = SafetyGate(
        SafetyGateConfig(
            max_gain_change_db=1.0,
            max_live_gain_increase_db=1.0,
            dry_run=True,
        )
    )
    executor = ActionPlanExecutor(mixer, gate, dry_run=True)
    plan = ActionPlan(
        plan_id="test",
        decisions=[
            ActionDecision(
                id="gain.safe",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"channel_id": 1, "gain_db": 0.5},
                reason="small lift",
                confidence=0.9,
                risk_level="low",
                source_modules=["test"],
                expected_audio_effect="small level lift",
                safe_to_apply=True,
            )
        ],
    )

    result = executor.execute(
        plan,
        current_state={"channel:1": {"true_peak_dbtp": -12.0}},
        live_mode=True,
    )

    assert result.sent == []
    assert result.recommended_only
    assert mixer.calls == []
