from automixer.decision.models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_PAN,
    ActionDecision,
    ActionPlan,
)
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

    def set_eq_band(self, *args, **kwargs):
        self.calls.append(("set_eq_band", args, kwargs))
        return True

    def set_compressor(self, *args, **kwargs):
        self.calls.append(("set_compressor", args, kwargs))
        return True

    def set_pan(self, *args, **kwargs):
        self.calls.append(("set_pan", args, kwargs))
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


def test_executor_rejects_live_dispatch_without_canonical_supervised_authority():
    mixer = FakeMixer()
    gate = SafetyGate(
        SafetyGateConfig(
            dry_run=False,
            min_interval_sec=0.0,
            cooldown_sec=0.0,
        )
    )
    executor = ActionPlanExecutor(mixer, gate, dry_run=False)
    plan = ActionPlan(
        plan_id="untrusted-live-plan",
        decisions=[
            ActionDecision(
                id="gain",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"channel_id": 1, "gain_db": -0.5},
                safe_to_apply=True,
                risk_level="low",
            ),
            ActionDecision(
                id="eq",
                action_type=ACTION_EQ,
                target="channel:1",
                parameters={
                    "channel_id": 1,
                    "band": 2,
                    "frequency_hz": 1000.0,
                    "gain_db": -1.0,
                    "q": 1.0,
                },
                safe_to_apply=True,
                risk_level="low",
            ),
            ActionDecision(
                id="compression",
                action_type=ACTION_COMPRESSION,
                target="channel:1",
                parameters={"channel_id": 1, "threshold_db": -18.0, "ratio": 2.0},
                safe_to_apply=True,
                risk_level="low",
            ),
            ActionDecision(
                id="pan",
                action_type=ACTION_PAN,
                target="channel:1",
                parameters={"channel_id": 1, "pan": 0.1},
                safe_to_apply=True,
                risk_level="low",
            ),
        ],
    )

    result = executor.execute(
        plan,
        current_state={
            "channel:1": {
                "true_peak_dbtp": -12.0,
                "compression_threshold_db": -18.0,
                "pan": 0.0,
            }
        },
        live_mode=True,
    )

    assert result.sent == []
    assert result.blocked == []
    assert result.safety.dry_run is True
    assert [item["reason"] for item in result.recommended_only] == [
        "live_write_authority_unavailable",
    ] * 4
    assert mixer.calls == []
