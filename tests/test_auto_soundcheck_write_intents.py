from unittest.mock import MagicMock

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo


class WingSoundcheckMixerDouble:
    def __init__(self):
        self.set_gain = MagicMock()
        self.set_fader = MagicMock()


def test_auto_soundcheck_gain_emits_replay_intent_and_skips_live_wing_write():
    engine = AutoSoundcheckEngine(auto_apply=False)
    engine.mixer_type = "wing"
    engine.mixer_client = WingSoundcheckMixerDouble()
    engine.replay_arrangement_id = "intro_a"
    engine.replay_scene_id = "soundcheck_scene_1"
    engine.replay_scene_label = "preflight"

    info = ChannelInfo(
        channel=1,
        name="Kick In",
        preset="kick",
        peak_db=-8.0,
        rms_db=-20.0,
        fader_db=-12.0,
    )

    engine._apply_input_gain(1, info, "kick")

    engine.mixer_client.set_gain.assert_not_called()

    intents = engine.get_replay_write_intents()
    assert len(intents) == 1
    record = intents[0]
    payload = record["timeline_artifact"]["events"][0]["source_payload"]

    assert record["write_kind"] == "gain"
    assert record["allow_live_write"] is False
    assert record["metadata"]["arrangement_id"] == "intro_a"
    assert record["metadata"]["scene_id"] == "soundcheck_scene_1"
    assert record["metadata"]["scene_label"] == "preflight"
    assert payload["readback_source"] == "analysis_snapshot"
    assert payload["requires_confirmation"] is True
    assert payload["recommended_target_trim_db"] < 0.0
    assert payload["autonomy_mode"] == "semi_autonomous_calibration"
    assert payload["required_approval_level"] == "operator_required"
    assert payload["action_envelope"]["transport_policy"] == "dry_run_only"
    assert payload["action_envelope"]["trust_gates"]["summary"]["effective_autonomy_mode"] == (
        "semi_autonomous_calibration"
    )
