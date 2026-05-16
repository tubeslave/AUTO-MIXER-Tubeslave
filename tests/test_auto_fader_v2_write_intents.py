from __future__ import annotations

import time
from unittest.mock import MagicMock

from auto_fader_v2.controller import AutoFaderControllerV2


class WingMixerDouble:
    def __init__(self, readback: float = -6.0):
        self.get_channel_fader = MagicMock(return_value=readback)
        self.set_channel_fader = MagicMock()
        self.state = {}


def _make_controller(readback: float = -6.0) -> tuple[AutoFaderControllerV2, WingMixerDouble]:
    mixer = WingMixerDouble(readback=readback)
    controller = AutoFaderControllerV2(
        mixer_client=mixer,
        config={
            "replay_write_intent": {
                "arrangement_id": "speech_bed_a",
                "scene_id": "scene_12",
                "scene_label": "rehearsal",
            },
            "automation": {
                "auto_fader": {},
            },
        },
    )
    controller.selected_channels = [1, 2]
    controller.instrument_types = {
        1: "leadVocal",
        2: "keys",
    }
    return controller, mixer


def test_reset_faders_to_zero_emits_replay_intents_and_skips_live_wing_writes(monkeypatch):
    controller, mixer = _make_controller(readback=-9.0)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    controller._reset_faders_to_zero()

    mixer.set_channel_fader.assert_not_called()
    intents = controller.get_replay_write_intents()
    assert len(intents) == 2
    assert [record["mixer_channel"] for record in intents] == [1, 2]
    assert all(record["allow_live_write"] is False for record in intents)

    first_payload = intents[0]["timeline_artifact"]["events"][0]["source_payload"]
    assert intents[0]["metadata"]["arrangement_id"] == "speech_bed_a"
    assert intents[0]["metadata"]["scene_id"] == "scene_12"
    assert first_payload["readback_source"] == "startup_fader_reset"
    assert first_payload["recommended_target_fader_db"] == 0.0
    assert first_payload["requires_confirmation"] is True


def test_send_fader_command_emits_replay_intent_with_ceiling_limited_target_for_wing():
    controller, mixer = _make_controller(readback=5.8)

    controller._send_fader_command(channel_id=1, adjustment_db=2.0)

    mixer.set_channel_fader.assert_not_called()
    intents = controller.get_replay_write_intents()
    assert len(intents) == 1

    record = intents[0]
    payload = record["timeline_artifact"]["events"][0]["source_payload"]
    assert record["write_kind"] == "fader"
    assert record["allow_live_write"] is False
    assert payload["readback_source"] == "controller_state"
    assert payload["current_fader_db"] == 5.8
    assert payload["recommended_target_fader_db"] == 6.0
    assert payload["requires_confirmation"] is True


def test_speech_priority_arbitration_emits_dry_run_intent_for_wing():
    controller, mixer = _make_controller(readback=-6.0)
    controller.config["automation"]["auto_fader"].update(
        {
            "ducking_enabled": True,
            "speech_priority_enabled": True,
            "ducking_amount_db": -4.0,
            "vocal_channels": [1],
        }
    )
    controller.ducking_enabled = True
    controller.speech_priority_enabled = True
    controller.ducking_amount_db = -4.0
    controller.vocal_channels = [1]

    controller._maybe_emit_speech_priority_arbitration(
        vocal_levels={1: -18.0},
        duck_targets=[2],
        activity_level=1.0,
        is_vocal_active=True,
    )

    mixer.set_channel_fader.assert_not_called()
    intents = controller.get_replay_write_intents()
    assert len(intents) == 1
    record = intents[0]
    payload = record["timeline_artifact"]["events"][0]["source_payload"]

    assert record["write_kind"] == "speech_arbitration"
    assert record["allow_live_write"] is False
    assert record["target_bus"] == "main"
    assert payload["environment_label"] == "speech_focus"
    assert payload["priority_source"] == "leadVocal:ch1"
    assert payload["requested_state"]["environment"] == "speech"
    assert payload["requested_state"]["duck_music_db"] == -4.0
    assert payload["action_envelope"]["kind"] == "speech_priority"
    assert payload["action_envelope"]["transport_policy"] == "dry_run_only"


def test_speech_priority_arbitration_dedupes_unchanged_state_and_emits_release():
    controller, _ = _make_controller(readback=-6.0)
    controller.ducking_enabled = True
    controller.speech_priority_enabled = True
    controller.ducking_amount_db = -4.0
    controller.vocal_channels = [1]

    controller._maybe_emit_speech_priority_arbitration(
        vocal_levels={1: -18.0},
        duck_targets=[2],
        activity_level=1.0,
        is_vocal_active=True,
    )
    controller._maybe_emit_speech_priority_arbitration(
        vocal_levels={1: -18.0},
        duck_targets=[2],
        activity_level=1.0,
        is_vocal_active=True,
    )
    controller._maybe_emit_speech_priority_arbitration(
        vocal_levels={1: -45.0},
        duck_targets=[2],
        activity_level=0.0,
        is_vocal_active=False,
    )

    intents = controller.get_replay_write_intents()
    assert len(intents) == 2
    assert [record["proposal"]["requested_state"]["environment"] for record in intents] == [
        "speech",
        "music",
    ]
    assert intents[1]["proposal"]["current_state"]["environment"] == "speech"
    assert intents[1]["proposal"]["requested_state"]["duck_music_db"] == 0.0
