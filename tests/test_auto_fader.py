from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from auto_fader import AutoFaderController


def _make_controller(readback):
    mixer = MagicMock()
    mixer.is_connected = True
    mixer.get_channel_fader = MagicMock(return_value=readback)
    mixer.set_channel_fader = MagicMock()
    controller = AutoFaderController(mixer_client=mixer, config={})
    controller.channels = {
        1: SimpleNamespace(
            mixer_channel=17,
            current_fader=8.0,
            instrument_type="vocal",
            locked=False,
        )
    }
    return controller, mixer


class WingMixerDouble:
    def __init__(self, readback):
        self.is_connected = True
        self.get_channel_fader = MagicMock(return_value=readback)
        self.set_channel_fader = MagicMock()


def test_apply_auto_balance_uses_precomputed_max_clipped_target_when_readback_missing():
    controller, mixer = _make_controller(None)
    controller.auto_balance_result = {
        1: {
            "correction": 2.0,
            "integrated_lufs": -20.0,
            "target_lufs": -18.0,
            "locked": False,
            "current_fader": 8.0,
            "target_fader": 10.0,
        }
    }

    assert controller.apply_auto_balance() is True
    mixer.set_channel_fader.assert_called_once_with(17, 10.0)


def test_apply_auto_balance_uses_precomputed_min_clipped_target_when_readback_invalid():
    controller, mixer = _make_controller(-120.0)
    controller.channels[1].current_fader = -55.0
    controller.auto_balance_result = {
        1: {
            "correction": -5.0,
            "integrated_lufs": -70.0,
            "target_lufs": -60.0,
            "locked": False,
            "current_fader": -55.0,
            "target_fader": -60.0,
        }
    }

    assert controller.apply_auto_balance() is True
    mixer.set_channel_fader.assert_called_once_with(17, -60.0)


def test_apply_auto_balance_emits_replay_intent_and_skips_live_wing_write():
    mixer = WingMixerDouble(readback=-6.0)
    controller = AutoFaderController(
        mixer_client=mixer,
        config={
            "replay_write_intent": {
                "arrangement_id": "chorus_a",
                "scene_id": "scene_7",
            }
        },
    )
    controller.channels = {
        1: SimpleNamespace(
            mixer_channel=17,
            current_fader=-6.0,
            instrument_type="vocal",
            locked=False,
        )
    }
    controller.auto_balance_result = {
        1: {
            "correction": 2.0,
            "integrated_lufs": -20.0,
            "target_lufs": -18.0,
            "locked": False,
            "current_fader": -6.0,
            "target_fader": -4.0,
        }
    }

    assert controller.apply_auto_balance() is True
    mixer.set_channel_fader.assert_not_called()

    intents = controller.get_replay_write_intents()
    assert len(intents) == 1
    record = intents[0]
    payload = record["timeline_artifact"]["events"][0]["source_payload"]

    assert record["write_kind"] == "fader"
    assert record["allow_live_write"] is False
    assert record["metadata"]["arrangement_id"] == "chorus_a"
    assert record["metadata"]["scene_id"] == "scene_7"
    assert payload["readback_source"] == "console_state"
    assert payload["requires_confirmation"] is True


def test_apply_auto_balance_uses_live_apply_adapter_when_present():
    controller, mixer = _make_controller(-6.0)
    controller.live_apply_adapter = MagicMock(return_value=True)
    controller.auto_balance_result = {
        1: {
            "correction": 2.0,
            "integrated_lufs": -20.0,
            "target_lufs": -18.0,
            "locked": False,
            "current_fader": -6.0,
            "target_fader": -4.0,
        }
    }

    assert controller.apply_auto_balance() is True
    mixer.set_channel_fader.assert_not_called()
    controller.live_apply_adapter.assert_called_once()
