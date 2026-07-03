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
    # Потолок автоматики — 0 dBFS (SAFETY инвариант 1): target_fader 10.0
    # клэмпится к 0.0, автоматика не поднимает фейдер выше unity
    mixer.set_channel_fader.assert_called_once_with(17, 0.0)


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


class _FakeLufsMeter:
    def __init__(self, value):
        self._value = value
    def get_integrated_lufs(self):
        return self._value
    def get_block_count(self):
        return 10


def _balance_controller(tap_point, measured_lufs, current_fader):
    """Контроллер с одним каналом для проверки компенсации фейдера по tap_point."""
    from auto_fader import BalanceProfile, GenreProfile
    mixer = MagicMock()
    mixer.is_connected = True
    mixer.get_channel_fader = MagicMock(return_value=current_fader)
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"audio": {"tap_point": tap_point}},
    )
    controller.profile = BalanceProfile.get_preset(GenreProfile.POP_ROCK)
    controller.bleed_service = None
    controller.bleed_threshold = -50.0
    controller.auto_balance_result = {}
    controller.auto_balance_pass = 1
    controller.channels = {
        1: SimpleNamespace(
            mixer_channel=1,
            current_fader=current_fader,
            instrument_type="leadVocal",
            locked=False,
            integrated_lufs=measured_lufs,
            integrated_lufs_meter=_FakeLufsMeter(measured_lufs),
            spectral_centroid=1000.0,
            band_energy={},
            band_energy_max={},
        )
    }
    return controller


def test_tap_point_pre_fader_skips_fader_compensation():
    """pre_fader: pre_fader == measured (фейдер не вычитается)."""
    # источник -22 LUFS, фейдер на -12. Для leadVocal цель -22.
    ctrl = _balance_controller("pre_fader", measured_lufs=-22.0, current_fader=-12.0)
    ctrl._compute_auto_balance()
    r = ctrl.auto_balance_result[1]
    # pre_fader = measured = -22, target = -22 → ideal_fader = 0, correction = 0 - (-12) = +12
    assert r["target_lufs"] == -22.0
    # ideal ~ 0, correction относительно current -12 → около +12
    assert abs(r["correction"] - 12.0) < 0.5, r


def test_tap_point_post_fader_applies_fader_compensation():
    """post_fader: pre_fader = measured - fader (компенсация)."""
    # источник измерен -34 post-fader, фейдер -12 → pre_fader = -34-(-12) = -22
    ctrl = _balance_controller("post_fader", measured_lufs=-34.0, current_fader=-12.0)
    ctrl._compute_auto_balance()
    r = ctrl.auto_balance_result[1]
    # pre_fader = -22, target -22 → ideal 0, correction = 0-(-12)=+12
    assert abs(r["correction"] - 12.0) < 0.5, r


def test_tap_point_defaults_to_post_fader():
    from auto_fader import AutoFaderController
    ctrl = AutoFaderController(mixer_client=None, config={})
    assert ctrl.tap_point == "post_fader"
    ctrl2 = AutoFaderController(mixer_client=None, config={"audio": {"tap_point": "pre_fader"}})
    assert ctrl2.tap_point == "pre_fader"
