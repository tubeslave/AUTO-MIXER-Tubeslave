from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import audio_devices
import handlers
from auto_soundcheck_engine import AutoSoundcheckEngine
from autofoh_safety import BusFaderMove, ChannelGainMove
from server import AutoMixerServer


def test_audio_device_discovery_does_not_invent_devices_without_pyaudio(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyaudio", None)

    assert audio_devices.get_audio_devices() == []


def test_audio_device_discovery_terminates_pyaudio_after_enumeration_failure(monkeypatch):
    pa = MagicMock()
    pa.get_device_count.side_effect = RuntimeError("backend unavailable")
    pyaudio = SimpleNamespace(PyAudio=MagicMock(return_value=pa))
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio)

    assert audio_devices.get_audio_devices() == []
    pa.terminate.assert_called_once_with()


def test_audio_device_discovery_discards_partial_results_after_backend_failure(monkeypatch):
    pa = MagicMock()
    pa.get_device_count.return_value = 2
    pa.get_device_info_by_index.side_effect = [
        {
            "name": "Real first device",
            "maxInputChannels": 2,
            "defaultSampleRate": 48000,
        },
        RuntimeError("enumeration interrupted"),
    ]
    pyaudio = SimpleNamespace(PyAudio=MagicMock(return_value=pa))
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio)

    assert audio_devices.get_audio_devices() == []
    pa.terminate.assert_called_once_with()


def test_channel_gain_readback_is_verified_against_the_requested_target():
    engine = AutoSoundcheckEngine.__new__(AutoSoundcheckEngine)
    engine.mixer_client = SimpleNamespace(get_channel_gain=lambda channel: 3.0)
    action = ChannelGainMove(reason="offline test", channel_id=4, target_db=3.0)

    applied, metrics = engine._read_control_state_for_action(action)

    assert applied is True
    assert metrics["control_state_verification_supported"] is True
    assert metrics["current_value_db"] == 3.0


def test_unimplemented_action_readback_fails_closed_instead_of_verifying():
    engine = AutoSoundcheckEngine.__new__(AutoSoundcheckEngine)
    engine.mixer_client = object()
    action = BusFaderMove(reason="offline test", bus_id=2, target_db=-6.0)

    applied, metrics = engine._read_control_state_for_action(action)

    assert applied is False
    assert metrics == {
        "action_type": "BusFaderMove",
        "control_state_verification_supported": False,
        "note": "control-state verification not implemented for action type",
    }


def test_register_all_handlers_rejects_duplicate_message_types(monkeypatch):
    async def first_handler(_websocket, _data):
        return None

    async def second_handler(_websocket, _data):
        return None

    monkeypatch.setattr(
        handlers,
        "_ALL_REGISTRARS",
        [
            lambda _server: {"same_type": first_handler},
            lambda _server: {"same_type": second_handler},
        ],
    )

    with pytest.raises(RuntimeError, match="Duplicate WebSocket handler type.*same_type"):
        handlers.register_all_handlers(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "response_type", "capability"),
    [
        ("get_auto_gate_status", "auto_gate_status", "auto_gate"),
        ("get_auto_panner_status", "auto_panner_status", "auto_panner"),
        ("get_auto_reverb_status", "auto_reverb_status", "auto_reverb"),
        ("get_auto_effects_status", "auto_effects_status", "auto_effects"),
        (
            "get_cross_adaptive_eq_status",
            "cross_adaptive_eq_status",
            "cross_adaptive_eq",
        ),
    ],
)
async def test_unimplemented_automation_status_is_explicitly_non_authoritative(
    method_name,
    response_type,
    capability,
):
    server = AutoMixerServer.__new__(AutoMixerServer)
    server.send_to_client = AsyncMock()

    await getattr(server, method_name)("ws", request_id="request-1")

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == response_type
    assert payload["capability"] == capability
    assert payload["status"] == "unavailable"
    assert payload["available"] is False
    assert payload["active"] is False
    assert payload["live_write"] is False
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
