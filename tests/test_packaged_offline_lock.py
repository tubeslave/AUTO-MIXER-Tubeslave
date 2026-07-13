"""Fail-closed contract for the packaged no-hardware runtime smoke lock."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from server import AutoMixerServer


@pytest.fixture
def offline_server():
    with (
        patch.dict(
            os.environ,
            {
                "AUTOMIXER_PACKAGED_OFFLINE_LOCK": "true",
                "AUTOMIXER_OPERATOR_MODE": "supervised",
                "AUTOMIXER_AUTO_MODE_UNLOCKED": "true",
                "AUTOMIXER_WING_OBSERVATION_MODE": "false",
                "AUTOMIXER_WING_RECORD_TELEMETRY": "true",
            },
        ),
        patch.object(AutoMixerServer, "_load_config", return_value={}),
        patch("server.BleedService"),
        patch.object(AutoMixerServer, "_init_training_service") as init_training,
    ):
        server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

    assert init_training.call_count == 0
    server.broadcast = AsyncMock()
    server.send_to_client = AsyncMock()
    return server


def test_offline_lock_forces_non_authoritative_runtime_state(offline_server):
    server = offline_server

    assert server._packaged_offline_lock is True
    assert server.operator_mode == "analyze"
    assert server._auto_mode_unlocked is False
    assert server._wing_observation_only is True
    assert server._wing_record_telemetry is False
    assert server.auto_soundcheck_observe_only is True

    mode = server.set_operator_mode("supervised", reason="attempted_unlock")
    assert mode["mode"] == "analyze"
    assert mode["reason"] == "packaged_offline_lock_active"
    assert mode["mixer_transport_locked"] is True

    supervision = server.get_manual_write_supervision_status()
    assert supervision["packaged_offline_lock"] is True
    assert supervision["mixer_transport_locked"] is True


@pytest.mark.asyncio
async def test_offline_lock_blocks_every_direct_mixer_transport_before_construction(
    offline_server,
):
    server = offline_server

    with (
        patch("server.EnhancedOSCClient") as wing_client,
        patch("server.DLiveClient") as dlive_client,
        patch("server.MixingStationClient") as mixing_station_client,
        patch("server.discover_mixing_station") as discover_station,
    ):
        results = [
            await server.connect_wing(
                "192.168.1.102",
                observation_only=False,
                record_telemetry=True,
            ),
            await server.connect_dlive("192.168.3.70"),
            await server.connect_mixing_station("127.0.0.1"),
            await server.discover_mixing_station(),
        ]

    assert results == [False, False, False, False]
    wing_client.assert_not_called()
    dlive_client.assert_not_called()
    mixing_station_client.assert_not_called()
    discover_station.assert_not_called()
    assert server.mixer_client is None
    assert server.connection_mode is None

    blocked_operations = [
        call.args[0]["operation"] for call in server.broadcast.await_args_list
    ]
    assert blocked_operations == [
        "connect_wing",
        "connect_dlive",
        "connect_mixing_station",
        "discover_mixing_station",
    ]
    assert all(
        call.args[0]["reason"] == "packaged_offline_lock_active"
        for call in server.broadcast.await_args_list
    )


@pytest.mark.asyncio
async def test_offline_lock_blocks_discovery_handlers_before_network_work(
    offline_server,
):
    server = offline_server
    websocket = object()

    with (
        patch("handlers.connection_handlers.discover_mixers") as discover_mixers,
        patch("handlers.connection_handlers.discover_mixer_auto") as discover_auto,
    ):
        await server._dispatch["scan_mixers"](websocket, {"full_scan": True})
        await server._dispatch["auto_connect"](websocket, {"full_scan": True})

    discover_mixers.assert_not_called()
    discover_auto.assert_not_called()
    blocked_operations = [
        call.args[1]["operation"] for call in server.send_to_client.await_args_list
    ]
    assert blocked_operations == ["scan_mixers", "auto_connect"]


@pytest.mark.asyncio
async def test_offline_lock_blocks_soundcheck_engine_and_manual_write(offline_server):
    server = offline_server
    websocket = object()

    with patch("handlers.soundcheck_handlers.AutoSoundcheckEngine") as engine:
        await server._dispatch["start_auto_soundcheck"](
            websocket,
            {
                "mixer_type": "wing",
                "mixer_ip": "192.168.1.102",
                "channels": [1],
            },
        )
    engine.assert_not_called()
    soundcheck_payload = server.send_to_client.await_args.args[1]
    assert soundcheck_payload["status"] == "blocked"
    assert soundcheck_payload["reason"] == "packaged_offline_lock_active"

    result = await server._apply_manual_console_write_unlocked(
        websocket,
        channel=1,
        value=-1.0,
        value_type="fader",
        source="packaged_runtime_smoke",
        approved=True,
        approval_id="must-never-write",
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "packaged_offline_lock_active"
    assert result["mixer_transport_locked"] is True
