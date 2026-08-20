"""Contract tests for preserving dLive identity fields through connection handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from handlers.connection_handlers import register_handlers


@pytest.mark.asyncio
async def test_connect_dlive_defaults_match_attended_surface_identity():
    server = SimpleNamespace(connect_configured_dlive=AsyncMock())
    handlers = register_handlers(server)
    websocket = object()

    await handlers["connect_dlive"](websocket, {})

    server.connect_configured_dlive.assert_awaited_once_with(
        owner_websocket=websocket
    )


@pytest.mark.asyncio
async def test_connect_dlive_ignores_ingress_destination_fields():
    server = SimpleNamespace(connect_configured_dlive=AsyncMock())
    handlers = register_handlers(server)
    websocket = object()

    await handlers["connect_dlive"](
        websocket,
        {
            "ip": "169.254.169.254",
            "port": 80,
            "tls": True,
            "endpoint": "surface",
            "tls_ca_file": "/etc/passwd",
            "tls_server_hostname": "metadata.invalid",
        },
    )

    server.connect_configured_dlive.assert_awaited_once_with(
        owner_websocket=websocket
    )


@pytest.mark.asyncio
async def test_connect_mixing_station_ignores_ingress_destination_fields():
    server = SimpleNamespace(connect_configured_mixing_station=AsyncMock())
    handlers = register_handlers(server)
    websocket = object()

    await handlers["connect_mixing_station"](
        websocket,
        {"host": "169.254.169.254", "osc_port": 53, "rest_port": 80},
    )

    server.connect_configured_mixing_station.assert_awaited_once_with(
        owner_websocket=websocket
    )


@pytest.mark.asyncio
async def test_scan_mixers_ignores_ingress_subnet_destination():
    server = SimpleNamespace(
        block_packaged_offline_transport=AsyncMock(return_value=False),
        send_to_client=AsyncMock(),
    )
    handlers = register_handlers(server)

    with patch(
        "handlers.connection_handlers.discover_mixers",
        return_value=[],
    ) as discover:
        await handlers["scan_mixers"](
            object(),
            {"full_scan": True, "subnet": "169.254.169"},
        )

    assert discover.call_args.kwargs == {
        "scan_subnet": True,
        "subnet": None,
        "timeout": 2.0,
    }


@pytest.mark.asyncio
async def test_auto_connect_preserves_dlive_midi_base_and_user_profile():
    discovered = SimpleNamespace(
        mixer_type="dlive",
        ip="192.168.3.71",
        port=51328,
        tls=False,
        endpoint="surface",
        name="C3500",
    )
    server = SimpleNamespace(
        block_packaged_offline_transport=AsyncMock(return_value=False),
        send_to_client=AsyncMock(),
        connect_dlive=AsyncMock(),
    )
    handlers = register_handlers(server)
    websocket = object()

    with patch(
        "handlers.connection_handlers.discover_mixer_auto",
        return_value=discovered,
    ) as discover_auto:
        await handlers["auto_connect"](
            websocket,
            {
                "preferred_type": "dlive",
                "preferred_ip": "192.168.3.71",
                "midi_base_channel": 11,
                "user_profile": 4,
            },
        )

    assert discover_auto.call_args.kwargs["preferred_ip"] is None

    server.connect_dlive.assert_awaited_once_with(
        ip="192.168.3.71",
        port=51328,
        tls=False,
        endpoint="surface",
        midi_base_channel=11,
        user_profile=4,
        owner_websocket=websocket,
    )
    found = server.send_to_client.await_args_list[1].args[1]
    assert found["midi_base_channel"] == 11
    assert found["user_profile"] == 4
