from unittest.mock import AsyncMock

import pytest

from handlers import channel_scan_handlers, eq_handlers, phase_handlers
from server import AutoMixerServer


@pytest.mark.parametrize(
    ("method_name", "args", "expected_type", "expected_surface"),
    [
        (
            "request_reset_trim",
            ("ws", [1, 2]),
            "reset_trim_result",
            "reset_trim",
        ),
        (
            "request_bypass_mixer",
            ("ws",),
            "bypass_result",
            "bypass_mixer",
        ),
        (
            "request_reset_all_eq",
            ("ws", {"channels": [1, 2]}),
            "reset_all_eq_result",
            "reset_all_eq",
        ),
        (
            "request_reset_phase_delay",
            ("ws", [1, 2]),
            "phase_alignment_reset_result",
            "reset_phase_delay",
        ),
    ],
)
@pytest.mark.asyncio
async def test_unpromoted_maintenance_requests_fail_closed_without_legacy_write(
    method_name,
    args,
    expected_type,
    expected_surface,
):
    server = AutoMixerServer.__new__(AutoMixerServer)
    server._block_quarantined_wing_write_surface = AsyncMock(return_value=False)
    server.send_to_client = AsyncMock()
    legacy_method = expected_surface
    setattr(server, legacy_method, AsyncMock())

    await getattr(server, method_name)(
        *args,
        dry_run=False,
        confirm_maintenance=True,
        rollback_snapshot_path="/not/exposed/in/response.json",
    )

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == expected_type
    assert payload["status"] == "blocked"
    assert payload["success"] is False
    assert payload["send_attempted"] is False
    assert payload["reason"] == "legacy_maintenance_surface_not_promoted"
    assert payload["surface"] == expected_surface
    assert payload["rollback_plan_present"] is True
    assert "/not/exposed" not in str(payload)
    getattr(server, legacy_method).assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_handlers_target_real_fail_closed_server_entrypoints():
    server = AsyncMock(spec=AutoMixerServer)
    handlers = {
        **channel_scan_handlers.register_handlers(server),
        **eq_handlers.register_handlers(server),
        **phase_handlers.register_handlers(server),
    }

    await handlers["reset_trim"]("ws", {"channels": [1], "dry_run": True})
    server.request_reset_trim.assert_awaited_once()

    await handlers["bypass_mixer"]("ws", {"dry_run": True})
    server.request_bypass_mixer.assert_awaited_once()

    await handlers["reset_all_eq"]("ws", {"channels": [1], "dry_run": True})
    server.request_reset_all_eq.assert_awaited_once()

    await handlers["reset_phase_delay"]("ws", {"channels": [1], "dry_run": True})
    server.request_reset_phase_delay.assert_awaited_once()
