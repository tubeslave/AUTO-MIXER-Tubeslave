"""Tests for tools/manual_write_recovery_operator.py — no live backend or console needed."""
from __future__ import annotations

import asyncio
import json

from websockets.asyncio.server import serve as ws_serve

from tools.manual_write_recovery_operator import (
    build_authenticate,
    build_parser,
    build_reconcile_request,
    build_reset_request,
    build_supervision_request,
    cmd_reconcile,
    cmd_reset,
    cmd_status,
    recv_until,
)

TOKEN = "t" * 43
APPROVAL = "pytest-operator-20260728"


def test_payload_builders_match_backend_contract():
    assert build_supervision_request() == {"type": "get_manual_write_supervision"}
    assert build_reconcile_request("tx-1", "op-1") == {
        "type": "reconcile_console_write_recovery",
        "transaction_id": "tx-1",
        "approved": True,
        "approval_id": "op-1",
    }
    assert build_reset_request("op-1", "why") == {
        "type": "reset_manual_write_emergency_stop",
        "approved": True,
        "approval_id": "op-1",
        "reason": "why",
    }
    assert build_authenticate(TOKEN, "operator-recovery") == {
        "type": "authenticate",
        "token": TOKEN,
        "client_id": "operator-recovery",
    }


class _FakeSocket:
    def __init__(self, messages):
        self._messages = list(messages)

    async def recv(self):
        await asyncio.sleep(0)
        if not self._messages:
            await asyncio.sleep(60)
        return self._messages.pop(0)


def test_recv_until_ignores_broadcasts():
    socket = _FakeSocket(
        [
            json.dumps({"type": "mixer_update"}),
            json.dumps({"type": "state_update"}),
            json.dumps({"type": "manual_write_supervision_status", "emergency_stopped": False}),
        ]
    )
    message = asyncio.run(recv_until(socket, {"manual_write_supervision_status"}, 1.0))
    assert message["emergency_stopped"] is False


def _supervision(unresolved, emergency_stopped):
    return {
        "type": "manual_write_supervision_status",
        "emergency_stopped": emergency_stopped,
        "reset_required": emergency_stopped,
        "wal_recovery_required": bool(unresolved),
        "wal_unresolved_transaction_ids": list(unresolved),
        "approved_live_write_kinds": [],
        "operator_mode": "supervised",
        "connection_mode": "wing",
    }


async def _run_against_fake_backend(argv, responder):
    requests = []

    async def handler(websocket):
        async for raw in websocket:
            message = json.loads(raw)
            requests.append(message)
            reply = responder(message, len(requests))
            if reply is not None:
                await websocket.send(json.dumps(reply))

    server = await ws_serve(handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        args = build_parser().parse_args(["--ws-url", f"ws://127.0.0.1:{port}", *argv])
        handlers = {"status": cmd_status, "reconcile": cmd_reconcile, "reset": cmd_reset}
        rc = await handlers[args.command](args, [])
    finally:
        server.close()
        await server.wait_closed()
    return rc, requests


def _status_ok_responder(message, _n):
    if message["type"] == "authenticate":
        return {"type": "authentication_status", "authenticated": True}
    if message["type"] == "get_manual_write_supervision":
        return _supervision([], False)
    return None


def test_status_with_token_authenticates_first():
    rc, requests = asyncio.run(
        _run_against_fake_backend(["--token", TOKEN, "status"], _status_ok_responder)
    )
    assert rc == 0
    assert requests[0]["type"] == "authenticate"
    assert requests[1]["type"] == "get_manual_write_supervision"


def _reconcile_ok_responder(message, n):
    if message["type"] == "get_manual_write_supervision":
        return _supervision(["tx-1"] if n == 1 else [], True)
    if message["type"] == "reconcile_console_write_recovery":
        return {
            "type": "console_write_recovery_result",
            "status": "reconciled",
            "success": True,
            "reason": "reconciled_committed",
            "transaction_id": message["transaction_id"],
            "approval_id": message["approval_id"],
            "physical_readback": [
                {
                    "address": "/main/1/mute",
                    "actual": 1,
                    "target_match": True,
                    "preimage_match": False,
                    "confirmed_round_trips": 2,
                }
            ],
            "emergency_reset_still_required": True,
        }
    return None


def test_reconcile_all_unresolved_success():
    rc, requests = asyncio.run(
        _run_against_fake_backend(
            ["reconcile", "--approval-id", APPROVAL], _reconcile_ok_responder
        )
    )
    assert rc == 0
    assert requests[0]["type"] == "get_manual_write_supervision"
    assert requests[1] == build_reconcile_request("tx-1", APPROVAL)


def _reconcile_auth_blocked_responder(message, _n):
    if message["type"] == "get_manual_write_supervision":
        return _supervision(["tx-1"], True)
    if message["type"] == "reconcile_console_write_recovery":
        return {
            "type": "control_authentication_required",
            "status": "blocked",
            "success": False,
            "reason": "authenticated_control_session_required",
        }
    return None


def test_reconcile_without_token_backend_blocks_with_hint(capsys):
    rc, _requests = asyncio.run(
        _run_against_fake_backend(
            ["reconcile", "--approval-id", APPROVAL], _reconcile_auth_blocked_responder
        )
    )
    assert rc == 2
    assert "AUTOMIXER_WS_AUTH_TOKEN" in capsys.readouterr().out


def _reset_pending_responder(message, _n):
    if message["type"] == "get_manual_write_supervision":
        return _supervision(["tx-1"], True)
    if message["type"] == "reset_manual_write_emergency_stop":
        return {
            "type": "manual_write_emergency_reset_result",
            "status": "blocked",
            "success": False,
            "reason": "console_write_recovery_pending",
        }
    return None


def test_reset_blocked_until_wal_reconciled(capsys):
    rc, _requests = asyncio.run(
        _run_against_fake_backend(
            ["reset", "--approval-id", APPROVAL], _reset_pending_responder
        )
    )
    assert rc == 2
    assert "reconcile" in capsys.readouterr().out


def _reset_ok_responder(message, n):
    if message["type"] == "get_manual_write_supervision":
        return _supervision([], n == 1)
    if message["type"] == "reset_manual_write_emergency_stop":
        return {
            "type": "manual_write_emergency_reset_result",
            "status": "reset",
            "success": True,
            "reason": message["reason"],
            "approval_id": message["approval_id"],
            "automation_remains_frozen": True,
        }
    return None


def test_reset_success_clears_latch():
    rc, requests = asyncio.run(
        _run_against_fake_backend(["reset", "--approval-id", APPROVAL], _reset_ok_responder)
    )
    assert rc == 0
    assert requests[1] == build_reset_request(APPROVAL, "operator_requested_emergency_reset")
