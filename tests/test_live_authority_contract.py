"""Static tripwires for the narrow live-console authority boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_canonical_soundcheck_is_proposal_only():
    source = _source("backend/handlers/soundcheck_handlers.py")
    engine_source = _source("backend/auto_soundcheck_engine.py")

    assert "auto_apply=True" not in source
    assert "auto_apply=False" in source
    assert "observe_only = True" in source
    assert "Observe-only session: skipping soundcheck profile auto-save" in engine_source


def test_frontend_cannot_request_force_auto_apply():
    source = _source("frontend/src/services/websocket.js")

    assert "force_auto_apply" not in source
    assert "forceAutoApply" not in source


def test_wing_send_does_not_promote_commands_to_observed_state():
    source = _source("backend/wing_client.py")
    send_body = source.split("    def send(self, address: str, *values):", 1)[1].split(
        "    def query_parameter", 1
    )[0]

    assert "_set_state_value" not in send_body
    assert "_set_commanded_value" in send_body


def test_remote_websocket_bind_is_fail_closed_and_local_default_is_loopback():
    source = _source("backend/server.py")

    assert 'os.environ.get("AUTOMIXER_WS_HOST", "127.0.0.1")' in source
    assert "Remote WebSocket bind requires AUTOMIXER_WS_AUTH_TOKEN" in source
    assert "if not await self._authenticate_websocket(websocket)" in source


def test_only_physical_readback_can_close_manual_write_proof():
    source = _source("backend/server.py")

    assert 'physically_verified = readback_status == "verified_from_physical_readback"' in source
    assert '"status": "applied" if physically_verified else "sent_unverified"' in source
    assert 'reason="rollback_transaction_mismatch"' in source
