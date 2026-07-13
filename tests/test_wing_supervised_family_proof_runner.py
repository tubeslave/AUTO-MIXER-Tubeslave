from argparse import Namespace
import json
from pathlib import Path

import pytest

from tools import wing_supervised_family_proof_runner as runner


def args(tmp_path, *, execute=False):
    return Namespace(
        family="fader",
        ip="192.0.2.10",
        port=2223,
        websocket_url="ws://127.0.0.1:8765",
        channel=3,
        bus=1,
        dca=1,
        main=1,
        send_bus=13,
        execute=execute,
        approval_id=None,
        confirm_live_write=None,
        confirm_restore=None,
        confirm_family=None,
        confirm_ip=None,
        confirm_target_scope=None,
        artifact_root=tmp_path,
    )


def test_plan_only_default_never_contacts_transport(monkeypatch, tmp_path):
    parsed = args(tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: parsed)

    assert runner.main() == 0
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "planned"
    assert summary["contacts_wing"] is False
    assert summary["contacts_backend"] is False
    assert summary["opens_udp_or_osc"] is False
    assert summary["live_write"] is False
    assert summary["accepted"] is False
    assert summary["canonical_runner"] == runner.CANONICAL_RUNNER


def test_execute_with_missing_typed_barriers_blocks_before_transport(
    monkeypatch,
    tmp_path,
):
    parsed = args(tmp_path, execute=True)
    monkeypatch.setattr(runner, "parse_args", lambda: parsed)

    assert runner.main() == 2
    summary_path = next(tmp_path.glob("*/summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert "operator_approval_id_required" in summary["authorization_failures"]
    assert summary["contacts_wing"] is False
    assert summary["readiness_matrix_update_allowed"] is False
    assert "canonical_attended_backend_session_required" in summary[
        "authorization_failures"
    ]


@pytest.mark.asyncio
async def test_even_fully_confirmed_legacy_execution_requires_canonical_session(
    tmp_path,
):
    parsed = args(tmp_path, execute=True)
    parsed.approval_id = "operator-reviewed"
    parsed.confirm_live_write = runner.LIVE_EXECUTION_CONFIRMATION
    parsed.confirm_restore = runner.RESTORE_CONFIRMATION
    parsed.confirm_family = parsed.family
    parsed.confirm_ip = parsed.ip
    parsed.confirm_target_scope = runner.target_scope_fingerprint(parsed)
    with pytest.raises(
        PermissionError,
        match="canonical_attended_backend_session_required",
    ):
        await runner.run(parsed)


def test_legacy_source_contains_no_live_transport_path():
    source = Path(runner.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "from wing_client import",
        "import websockets",
        ".connect(",
        "query_parameter(",
        "send_parameter(",
        "build_wing_supervised_family_adapters",
        "SupervisedConsoleTransactionCoordinator",
    ):
        assert forbidden not in source
