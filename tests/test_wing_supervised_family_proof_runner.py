from argparse import Namespace
from dataclasses import replace
import json

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


def test_plan_only_default_never_constructs_wing_transport(monkeypatch, tmp_path):
    parsed = args(tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: parsed)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("plan-only mode must not construct WingClient")

    monkeypatch.setattr(runner, "WingClient", forbidden_transport)

    assert runner.main() == 0
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "planned"
    assert summary["contacts_wing"] is False
    assert summary["live_write"] is False
    assert summary["accepted"] is False


def test_execute_with_missing_typed_barriers_blocks_before_transport(
    monkeypatch,
    tmp_path,
):
    parsed = args(tmp_path, execute=True)
    monkeypatch.setattr(runner, "parse_args", lambda: parsed)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("invalid authorization must block before WingClient")

    monkeypatch.setattr(runner, "WingClient", forbidden_transport)

    assert runner.main() == 2
    summary_path = next(tmp_path.glob("*/summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert "operator_approval_id_required" in summary["authorization_failures"]
    assert summary["contacts_wing"] is False
    assert summary["readiness_matrix_update_allowed"] is False


@pytest.mark.asyncio
async def test_canonical_proposal_only_demotion_blocks_before_wing_client(
    monkeypatch,
    tmp_path,
):
    parsed = args(tmp_path, execute=True)
    parsed.approval_id = "operator-reviewed"
    parsed.confirm_live_write = runner.LIVE_EXECUTION_CONFIRMATION
    parsed.confirm_restore = runner.RESTORE_CONFIRMATION
    parsed.confirm_family = parsed.family
    parsed.confirm_ip = parsed.ip
    parsed.confirm_target_scope = runner.target_scope_fingerprint(parsed)
    contracts = tuple(
        replace(item, lifecycle="proposal_only", hardware_proof="missing")
        if item.family_id == parsed.family
        else item
        for item in runner.FAMILY_CONTRACTS
    )
    monkeypatch.setattr(runner, "FAMILY_CONTRACTS", contracts)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("proposal-only lifecycle must block before WingClient")

    monkeypatch.setattr(runner, "WingClient", forbidden_transport)

    with pytest.raises(PermissionError, match="canonical_family_lifecycle"):
        await runner.run(parsed)
