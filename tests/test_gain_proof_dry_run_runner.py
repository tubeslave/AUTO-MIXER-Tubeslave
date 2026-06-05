"""Offline tests for the fake gain proof readiness runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "gain_proof_dry_run_runner.py"
SPEC = importlib.util.spec_from_file_location("gain_proof_dry_run_runner", RUNNER_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def parse_args(tmp_path: Path, *argv: str):
    return runner.build_parser().parse_args(
        [
            "--artifact-root",
            str(tmp_path),
            *argv,
        ]
    )


def read_summary(artifact_root: Path) -> dict:
    summaries = list(artifact_root.glob("*/summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def read_events(summary: dict) -> list[dict]:
    events_path = Path(summary["events_path"])
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (["--channel", "0"], "--channel must be in 1..48"),
        (["--channel", "49"], "--channel must be in 1..48"),
        (["--delta-db", "0"], "--delta-db absolute value must be in 0..0.5 dB"),
        (["--delta-db", "-0.75"], "--delta-db absolute value must be in 0..0.5 dB"),
        (["--cooldown-sec", "31"], "--cooldown-sec must be between 0 and 30"),
    ],
)
def test_validate_args_blocks_unsafe_invocations(tmp_path: Path, argv: list[str], expected_error: str) -> None:
    args = parse_args(tmp_path, *argv)

    assert expected_error in runner.validate_args(args)


@pytest.mark.asyncio
async def test_gain_proof_dry_run_uses_operator_proposal_path_only(tmp_path: Path) -> None:
    args = parse_args(
        tmp_path,
        "--channel",
        "3",
        "--initial-gain-db",
        "1.0",
        "--delta-db",
        "-0.25",
        "--cooldown-sec",
        "5",
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["proof_artifact_schema"] == "gain_proof_dry_run.v1"
    assert summary["proof_runner"] == "gain_proof_dry_run_runner"
    assert summary["proof_mode"] == "dry_run"
    assert summary["physical_wing"] is False
    assert summary["write_policy"] == "fake_mixer_apply_operator_proposal_only"
    assert summary["pre_read_source"] == "fake_mixer"
    assert summary["readback_source"] == "fake_mixer"
    assert summary["rollback_readback_source"] == "fake_mixer"
    assert summary["operator_approval_recorded"] is True
    assert summary["idempotency_key_present"] is True
    apply_result = summary["apply_result"]["apply_result"]
    assert apply_result["pre_read_status"] == "verified_from_physical_readback"
    assert apply_result["readback_status"] == "verified_from_physical_readback"
    assert apply_result["proof_status"] == "verified"
    assert apply_result["rollback_available"] is True
    assert summary["cooldown_apply_result"]["apply_result"]["reason"] == "cooldown"
    assert summary["rollback_result"]["rollback_readback_status"] == "verified_from_physical_readback"
    assert summary["rollback_result"]["rollback_proof_status"] == "verified"
    assert summary["post_emergency_apply_result"]["apply_result"]["reason"] == "emergency_stop_active"
    assert [call["method"] for call in summary["fake_mixer_write_calls"]] == [
        "set_channel_gain",
        "set_channel_gain",
    ]

    sent_types = [event["payload"]["type"] for event in read_events(summary) if event["event_type"] == "send"]
    assert sent_types == [
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
        "create_operator_proposal",
        "apply_operator_proposal",
        "rollback_last_manual_write",
        "emergency_stop",
        "create_operator_proposal",
        "apply_operator_proposal",
    ]
    forbidden_direct_writes = {
        "set_gain",
        "set_fader",
        "set_eq",
        "set_compressor",
        "route_output",
        "route_multiple_outputs",
        "set_channel_input",
        "set_channel_alt_input",
        "load_snap",
        "save_snap",
        "undo_restore_snapshot",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
        "reset_phase_delay",
        "start_auto_eq",
        "apply_eq_correction",
        "start_realtime_fader",
        "apply_auto_balance",
        "start_auto_soundcheck",
        "start_auto_compressor",
        "start_auto_compressor_soundcheck",
        "start_auto_compressor_live",
    }
    assert not forbidden_direct_writes & set(sent_types)
