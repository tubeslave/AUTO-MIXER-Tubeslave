"""Offline tests for the fake gain proof readiness runner."""

from __future__ import annotations

import importlib.util
import json
import os
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
    process_artifacts_dir = Path(os.environ["AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR"])
    process_artifacts_dir.mkdir(parents=True, exist_ok=True)
    process_artifacts = {
        filename: process_artifacts_dir / filename
        for filename in (
            "manual_write_events.jsonl",
            "decision_traces.jsonl",
            "decision_memory.jsonl",
        )
    }
    for filename, artifact_path in process_artifacts.items():
        artifact_path.write_text(
            json.dumps({"sentinel": filename}) + "\n",
            encoding="utf-8",
        )
    process_artifacts_before = {
        filename: artifact_path.read_bytes()
        for filename, artifact_path in process_artifacts.items()
    }
    process_runtime_dir = Path(os.environ["AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR"])
    process_runtime_dir.mkdir(parents=True, exist_ok=True)
    process_wal = process_runtime_dir / "manual_write_wal.jsonl"
    process_wal.write_text('{"sentinel":"must-remain-unchanged"}\n', encoding="utf-8")
    process_wal_before = process_wal.read_bytes()
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
    assert summary["console_runtime_isolated"] is True
    assert summary["evidence_stores_isolated"] is True
    assert summary["operator_proposal_result_runtime_isolated"] is True
    assert summary["operator_proposal_result_runtime_cleaned"] is True
    assert summary["operator_proposal_result_store_limits"] == {
        "max_artifact_bytes": 1 * 1024 * 1024,
        "max_total_bytes": 8 * 1024 * 1024,
        "min_free_bytes": 64 * 1024 * 1024,
        "scope": "isolated_fake_dry_run_only",
    }
    assert not Path(summary["operator_proposal_result_runtime_dir"]).exists()
    assert Path(summary["operator_proposal_result_dir"]).parent == Path(
        summary["operator_proposal_result_runtime_dir"]
    )
    assert Path(summary["console_runtime_dir"]) == (
        Path(summary["summary_path"]).parent / "console_runtime"
    )
    assert Path(summary["console_wal_path"]).parent == Path(summary["console_runtime_dir"])
    assert process_wal.read_bytes() == process_wal_before
    for filename, artifact_path in process_artifacts.items():
        assert artifact_path.read_bytes() == process_artifacts_before[filename]
    run_manual_dir = Path(summary["manual_write_artifacts_dir"])
    assert run_manual_dir == Path(summary["summary_path"]).parent / "manual_write_gate"
    assert Path(summary["decision_trace_path"]) == run_manual_dir / "decision_traces.jsonl"
    assert Path(summary["decision_memory_path"]) == run_manual_dir / "decision_memory.jsonl"
    for filename in (
        "manual_write_events.jsonl",
        "decision_traces.jsonl",
        "decision_memory.jsonl",
    ):
        assert (run_manual_dir / filename).is_file()
    assert summary["write_policy"] == "fake_mixer_apply_operator_proposal_only"
    assert summary["pre_read_source"] == "fake_mixer"
    assert summary["readback_source"] == "fake_mixer"
    assert summary["rollback_readback_source"] == "not_attempted"
    assert summary["operator_approval_recorded"] is True
    assert summary["idempotency_key_present"] is True
    apply_result = summary["apply_result"]["apply_result"]
    assert apply_result["status"] == "blocked"
    assert apply_result["reason"] == "pre_read_unavailable"
    assert apply_result["pre_read_status"] == "readback_unavailable"
    assert apply_result["pre_read_error"] == "physical_query_provenance_unverified"
    assert apply_result["proof_status"] == "blocked"
    assert apply_result["no_console_send_proven"] is True
    assert summary["fake_physical_proof_rejection_status"] == "verified"
    assert summary["physical_proof_claimed"] is False
    assert summary["rollback_result"]["status"] == "not_attempted"
    assert summary["post_emergency_apply_result"]["apply_result"]["reason"] == (
        "operator_proposal_global_hazard_active"
    )
    assert summary["emergency_stop_proof_status"] == "verified"
    assert summary["fake_mixer_write_calls"] == []

    sent_types = [event["payload"]["type"] for event in read_events(summary) if event["event_type"] == "send"]
    assert sent_types == [
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
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


@pytest.mark.asyncio
async def test_same_second_dry_runs_use_exclusive_uuid_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "utc_stamp", lambda: "20260712T000000Z")
    args = parse_args(tmp_path, "--cooldown-sec", "1")

    assert await runner.run(args) == 0
    assert await runner.run(args) == 0

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    assert all((run_dir / "summary.json").is_file() for run_dir in run_dirs)
