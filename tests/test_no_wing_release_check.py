import json
import sys
from pathlib import Path

import pytest

from tools import no_wing_release_check as release_check
from tools.no_wing_release_check import (
    CheckResult,
    CommandCheck,
    build_checks,
    run_command_check,
    run_release_checks,
    validate_no_live_commands,
    write_summary,
)


def test_no_wing_release_check_commands_have_no_live_write_tokens(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_package=True)

    validate_no_live_commands(checks)

    command_text = "\n".join(check.shell_display() for check in checks)
    assert "wing_operator_proposal_validation_runner.py" not in command_text
    assert "wing_supervised_smoke_runner.py" not in command_text
    assert "apply_operator_proposal" not in command_text
    assert "set_fader" not in command_text
    assert "set_gain" not in command_text
    assert "set_eq" not in command_text
    assert "set_compressor" not in command_text
    assert "route_output" not in command_text
    assert "reset" not in command_text
    assert "mute" not in command_text
    assert any(check.name == "frontend_dist_package" for check in checks)
    assert all(check.timeout_sec == release_check.DEFAULT_COMMAND_TIMEOUT_SEC for check in checks)


def test_no_wing_release_check_can_disable_command_timeouts(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, command_timeout_sec=None)

    assert checks
    assert all(check.timeout_sec is None for check in checks)


def test_no_wing_release_check_rejects_forbidden_live_command():
    checks = [
        CommandCheck(
            name="bad_live_runner",
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_supervised_smoke_runner",
            command=("python3", "tools/wing_supervised_smoke_runner.py"),
            cwd=".",
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "forbidden live command tokens" in message
    assert "bad_live_runner:wing_operator_proposal_validation_runner.py" in message
    assert "bad_supervised_smoke_runner:wing_supervised_smoke_runner.py" in message


def test_no_wing_release_check_rejects_expanded_forbidden_live_commands():
    checks = [
        CommandCheck(
            name="bad_eq_runner",
            command=("python3", "tools/apply_set_eq_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_compressor_runner",
            command=("python3", "tools/apply_set_compressor_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_runner",
            command=("python3", "tools/reset_eq.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_mute_runner",
            command=("python3", "tools/mute_channels.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_eq_runner:set_eq" in message
    assert "bad_compressor_runner:set_compressor" in message
    assert "bad_reset_runner:reset" in message
    assert "bad_mute_runner:mute" in message


def test_no_wing_release_check_rejects_live_write_method_variants():
    checks = [
        CommandCheck(
            name="bad_channel_gain_runner",
            command=("python3", "tools/proof_set_channel_gain.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_fx_runner",
            command=("python3", "tools/apply_set_fx_model_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_route_runner",
            command=("python3", "tools/route_multiple_outputs_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_input_patch_runner",
            command=("python3", "tools/set_channel_input_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_phase_runner",
            command=("python3", "tools/reset_phase_delay.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_channel_gain_runner:set_channel_gain" in message
    assert "bad_fx_runner:set_fx_model" in message
    assert "bad_route_runner:route_multiple_outputs" in message
    assert "bad_input_patch_runner:set_channel_input" in message
    assert "bad_phase_runner:reset_phase_delay" in message


def test_no_wing_release_check_summary_reports_failed_checks(tmp_path):
    result = CheckResult(
        name="example_check",
        status="failed",
        started_at="2026-05-26T00:00:00Z",
        finished_at="2026-05-26T00:00:01Z",
        duration_sec=1.0,
        returncode=1,
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-05-26T00:00:00Z")

    assert summary["status"] == "failed"
    assert summary["live_write"] is False
    assert summary["failed_checks"] == ["example_check"]
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"
    assert "`example_check`: `failed`" in (tmp_path / "summary.md").read_text()


def test_no_wing_release_check_records_command_timeout(tmp_path):
    check = CommandCheck(
        name="slow_check",
        command=(
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(2)",
        ),
        cwd=str(Path.cwd()),
        timeout_sec=0.1,
    )

    result = run_command_check(check, tmp_path)

    assert result.status == "timeout"
    assert result.returncode is None
    assert result.detail == {"timeout_sec": 0.1, "reason": "command_timeout"}
    assert "started" in Path(result.stdout_log).read_text()
    assert "Command timed out after 0.1 seconds." in Path(result.stderr_log).read_text()


def test_no_wing_release_check_stops_when_backend_port_is_busy(tmp_path, monkeypatch):
    def fake_busy_port_check(port: int, host: str = "127.0.0.1") -> CheckResult:
        return CheckResult(
            name="backend_port_free",
            status="failed",
            started_at="2026-05-26T00:00:00Z",
            finished_at="2026-05-26T00:00:01Z",
            duration_sec=1.0,
            detail={
                "host": host,
                "listener_detected": True,
                "policy": "port_must_not_have_unexpected_listener",
                "port": port,
            },
        )

    monkeypatch.setattr(release_check, "check_port_free", fake_busy_port_check)

    summary = run_release_checks(
        artifact_root=tmp_path,
        include_frontend=False,
        quick=True,
        backend_port=8765,
    )

    assert summary["status"] == "failed"
    assert summary["failed_checks"] == ["backend_port_free"]
    assert summary["live_write"] is False
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "logs").exists()
