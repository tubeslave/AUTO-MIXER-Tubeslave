import json
from pathlib import Path

import pytest

from tools import no_wing_release_check as release_check
from tools.no_wing_release_check import (
    CheckResult,
    CommandCheck,
    build_checks,
    run_release_checks,
    validate_no_live_commands,
    write_summary,
)


def test_no_wing_release_check_commands_have_no_live_write_tokens(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_package=True)

    validate_no_live_commands(checks)

    command_text = "\n".join(check.shell_display() for check in checks)
    assert "wing_operator_proposal_validation_runner.py" not in command_text
    assert "apply_operator_proposal" not in command_text
    assert "set_fader" not in command_text
    assert "set_gain" not in command_text
    assert "route_output" not in command_text
    assert any(check.name == "frontend_dist_package" for check in checks)


def test_no_wing_release_check_rejects_forbidden_live_command():
    checks = [
        CommandCheck(
            name="bad_live_runner",
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            cwd=".",
        )
    ]

    with pytest.raises(ValueError, match="forbidden live command tokens"):
        validate_no_live_commands(checks)


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
