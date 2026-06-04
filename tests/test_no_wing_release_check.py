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
    assert any(check.name == "write_surface_static_scan" for check in checks)
    assert any(check.name == "memory_lint" for check in checks)
    assert any(check.name == "frontend_dist_package" for check in checks)
    assert all(check.timeout_sec == release_check.DEFAULT_COMMAND_TIMEOUT_SEC for check in checks)


def test_no_wing_release_check_allows_benign_words_containing_forbidden_tokens():
    checks = [
        CommandCheck(
            name="resettable_fixture",
            command=("python3", "tools/offline_resettable_fixture.py"),
            cwd=".",
        ),
        CommandCheck(
            name="commute_report_check",
            command=("python3", "tools/commute_report.py"),
            cwd=".",
        ),
        CommandCheck(
            name="set_gainstaging_fixture",
            command=("python3", "tools/offline_set_gainstaging.py"),
            cwd=".",
        ),
        CommandCheck(
            name="wingclientless_parser",
            command=("python3", "tools/wingclientless_parser.py"),
            cwd=".",
        ),
    ]

    validate_no_live_commands(checks)


def test_no_wing_release_check_can_disable_command_timeouts(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, command_timeout_sec=None)

    assert checks
    assert all(check.timeout_sec is None for check in checks)


def test_no_wing_release_check_quick_mode_still_runs_write_surface_scan(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, quick=True)

    check_names = [check.name for check in checks]
    assert "write_surface_static_scan" in check_names
    assert "memory_lint" in check_names
    assert "supervised_proof_artifact_report" in check_names
    assert check_names.index("write_surface_static_scan") < check_names.index("backend_pytest")
    assert check_names.index("offline_operator_validation_harness") < check_names.index(
        "supervised_proof_artifact_report"
    )
    assert check_names.index("supervised_proof_artifact_report") < check_names.index(
        "memory_lint"
    )
    assert check_names.index("memory_lint") < check_names.index("diff_check")
    assert any(
        "tests/test_write_surface_inventory_static_scan.py" in check.command
        for check in checks
        if check.name == "write_surface_static_scan"
    )
    report_check = next(check for check in checks if check.name == "supervised_proof_artifact_report")
    assert "tools/evaluate_supervised_proof_artifacts.py" in report_check.command
    assert "--quiet" in report_check.command


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


def test_no_wing_release_check_rejects_forbidden_live_command_in_check_name():
    checks = [
        CommandCheck(
            name="planned_set_gain_check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
        CommandCheck(
            name="planned_load_scene_check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "planned_set_gain_check:set_gain" in message
    assert "planned_load_scene_check:load_scene" in message


def test_no_wing_release_check_rejects_forbidden_live_tokens_case_insensitively():
    checks = [
        CommandCheck(
            name="Planned_Set_Gain_Check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
        CommandCheck(
            name="mixed_case_wrapper",
            command=("python3", "tools/LOAD_SNAP_PLAN.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "Planned_Set_Gain_Check:set_gain" in message
    assert "mixed_case_wrapper:load_snap" in message


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


def test_no_wing_release_check_rejects_unmute_as_live_write():
    checks = [
        CommandCheck(
            name="bad_unmute_runner",
            command=("python3", "tools/unmute_channels.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    assert "bad_unmute_runner:unmute" in str(excinfo.value)


def test_no_wing_release_check_rejects_mute_aliases_as_live_writes():
    checks = [
        CommandCheck(
            name="bad_set_mute_runner",
            command=("python3", "tools/apply_set_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_channel_mute_runner",
            command=("python3", "tools/apply_set_channel_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_bus_mute_runner",
            command=("python3", "tools/apply_set_bus_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_main_mute_runner",
            command=("python3", "tools/apply_set_main_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_mute_channel_runner",
            command=("python3", "tools/mute_channel_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_unmute_channel_runner",
            command=("python3", "tools/unmute_channel_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_set_mute_runner:set_mute" in message
    assert "bad_set_channel_mute_runner:set_channel_mute" in message
    assert "bad_set_bus_mute_runner:set_bus_mute" in message
    assert "bad_set_main_mute_runner:set_main_mute" in message
    assert "bad_mute_channel_runner:mute_channel" in message
    assert "bad_unmute_channel_runner:unmute_channel" in message


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


def test_no_wing_release_check_rejects_group_dca_matrix_balance_aliases():
    checks = [
        CommandCheck(
            name="bad_group_balance_runner",
            command=("python3", "tools/apply_set_group_fader_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_balance_runner",
            command=("python3", "tools/apply_set_matrix_fader_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_dca_level_runner",
            command=("python3", "tools/apply_set_dca_level_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_dca_mute_runner",
            command=("python3", "tools/apply_set_dca_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_group_mute_runner",
            command=("python3", "tools/apply_set_group_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_mute_runner",
            command=("python3", "tools/apply_set_matrix_mute_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_group_balance_runner:set_group_fader" in message
    assert "bad_matrix_balance_runner:set_matrix_fader" in message
    assert "bad_dca_level_runner:set_dca_level" in message
    assert "bad_dca_mute_runner:set_dca_mute" in message
    assert "bad_group_mute_runner:set_group_mute" in message
    assert "bad_matrix_mute_runner:set_matrix_mute" in message


def test_no_wing_release_check_rejects_bus_main_matrix_processing_aliases():
    checks = [
        CommandCheck(
            name="bad_bus_eq_runner",
            command=("python3", "tools/apply_set_bus_eq_on_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_main_eq_runner",
            command=("python3", "tools/apply_set_main_eq_band_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_eq_runner",
            command=("python3", "tools/apply_set_matrix_eq_band_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_bus_dynamics_runner",
            command=("python3", "tools/apply_set_bus_compressor_gain_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_bus_pan_runner",
            command=("python3", "tools/apply_set_bus_pan_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_main_pan_runner",
            command=("python3", "tools/apply_set_main_pan_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_bus_eq_runner:set_bus_eq_on" in message
    assert "bad_main_eq_runner:set_main_eq_band" in message
    assert "bad_matrix_eq_runner:set_matrix_eq_band" in message
    assert "bad_bus_dynamics_runner:set_bus_compressor_gain" in message
    assert "bad_bus_pan_runner:set_bus_pan" in message
    assert "bad_main_pan_runner:set_main_pan" in message


def test_no_wing_release_check_rejects_fx_return_and_send_aliases():
    checks = [
        CommandCheck(
            name="bad_fx_return_runner",
            command=("python3", "tools/apply_set_fx_return_level_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_channel_main_send_runner",
            command=("python3", "tools/apply_set_channel_main_send_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reverb_send_runner",
            command=("python3", "tools/apply_set_reverb_send_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_fx_return_runner:set_fx_return_level" in message
    assert "bad_channel_main_send_runner:set_channel_main_send" in message
    assert "bad_reverb_send_runner:set_reverb_send" in message


def test_no_wing_release_check_rejects_trim_and_preamp_aliases():
    checks = [
        CommandCheck(
            name="bad_trim_runner",
            command=("python3", "tools/apply_set_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_input_trim_runner",
            command=("python3", "tools/apply_set_input_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_channel_trim_runner",
            command=("python3", "tools/apply_set_channel_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_preamp_trim_runner",
            command=("python3", "tools/apply_set_preamp_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_preamp_gain_runner",
            command=("python3", "tools/apply_set_preamp_gain_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_trim_runner:set_trim" in message
    assert "bad_input_trim_runner:set_input_trim" in message
    assert "bad_channel_trim_runner:set_channel_trim" in message
    assert "bad_preamp_trim_runner:set_preamp_trim" in message
    assert "bad_preamp_gain_runner:set_preamp_gain" in message


def test_no_wing_release_check_rejects_channel_reset_variants():
    checks = [
        CommandCheck(
            name="bad_reset_eq_runner",
            command=("python3", "tools/reset_channel_eq.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_hpf_runner",
            command=("python3", "tools/reset_channel_hpf.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_dynamics_runner",
            command=("python3", "tools/reset_channel_dynamics.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_processing_runner",
            command=("python3", "tools/reset_channel_processing.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_reset_eq_runner:reset_channel_eq" in message
    assert "bad_reset_hpf_runner:reset_channel_hpf" in message
    assert "bad_reset_dynamics_runner:reset_channel_dynamics" in message
    assert "bad_reset_processing_runner:reset_channel_processing" in message


def test_no_wing_release_check_rejects_scene_and_snapshot_aliases():
    checks = [
        CommandCheck(
            name="bad_recall_snap_runner",
            command=("python3", "tools/recall_snap_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_store_snap_runner",
            command=("python3", "tools/store_snap_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_restore_snapshot_runner",
            command=("python3", "tools/restore_snapshot_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_load_scene_runner",
            command=("python3", "tools/load_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_save_scene_runner",
            command=("python3", "tools/save_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_recall_scene_runner",
            command=("python3", "tools/recall_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_restore_scene_runner",
            command=("python3", "tools/restore_scene_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_recall_snap_runner:recall_snap" in message
    assert "bad_store_snap_runner:store_snap" in message
    assert "bad_restore_snapshot_runner:restore_snapshot" in message
    assert "bad_load_scene_runner:load_scene" in message
    assert "bad_save_scene_runner:save_scene" in message
    assert "bad_recall_scene_runner:recall_scene" in message
    assert "bad_restore_scene_runner:restore_scene" in message


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
            "import time; print('started', flush=True); time.sleep(10)",
        ),
        cwd=str(Path.cwd()),
        timeout_sec=2.0,
    )

    result = run_command_check(check, tmp_path)

    assert result.status == "timeout"
    assert result.returncode is None
    assert result.detail == {"timeout_sec": 2.0, "reason": "command_timeout"}
    assert "started" in Path(result.stdout_log).read_text()
    assert "Command timed out after 2.0 seconds." in Path(result.stderr_log).read_text()


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
