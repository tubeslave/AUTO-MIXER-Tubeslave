import json
import sys
from pathlib import Path

import pytest

from backend.operator_mode_policy import MODE_SUPERVISED, build_operator_mode_status
from tools import no_wing_release_check as release_check
from tools.no_wing_release_check import (
    CheckResult,
    CommandCheck,
    autonomy_evidence_overlay_summary,
    autonomy_gap_rollup_summary,
    bus_group_main_dca_readback_contract_summary,
    build_checks,
    channel_basic_controls_readback_contract_summary,
    channel_processing_readback_contract_summary,
    current_supervised_live_write_kinds,
    fx_slots_inserts_readback_contract_summary,
    gain_proof_artifact_root_for_run,
    matrix_state_readback_contract_summary,
    next_wing_proof_plan_summary,
    proof_report_path_for_run,
    raw_mixer_readback_contract_summary,
    read_surface_fixture_evidence_summary,
    read_surface_remediation_plan_summary,
    routing_sends_readback_contract_summary,
    run_command_check,
    run_release_checks,
    supervised_gain_dry_run_summary,
    user_presets_readback_contract_summary,
    validate_repo_local_commands,
    validate_no_live_commands,
    validate_bus_group_main_dca_physical_readback_contract_snapshot,
    validate_channel_basic_controls_physical_readback_contract_snapshot,
    validate_channel_processing_physical_readback_contract_snapshot,
    validate_fx_slots_inserts_physical_readback_contract_snapshot,
    validate_matrix_state_physical_readback_contract_snapshot,
    validate_read_surface_fixture_evidence_snapshot,
    validate_raw_mixer_state_physical_readback_contract_snapshot,
    validate_read_surface_remediation_plan_snapshot,
    validate_routing_sends_physical_readback_contract_snapshot,
    validate_snapshots_scenes_physical_readback_contract_snapshot,
    validate_user_presets_physical_readback_contract_snapshot,
    validate_supervised_proof_report_autonomy_overlay,
    validate_supervised_proof_report_next_wing_plan,
    validate_supervised_proof_report_gain_dry_run,
    validate_supervised_proof_report_gap_rollup,
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
    command_tokens = [token for check in checks for token in check.command]
    assert "reset" not in command_tokens
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


def test_no_wing_release_check_allows_configured_git_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIXER_GIT_BIN", "/custom/git")

    checks = build_checks(run_dir=tmp_path, include_frontend=False)
    diff_check = next(check for check in checks if check.name == "diff_check")

    assert diff_check.command == ("/custom/git", "diff", "--check")


def test_no_wing_release_check_repo_local_command_plan_passes(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=True, include_package=True)

    validate_repo_local_commands(checks)


def test_no_wing_release_check_rejects_external_project_command_paths(tmp_path):
    repo_root = tmp_path / "AUTO-MIXER-Tubeslave-main"
    external_project = tmp_path / "paperclip"
    repo_root.mkdir()
    external_project.mkdir()
    checks = [
        CommandCheck(
            name="bad_cwd",
            command=("python3", "tools/offline_check.py"),
            cwd=str(external_project),
        ),
        CommandCheck(
            name="bad_absolute_script",
            command=("python3", str(external_project / "runner.py")),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="bad_relative_script",
            command=("python3", "../paperclip/runner.py"),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="bad_env_path",
            command=("python3", "tools/offline_check.py"),
            cwd=str(repo_root),
            env={"PYTHONPATH": "../paperclip"},
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_repo_local_commands(checks, repo_root=repo_root)

    message = str(excinfo.value)
    assert "repo boundary" in message
    assert "bad_cwd:cwd:" in message
    assert "bad_absolute_script:command_path:" in message
    assert "bad_relative_script:command_path:../paperclip/runner.py" in message
    assert "bad_env_path:env_path:PYTHONPATH" in message


def test_no_wing_release_check_derives_supervised_boundary_from_operator_policy():
    policy_status = build_operator_mode_status(
        mode=MODE_SUPERVISED,
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=False,
    )

    assert current_supervised_live_write_kinds() == tuple(
        policy_status["capabilities"]["allowed_live_write_kinds"]
    )
    assert current_supervised_live_write_kinds() == ("fader", "gain")


def test_no_wing_release_check_validates_read_surface_fixture_evidence_snapshot():
    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["snapshot_status"] == "fixture_evidence_incomplete"
    assert result.detail["surface_count"] >= 1
    assert result.detail["surface_with_fixture_count"] >= 1
    assert result.detail["status_counts"]["partial_fixture_evidence"] >= 1
    assert result.detail["physical_fixture_missing_surface_ids"]
    assert (
        result.detail["next_physical_readback_surface_id"]
        == result.detail["physical_fixture_missing_surface_ids"][0]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_read_surface_fixture_summary_row_drift(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))
    corrupted["summary"]["surface_count"] += 1
    corrupted["summary"]["surface_with_fixture_count"] = 0
    corrupted["summary"]["status_counts"]["partial_fixture_evidence"] = 0
    corrupted["summary"]["status_counts"]["fixture_evidence_present"] = "not-an-int"
    corrupted["summary"]["physical_fixture_missing_surface_ids"] = []
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["surfaces"][0]["status"] = "unsupported_fixture_status"

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_raw_mixer_state_cache_status_unknown" in result.detail["violations"]
    assert "read_surface_fixture_status_count_fixture_evidence_present_not_integer" in result.detail["violations"]
    assert "read_surface_fixture_surface_count_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_surface_with_fixture_count_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_status_counts_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_physical_missing_surface_ids_mismatch" in result.detail["violations"]
    assert (
        "read_surface_fixture_next_physical_readback_surface_id_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_read_surface_fixture_stale_declarations(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))

    channel_row = next(
        row
        for row in corrupted["surfaces"]
        if row["surface_id"] == "channel_identity_and_basic_controls"
    )
    channel_row["fixture_paths"] = ["tests/missing_read_surface_fixture.py"]
    channel_row["test_names"] = ["test_missing_read_surface_fixture"]

    raw_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "raw_mixer_state_cache")
    raw_row["fixture_paths"] = ["backend/server.py"]
    raw_row["test_names"] = ["test_read_surface_inventory_is_read_only_and_unique"]

    dlive_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "dlive_state")
    dlive_row["status"] = "missing_fixture_evidence"

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_fixture_path_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_test_name_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_raw_mixer_state_cache_fixture_path_not_tests" in result.detail["violations"]
    assert "read_surface_fixture_row_raw_mixer_state_cache_test_name_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_fixture_paths" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_test_names" in result.detail["violations"]


def test_no_wing_release_check_rejects_read_surface_contract_gap_inconsistency(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))

    channel_row = next(
        row
        for row in corrupted["surfaces"]
        if row["surface_id"] == "channel_identity_and_basic_controls"
    )
    channel_row["status"] = "partial_fixture_evidence"
    channel_row["covered_contracts"].append("physical_readback_fixture")
    channel_row["remaining_gaps"] = [
        gap
        for gap in channel_row["remaining_gaps"]
        if gap not in {"physical_readback_fixture_missing", "hardware_parity_unproven"}
    ]

    matrix_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "matrix_state")
    matrix_row["status"] = "fixture_evidence_present"
    matrix_row["covered_contracts"] = [
        contract
        for contract in matrix_row["covered_contracts"]
        if contract != "physical_readback_fixture"
    ]
    matrix_row["remaining_gaps"].append("physical_readback_fixture_missing")
    matrix_row["remaining_gaps"].append("hardware_parity_unproven")

    dlive_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "dlive_state")
    dlive_row["status"] = "missing_fixture_evidence"
    dlive_row["covered_contracts"] = ["cached_dlive_midi_state"]
    dlive_row["fixture_paths"] = []
    dlive_row["test_names"] = []

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_has_physical_contract" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_missing_physical_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_missing_hardware_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_missing_physical_contract" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_has_physical_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_has_hardware_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_covered_contracts" in result.detail["violations"]


def test_no_wing_release_check_validates_read_surface_remediation_plan_alignment():
    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "passed"
    assert result.detail["plan_status"] == "read_remediation_required"
    assert result.detail["physical_fixture_missing_surface_ids"]
    assert (
        result.detail["next_physical_readback_surface_id"]
        == result.detail["expected_next_physical_readback_surface_id"]
    )
    assert (
        result.detail["next_physical_readback_action_id"]
        == result.detail["expected_next_physical_readback_action_id"]
    )
    assert result.detail["next_physical_readback_target_action_ids"] == [
        result.detail["expected_next_physical_readback_action_id"]
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_read_surface_remediation_target_drift(monkeypatch):
    fixture_snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    plan = release_check.build_read_surface_remediation_plan(
        fixture_evidence_snapshot=fixture_snapshot
    )
    corrupted = json.loads(json.dumps(plan))
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["next_physical_readback_action_id"] = "remediate_dlive_state"
    next_action = next(
        action
        for action in corrupted["actions"]
        if action["id"] == "remediate_raw_mixer_state_cache"
    )
    next_action["next_physical_readback_target"] = False
    next_action["missing_contracts"] = []

    monkeypatch.setattr(
        release_check,
        "build_read_surface_remediation_plan",
        lambda **_: corrupted,
    )

    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "failed"
    assert "read_surface_remediation_next_target_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_id_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_surface_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_target_flag_missing" in result.detail["violations"]


def test_no_wing_release_check_rejects_multiple_remediation_next_target_flags(monkeypatch):
    fixture_snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    plan = release_check.build_read_surface_remediation_plan(
        fixture_evidence_snapshot=fixture_snapshot
    )
    corrupted = json.loads(json.dumps(plan))
    expected_next_action_id = corrupted["summary"]["next_physical_readback_action_id"]
    extra_action = next(
        action
        for action in corrupted["actions"]
        if action["id"] != expected_next_action_id
    )
    extra_action["next_physical_readback_target"] = True

    monkeypatch.setattr(
        release_check,
        "build_read_surface_remediation_plan",
        lambda **_: corrupted,
    )

    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "failed"
    assert (
        "read_surface_remediation_next_target_action_flags_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_raw_mixer_readback_contract_snapshot():
    result = validate_raw_mixer_state_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert result.detail["surface_id"] == "raw_mixer_state_cache"
    assert result.detail["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert result.detail["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["fixture_next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["remediation_next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "manual_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_raw_mixer_contract_drift(monkeypatch):
    contract = release_check.build_raw_mixer_state_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "raw_mixer_state_physical_readback.v0"
    corrupted["surface_id"] = "channel_identity_and_basic_controls"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True

    monkeypatch.setattr(
        release_check,
        "build_raw_mixer_state_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_raw_mixer_state_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "raw_mixer_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_live_write_not_false" in result.detail["violations"]
    assert "raw_mixer_readback_required_top_level_fields_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_allows_cached_source" in result.detail["violations"]
    assert "raw_mixer_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_forbidden_mutation_fields_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_next_target_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_live_authority_not_false" in result.detail["violations"]


def test_no_wing_release_check_validates_channel_basic_controls_readback_contract_snapshot():
    result = validate_channel_basic_controls_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "channel_basic_controls_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_identity_and_basic_controls"
    assert result.detail["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert result.detail["physical_readback_surface_id"] == "channel_identity_and_basic_controls"
    assert result.detail["physical_readback_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_channel_statuses"]
    assert "manual_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "cache_matches_physical_readback" in result.detail["required_coverage_flags"]
    assert result.detail["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_basic_controls_contract_drift(monkeypatch):
    contract = release_check.build_channel_basic_controls_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "channel_basic_controls_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_coverage_flags"] = []
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_channel_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["fader"]

    monkeypatch.setattr(
        release_check,
        "build_channel_basic_controls_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_channel_basic_controls_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "channel_basic_controls_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_basic_controls_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_controls_readback_required_control_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_controls_readback_required_coverage_flags_mismatch"
        in result.detail["violations"]
    )
    assert "channel_basic_controls_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_allows_cached_source" in result.detail["violations"]
    assert "channel_basic_controls_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert (
        "channel_basic_controls_readback_forbidden_mutation_fields_mismatch"
        in result.detail["violations"]
    )
    assert "channel_basic_controls_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_contract_live_authority_not_false" in result.detail["violations"]
    assert "channel_basic_controls_promotion_family_missing:gain" in result.detail["violations"]
    assert "channel_basic_controls_promotion_family_missing:mute" in result.detail["violations"]


def test_no_wing_release_check_validates_channel_processing_readback_contract_snapshot():
    result = validate_channel_processing_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "channel_processing_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_eq_dynamics_gate"
    assert result.detail["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert result.detail["physical_readback_surface_id"] == "channel_eq_dynamics_gate"
    assert result.detail["physical_readback_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_channel_statuses"]
    assert "eq_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_hpf_fields"] == ["enabled", "frequency_hz"]
    assert result.detail["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in result.detail["required_compressor_fields"]
    assert "range_db" in result.detail["required_gate_fields"]
    assert result.detail["minimum_eq_band_count"] == 4
    assert result.detail["promotion_families_requiring_surface"][:4] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_processing_contract_drift(monkeypatch):
    contract = release_check.build_channel_processing_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "channel_processing_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_eq_band_fields"] = ["band"]
    corrupted["artifact_contract"]["required_compressor_fields"] = ["enabled"]
    corrupted["artifact_contract"]["required_gate_fields"] = ["enabled"]
    corrupted["artifact_contract"]["minimum_eq_band_count"] = 2
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_channel_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["eq"]

    monkeypatch.setattr(
        release_check,
        "build_channel_processing_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_channel_processing_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "channel_processing_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_processing_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_readback_required_eq_band_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_readback_required_compressor_fields_mismatch"
        in result.detail["violations"]
    )
    assert "channel_processing_readback_required_gate_fields_mismatch" in result.detail["violations"]
    assert (
        "channel_processing_readback_artifact_minimum_eq_band_count_mismatch"
        in result.detail["violations"]
    )
    assert "channel_processing_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_allows_cached_source" in result.detail["violations"]
    assert "channel_processing_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_forbidden_mutation_fields_mismatch" in result.detail["violations"]
    assert "channel_processing_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "channel_processing_contract_live_authority_not_false" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:compressor" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:gate" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:deessing" in result.detail["violations"]


def test_no_wing_release_check_validates_bus_group_main_dca_readback_contract_snapshot():
    result = validate_bus_group_main_dca_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "bus_group_main_dca_physical_readback.v1"
    assert result.detail["surface_id"] == "bus_group_main_dca_state"
    assert result.detail["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert result.detail["physical_readback_surface_id"] == "bus_group_main_dca_state"
    assert result.detail["physical_readback_action_id"] == "remediate_bus_group_main_dca_state"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_surface_statuses"]
    assert "dca_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_bus_control_fields"] == [
        "fader_db",
        "mute",
        "pan",
        "width",
        "mono",
    ]
    assert "true_peak_dbtp" in result.detail["required_metering_fields"]
    assert "member_buses" in result.detail["required_dca_assignment_fields"]
    assert result.detail["minimum_bus_count"] == 1
    assert result.detail["minimum_main_count"] == 1
    assert result.detail["minimum_dca_count"] == 1
    assert result.detail["promotion_families_requiring_surface"][:3] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_bus_group_main_dca_contract_drift(monkeypatch):
    contract = release_check.build_bus_group_main_dca_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "bus_group_main_dca_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_bus_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_metering_fields"] = ["lufs"]
    corrupted["artifact_contract"]["minimum_bus_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_surface_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["dca"]

    monkeypatch.setattr(
        release_check,
        "build_bus_group_main_dca_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_bus_group_main_dca_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "bus_group_main_dca_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "bus_group_main_dca_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "bus_group_main_dca_readback_required_bus_control_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "bus_group_main_dca_readback_required_metering_fields_mismatch"
        in result.detail["violations"]
    )
    assert "bus_group_main_dca_readback_artifact_minimum_bus_count_mismatch" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_allows_cached_source" in result.detail["violations"]
    assert "bus_group_main_dca_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_contract_live_authority_not_false" in result.detail["violations"]
    assert "bus_group_main_dca_promotion_family_missing:bus_group_matrix" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_promotion_family_missing:master_bus" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_matrix_state_readback_contract_snapshot():
    result = validate_matrix_state_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "matrix_state_physical_readback.v1"
    assert result.detail["surface_id"] == "matrix_state"
    assert result.detail["remediation_action_id"] == "remediate_matrix_state"
    assert result.detail["physical_readback_surface_id"] == "matrix_state"
    assert result.detail["physical_readback_action_id"] == "remediate_matrix_state"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_matrix_statuses"]
    assert "matrix_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert result.detail["required_eq_band_fields"] == [
        "band",
        "frequency_hz",
        "gain_db",
        "q",
    ]
    assert "gain_reduction_db" in result.detail["required_dynamics_fields"]
    assert "post_insert_weight" in result.detail["required_insert_fields"]
    assert "true_peak_dbtp" in result.detail["required_metering_fields"]
    assert result.detail["minimum_matrix_count"] == 1
    assert result.detail["minimum_eq_band_count"] == 8
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_matrix_state_contract_drift(monkeypatch):
    contract = release_check.build_matrix_state_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "matrix_state_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_metering_fields"] = ["lufs"]
    corrupted["artifact_contract"]["minimum_matrix_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_matrix_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_matrix_state_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_matrix_state_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "matrix_state_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_contract_live_write_not_false" in result.detail["violations"]
    assert "matrix_state_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_required_control_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_required_metering_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_artifact_minimum_matrix_count_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_allows_cached_source" in result.detail["violations"]
    assert "matrix_state_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "matrix_state_contract_live_authority_not_false" in result.detail["violations"]
    assert "matrix_state_promotion_family_missing:bus_group_matrix" in result.detail[
        "violations"
    ]
    assert "matrix_state_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_fx_slots_inserts_readback_contract_snapshot():
    result = validate_fx_slots_inserts_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "fx_slots_inserts_physical_readback.v1"
    assert result.detail["surface_id"] == "fx_slots_and_inserts"
    assert result.detail["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert result.detail["physical_readback_surface_id"] == "fx_slots_and_inserts"
    assert result.detail["physical_readback_action_id"] == "remediate_fx_slots_and_inserts"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "fx_parameter_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "parameter_schema" in result.detail["required_fx_slot_fields"]
    assert "safe_range" in result.detail["required_parameter_fields"]
    assert "physical_model_schema_verified" in result.detail[
        "required_parameter_schema_fields"
    ]
    assert "scoped_diff_marker" in result.detail["required_insert_coverage_flags"]
    assert result.detail["minimum_fx_slot_count"] == 1
    assert result.detail["minimum_fx_parameter_count"] == 4
    assert result.detail["minimum_insert_count"] == 1
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_fx_slots_inserts_contract_drift(monkeypatch):
    contract = release_check.build_fx_slots_inserts_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "fx_slots_inserts_physical_readback.v0"
    corrupted["surface_id"] = "matrix_state"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_fx_slot_fields"] = ["slot"]
    corrupted["artifact_contract"]["required_parameter_fields"] = ["value"]
    corrupted["artifact_contract"]["minimum_fx_parameter_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "routing_and_sends"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_fx_slots_inserts_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_fx_slots_inserts_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "fx_slots_inserts_readback_contract_schema_id_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_contract_surface_id_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_contract_live_write_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_fx_slot_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_parameter_fields_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "fx_slots_inserts_readback_artifact_minimum_fx_parameter_count_mismatch"
        in result.detail["violations"]
    )
    assert "fx_slots_inserts_readback_allowed_sources_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_allows_cached_source" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_accepted_statuses_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_contract_surface_summary_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_contract_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_promotion_family_missing:mix_balance" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_routing_sends_readback_contract_snapshot():
    result = validate_routing_sends_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "routing_sends_physical_readback.v1"
    assert result.detail["surface_id"] == "routing_and_sends"
    assert result.detail["remediation_action_id"] == "remediate_routing_and_sends"
    assert result.detail["physical_readback_surface_id"] == "routing_and_sends"
    assert result.detail["physical_readback_action_id"] == "remediate_routing_and_sends"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "routing_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "wing_routing_hint" in result.detail["required_static_route_fields"]
    assert "pre_on" in result.detail["required_bus_send_fields"]
    assert "main" in result.detail["required_main_send_fields"]
    assert "target_matrix" in result.detail["required_matrix_send_fields"]
    assert "source_channel" in result.detail["required_output_route_fields"]
    assert "cache_matches_physical_readback" in result.detail["required_send_coverage_flags"]
    assert (
        "cache_matches_physical_readback"
        in result.detail["required_output_route_coverage_flags"]
    )
    assert result.detail["minimum_static_dante_route_count"] == 1
    assert result.detail["minimum_channel_bus_send_count"] == 1
    assert result.detail["minimum_channel_main_send_count"] == 1
    assert result.detail["minimum_channel_matrix_send_count"] == 1
    assert result.detail["minimum_output_route_count"] == 1
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_routing_sends_contract_drift(monkeypatch):
    contract = release_check.build_routing_sends_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "routing_sends_physical_readback.v0"
    corrupted["surface_id"] = "matrix_state"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_static_route_fields"] = ["start"]
    corrupted["artifact_contract"]["required_bus_send_fields"] = ["channel"]
    corrupted["artifact_contract"]["minimum_channel_bus_send_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_routing_sends_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_routing_sends_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "routing_sends_readback_contract_schema_id_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_contract_surface_id_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_contract_live_write_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_static_route_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_bus_send_fields_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "routing_sends_readback_artifact_minimum_channel_bus_send_count_mismatch"
        in result.detail["violations"]
    )
    assert "routing_sends_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "routing_sends_readback_allows_cached_source" in result.detail["violations"]
    assert "routing_sends_readback_accepted_statuses_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_contract_surface_summary_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_contract_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]
    assert "routing_sends_promotion_family_missing:scene_preset" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_snapshots_scenes_readback_contract_snapshot():
    result = validate_snapshots_scenes_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "snapshots_scenes_physical_readback.v1"
    assert result.detail["surface_id"] == "snapshots_scenes"
    assert result.detail["remediation_action_id"] == "remediate_snapshots_scenes"
    assert result.detail["physical_readback_surface_id"] == "snapshots_scenes"
    assert result.detail["physical_readback_action_id"] == "remediate_snapshots_scenes"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "scene_recall_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "active_snapshot_index" in result.detail["required_active_scene_fields"]
    assert "scene_content_fingerprint" in result.detail["required_snapshot_fields"]
    assert "rollback_reference" in result.detail["required_coverage_flags"]
    assert "scene_load_allowed" in result.detail["required_false_safety_flags"]
    assert result.detail["minimum_snapshot_count"] == 1
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_user_presets_readback_contract_snapshot():
    result = validate_user_presets_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "user_presets_physical_readback.v1"
    assert result.detail["surface_id"] == "user_presets"
    assert result.detail["remediation_action_id"] == "remediate_user_presets"
    assert result.detail["physical_readback_surface_id"] == "user_presets"
    assert result.detail["physical_readback_action_id"] == "remediate_user_presets"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "preset_store_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "preset_content_fingerprint" in result.detail["required_preset_fields"]
    assert "rollback_reference" in result.detail["required_coverage_flags"]
    assert "preset_recall_allowed" in result.detail["required_false_safety_flags"]
    assert result.detail["minimum_preset_count"] == 1
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_quick_mode_still_runs_write_surface_scan(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, quick=True)

    check_names = [check.name for check in checks]
    assert "write_surface_static_scan" in check_names
    assert "read_surface_fixture_evidence" in check_names
    assert "memory_lint" in check_names
    assert "gain_dry_run_proof" in check_names
    assert "supervised_proof_artifact_report" in check_names
    assert check_names.index("write_surface_static_scan") < check_names.index("backend_pytest")
    assert check_names.index("read_surface_fixture_evidence") < check_names.index("backend_pytest")
    assert check_names.index("write_surface_static_scan") < check_names.index(
        "read_surface_fixture_evidence"
    )
    assert check_names.index("offline_operator_validation_harness") < check_names.index(
        "gain_dry_run_proof"
    )
    assert check_names.index("gain_dry_run_proof") < check_names.index(
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
    fixture_check = next(check for check in checks if check.name == "read_surface_fixture_evidence")
    assert "tests/test_read_surface_fixture_evidence.py" in fixture_check.command
    assert "tests/test_raw_mixer_state_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_channel_basic_controls_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_channel_processing_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_bus_group_main_dca_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_matrix_state_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_fx_slots_inserts_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_routing_sends_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_snapshots_scenes_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_user_presets_physical_readback_contract.py" in fixture_check.command
    report_check = next(check for check in checks if check.name == "supervised_proof_artifact_report")
    assert "tools/evaluate_supervised_proof_artifacts.py" in report_check.command
    assert "--quiet" in report_check.command
    assert "--require-physical" in report_check.command
    assert str(proof_report_path_for_run(tmp_path)) in report_check.command
    assert str(gain_proof_artifact_root_for_run(tmp_path)) in report_check.command
    assert report_check.command.count("--artifact-root") == 2
    gain_check = next(check for check in checks if check.name == "gain_dry_run_proof")
    assert "tools/gain_proof_dry_run_runner.py" in gain_check.command
    assert str(gain_proof_artifact_root_for_run(tmp_path)) in gain_check.command


def valid_proof_report_with_gap_rollup() -> dict:
    return {
        "type": "supervised_proof_artifact_report",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "value_type_counts": {"gain": 1},
            "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
            "autonomy_gap_rollup": {
                "read_only": True,
                "live_write": False,
                "live_widening_allowed": False,
                "full_autonomous_ready": False,
                "source": "autonomy_mode_transition_gate.rows[mode=auto]",
                "full_autonomous_blocked": True,
                "full_auto_gap_bucket_counts": {"read_surface": 1, "evidence_requirement": 12},
                "evidence_requirement_ids_by_action_family": {
                    "bus_group_matrix": [
                        "bus_group_assignment_cache_freshness",
                        "bus_group_fader_readback",
                        "matrix_send_readback",
                        "main_bus_true_peak_headroom",
                        "bus_group_physical_fixture",
                        "hardware_readback_and_rollback_proof",
                    ],
                    "mute": [
                        "current_mute_cache_freshness",
                        "mute_owner_intent_evidence",
                        "signal_return_or_idle_window_evidence",
                        "manual_mute_authority_review",
                        "mute_physical_fixture",
                        "hardware_readback_rollback_and_emergency_proof",
                    ],
                },
                "next_required_evidence_by_action_family": {
                    "bus_group_matrix": [
                        "Add bus/group/matrix safety review before promotion.",
                    ],
                    "mute": [
                        "Add fresh mute-state intent and physical proof before promotion.",
                    ],
                },
                "supervised_current_live_write_kinds": ["fader", "gain"],
                "next_required_evidence": [
                    "Reduce every Full Autonomous readiness gap bucket to zero before reviewing auto mode.",
                    "Keep live apply limited to current supervised write kinds until physical readback and rollback evidence exists.",
                ],
                "claim": (
                    "Full Autonomous remains blocked; this rollup is report-only and does not widen "
                    "the current supervised fader/gain live-write boundary."
                ),
            }
        },
        "rows": [
            {
                "proof_id": "gain-proof-dry-run",
                "value_type": "gain",
                "channel": 1,
                "proof_mode": "dry_run",
                "status": "dry_run_ready",
                "promotion_allowed": False,
                "autonomous_ready": False,
                "blockers": ["physical_proof_required_for_promotion"],
                "evidence": {
                    "pre_read_status": "verified_from_physical_readback",
                    "pre_read_source": "fake_mixer",
                    "readback_status": "verified_from_physical_readback",
                    "readback_source": "fake_mixer",
                    "rollback_readback_status": "verified_from_physical_readback",
                    "rollback_readback_source": "fake_mixer",
                    "operator_approval_recorded": True,
                    "idempotency_key_present": True,
                },
            }
        ],
        "autonomy_mode_transition_gate": {
            "summary": {
                "autonomous_apply_allowed_modes": [],
                "blocked_modes": ["auto"],
            }
        },
        "autonomy_evidence_overlay": {
            "type": "autonomy_evidence_overlay",
            "status": "dry_run_only",
            "safety": {
                "read_only": True,
                "live_write": False,
                "live_widening_allowed": False,
                "automatic_safety_update_allowed": False,
                "full_autonomous_ready": False,
                "operator_required": True,
            },
            "summary": {
                "target_count": 2,
                "status_counts": {
                    "dry_run_only": 1,
                    "manual_safety_review_candidate": 0,
                    "no_current_evidence": 1,
                    "remediation_required": 0,
                },
                "manual_safety_review_candidate_target_ids": [],
                "remediation_required_target_ids": [],
                "dry_run_only_target_ids": ["gain_trim_supervised_promotion"],
                "no_current_evidence_target_ids": ["fader_supervised_promotion"],
                "autonomous_ready_target_ids": [],
                "global_blockers": [
                    "overlay_does_not_modify_safety_rows",
                    "manual_safety_review_required_before_promotion",
                    "full_autonomous_mode_remains_blocked",
                ],
            },
            "rows": [
                {
                    "id": "fader_autonomy_evidence_overlay",
                    "target_id": "fader_supervised_promotion",
                    "value_type": "fader",
                    "status": "no_current_evidence",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": True,
                    "autonomous_ready": False,
                    "proof_artifact_count": 0,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 0,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": [],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["proof_artifact_missing"],
                    "next_required_actions": ["Run one physical supervised fader proof."],
                },
                {
                    "id": "gain_autonomy_evidence_overlay",
                    "target_id": "gain_trim_supervised_promotion",
                    "value_type": "gain",
                    "status": "dry_run_only",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": False,
                    "autonomous_ready": False,
                    "proof_artifact_count": 1,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 1,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["physical_proof_required_for_promotion"],
                    "next_required_actions": ["Run one small supervised gain proof on WING."],
                },
            ],
        },
        "next_wing_proof_plan": {
            "type": "supervised_wing_proof_plan",
            "status": "operator_proof_required",
            "safety": {
                "read_only": True,
                "live_write": False,
                "execution_allowed_by_plan": False,
                "live_widening_allowed": False,
                "automatic_safety_update_allowed": False,
                "full_autonomous_ready": False,
                "operator_required": True,
            },
            "summary": {
                "step_count": 1,
                "planned_value_types": ["gain"],
                "planned_channels": [3],
                "global_blockers": [
                    "operator_must_approve_and_execute_manually",
                    "proof_plan_does_not_contact_mixer",
                ],
            },
            "steps": [
                {
                    "id": "gain_physical_proof_ch3",
                    "value_type": "gain",
                    "channel": 3,
                    "delta_db": -0.25,
                    "live_execution_required": True,
                    "execution_allowed_by_plan": False,
                    "operator_approval_required": True,
                    "command": ["python", "tools/wing_operator_proposal_validation_runner.py"],
                    "rationale": "physical proof required",
                    "prerequisites": ["operator approval"],
                    "abort_conditions": ["operator concern"],
                    "expected_artifacts": ["summary.json"],
                    "evidence_blockers_to_review": ["physical_candidate_missing"],
                }
            ],
        },
    }


def test_no_wing_release_check_validates_proof_report_gap_rollup_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "passed"
    assert result.detail["autonomy_gap_rollup_present"] is True
    assert result.detail["full_auto_gap_bucket_counts"] == {
        "read_surface": 1,
        "evidence_requirement": 12,
    }
    assert result.detail["evidence_requirement_ids_by_action_family"]["bus_group_matrix"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert result.detail["evidence_requirement_ids_by_action_family"]["mute"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert "bus_group_matrix" in result.detail["next_required_evidence_by_action_family"]
    assert result.detail["supervised_current_live_write_kinds"] == ["fader", "gain"]
    assert result.detail["policy_supervised_live_write_kinds"] == ["fader", "gain"]


def test_no_wing_release_check_validates_gain_dry_run_report_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_gain_dry_run(report_path)

    assert result.status == "passed"
    assert result.detail["gain_dry_run_row_count"] == 1
    assert result.detail["dry_run_ready_proof_ids"] == ["gain-proof-dry-run"]
    assert result.detail["value_type_counts"] == {"gain": 1}


def test_no_wing_release_check_validates_autonomy_evidence_overlay_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "passed"
    assert result.detail["overlay_status"] == "dry_run_only"
    assert result.detail["row_count"] == 2
    assert result.detail["dry_run_only_target_ids"] == ["gain_trim_supervised_promotion"]
    assert result.detail["no_current_evidence_target_ids"] == ["fader_supervised_promotion"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_next_wing_proof_plan_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "passed"
    assert result.detail["plan_status"] == "operator_proof_required"
    assert result.detail["step_count"] == 1
    assert result.detail["planned_value_types"] == ["gain"]
    assert result.detail["planned_channels"] == [3]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_next_wing_proof_plan_that_grants_execution(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    plan = payload["next_wing_proof_plan"]
    plan["safety"]["execution_allowed_by_plan"] = True
    plan["safety"]["live_write"] = True
    plan["safety"]["operator_required"] = False
    plan["summary"]["global_blockers"] = []
    plan["steps"][0]["execution_allowed_by_plan"] = True
    plan["steps"][0]["operator_approval_required"] = False
    plan["steps"][0]["delta_db"] = 0.25
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert "next_wing_proof_plan_execution_allowed_by_plan_not_false" in result.detail["violations"]
    assert "next_wing_proof_plan_live_write_not_false" in result.detail["violations"]
    assert "next_wing_proof_plan_operator_required_not_true" in result.detail["violations"]
    assert "next_wing_proof_plan_operator_blocker_missing" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_execution_allowed" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_operator_approval_not_required" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_delta_not_safe_negative_proof" in result.detail["violations"]


def test_no_wing_release_check_rejects_autonomy_overlay_that_claims_authority(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    overlay = payload["autonomy_evidence_overlay"]
    overlay["safety"]["live_write"] = True
    overlay["safety"]["live_widening_allowed"] = True
    overlay["safety"]["automatic_safety_update_allowed"] = True
    overlay["safety"]["full_autonomous_ready"] = True
    overlay["safety"]["operator_required"] = False
    overlay["summary"]["autonomous_ready_target_ids"] = ["gain_trim_supervised_promotion"]
    overlay["summary"]["global_blockers"] = []
    overlay["rows"][1]["autonomous_ready"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "failed"
    assert "autonomy_overlay_live_write_not_false" in result.detail["violations"]
    assert "autonomy_overlay_live_widening_allowed_not_false" in result.detail["violations"]
    assert "autonomy_overlay_automatic_safety_update_allowed_not_false" in result.detail["violations"]
    assert "autonomy_overlay_full_autonomous_ready_not_false" in result.detail["violations"]
    assert "autonomy_overlay_operator_required_not_true" in result.detail["violations"]
    assert "autonomy_overlay_autonomous_ready_targets_not_empty" in result.detail["violations"]
    assert "autonomy_overlay_row_1_autonomous_ready_not_false" in result.detail["violations"]


def test_no_wing_release_check_rejects_autonomy_overlay_summary_row_drift(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    overlay = payload["autonomy_evidence_overlay"]
    overlay["summary"]["target_count"] = 3
    overlay["summary"]["status_counts"]["dry_run_only"] = 2
    overlay["summary"]["dry_run_only_target_ids"] = [
        "gain_trim_supervised_promotion",
        "fader_supervised_promotion",
    ]
    overlay["summary"]["no_current_evidence_target_ids"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "failed"
    assert "autonomy_overlay_target_count_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_status_counts_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_dry_run_only_target_ids_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_no_current_evidence_target_ids_mismatch" in result.detail["violations"]


def test_no_wing_release_check_rejects_gain_dry_run_report_that_claims_promotion(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["rows"][0]["promotion_allowed"] = True
    payload["rows"][0]["autonomous_ready"] = True
    payload["rows"][0]["blockers"] = []
    payload["rows"][0]["evidence"]["pre_read_source"] = "physical_readback"
    payload["summary"]["dry_run_ready_proof_ids"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_gain_dry_run(report_path)

    assert result.status == "failed"
    assert "summary_dry_run_ready_proof_ids_missing" in result.detail["violations"]
    assert "gain_dry_run_promotion_allowed_not_false" in result.detail["violations"]
    assert "gain_dry_run_autonomous_ready_not_false" in result.detail["violations"]
    assert "gain_dry_run_physical_proof_blocker_missing" in result.detail["violations"]
    assert "gain_dry_run_pre_read_source_not_fake_mixer" in result.detail["violations"]


def test_no_wing_release_check_rejects_missing_proof_report_gap_rollup(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(
        json.dumps(
            {
                "type": "supervised_proof_artifact_report",
                "summary": {},
                "autonomy_mode_transition_gate": {
                    "summary": {"autonomous_apply_allowed_modes": [], "blocked_modes": ["auto"]}
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "failed"
    assert "autonomy_gap_rollup_read_only_not_true" in result.detail["violations"]
    assert "autonomy_gap_rollup_bucket_counts_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_source_not_auto_mode_gate_row" in result.detail["violations"]
    assert "autonomy_gap_rollup_evidence_by_family_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_next_by_family_missing" in result.detail["violations"]


def test_no_wing_release_check_rejects_misleading_proof_report_gap_rollup_claims(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    rollup = payload["summary"]["autonomy_gap_rollup"]
    rollup["source"] = "manual_summary"
    rollup["claim"] = "Full Autonomous is ready for wider live apply."
    rollup["next_required_evidence"] = []
    rollup["supervised_current_live_write_kinds"] = ["fader", "gain", "eq"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "failed"
    assert "autonomy_gap_rollup_source_not_auto_mode_gate_row" in result.detail["violations"]
    assert "autonomy_gap_rollup_claim_does_not_preserve_auto_block" in result.detail["violations"]
    assert "autonomy_gap_rollup_claim_does_not_preserve_live_boundary" in result.detail["violations"]
    assert "autonomy_gap_rollup_next_required_evidence_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_unexpected_supervised_live_write_kinds" in result.detail["violations"]
    assert result.detail["unexpected_supervised_live_write_kinds"] == ["eq"]


def test_no_wing_release_check_runs_gap_rollup_schema_after_report(tmp_path, monkeypatch):
    def fake_free_port(port: int, host: str = "127.0.0.1") -> CheckResult:
        return CheckResult(
            name="backend_port_free",
            status="passed",
            started_at="2026-06-05T00:00:00Z",
            finished_at="2026-06-05T00:00:00Z",
            duration_sec=0.0,
            detail={"host": host, "port": port, "listener_detected": False},
        )

    def fake_command_check(check: CommandCheck, log_dir: Path) -> CheckResult:
        if check.name == "supervised_proof_artifact_report":
            report_path = proof_report_path_for_run(log_dir.parent)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")
        return CheckResult(
            name=check.name,
            status="passed",
            started_at="2026-06-05T00:00:00Z",
            finished_at="2026-06-05T00:00:00Z",
            duration_sec=0.0,
            command=check.shell_display(),
            cwd=check.cwd,
            returncode=0,
        )

    monkeypatch.setattr(release_check, "check_port_free", fake_free_port)
    monkeypatch.setattr(release_check, "run_command_check", fake_command_check)

    summary = run_release_checks(
        artifact_root=tmp_path,
        include_frontend=False,
        quick=True,
        command_timeout_sec=None,
    )

    result_names = [result["name"] for result in summary["results"]]
    assert summary["status"] == "passed"
    assert "read_surface_fixture_evidence_schema" in result_names
    assert "read_surface_remediation_plan_schema" in result_names
    assert "raw_mixer_state_physical_readback_contract_schema" in result_names
    assert "channel_basic_controls_physical_readback_contract_schema" in result_names
    assert "channel_processing_physical_readback_contract_schema" in result_names
    assert "bus_group_main_dca_physical_readback_contract_schema" in result_names
    assert "matrix_state_physical_readback_contract_schema" in result_names
    assert "fx_slots_inserts_physical_readback_contract_schema" in result_names
    assert "routing_sends_physical_readback_contract_schema" in result_names
    assert "snapshots_scenes_physical_readback_contract_schema" in result_names
    assert "user_presets_physical_readback_contract_schema" in result_names
    assert "supervised_proof_report_gain_dry_run_schema" in result_names
    assert "autonomy_evidence_overlay_schema" in result_names
    assert "next_wing_proof_plan_schema" in result_names
    assert "supervised_proof_report_gap_rollup_schema" in result_names
    assert result_names.index("read_surface_fixture_evidence") < result_names.index(
        "read_surface_fixture_evidence_schema"
    )
    assert result_names.index("read_surface_fixture_evidence_schema") < result_names.index(
        "read_surface_remediation_plan_schema"
    )
    assert result_names.index("read_surface_remediation_plan_schema") < result_names.index(
        "raw_mixer_state_physical_readback_contract_schema"
    )
    assert result_names.index("raw_mixer_state_physical_readback_contract_schema") < result_names.index(
        "channel_basic_controls_physical_readback_contract_schema"
    )
    assert result_names.index("channel_basic_controls_physical_readback_contract_schema") < result_names.index(
        "channel_processing_physical_readback_contract_schema"
    )
    assert result_names.index("channel_processing_physical_readback_contract_schema") < result_names.index(
        "bus_group_main_dca_physical_readback_contract_schema"
    )
    assert result_names.index("bus_group_main_dca_physical_readback_contract_schema") < result_names.index(
        "matrix_state_physical_readback_contract_schema"
    )
    assert result_names.index("matrix_state_physical_readback_contract_schema") < result_names.index(
        "fx_slots_inserts_physical_readback_contract_schema"
    )
    assert result_names.index("fx_slots_inserts_physical_readback_contract_schema") < result_names.index(
        "routing_sends_physical_readback_contract_schema"
    )
    assert result_names.index("routing_sends_physical_readback_contract_schema") < result_names.index(
        "snapshots_scenes_physical_readback_contract_schema"
    )
    assert result_names.index("snapshots_scenes_physical_readback_contract_schema") < result_names.index(
        "user_presets_physical_readback_contract_schema"
    )
    assert result_names.index("user_presets_physical_readback_contract_schema") < result_names.index(
        "backend_pytest"
    )
    assert result_names.index("supervised_proof_artifact_report") < result_names.index(
        "supervised_proof_report_gain_dry_run_schema"
    )
    assert result_names.index("supervised_proof_report_gain_dry_run_schema") < result_names.index(
        "autonomy_evidence_overlay_schema"
    )
    assert result_names.index("autonomy_evidence_overlay_schema") < result_names.index(
        "next_wing_proof_plan_schema"
    )
    assert result_names.index("next_wing_proof_plan_schema") < result_names.index(
        "supervised_proof_report_gap_rollup_schema"
    )


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


def test_no_wing_release_check_rejects_current_cycle_forbidden_command_boundary():
    current_cycle_forbidden_tokens = (
        "set_fader",
        "set_gain",
        "set_trim",
        "set_input_trim",
        "set_channel_trim",
        "set_preamp_trim",
        "set_preamp_gain",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "reset",
        "mute",
        "unmute",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
    )
    missing = [
        token
        for token in current_cycle_forbidden_tokens
        if token not in release_check.FORBIDDEN_COMMAND_TOKENS
    ]
    assert missing == []

    checks = [
        CommandCheck(
            name=f"bad_{token}_runner",
            command=("python3", f"tools/{token}_audit.py"),
            cwd=".",
        )
        for token in current_cycle_forbidden_tokens
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    for token in current_cycle_forbidden_tokens:
        assert f"bad_{token}_runner:{token}" in message


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


def test_no_wing_release_check_summary_embeds_autonomy_gap_rollup(tmp_path):
    result = CheckResult(
        name="supervised_proof_report_gap_rollup_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "autonomy_gap_rollup_present": True,
            "full_auto_gap_bucket_counts": {
                "read_surface": 14,
                "write_surface": 19,
                "evidence_requirement": 12,
            },
            "evidence_requirement_ids_by_action_family": {
                "bus_group_matrix": [
                    "bus_group_assignment_cache_freshness",
                    "bus_group_fader_readback",
                ],
                "mute": [
                    "current_mute_cache_freshness",
                    "mute_owner_intent_evidence",
                ],
            },
            "next_required_evidence_by_action_family": {
                "bus_group_matrix": ["Add bus/group/matrix safety review before promotion."],
                "mute": ["Add fresh mute-state intent before promotion."],
            },
            "supervised_current_live_write_kinds": ["fader", "gain"],
            "policy_supervised_live_write_kinds": ["fader", "gain"],
            "unexpected_supervised_live_write_kinds": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert autonomy_gap_rollup_summary([result]) == summary["autonomy_gap_rollup"]
    assert json_summary["autonomy_gap_rollup"]["full_auto_gap_bucket_counts"] == {
        "read_surface": 14,
        "write_surface": 19,
        "evidence_requirement": 12,
    }
    assert json_summary["autonomy_gap_rollup"]["evidence_requirement_ids_by_action_family"][
        "bus_group_matrix"
    ] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
    ]
    assert json_summary["autonomy_gap_rollup"]["next_required_evidence_by_action_family"][
        "mute"
    ] == ["Add fresh mute-state intent before promotion."]
    assert json_summary["autonomy_gap_rollup"]["live_write"] is False
    assert json_summary["autonomy_gap_rollup"]["full_autonomous_ready"] is False
    assert "## Autonomy Gap Rollup" in markdown_summary
    assert "full_auto_gap_bucket_counts" in markdown_summary
    assert '"read_surface": 14' in markdown_summary
    assert "evidence_requirement_ids_by_action_family" in markdown_summary
    assert "bus_group_assignment_cache_freshness" in markdown_summary
    assert "next_required_evidence_by_action_family" in markdown_summary
    assert 'supervised_current_live_write_kinds: `["fader", "gain"]`' in markdown_summary


def test_no_wing_release_check_summary_embeds_gain_dry_run_proof(tmp_path):
    result = CheckResult(
        name="supervised_proof_report_gain_dry_run_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "gain_dry_run_row_count": 1,
            "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
            "value_type_counts": {"gain": 1},
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert supervised_gain_dry_run_summary([result]) == summary["supervised_gain_dry_run"]
    assert json_summary["supervised_gain_dry_run"]["live_write"] is False
    assert json_summary["supervised_gain_dry_run"]["promotion_allowed"] is False
    assert json_summary["supervised_gain_dry_run"]["autonomous_ready"] is False
    assert json_summary["supervised_gain_dry_run"]["gain_dry_run_row_count"] == 1
    assert "## Supervised Gain Dry-Run Proof" in markdown_summary
    assert 'dry_run_ready_proof_ids: `["gain-proof-dry-run"]`' in markdown_summary


def test_no_wing_release_check_summary_embeds_autonomy_evidence_overlay(tmp_path):
    result = CheckResult(
        name="autonomy_evidence_overlay_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "overlay_status": "dry_run_only",
            "row_count": 2,
            "status_counts": {
                "dry_run_only": 1,
                "manual_safety_review_candidate": 0,
                "no_current_evidence": 1,
                "remediation_required": 0,
            },
            "manual_safety_review_candidate_target_ids": [],
            "remediation_required_target_ids": [],
            "dry_run_only_target_ids": ["gain_trim_supervised_promotion"],
            "no_current_evidence_target_ids": ["fader_supervised_promotion"],
            "global_blockers": [
                "overlay_does_not_modify_safety_rows",
                "manual_safety_review_required_before_promotion",
                "full_autonomous_mode_remains_blocked",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert autonomy_evidence_overlay_summary([result]) == summary["autonomy_evidence_overlay"]
    assert json_summary["autonomy_evidence_overlay"]["live_write"] is False
    assert json_summary["autonomy_evidence_overlay"]["live_widening_allowed"] is False
    assert json_summary["autonomy_evidence_overlay"]["automatic_safety_update_allowed"] is False
    assert json_summary["autonomy_evidence_overlay"]["full_autonomous_ready"] is False
    assert json_summary["autonomy_evidence_overlay"]["operator_required"] is True
    assert json_summary["autonomy_evidence_overlay"]["dry_run_only_target_ids"] == [
        "gain_trim_supervised_promotion"
    ]
    assert "## Autonomy Evidence Overlay" in markdown_summary
    assert 'dry_run_only_target_ids: `["gain_trim_supervised_promotion"]`' in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_fixture_evidence(tmp_path):
    result = CheckResult(
        name="read_surface_fixture_evidence_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "snapshot_status": "fixture_evidence_incomplete",
            "surface_count": 10,
            "surface_with_fixture_count": 10,
            "declared_fixture_path_count": 10,
            "declared_test_name_count": 25,
            "status_counts": {"partial_fixture_evidence": 10},
            "physical_fixture_missing_surface_ids": ["channel_identity_and_basic_controls"],
            "next_physical_readback_surface_id": "channel_identity_and_basic_controls",
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_fixture_evidence_summary([result]) == summary["read_surface_fixture_evidence"]
    assert json_summary["read_surface_fixture_evidence"]["live_write"] is False
    assert json_summary["read_surface_fixture_evidence"]["full_autonomous_ready"] is False
    assert json_summary["read_surface_fixture_evidence"]["surface_count"] == 10
    assert json_summary["read_surface_fixture_evidence"]["declared_fixture_path_count"] == 10
    assert json_summary["read_surface_fixture_evidence"]["declared_test_name_count"] == 25
    assert json_summary["read_surface_fixture_evidence"]["physical_gap_surface_count"] == 1
    assert json_summary["read_surface_fixture_evidence"]["summary_recomputed_from_rows"] is True
    assert json_summary["read_surface_fixture_evidence"]["fixture_declaration_validation"] is True
    assert json_summary["read_surface_fixture_evidence"]["contract_gap_consistency_enforced"] is True
    assert json_summary["read_surface_fixture_evidence"]["physical_fixture_missing_surface_ids"] == [
        "channel_identity_and_basic_controls"
    ]
    assert (
        json_summary["read_surface_fixture_evidence"]["next_physical_readback_surface_id"]
        == "channel_identity_and_basic_controls"
    )
    assert "## Read Surface Fixture Evidence" in markdown_summary
    assert "declared_fixture_path_count" in markdown_summary
    assert "contract_gap_consistency_enforced: `true`" in markdown_summary
    assert "physical_fixture_missing_surface_ids" in markdown_summary
    assert "next_physical_readback_surface_id" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_remediation_plan(tmp_path):
    result = CheckResult(
        name="read_surface_remediation_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "read_remediation_required",
            "action_count": 10,
            "incomplete_surface_count": 10,
            "physical_fixture_missing_surface_ids": ["raw_mixer_state_cache"],
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "next_physical_readback_target_action_ids": [
                "remediate_raw_mixer_state_cache"
            ],
            "expected_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "expected_next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_remediation_plan_summary([result]) == summary["read_surface_remediation_plan"]
    assert json_summary["read_surface_remediation_plan"]["read_only"] is True
    assert json_summary["read_surface_remediation_plan"]["live_write"] is False
    assert json_summary["read_surface_remediation_plan"]["mutates_console_state"] is False
    assert json_summary["read_surface_remediation_plan"]["full_autonomous_ready"] is False
    assert json_summary["read_surface_remediation_plan"]["physical_gap_surface_count"] == 1
    assert (
        json_summary["read_surface_remediation_plan"]["next_physical_readback_action_id"]
        == "remediate_raw_mixer_state_cache"
    )
    assert json_summary["read_surface_remediation_plan"][
        "next_physical_readback_target_action_ids"
    ] == ["remediate_raw_mixer_state_cache"]
    assert json_summary["read_surface_remediation_plan"]["target_action_consistency"] is True
    assert "## Read Surface Remediation Plan" in markdown_summary
    assert "next_physical_readback_action_id" in markdown_summary
    assert "next_physical_readback_target_action_ids" in markdown_summary
    assert "target_action_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_raw_mixer_readback_contract(tmp_path):
    result = CheckResult(
        name="raw_mixer_state_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "raw_mixer_state_physical_readback.v1",
            "surface_id": "raw_mixer_state_cache",
            "remediation_action_id": "remediate_raw_mixer_state_cache",
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "fixture_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "remediation_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "remediation_next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "required_top_level_field_count": 10,
            "required_row_field_count": 5,
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert raw_mixer_readback_contract_summary([result]) == summary[
        "raw_mixer_state_physical_readback_contract"
    ]
    contract = json_summary["raw_mixer_state_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "raw_mixer_state_cache"
    assert contract["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert "## Raw Mixer State Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_basic_controls_readback_contract(tmp_path):
    result = CheckResult(
        name="channel_basic_controls_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "channel_basic_controls_physical_readback.v1",
            "surface_id": "channel_identity_and_basic_controls",
            "remediation_action_id": "remediate_channel_identity_and_basic_controls",
            "physical_readback_surface_id": "channel_identity_and_basic_controls",
            "physical_readback_action_id": "remediate_channel_identity_and_basic_controls",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "pan_phase_delay",
            ],
            "required_top_level_field_count": 10,
            "required_channel_field_count": 8,
            "required_identity_fields": ["source_name", "source_role", "input_source"],
            "required_control_fields": ["fader_db", "trim_db", "mute", "pan"],
            "required_coverage_flags": [
                "identity",
                "basic_controls",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_channel_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "mute_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_basic_controls_readback_contract_summary([result]) == summary[
        "channel_basic_controls_physical_readback_contract"
    ]
    contract = json_summary["channel_basic_controls_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "channel_identity_and_basic_controls"
    assert contract["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert contract["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert contract["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "## Channel Basic Controls Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "promotion_families_requiring_surface" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_processing_readback_contract(tmp_path):
    result = CheckResult(
        name="channel_processing_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "channel_processing_physical_readback.v1",
            "surface_id": "channel_eq_dynamics_gate",
            "remediation_action_id": "remediate_channel_eq_dynamics_gate",
            "physical_readback_surface_id": "channel_eq_dynamics_gate",
            "physical_readback_action_id": "remediate_channel_eq_dynamics_gate",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "eq",
                "compressor",
                "gate",
                "deessing",
            ],
            "required_top_level_field_count": 10,
            "required_channel_field_count": 10,
            "required_hpf_fields": ["enabled", "frequency_hz"],
            "required_eq_fields": ["enabled", "bands"],
            "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
            "required_compressor_fields": ["enabled", "model", "gain_reduction_db"],
            "required_gate_fields": ["enabled", "model", "range_db"],
            "required_coverage_flags": [
                "hpf",
                "eq",
                "compressor",
                "gate",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_eq_band_count": 4,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_channel_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "eq_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_processing_readback_contract_summary([result]) == summary[
        "channel_processing_physical_readback_contract"
    ]
    contract = json_summary["channel_processing_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "channel_eq_dynamics_gate"
    assert contract["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert contract["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
    ]
    assert contract["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in contract["required_compressor_fields"]
    assert "range_db" in contract["required_gate_fields"]
    assert contract["minimum_eq_band_count"] == 4
    assert "## Channel Processing Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_eq_band_count: `4`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_bus_group_main_dca_readback_contract(tmp_path):
    result = CheckResult(
        name="bus_group_main_dca_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "bus_group_main_dca_physical_readback.v1",
            "surface_id": "bus_group_main_dca_state",
            "remediation_action_id": "remediate_bus_group_main_dca_state",
            "physical_readback_surface_id": "bus_group_main_dca_state",
            "physical_readback_action_id": "remediate_bus_group_main_dca_state",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "dca",
                "master_bus",
            ],
            "required_top_level_field_count": 14,
            "required_bus_field_count": 9,
            "required_main_field_count": 8,
            "required_dca_field_count": 8,
            "required_bus_control_fields": ["fader_db", "mute", "pan", "width", "mono"],
            "required_main_control_fields": ["fader_db", "mute", "pan"],
            "required_dca_control_fields": ["fader_db", "mute", "monitor"],
            "required_bus_assignment_fields": [
                "dca_assignments",
                "main_send_on",
                "main_send_level_db",
                "main_send_pre",
            ],
            "required_dca_assignment_fields": ["member_channels", "member_buses"],
            "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
            "required_bus_coverage_flags": [
                "controls",
                "assignments",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_main_coverage_flags": [
                "controls",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_dca_coverage_flags": [
                "identity",
                "controls",
                "assignments",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_bus_count": 1,
            "minimum_main_count": 1,
            "minimum_dca_count": 1,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_surface_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "dca_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert bus_group_main_dca_readback_contract_summary([result]) == summary[
        "bus_group_main_dca_physical_readback_contract"
    ]
    contract = json_summary["bus_group_main_dca_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "bus_group_main_dca_state"
    assert contract["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
    ]
    assert contract["required_bus_control_fields"] == ["fader_db", "mute", "pan", "width", "mono"]
    assert "true_peak_dbtp" in contract["required_metering_fields"]
    assert "member_buses" in contract["required_dca_assignment_fields"]
    assert contract["minimum_bus_count"] == 1
    assert contract["minimum_main_count"] == 1
    assert contract["minimum_dca_count"] == 1
    assert "## Bus Group Main DCA Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_bus_count: `1`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_matrix_state_readback_contract(tmp_path):
    result = CheckResult(
        name="matrix_state_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "matrix_state_physical_readback.v1",
            "surface_id": "matrix_state",
            "remediation_action_id": "remediate_matrix_state",
            "physical_readback_surface_id": "matrix_state",
            "physical_readback_action_id": "remediate_matrix_state",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "fx_send_insert",
                "routing",
            ],
            "required_top_level_field_count": 10,
            "required_matrix_field_count": 11,
            "required_control_fields": ["fader_db", "mute", "pan"],
            "required_eq_fields": ["enabled", "bands"],
            "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
            "required_dynamics_fields": ["enabled", "model", "gain_reduction_db"],
            "required_insert_fields": ["pre_insert_on", "post_insert_weight"],
            "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
            "required_coverage_flags": [
                "controls",
                "eq",
                "dynamics",
                "inserts",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_matrix_count": 1,
            "minimum_eq_band_count": 8,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_matrix_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "matrix_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert matrix_state_readback_contract_summary([result]) == summary[
        "matrix_state_physical_readback_contract"
    ]
    contract = json_summary["matrix_state_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "matrix_state"
    assert contract["remediation_action_id"] == "remediate_matrix_state"
    assert contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "fx_send_insert",
        "routing",
    ]
    assert contract["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert "gain_reduction_db" in contract["required_dynamics_fields"]
    assert "post_insert_weight" in contract["required_insert_fields"]
    assert "true_peak_dbtp" in contract["required_metering_fields"]
    assert contract["minimum_matrix_count"] == 1
    assert contract["minimum_eq_band_count"] == 8
    assert "## Matrix State Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_matrix_count: `1`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_fx_slots_inserts_readback_contract(tmp_path):
    result = CheckResult(
        name="fx_slots_inserts_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "fx_slots_inserts_physical_readback.v1",
            "surface_id": "fx_slots_and_inserts",
            "remediation_action_id": "remediate_fx_slots_and_inserts",
            "physical_readback_surface_id": "fx_slots_and_inserts",
            "physical_readback_action_id": "remediate_fx_slots_and_inserts",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fx_send_insert",
                "routing",
                "mix_balance",
            ],
            "required_top_level_field_count": 12,
            "required_fx_slot_field_count": 16,
            "required_fx_slot_fields": ["slot", "slot_number", "parameter_schema"],
            "required_parameter_fields": ["parameter_index", "value", "safe_range"],
            "required_parameter_safe_range_fields": ["min", "max"],
            "required_parameter_schema_fields": [
                "status",
                "physical_model_schema_verified",
            ],
            "required_fx_slot_coverage_flags": [
                "identity",
                "assignment",
                "cache_matches_physical_readback",
            ],
            "required_insert_fields": ["target_type", "position", "linked_fx_slot"],
            "required_insert_coverage_flags": [
                "assignment",
                "status",
                "scoped_diff_marker",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_fx_slot_count": 1,
            "minimum_fx_parameter_count": 4,
            "minimum_insert_count": 1,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "allowed_insert_target_types": ["channel", "bus", "main", "matrix"],
            "allowed_insert_positions": ["pre", "post"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "fx_parameter_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert fx_slots_inserts_readback_contract_summary([result]) == summary[
        "fx_slots_inserts_physical_readback_contract"
    ]
    contract = json_summary["fx_slots_inserts_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "fx_slots_and_inserts"
    assert contract["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert contract["promotion_families_requiring_surface"] == [
        "fx_send_insert",
        "routing",
        "mix_balance",
    ]
    assert "parameter_schema" in contract["required_fx_slot_fields"]
    assert "safe_range" in contract["required_parameter_fields"]
    assert "physical_model_schema_verified" in contract["required_parameter_schema_fields"]
    assert "scoped_diff_marker" in contract["required_insert_coverage_flags"]
    assert contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert contract["minimum_fx_slot_count"] == 1
    assert contract["minimum_fx_parameter_count"] == 4
    assert contract["minimum_insert_count"] == 1
    assert "## FX Slots Inserts Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_fx_parameter_count: `4`" in markdown_summary
    assert "required_false_safety_flags" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_routing_sends_readback_contract(tmp_path):
    result = CheckResult(
        name="routing_sends_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "routing_sends_physical_readback.v1",
            "surface_id": "routing_and_sends",
            "remediation_action_id": "remediate_routing_and_sends",
            "physical_readback_surface_id": "routing_and_sends",
            "physical_readback_action_id": "remediate_routing_and_sends",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "routing",
                "fx_send_insert",
                "bus_group_matrix",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "required_top_level_field_count": 18,
            "required_static_route_field_count": 13,
            "required_static_route_fields": ["start", "end", "role", "wing_routing_hint"],
            "required_bus_send_fields": ["channel", "send", "target_bus", "pre_on"],
            "required_main_send_fields": ["channel", "main", "pre"],
            "required_matrix_send_fields": ["channel", "send", "target_matrix"],
            "required_output_route_fields": ["output_group", "output_number", "source_channel"],
            "required_static_route_coverage_flags": [
                "identity",
                "static_config_match",
                "physical_route_mapping",
                "scoped_diff_marker",
            ],
            "required_send_coverage_flags": [
                "assignment",
                "level",
                "mode",
                "scoped_diff_marker",
                "cache_matches_physical_readback",
            ],
            "required_output_route_coverage_flags": [
                "assignment",
                "source",
                "scoped_diff_marker",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_static_dante_route_count": 1,
            "minimum_channel_bus_send_count": 1,
            "minimum_channel_main_send_count": 1,
            "minimum_channel_matrix_send_count": 1,
            "minimum_output_route_count": 1,
            "allowed_sources": [
                "physical_readback",
                "wing_physical_readback",
                "dlive_physical_readback",
            ],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "routing_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert routing_sends_readback_contract_summary([result]) == summary[
        "routing_sends_physical_readback_contract"
    ]
    contract = json_summary["routing_sends_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["live_widening_allowed"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "routing_and_sends"
    assert contract["remediation_action_id"] == "remediate_routing_and_sends"
    assert "routing" in contract["promotion_families_requiring_surface"]
    assert "wing_routing_hint" in contract["required_static_route_fields"]
    assert "pre_on" in contract["required_bus_send_fields"]
    assert "main" in contract["required_main_send_fields"]
    assert "target_matrix" in contract["required_matrix_send_fields"]
    assert "source_channel" in contract["required_output_route_fields"]
    assert "cache_matches_physical_readback" in contract["required_send_coverage_flags"]
    assert (
        "cache_matches_physical_readback"
        in contract["required_output_route_coverage_flags"]
    )
    assert contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert contract["minimum_static_dante_route_count"] == 1
    assert contract["minimum_channel_bus_send_count"] == 1
    assert contract["minimum_channel_main_send_count"] == 1
    assert contract["minimum_channel_matrix_send_count"] == 1
    assert contract["minimum_output_route_count"] == 1
    assert "## Routing Sends Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_channel_bus_send_count: `1`" in markdown_summary
    assert "required_false_safety_flags" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_user_presets_readback_contract(tmp_path):
    result = CheckResult(
        name="user_presets_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "user_presets_physical_readback.v1",
            "surface_id": "user_presets",
            "remediation_action_id": "remediate_user_presets",
            "physical_readback_surface_id": "user_presets",
            "physical_readback_action_id": "remediate_user_presets",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "scene_preset",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "required_top_level_field_count": 10,
            "required_preset_fields": [
                "preset_index",
                "preset_name",
                "preset_content_fingerprint",
                "rollback_reference",
            ],
            "required_coverage_flags": [
                "catalog_identity",
                "preset_content_fingerprint",
                "rollback_reference",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
                "preset_recall_allowed",
                "preset_store_allowed",
            ],
            "minimum_preset_count": 1,
            "allowed_sources": [
                "physical_readback",
                "wing_physical_readback",
                "dlive_physical_readback",
            ],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "preset_recall_events",
                "preset_store_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert user_presets_readback_contract_summary([result]) == summary[
        "user_presets_physical_readback_contract"
    ]
    contract = json_summary["user_presets_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["preset_recall_allowed"] is False
    assert contract["preset_store_allowed"] is False
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "user_presets"
    assert "scene_preset" in contract["promotion_families_requiring_surface"]
    assert "preset_content_fingerprint" in contract["required_preset_fields"]
    assert "rollback_reference" in contract["required_coverage_flags"]
    assert "preset_store_allowed" in contract["required_false_safety_flags"]
    assert contract["minimum_preset_count"] == 1
    assert "## User Presets Physical Readback Contract" in markdown_summary
    assert "preset_recall_allowed: `false`" in markdown_summary
    assert "preset_store_allowed: `false`" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_next_wing_proof_plan(tmp_path):
    result = CheckResult(
        name="next_wing_proof_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "operator_proof_required",
            "step_count": 1,
            "planned_value_types": ["gain"],
            "planned_channels": [3],
            "global_blockers": ["operator_must_approve_and_execute_manually"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert next_wing_proof_plan_summary([result]) == summary["next_wing_proof_plan"]
    assert json_summary["next_wing_proof_plan"]["live_write"] is False
    assert json_summary["next_wing_proof_plan"]["execution_allowed_by_plan"] is False
    assert json_summary["next_wing_proof_plan"]["operator_required"] is True
    assert json_summary["next_wing_proof_plan"]["planned_value_types"] == ["gain"]
    assert "## Next WING Proof Plan" in markdown_summary
    assert 'planned_value_types: `["gain"]`' in markdown_summary


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
