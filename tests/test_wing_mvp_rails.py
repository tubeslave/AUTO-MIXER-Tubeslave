import importlib.util
import json
import re
import sys
from pathlib import Path

from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
from backend.evaluation_integration_bundle_contract import (
    REQUIRED_INTEGRATION_CHECKS,
    evaluate_integration_bundle_contract,
)
from backend.fault_injection_recovery_contract import (
    REQUIRED_AUDIT_FIELDS as RECOVERY_REQUIRED_AUDIT_FIELDS,
    REQUIRED_EVALUATION_TYPES as RECOVERY_REQUIRED_EVALUATION_TYPES,
    REQUIRED_FAULT_SCENARIOS as RECOVERY_REQUIRED_FAULT_SCENARIOS,
    REQUIRED_RECOVERY_OUTCOMES,
    evaluate_fault_injection_recovery_contract,
)
from backend.integration_policy_contract import (
    REQUIRED_INTEGRATION_CONTROLS as POLICY_REQUIRED_INTEGRATION_CONTROLS,
    REQUIRED_INTEGRATION_EVIDENCE as POLICY_REQUIRED_INTEGRATION_EVIDENCE,
    evaluate_integration_policy_contract,
)
from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
    evaluate_offline_replay_evaluation_contract,
)
from backend.quality_threshold_policy_contract import (
    REQUIRED_POLICY_CHECKS as THRESHOLD_REQUIRED_POLICY_CHECKS,
    REQUIRED_THRESHOLD_AXES,
    evaluate_quality_threshold_policy_contract,
)
from backend.recovery_policy_contract import (
    REQUIRED_RECOVERY_CONTROLS as POLICY_REQUIRED_RECOVERY_CONTROLS,
    REQUIRED_RECOVERY_EVIDENCE as POLICY_REQUIRED_RECOVERY_EVIDENCE,
    evaluate_recovery_policy_contract,
)
from backend.safety_kernel_trace_contract import (
    REQUIRED_AUDIT_FIELDS as SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS,
    REQUIRED_GUARD_CHECKS as SAFETY_KERNEL_REQUIRED_GUARD_CHECKS,
    REQUIRED_TRACE_SCENARIOS as SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS,
    evaluate_safety_kernel_trace_contract,
)
from backend import wing_readonly_preflight_contract as readonly_contract
from tools import no_wing_release_check as release_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVP_DOC = PROJECT_ROOT / "Docs" / "MVP_WING_SUPERVISED.md"
CORRECTION_ROADMAP = PROJECT_ROOT / "Docs" / "CORRECTION_METHODS_ROADMAP.md"
BLOCKERS_DOC = PROJECT_ROOT / ".ai" / "project-layer" / "wing_mvp_blockers.md"
CORRECTION_BACKLOG = PROJECT_ROOT / ".ai" / "project-layer" / "correction_methods_backlog.md"
READONLY_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_READONLY_PREFLIGHT.md"
SUPERVISED_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_SUPERVISED_FADER_GAIN_PROOF.md"
DOCS_INDEX = PROJECT_ROOT / "Docs" / "INDEX.md"
CURRENT_STATE_DOC = PROJECT_ROOT / "Docs" / "CURRENT_STATE.md"
ROADMAP_DOC = PROJECT_ROOT / "Docs" / "ROADMAP.md"
ACTIVE_MILESTONE_DOC = PROJECT_ROOT / "Docs" / "ACTIVE_MILESTONE.md"
PROJECT_EXECUTION_CONTRACT = PROJECT_ROOT / "Docs" / "PROJECT_EXECUTION_CONTRACT.md"
PROJECT_GOAL = PROJECT_ROOT / "Docs" / "PROJECT_GOAL.md"
PROJECT_COMPLETION_GATE = PROJECT_ROOT / "Docs" / "PROJECT_COMPLETION_GATE.md"
SAFETY_CONTRACT = PROJECT_ROOT / "Docs" / "SAFETY_CONTRACT.md"
CAPABILITY_MATRIX = PROJECT_ROOT / "Docs" / "CAPABILITY_MATRIX.md"
CONSOLE_CAPABILITY_MATRIX = PROJECT_ROOT / "Docs" / "CONSOLE_CAPABILITY_MATRIX.md"
M001_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M001-wing-readonly-preflight-evidence-gate.md"
M011_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M011-offline-replay-evaluation-contract.md"
M011_ADR = PROJECT_ROOT / "Docs" / "adr" / "offline-replay-evaluation-contract-20260619.md"
M012_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M012-fault-injection-recovery-contract.md"
M012_ADR = PROJECT_ROOT / "Docs" / "adr" / "fault-injection-recovery-contract-20260619.md"
M013_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M013-integration-evaluation-bundle-contract.md"
M013_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-evaluation-bundle-contract-20260619.md"
M014_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M014-quality-threshold-policy-contract.md"
M014_ADR = PROJECT_ROOT / "Docs" / "adr" / "quality-threshold-policy-contract-20260619.md"
M015_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M015-replay-threshold-policy-binding.md"
M015_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-threshold-policy-binding-20260619.md"
M016_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M016-recovery-policy-contract.md"
M016_ADR = PROJECT_ROOT / "Docs" / "adr" / "recovery-policy-contract-20260619.md"
M017_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M017-fault-recovery-policy-binding.md"
M017_ADR = PROJECT_ROOT / "Docs" / "adr" / "fault-recovery-policy-binding-20260619.md"
M018_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M018-integration-policy-contract.md"
M018_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-policy-contract-20260619.md"
M019_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M019-integration-policy-binding.md"
M019_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-policy-binding-20260619.md"
M020_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M020-safety-kernel-trace-contract.md"
M020_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-kernel-trace-contract-20260619.md"
AUTOPILOT_SKILL = PROJECT_ROOT / ".agents" / "skills" / "automixer-autopilot" / "SKILL.md"
AUTOPILOT_BLOCKERS = PROJECT_ROOT / "ops" / "autopilot" / "BLOCKERS.md"
AUTOPILOT_OPERATIONS = PROJECT_ROOT / "ops" / "autopilot" / "OPERATIONS.md"
AUTOPILOT_HEARTBEAT = PROJECT_ROOT / "ops" / "autopilot" / "HEARTBEAT.md"
AUTOPILOT_LAST_REPORT = PROJECT_ROOT / "ops" / "autopilot" / "LAST_REPORT.md"
AUTOPILOT_STATE = PROJECT_ROOT / "ops" / "autopilot" / "STATE.json"
REAL_WING_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "wing_readonly_preflight"
VALIDATE_INPUTS_READY_SUMMARY = (
    PROJECT_ROOT
    / "ops"
    / "autopilot"
    / "evidence"
    / "M001_CP04_validate_inputs_ready_for_websocket"
    / "summary.json"
)
VALIDATE_INPUTS_MISMATCH_SUMMARY = (
    PROJECT_ROOT
    / "ops"
    / "autopilot"
    / "evidence"
    / "M001_CP04_validate_inputs_expected_plan_mismatch"
    / "summary.json"
)
CP04_HANDOFF_REPORT = (
    PROJECT_ROOT
    / "Docs"
    / "reports"
    / "AUTOMIXER_AUTOPILOT_M001_CP04_WING_READONLY_HANDOFF_20260618.md"
)
PROGRESS_TOOL = PROJECT_ROOT / "tools" / "codex_autoloop" / "update_mvp_progress.py"
RUNNER_TOOL = PROJECT_ROOT / "tools" / "wing_readonly_preflight_runner.py"
PROMPT = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"

DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS = (
    "declared_operator_sequence_count: 12",
    "declared_expected_operator_sequence_count: 12",
    "declared_operator_sequence_count_contract_flag: true",
    "declared_operator_sequence_count_matches_actual: true",
    "declared_operator_sequence_count_matches_contract: true",
    "declared_expected_operator_sequence_count_matches_contract: true",
    "declared_operator_sequence_count_contract_flag_matches_declared_count: true",
)

OPERATOR_SEQUENCE_SHA_MARKERS = (
    "operator_sequence_sha256",
    "actual_operator_sequence_sha256",
    "operator_sequence_sha256_matches_actual: true",
)

OPERATOR_SEQUENCE_CONTRACT_MARKERS = (
    "expected_operator_sequence_sha256",
    "actual_expected_operator_sequence_sha256",
    "operator_sequence_matches_contract: true",
    "operator_sequence_sha256_matches_contract: true",
    "declared_expected_operator_sequence_sha256_matches_contract: true",
    "declared_operator_sequence_matches_contract_flag: true",
    "declared_operator_sequence_matches_contract_flag_matches_actual: true",
    "declared_operator_sequence_sha256_matches_contract_flag: true",
    "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true",
)

DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS = tuple(
    marker.split(":", 1)[0] for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS
)

ENVIRONMENT_BLOCKED_FRONTEND_MARKERS = (
    "environment_blocked_checks",
    "environment_blocked_check_details",
    "local_dev_server_bind_denied",
    "127.0.0.1:3000",
)

ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS = (
    "artifact-manifest.json",
    "same no-WING artifact directory",
    "artifact_manifest_self_consistency_summary_surface_match: true",
    "artifact_manifest_self_consistency_summary_surface_json_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false",
    "artifact_manifest_self_consistency_summary_surface_live_authority_granted: false",
    "artifact_manifest_self_consistency_match: true",
    "artifact_manifest_self_consistency_mismatches: []",
    "artifact_manifest_contract_cross_surface_match: true",
    "artifact_manifest_contract_cross_surface_mismatches: []",
)

ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS = (
    "status: passed",
    "failed_checks: []",
    "read_only: true",
    "live_authority_granted: false",
    "readiness_closure_allowed: false",
)

ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS = (
    "PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py",
    "status: passed",
    "failed_checks: []",
    "failed_checks_missing_check_entries: []",
    "failed_checks_have_check_entries: true",
    "summary_json_exists: true",
    "summary_md_exists: true",
    "artifact_manifest_exists: true",
    "artifact_bundle_loadable: true",
    "manifest_type: true",
    "manifest_required_artifact_ids_match_contract: true",
    "manifest_required_artifact_paths_match_contract: true",
    "manifest_records_required_artifact_ids: true",
    "summary_json_path_matches_contract: true",
    "summary_json_sha256_matches_actual: true",
    "summary_md_path_matches_contract: true",
    "summary_md_sha256_matches_actual: true",
    "summary_json_sha256_matches_manifest: true",
    "summary_md_sha256_matches_manifest: true",
    "manifest_denies_readiness_matrix_update: true",
    "manifest_denies_readiness_closure: true",
    "manifest_denies_live_authority: true",
    "summary_denies_readiness_matrix_update: true",
    "summary_denies_readiness_closure: true",
    "summary_denies_live_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "summary_json_artifact_manifest_verifier_fields_present: true",
    "manifest_artifact_manifest_verifier_fields_present: true",
    "summary_md_artifact_manifest_verifier_fields_rendered: true",
    "manifest_artifact_manifest_verifier_command_matches_expected: true",
    "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "verifier_report_required_fields_present: true",
    "verifier_report_required_fields_missing: []",
    "verifier_report_required_fields_match_contract: true",
    "verifier_report_readiness_closure_allowed_matches_expected: true",
    "verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_mismatches: []",
    "artifact_manifest_verifier_check_summary_json_exists: true",
    "artifact_manifest_verifier_check_summary_md_exists: true",
    "artifact_manifest_verifier_check_artifact_manifest_exists: true",
    "artifact_manifest_verifier_check_artifact_bundle_loadable: true",
    "artifact_manifest_verifier_check_manifest_type: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_ids_match_contract: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_paths_match_contract: true",
    "artifact_manifest_verifier_check_manifest_records_required_artifact_ids: true",
    "artifact_manifest_verifier_check_summary_json_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_md_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_manifest_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_summary_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_fields_rendered: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_command_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_required_fields_present: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_contract_cross_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_failed_checks_have_check_entries: true",
    "artifact_manifest_verifier_report_required_fields",
    "checks",
    "verifier_report_required_fields",
    "verifier_report_required_fields_missing",
    "verifier_report_required_fields_match_contract",
)


def load_progress_module():
    spec = importlib.util.spec_from_file_location("update_mvp_progress", PROGRESS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "wing_readonly_preflight_runner_for_mvp_rails", RUNNER_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def covered_verifier_checks(markers):
    covered: set[str] = set()
    for marker in markers:
        if not marker.endswith(": true"):
            continue
        raw_marker = marker.removesuffix(": true")
        covered.add(raw_marker)
        if raw_marker.startswith("artifact_manifest_verifier_check_"):
            covered.add(raw_marker.removeprefix("artifact_manifest_verifier_check_"))
    return covered


def current_wing_mvp_20260614_index_report_names():
    index_text = DOCS_INDEX.read_text(encoding="utf-8")
    report_paths = re.findall(
        r"`(Docs/reports/AUTOMIXER_WING_MVP_[^`]+_20260614\.md)`",
        index_text,
    )
    return sorted({Path(path).name for path in report_paths})


def markdown_table_row(text: str, first_cell: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(f"| {first_cell} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"missing markdown table row for {first_cell!r}")


def files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def test_progress_baseline_covers_runner_verifier_checks():
    progress = load_progress_module()
    runner = load_runner_module()
    missing_checks = sorted(
        set(runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS)
        - covered_verifier_checks(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)
    )
    assert missing_checks == []


def test_current_state_doc_matches_autopilot_state_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    authority = summary["authority"]
    text = CURRENT_STATE_DOC.read_text(encoding="utf-8")

    latest_commit = re.search(
        r"Latest verified product/runtime commit:\s*`([0-9a-f]+) ",
        text,
    )
    assert latest_commit is not None
    assert latest_commit.group(1) == state["last_verified_commit"]

    latest_full_suite = re.search(
        r"latest verified full-suite run for `([0-9a-f]+)` passed with\s*`([^`]+)`",
        text,
    )
    assert latest_full_suite is not None
    assert latest_full_suite.group(1) == state["last_verified_commit"]
    assert summary["pytest"].rsplit(": ", 1)[-1] == latest_full_suite.group(2)

    assert summary["no_wing_artifact"] in text
    assert "tools/verify_no_wing_artifact_manifest.py" in text
    assert "failed_checks: []" in text

    for flag in ("live_authority_granted", "contacts_wing", "contacts_backend"):
        expected = f"{flag}: {str(authority[flag]).lower()}"
        assert expected in text

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert re.search(r"waiting for external\s+WING/operator validation", text)
    assert "No real hardware evidence was produced" in text


def test_milestone_and_roadmap_stay_waiting_for_real_wing_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_milestone"] == "M001"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    assert "ID: M001" in active
    assert "Status: WAITING_FOR_EXTERNAL_VALIDATION" in active
    assert "A real hardware run remains external validation" in active
    assert "M001 autonomous work is complete" in active
    assert "Real hardware proof requires WING Rack" in active

    assert "Exactly one milestone is active: `M001`" in roadmap
    assert "`WAITING_FOR_EXTERNAL_VALIDATION`" in roadmap
    assert markdown_table_row(roadmap, "M001")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M002")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    for milestone in ("M003", "M004", "M005", "M006"):
        assert markdown_table_row(roadmap, milestone)[-1] == "BLOCKED_BY_M002"
    assert "Real read-only WING artifact is reviewed" in markdown_table_row(
        roadmap, "M002"
    )[2]

    assert "Full Autonomous Mixing Mode | NOT_STARTED" in capability
    assert "Locked until all safety and hardware gates pass" in capability
    assert "Supervised AutoMix Mode | SPECIFIED" in capability
    assert "WING live write scope is only `fader`/`gain`; real proof missing" in capability

    raw_read = markdown_table_row(console, "Raw state/read-only connection")
    physical = markdown_table_row(console, "Physical readback")
    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    assert raw_read[-2:] == ["M001/M002 real artifact", "none in setup"]
    assert physical[-2:] == ["accepted real artifact review", "none"]
    assert fader[-2:] == ["supervised fader proof", "none"]
    assert gain[-2:] == ["supervised gain-down proof", "none"]
    assert fader[3] == "no live write"
    assert gain[3] == "no live write"

    assert "real WING Rack read-only artifact" in blockers
    assert "Control-plane docs/state can proceed without live hardware contact" in blockers


def test_autopilot_stop_lease_and_recovery_contract_stays_bound_to_state():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    contract = PROJECT_EXECUTION_CONTRACT.read_text(encoding="utf-8")
    skill = AUTOPILOT_SKILL.read_text(encoding="utf-8")
    plan = M001_PLAN.read_text(encoding="utf-8")
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    operations = AUTOPILOT_OPERATIONS.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")

    assert state["stop_file"] == "ops/autopilot/STOP"
    assert state["writer_lease_dir"] == "ops/autopilot/WRITER_LEASE"
    assert state["writer_lease_stale_after_minutes"] == 30
    assert state["current_writer_lease"] is None

    stop_path = state["stop_file"]
    lease_path = state["writer_lease_dir"]
    owner_path = f"{lease_path}/owner.json"

    for text in (contract, skill, operations):
        assert stop_path in text
        assert lease_path in text

    assert owner_path in operations
    assert ".codex_autoloop.lock" in contract
    assert ".codex_autoloop.lock" in skill
    assert "RUN_LEDGER.jsonl" in contract
    assert "RUN_LEDGER.jsonl" in operations
    assert "STOPPED" in contract
    assert "STOPPED" in skill
    assert "30 minutes" in operations
    assert "30" in contract

    assert "STOP, lock, stale-run and recovery behavior remain documented" in plan
    assert "Stop if `ops/autopilot/STOP` exists" in active
    assert "another writer owns a non-stale lease" in active
    assert "Writer lease: none" in heartbeat
    assert "`ops/autopilot/STOP` does not exist after setup" in blockers

    assert "## Stop procedure" in report
    assert "Create `ops/autopilot/STOP`" in report
    assert re.search(r"release\s+lease", report)
    assert "set status to `STOPPED`" in report
    assert "## Recovery procedure" in report
    for marker in ("STATE.json", "RUN_LEDGER.jsonl", "active plan", "git\ndiff"):
        assert marker in report
    assert "Resume from the first checkpoint whose required evidence is missing" in report


def test_goal_transition_fallback_contract_stays_report_only_without_native_goal_tools():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    contract = PROJECT_EXECUTION_CONTRACT.read_text(encoding="utf-8")
    skill = AUTOPILOT_SKILL.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_milestone"] == "M001"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["product_runtime_changed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert "operator-supervised WING read-only preflight" in state["next_action"]

    assert "## Goal transition" in contract
    assert "Native thread Goal management" in contract
    assert "goal tools" in contract
    assert "native Goal update unavailable" in contract
    assert "`LAST_REPORT.md`" in contract
    assert "`/goal`" in contract
    assert "state\nchecks" in contract
    assert "Codex threads/goals" in contract
    assert "Automixer product files" in contract
    assert "hardware" in contract
    assert "single-writer serialization" in contract

    assert "active Goal" in contract
    assert "RUN_LEDGER.jsonl" in contract
    assert "Do not mark a checkpoint complete while required checks fail" in skill
    assert re.search(
        r"Always name exact files, commands, evidence paths, blockers and next\s+action",
        skill,
    )

    assert "Future heartbeat wake-ups must verify native Goal transition tools before" in blockers
    assert "replacing a thread Goal" in blockers
    assert "Control-plane docs/state can proceed without live hardware contact" in blockers

    assert "Writer lease: none" in heartbeat
    assert "This heartbeat made no product/runtime\nchange" in heartbeat
    assert "M001 still has no real WING read-only artifact" in heartbeat

    assert "## First Goal" in report
    assert "The next finite Goal should only be created after a real\nWING read-only artifact" in report
    assert "another explicitly safe\nindependent roadmap item" in report
    assert "Do not contact\nWING from unattended automation" in report
    assert "There is no dependency-ready autonomous\nroadmap milestone" in report


def test_project_completion_gate_stays_closed_without_real_hardware_and_owner_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    gate = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert summary["hardware_evidence"] is False
    assert summary["authority"]["live_authority_granted"] is False
    assert summary["authority"]["readiness_closure_allowed"] is False
    assert summary["authority"]["contacts_wing"] is False
    assert summary["authority"]["contacts_backend"] is False

    assert "Status: authoritative completion gate" in gate
    assert "evidence-backed" in gate
    assert "hardware, listening или owner validation" in gate
    assert "WAITING_FOR_EXTERNAL_VALIDATION`, а не `COMPLETE`" in gate
    assert "## Gate Conditions" in gate
    assert "Full\n   Autonomous Mixing Mode" in gate
    assert "staged real-hardware evidence" in gate
    assert "High-risk console actions" in gate
    assert "human supervision" in gate
    assert "Threshold Policy" in gate
    assert "PENDING_APPROVAL" in gate
    assert "COMPLETE` запрещен" in gate
    assert "UNKNOWN" in gate

    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal
    assert "зеленому unit test" in goal
    assert "Offline, shadow, replay, mock и simulator evidence" in goal
    assert "не являются hardware proof" in goal
    assert "supervised `fader`/`gain` rail" in goal

    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    multichannel = markdown_table_row(capability, "Multichannel audio analysis")
    assert full_auto[1] == "NOT_STARTED"
    assert full_auto[-1] == "none"
    assert "all safety and hardware gates pass" in full_auto[4]
    assert full_auto[-2] == "completion gate review"
    assert supervised[1] == "SPECIFIED"
    assert "real proof missing" in supervised[4]
    assert supervised[-1] == "none"
    assert multichannel[-1] == "UNKNOWN"
    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence" in capability

    assert markdown_table_row(roadmap, "M001")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M002")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M010")[-1] == "PENDING_APPROVAL"
    assert "Full autonomous eligibility review" in markdown_table_row(roadmap, "M010")[1]
    assert "M001-M009" in markdown_table_row(roadmap, "M010")[3]
    assert "AUTONOMOUS_ELIGIBLE only if all gates pass" in markdown_table_row(
        roadmap, "M010"
    )[5]

    for category in (
        "Raw state/read-only connection",
        "Physical readback",
        "Channel fader",
        "Preamp trim/gain-down",
        "Rollback",
        "Emergency stop",
    ):
        assert markdown_table_row(console, category)[-1] in {"none", "none in setup"}

    assert "Full Autonomous Mixing Mode is not ready" in current
    assert "No real hardware evidence was produced" in current
    assert "Goal transition fallback behavior is now covered" in current
    assert "Hardware input required now: yes" in report
    assert "These commits do not create hardware proof" in report
    assert "grant live authority" in report
    assert "Do not contact\nWING from unattended automation" in report


def test_validate_only_preflight_evidence_stays_out_of_real_wing_artifact_root():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    authority = summary["authority"]
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    handoff = CP04_HANDOFF_REPORT.read_text(encoding="utf-8")
    ready = json.loads(VALIDATE_INPUTS_READY_SUMMARY.read_text(encoding="utf-8"))
    mismatch = json.loads(VALIDATE_INPUTS_MISMATCH_SUMMARY.read_text(encoding="utf-8"))

    validate_only_artifacts = summary["cp04_validate_input_artifacts"]
    expected_artifacts = [
        "ops/autopilot/evidence/M001_CP04_validate_inputs_expected_plan_mismatch/summary.json",
        "ops/autopilot/evidence/M001_CP04_validate_inputs_ready_for_websocket/summary.json",
    ]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert summary["hardware_evidence"] is False
    assert summary["product_runtime_changed"] is False
    assert summary["milestone_decision"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert authority["contacts_wing"] is False
    assert authority["contacts_backend"] is False
    assert authority["live_write"] is False
    assert authority["live_authority_granted"] is False
    assert authority["readiness_closure_allowed"] is False

    assert summary["evidence_placement"] == (
        "validate-only summaries are stored under ops/autopilot/evidence, "
        "not artifacts/wing_readonly_preflight"
    )
    assert validate_only_artifacts == expected_artifacts
    assert all(
        not artifact.startswith("artifacts/wing_readonly_preflight")
        for artifact in validate_only_artifacts
    )
    assert files_under(REAL_WING_ARTIFACT_ROOT) == []

    for payload in (ready, mismatch):
        assert payload["validation_only"] is True
        assert payload["read_only"] is True
        assert payload["websocket_opened"] is False
        assert payload["write_request_sent"] is False
        assert payload["contacts_wing"] is False
        assert payload["contacts_backend"] is False
        assert payload["live_write"] is False
        assert payload["mutates_console_state"] is False
        assert payload["queries_console"] is False
        assert payload["readiness_closure_allowed"] is False
        assert payload["readiness_matrix_update_allowed"] is False
        assert payload["live_authority_granted"] is False
        assert payload["artifact_root"] == "artifacts/wing_readonly_preflight"

    assert ready["status"] == "ready_for_websocket"
    assert ready["reason"] == "run_inputs_validated_no_websocket"
    assert ready["expected_plan_match"] is True
    assert ready["execution_allowed_by_validation"] is False
    assert mismatch["status"] == "blocked"
    assert mismatch["reason"] == "expected_plan_mismatch"
    assert mismatch["expected_plan_match"] is False

    for artifact in expected_artifacts:
        assert artifact in report
        assert artifact in handoff
    assert "Validate-only summaries were moved to `ops/autopilot/evidence/`" in report
    assert "not treated as real `artifacts/wing_readonly_preflight` candidates" in report
    assert "M001 still has no real WING read-only artifact under\n" in heartbeat
    assert "`artifacts/wing_readonly_preflight/`" in heartbeat
    assert "No real hardware evidence was produced" in current
    assert "`artifacts/wing_readonly_preflight` root" in current
    assert "websocket_opened: false" in handoff
    assert "write_request_sent: false" in handoff


def test_m002_physical_readback_waits_for_accepted_real_wing_review():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    autopilot_blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    plan = M001_PLAN.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_checkpoint"] == "M001-EXTERNAL-WING-READONLY-ARTIFACT"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["next_action"] == (
        "Wait for operator-supervised WING read-only preflight using "
        "ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json."
    )
    assert state["external_validation_requirements"] == [
        "Real WING Rack read-only artifact",
        "Operator review of accepted read-only preflight candidate",
        "Explicit operator approval before supervised fader/gain proof",
    ]

    m002 = markdown_table_row(roadmap, "M002")
    m003 = markdown_table_row(roadmap, "M003")
    m004 = markdown_table_row(roadmap, "M004")
    m005 = markdown_table_row(roadmap, "M005")
    m006 = markdown_table_row(roadmap, "M006")
    assert m002[1] == "Physical console readback intake"
    assert m002[2] == (
        "Real read-only WING artifact is reviewed and bound to accepted "
        "decision records"
    )
    assert m002[3] == "M001 plus WING hardware/operator"
    assert m002[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    for row in (m003, m004, m005, m006):
        assert row[-1] == "BLOCKED_BY_M002"
    assert m006[2] == "MVP readiness matrix updates only from accepted real proof"

    readonly = markdown_table_row(project_blockers, "WING read-only connection proof")
    physical_readback = markdown_table_row(project_blockers, "Physical console readback proof")
    readiness = markdown_table_row(project_blockers, "Readiness matrix update based on real proof")
    assert readonly[1] == "tooling_ready_waits_real_wing"
    assert physical_readback[1] == "tooling_ready_waits_real_wing"
    assert "Run read-only preflight with `--require-physical-readback`" in physical_readback[3]
    assert readiness[1] == "blocked_by_real_artifact_review"
    assert "Update readiness only after manual review of real WING artifacts" in readiness[3]
    assert "do not close rows from offline evidence" in readiness[3]

    assert (
        "real WING Rack read-only artifact\n  and operator review are required "
        "before physical readback intake can begin"
    ) in autopilot_blockers
    assert "presence and accepted review records" in active
    assert "operator captures and reviews a real WING read-only artifact" in plan
    assert "M001 autonomous pre-hardware work is complete and waiting for external" in current
    assert "WING/operator validation" in current
    assert "Hardware input required now: yes" in report
    assert "intended backend/WING session and operator review" in report
    assert "Wait for the operator-supervised real WING read-only preflight" in report
    assert "There is no dependency-ready autonomous\nroadmap milestone" in report
    assert "until that external validation exists" in report


def test_supervised_fader_gain_proof_stays_blocked_until_m002_and_operator_approval():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    supervised_runbook = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m003 = markdown_table_row(roadmap, "M003")
    m004 = markdown_table_row(roadmap, "M004")
    m009 = markdown_table_row(roadmap, "M009")
    assert m003[1] == "Supervised fader proof"
    assert m003[2] == "Exact bounded fader move is approved, applied, read back and rolled back"
    assert m003[3] == "M002"
    assert m003[4] == "critical live write"
    assert m003[5] == "SUPERVISED_LIVE_VERIFIED for fader only"
    assert m003[-1] == "BLOCKED_BY_M002"
    assert m004[1] == "Supervised gain-down/trim proof"
    assert m004[2] == (
        "Exact bounded gain-down/trim move is approved, applied, read back "
        "and rolled back"
    )
    assert m004[3] == "M002"
    assert m004[4] == "critical live write"
    assert m004[5] == "SUPERVISED_LIVE_VERIFIED for gain-down only"
    assert m004[-1] == "BLOCKED_BY_M002"
    assert m009[1] == "Supervised non-fader/gain live expansion"
    assert "separately approved ADR" in m009[2]
    assert m009[-1] == "PENDING_APPROVAL"

    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    pan = markdown_table_row(console, "Pan")
    rollback = markdown_table_row(console, "Rollback")
    assert fader[3] == "no live write"
    assert fader[4] == "yes only after explicit approval"
    assert fader[-2:] == ["supervised fader proof", "none"]
    assert gain[3] == "no live write"
    assert gain[4] == "yes only after explicit approval and gain-down bounds"
    assert gain[-2:] == ["supervised gain-down proof", "none"]
    assert pan[4] == "no for WING MVP"
    assert "separate ADR and hardware proof" in pan[5]
    assert rollback[4] == "required before closure"
    assert rollback[-1] == "none"

    fader_blocker = markdown_table_row(project_blockers, "Supervised fader write proof")
    gain_blocker = markdown_table_row(project_blockers, "Supervised gain-down / trim proof")
    rollback_blocker = markdown_table_row(project_blockers, "Rollback proof")
    for row in (fader_blocker, gain_blocker, rollback_blocker):
        assert row[1] == "blocked_by_readonly_preflight"
    assert "run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type fader`" in fader_blocker[3]
    assert (
        "run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for "
        "`--value-type gain` with a negative delta only"
    ) in gain_blocker[3]
    assert "Preserve rollback request, rollback readback, and artifact review" in rollback_blocker[3]

    assert "Status: blocked until a clean read-only preflight candidate" in supervised_runbook
    assert "Allowed write kinds for this MVP:" in supervised_runbook
    assert "- `fader`" in supervised_runbook
    assert "- `gain` with a negative delta only" in supervised_runbook
    for forbidden in ("EQ", "compressor", "gate", "routing", "mute/unmute", "FX", "scenes"):
        assert f"- {forbidden}" in supervised_runbook
    assert "The operator explicitly approves the exact channel, value type, and delta" in supervised_runbook
    assert "The selected delta is small and negative" in supervised_runbook
    assert "rollback went through `rollback_last_manual_write`" in supervised_runbook
    assert "rollback readback status is `verified_from_physical_readback`" in supervised_runbook
    assert "the delta is positive" in supervised_runbook
    assert "the value type is not `fader` or `gain`" in supervised_runbook
    assert "Do not claim full autonomous readiness" in supervised_runbook

    assert "Supervised fader write proof with explicit operator approval" in goal
    assert "Supervised gain-down / trim proof with explicit operator approval" in goal
    assert "supervised `fader`/`gain` rail с явным approval, readback и rollback" in goal


def test_emergency_stop_timeout_proof_stays_blocked_until_m002_and_real_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    supervised_runbook = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    authority = state["last_validation_summary"]["authority"]
    for key in (
        "live_authority_granted",
        "readiness_closure_allowed",
        "contacts_wing",
        "contacts_backend",
        "live_write",
    ):
        assert authority[key] is False

    m005 = markdown_table_row(roadmap, "M005")
    m006 = markdown_table_row(roadmap, "M006")
    assert m005[1] == "Emergency stop and timeout proof"
    assert m005[2] == (
        "Emergency stop, disconnect, timeout and stale-state protections "
        "are verified"
    )
    assert m005[3] == "M002"
    assert m005[4] == "critical safety"
    assert m005[5] == "SIMULATOR_VERIFIED plus supervised hardware evidence"
    assert m005[-1] == "BLOCKED_BY_M002"
    assert m006[1] == "Operator evidence and readiness update"
    assert m006[3] == "M002-M005"
    assert "accepted real proof" in m006[2]
    assert m006[-1] == "BLOCKED_BY_M002"

    rollback = markdown_table_row(console, "Rollback")
    emergency = markdown_table_row(console, "Emergency stop")
    stale_disconnect = markdown_table_row(console, "Stale state / disconnect fail-closed")
    assert rollback[3] == "no live action unattended"
    assert rollback[4] == "required before closure"
    assert rollback[-2:] == ["supervised rollback proof", "none"]
    assert emergency[3] == "no live action unattended"
    assert emergency[4] == "required before closure"
    assert emergency[-2:] == ["emergency-stop proof", "none"]
    assert stale_disconnect[3] == "no hardware unattended"
    assert stale_disconnect[4] == "required"
    assert stale_disconnect[-2:] == ["timeout/fail-closed proof", "none"]

    rollback_blocker = markdown_table_row(project_blockers, "Rollback proof")
    emergency_blocker = markdown_table_row(project_blockers, "Emergency-stop proof")
    timeout_blocker = markdown_table_row(
        project_blockers, "Connection-loss / timeout fail-closed proof"
    )
    assert rollback_blocker[1] == "blocked_by_readonly_preflight"
    assert emergency_blocker[1] == "blocked_by_readonly_preflight"
    assert timeout_blocker[1] == "tooling_ready_waits_real_wing"
    assert "rollback through `rollback_last_manual_write`" in rollback_blocker[2]
    assert "runner supports `--prove-emergency-stop`" in emergency_blocker[2]
    assert "blocks later apply with `emergency_stop_active`" in emergency_blocker[2]
    assert (
        "Use `--prove-emergency-stop` only inside the explicit supervised proof run."
        in emergency_blocker[3]
    )
    assert "controlled real-run timeout/connection-loss artifact" in timeout_blocker[3]
    assert "verify fail-closed semantics" in timeout_blocker[3]

    assert "Rollback proof." in goal
    assert "Emergency stop proof." in goal
    assert "Connection-loss / timeout fail-closed proof." in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal

    assert "Unknown and insufficient evidence are valid safe outcomes." in safety
    assert "send live OSC writes;" in safety
    assert "control a real console;" in safety
    assert "modify the user's WING scene;" in safety
    assert "stale-state protection" in safety
    assert "readback, rollback or audit logging" in safety
    assert "describe simulated validation as real hardware validation;" in safety
    assert "weaken safety tests to pass a change." in safety

    assert supervised_runbook.count("--prove-emergency-stop") == 2
    assert "rollback went through `rollback_last_manual_write`" in supervised_runbook
    assert "rollback readback status is `verified_from_physical_readback`" in supervised_runbook
    assert "emergency stop was applied" in supervised_runbook
    assert "later apply was blocked with `emergency_stop_active`" in supervised_runbook
    assert "emergency stop does not block a later apply" in supervised_runbook
    assert "timeout happens after `apply_operator_proposal`" in supervised_runbook
    assert "physical state is not\n  manually inspected" in supervised_runbook
    assert "any direct write path is used instead of `apply_operator_proposal`" in supervised_runbook
    assert "Do not claim full autonomous readiness." in supervised_runbook


def test_m006_readiness_update_stays_report_only_until_real_m002_m005_proof():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m006 = markdown_table_row(roadmap, "M006")
    assert m006[1] == "Operator evidence and readiness update"
    assert m006[2] == "MVP readiness matrix updates only from accepted real proof"
    assert m006[3] == "M002-M005"
    assert m006[4] == "high audit"
    assert m006[5] == "HARDWARE_READ_VERIFIED or SUPERVISED_LIVE_VERIFIED evidence surfaced"
    assert m006[-1] == "BLOCKED_BY_M002"

    operator_report = markdown_table_row(project_blockers, "Operator-visible evidence report")
    readiness = markdown_table_row(project_blockers, "Readiness matrix update based on real proof")
    assert operator_report[1] == "tooling_ready_waits_real_wing"
    assert "Save preflight/proof artifacts" in operator_report[3]
    assert "manual review" in operator_report[3]
    assert readiness[1] == "blocked_by_real_artifact_review"
    assert "Update readiness only after manual review of real WING artifacts" in readiness[3]
    assert "do not close rows from offline evidence" in readiness[3]

    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    audit_logging = markdown_table_row(console, "Audit logging")
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert full_auto[1] == "NOT_STARTED"
    assert "Locked until all safety and hardware gates pass" in full_auto[4]
    assert full_auto[8] == "none"
    assert audit_logging[4] == "required"
    assert audit_logging[-2:] == ["artifact review", "none"]

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert "real_wing_and_emergency_stop_proof_incomplete" in snapshot["summary"]["global_blockers"]
    assert preflight["read_only"] is True
    assert preflight["live_write"] is False
    assert preflight["live_widening_allowed"] is False
    assert preflight["automatic_safety_update_allowed"] is False
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["closed_blocker_ids"] == []
    assert "Do not update readiness rows from missing or invalid preflight evidence." in (
        preflight["next_required_actions"]
    )
    assert "manual readiness review only" in preflight["claim"]

    assert "Operator-visible evidence report." in goal
    assert "Readiness matrix update based only on real proof." in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "MVP readiness/operator evidence update is now covered" in current
    assert "readiness/operator evidence update gate rail is now automated" in report


def test_m007_offline_promotion_framework_stays_replay_bound_before_live_expansion():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m007 = markdown_table_row(roadmap, "M007")
    m008 = markdown_table_row(roadmap, "M008")
    m009 = markdown_table_row(roadmap, "M009")
    m010 = markdown_table_row(roadmap, "M010")
    assert m007[1] == "Offline capability promotion framework"
    assert m007[2] == (
        "Long-term processing families have replay/fault/evaluation gates before any live expansion"
    )
    assert m007[3] == "M001-M006"
    assert m007[4] == "high offline"
    assert m007[5] == "PROTOTYPE/UNIT_VERIFIED to REPLAY_VERIFIED"
    assert m007[-1] == "PENDING"
    assert m008[3] == "M007"
    assert m008[-1] == "PENDING"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[3] == "M006-M008 plus hardware"
    assert m009[4] == "critical live write"
    assert m009[-1] == "PENDING_APPROVAL"
    assert m010[5] == "AUTONOMOUS_ELIGIBLE only if all gates pass"
    assert m010[-1] == "PENDING_APPROVAL"

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }
    assert snapshot["summary"]["next_global_step"] == (
        "Run proof gates per action family before widening supervised or autonomous live writes."
    )

    replay_bound_capabilities = {
        "Assisted Mixing Mode": ("INTEGRATION_VERIFIED", "replay accepted/rejected proposals", "none"),
        "Full Autonomous Mixing Mode": ("NOT_STARTED", "completion gate review", "none"),
        "Multichannel audio analysis": (
            "UNIT_VERIFIED",
            "replay and simulator validation",
            "UNKNOWN",
        ),
        "Source and instrument classification": ("UNIT_VERIFIED", "replay calibration gate", "none"),
        "Arrangement-role detection": ("PROTOTYPE", "replay accepted role decisions", "none"),
        "Mix balance": ("PROTOTYPE", "replay bus/stem validation", "none"),
        "EQ": ("PROTOTYPE", "assisted replay gate", "none"),
        "Compression": ("PROTOTYPE", "replay and safety audit", "none"),
        "Gating": ("PROTOTYPE", "replay and fault injection", "none"),
        "Panning": ("PROTOTYPE", "assisted replay only", "none"),
        "Frequency separation": ("PROTOTYPE", "replay validation", "none"),
        "Cross-channel unmasking": ("PROTOTYPE", "replay/fault validation", "none"),
        "Effects and sends": ("PROTOTYPE", "assisted replay", "none"),
        "Master-bus processing": ("PROTOTYPE", "offline and listening gate", "none"),
        "Adaptive live mixing": ("SPECIFIED", "simulator and supervised staged proof", "none"),
        "Long-term memory and evaluation": ("PROTOTYPE", "lint plus replay promotion", "none"),
    }
    for capability_name, (evidence_level, next_gate, hardware_evidence) in replay_bound_capabilities.items():
        row = markdown_table_row(capability, capability_name)
        assert row[1] == evidence_level
        assert row[7] == next_gate
        assert row[8] == hardware_evidence

    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence." in capability
    assert "Proposal-only, not apply authority" in markdown_table_row(
        capability, "Arrangement-role detection"
    )[4]
    assert "Live EQ writes forbidden" in markdown_table_row(capability, "EQ")[4]
    assert "Live FX writes forbidden" in markdown_table_row(capability, "Effects and sends")[4]
    assert "No autonomous live authority" in markdown_table_row(
        capability, "Adaptive live mixing"
    )[4]
    assert "Memory cannot mutate console directly" in markdown_table_row(
        capability, "Long-term memory and evaluation"
    )[4]

    live_forbidden_console_categories = {
        "Pan": "no for WING MVP",
        "EQ / HPF": "no",
        "Compressor / dynamics": "no",
        "Gate / expander": "no",
        "FX sends / returns": "no",
        "Routing / patch": "no",
        "Scenes / snapshots / presets": "no",
    }
    for console_category, supervised_live_state in live_forbidden_console_categories.items():
        row = markdown_table_row(console, console_category)
        assert row[3] == "no"
        assert row[4] == supervised_live_state
        assert row[6] == "none"

    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[3] == "docs/tests only"
    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[4] == "required"
    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[5] == (
        "fault-injection validation"
    )

    assert "offline, replay," in completion
    assert "fault-injection и integration evaluations" in completion
    assert "High-risk console actions прошли staged real-hardware testing" in completion
    assert "PENDING_APPROVAL" in completion
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal

    assert "Offline capability promotion framework entry is now covered" in current
    assert "M007 offline capability promotion framework rail is now automated" in report


def test_m008_assisted_processing_expansion_stays_proposal_only_with_identity_gates():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m008 = markdown_table_row(roadmap, "M008")
    m009 = markdown_table_row(roadmap, "M009")
    assert m008[1] == "Assisted processing family expansion"
    assert m008[2] == (
        "EQ, dynamics, pan, FX and balance stay proposal-only with source identity and safety gates"
    )
    assert m008[3] == "M007"
    assert m008[4] == "high assisted"
    assert m008[5] == "UNIT_VERIFIED to INTEGRATION_VERIFIED"
    assert m008[-1] == "PENDING"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[-1] == "PENDING_APPROVAL"

    assisted = markdown_table_row(capability, "Assisted Mixing Mode")
    source_identity = markdown_table_row(capability, "Source and instrument classification")
    arrangement_role = markdown_table_row(capability, "Arrangement-role detection")
    assert assisted[1] == "INTEGRATION_VERIFIED"
    assert assisted[4] == "Proposal freshness and source identity vary by family"
    assert assisted[7] == "replay accepted/rejected proposals"
    assert assisted[8] == "none"
    assert source_identity[7] == "replay calibration gate"
    assert arrangement_role[4] == "Proposal-only, not apply authority"
    assert arrangement_role[7] == "replay accepted role decisions"

    assisted_family_rows = {
        "Mix balance": ("PROTOTYPE", "source/stem identity", "replay bus/stem validation"),
        "EQ": ("PROTOTYPE", "feedback, band identity", "assisted replay gate"),
        "Compression": ("PROTOTYPE", "true peak, GR readback", "replay and safety audit"),
        "Gating": ("PROTOTYPE", "source identity, hysteresis", "replay and fault injection"),
        "Panning": ("PROTOTYPE", "mono/phase safety", "assisted replay only"),
        "Effects and sends": ("PROTOTYPE", "routing/readback", "assisted replay"),
        "Cross-channel unmasking": ("PROTOTYPE", "arrangement role, EQ safety", "replay/fault validation"),
    }
    for capability_name, (evidence_level, dependency, next_gate) in assisted_family_rows.items():
        row = markdown_table_row(capability, capability_name)
        assert row[1] == evidence_level
        assert row[6] == dependency
        assert row[7] == next_gate
        assert row[8] == "none"

    assert "Live EQ writes forbidden" in markdown_table_row(capability, "EQ")[4]
    assert "Makeup cannot bypass headroom/feedback" in markdown_table_row(
        capability, "Compression"
    )[4]
    assert "Chatter/bleed risk in live mode" in markdown_table_row(capability, "Gating")[4]
    assert "WING live scope is not approved for pan" in markdown_table_row(
        capability, "Panning"
    )[4]
    assert "Live FX writes forbidden" in markdown_table_row(capability, "Effects and sends")[4]

    assisted_snapshot_rows = {
        "eq_supervised_promotion": "read_coverage_blocked",
        "dynamics_supervised_promotion": "read_coverage_blocked",
        "gate_supervised_promotion": "read_coverage_blocked",
        "pan_phase_supervised_promotion": "manual_only_blocked",
        "fx_supervised_promotion": "manual_only_blocked",
        "bus_group_matrix_supervised_promotion": "manual_only_blocked",
    }
    rows_by_id = {row["id"]: row for row in snapshot["rows"]}
    for row_id, expected_status in assisted_snapshot_rows.items():
        row = rows_by_id[row_id]
        assert row["status"] == expected_status
        assert row["supervised_live_allowed"] is False
        assert row["autonomous_ready"] is False
        assert set(row["capability_statuses"]) == {"proposal_only"}
        assert "canonical_supervised" not in row["write_surface_statuses"]
        assert set(row["write_surface_statuses"]) <= {"proposal_only", "quarantined"}
        assert "safety_live_not_allowed" in row["blockers"]
        assert "not_autonomous_ready" in row["blockers"]

    console_assisted_boundaries = {
        "EQ / HPF": ("no", "no", "protected-band, feedback, readback, rollback proof"),
        "Compressor / dynamics": ("no", "no", "headroom, GR and readback proof"),
        "Gate / expander": ("no", "no", "chatter/bleed replay plus hardware proof"),
        "Pan": ("no", "no for WING MVP", "separate ADR and hardware proof"),
        "FX sends / returns": ("no", "no", "readback and rollback proof"),
        "Bus/group/main/DCA faders": (
            "no",
            "no except later approved supervised proof",
            "physical readback",
        ),
    }
    for category, (unattended, supervised, next_gate) in console_assisted_boundaries.items():
        row = markdown_table_row(console, category)
        assert row[3] == unattended
        assert row[4] == supervised
        assert row[5] == next_gate
        assert row[6] == "none"

    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"]["live_write"] is False
    assert snapshot["safety"]["live_widening_allowed"] is False
    assert "Assisted processing family expansion entry is now covered" in current
    assert "M008 assisted processing family expansion rail is now automated" in report


def test_m009_non_fader_gain_live_expansion_requires_adr_hardware_and_family_proof():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m009 = markdown_table_row(roadmap, "M009")
    m010 = markdown_table_row(roadmap, "M010")
    assert m009[1] == "Supervised non-fader/gain live expansion"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[3] == "M006-M008 plus hardware"
    assert m009[4] == "critical live write"
    assert m009[5] == "SUPERVISED_LIVE_VERIFIED per family"
    assert m009[-1] == "PENDING_APPROVAL"
    assert m010[3] == "M001-M009"
    assert m010[-1] == "PENDING_APPROVAL"

    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    assert supervised[1] == "SPECIFIED"
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert full_auto[1] == "NOT_STARTED"
    assert "Locked until all safety and hardware gates pass" in full_auto[4]
    assert full_auto[8] == "none"

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["supervised_live_allowed_target_ids"] == [
        "fader_supervised_promotion"
    ]
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }

    rows_by_id = {row["id"]: row for row in snapshot["rows"]}
    non_fader_gain_rows = {
        "eq_supervised_promotion": "eq",
        "dynamics_supervised_promotion": "compressor",
        "gate_supervised_promotion": "gate",
        "pan_phase_supervised_promotion": "pan_phase_delay",
        "fx_supervised_promotion": "fx_send_insert",
        "bus_group_matrix_supervised_promotion": "bus_group_matrix",
        "mute_supervised_promotion": "mute",
        "routing_supervised_promotion": "routing",
        "scene_preset_supervised_promotion": "scene_preset",
    }
    for row_id, action_family in non_fader_gain_rows.items():
        row = rows_by_id[row_id]
        assert row["action_family"] == action_family
        assert row["supervised_live_allowed"] is False
        assert row["autonomous_ready"] is False
        assert "canonical_supervised" not in row["write_surface_statuses"]
        assert set(row["write_surface_statuses"]) <= {"proposal_only", "quarantined"}
        assert "no_allowed_live_write_surface" in row["blockers"]
        assert "safety_live_not_allowed" in row["blockers"]
        assert any(
            blocker in row["blockers"]
            for blocker in ("real_wing_proof_missing", "real_wing_proof_not_allowed")
        )
        assert "rollback_readback_missing" in row["blockers"]
        assert "emergency_stop_proof_missing" in row["blockers"]

    allowed_fader = rows_by_id["fader_supervised_promotion"]
    gain_trim = rows_by_id["gain_trim_supervised_promotion"]
    assert allowed_fader["action_family"] == "fader"
    assert allowed_fader["supervised_live_allowed"] is True
    assert gain_trim["action_family"] == "gain"
    assert gain_trim["supervised_live_allowed"] is False
    assert "real_wing_proof_missing" in gain_trim["blockers"]
    assert "apply_readback_missing" in gain_trim["blockers"]

    console_live_expansion_rows = {
        "Pan": ("no", "no for WING MVP", "separate ADR and hardware proof"),
        "Mute/unmute": ("no", "no", "separate ADR and proof"),
        "EQ / HPF": ("no", "no", "protected-band, feedback, readback, rollback proof"),
        "Compressor / dynamics": ("no", "no", "headroom, GR and readback proof"),
        "Gate / expander": ("no", "no", "chatter/bleed replay plus hardware proof"),
        "De-esser": ("no", "no", "unit/replay first"),
        "FX sends / returns": ("no", "no", "readback and rollback proof"),
        "Routing / patch": ("no", "no", "read-only physical readback first"),
        "Bus/group/main/DCA faders": (
            "no",
            "no except later approved supervised proof",
            "physical readback",
        ),
        "Matrices": ("no", "no", "physical readback"),
        "Scenes / snapshots / presets": ("no", "no", "destructive-action ADR and proof"),
    }
    for category, (unattended, supervised_live, next_gate) in console_live_expansion_rows.items():
        row = markdown_table_row(console, category)
        assert row[3] == unattended
        assert row[4] == supervised_live
        assert row[5] == next_gate
        assert row[6] == "none"

    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    assert fader[4] == "yes only after explicit approval"
    assert fader[5:] == ["supervised fader proof", "none"]
    assert gain[4] == "yes only after explicit approval and gain-down bounds"
    assert gain[5:] == ["supervised gain-down proof", "none"]

    assert "High-risk console actions прошли staged real-hardware testing" in completion
    assert "Recovery, rollback, emergency-stop and audit behavior verified" in completion
    assert "Любая live WING запись до отдельного ADR и hardware proof ограничена только" in goal
    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Supervised non-fader/gain live expansion entry is now covered" in current
    assert "M009 supervised non-fader/gain live expansion rail is now automated" in report


def test_m010_full_autonomous_eligibility_stays_locked_until_completion_gates_pass():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"] == {
        "contacts_backend": False,
        "contacts_wing": False,
        "live_authority_granted": False,
        "live_write": False,
        "readiness_closure_allowed": False,
    }

    m010 = markdown_table_row(roadmap, "M010")
    assert m010[1] == "Full autonomous eligibility review"
    assert m010[2] == "Full Autonomous Mixing Mode eligibility is reviewed against all completion gates"
    assert m010[3] == "M001-M009"
    assert m010[4] == "critical autonomous"
    assert m010[5] == "AUTONOMOUS_ELIGIBLE only if all gates pass"
    assert m010[-1] == "PENDING_APPROVAL"

    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    multichannel = markdown_table_row(capability, "Multichannel audio analysis")
    assert full_auto == [
        "Full Autonomous Mixing Mode",
        "NOT_STARTED",
        "`backend/operator_mode_policy.py`, `Docs/PROJECT_COMPLETION_GATE.md`",
        "none sufficient",
        "Locked until all safety and hardware gates pass",
        "critical",
        "all mandatory capabilities",
        "completion gate review",
        "none",
    ]
    assert supervised[1] == "SPECIFIED"
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert multichannel[1] == "UNIT_VERIFIED"
    assert multichannel[-1] == "UNKNOWN"

    for capability_name in (
        "Gain staging",
        "Mix balance",
        "EQ",
        "Compression",
        "Gating",
        "Panning",
        "Effects and sends",
        "Master-bus processing",
        "Long-term memory and evaluation",
    ):
        row = markdown_table_row(capability, capability_name)
        assert row[1] != "AUTONOMOUS_ELIGIBLE"
        assert row[-1] == "none"

    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence" in capability
    assert "Все mandatory capabilities" in completion
    assert "Analysis Mode, Assisted Mixing Mode, Supervised AutoMix Mode и Full" in completion
    assert "Autonomous Mixing Mode" in completion
    assert "WING Rack read/write protocol coverage" in completion
    assert "deterministic Safety Kernel" in completion
    assert "offline, replay,\n   simulator, fault-injection" in completion
    assert "staged real-hardware testing" in completion
    assert "PENDING_APPROVAL" in completion
    assert "COMPLETE` запрещен" in completion

    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal
    assert "mock и simulator evidence не являются hardware proof" in goal
    assert "Любая live WING запись до отдельного ADR и hardware proof" in goal

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["summary"]["supervised_live_allowed_target_ids"] == [
        "fader_supervised_promotion"
    ]
    assert snapshot["summary"]["next_global_step"] == (
        "Run proof gates per action family before widening supervised or "
        "autonomous live writes."
    )
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }

    for row in snapshot["rows"]:
        assert row["autonomous_ready"] is False
        assert "not_autonomous_ready" in row["blockers"]

    for category in (
        "EQ / HPF",
        "Compressor / dynamics",
        "Gate / expander",
        "FX sends / returns",
        "Routing / patch",
        "Scenes / snapshots / presets",
        "Matrices",
    ):
        row = markdown_table_row(console, category)
        assert row[3] == "no"
        assert row[4] == "no"
        assert row[-1] == "none"

    assert "Full autonomous eligibility review entry is now covered" in current
    assert "completion gate cannot close from docs/unit/mock/replay/no-WING evidence" in current
    assert "M010 full autonomous eligibility rail is now automated" in report
    assert "no autonomous target is reported ready" in report


def test_m011_offline_replay_evaluation_contract_stays_offline_and_threshold_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M011_PLAN.read_text(encoding="utf-8")
    adr = M011_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False

    m011 = markdown_table_row(roadmap, "M011")
    assert m011[1] == "Offline replay evaluation contract"
    assert m011[2] == (
        "Soundcheck, studio and concert replay manifests fail closed until "
        "approved thresholds and complete scenario evidence exist"
    )
    assert m011[3] == "M007-M010"
    assert m011[4] == "high offline"
    assert (
        m011[5]
        == "SPECIFIED to UNIT_VERIFIED for gate contract; replay evidence remains pending"
    )
    assert m011[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert memory_eval[1] == "PROTOTYPE"
    assert "`backend/offline_replay_evaluation_contract.py`" in memory_eval[2]
    assert "tests/test_offline_replay_evaluation_contract.py" in memory_eval[3]
    assert "approved thresholds are still required" in memory_eval[4]
    assert memory_eval[-1] == "none"

    pending_manifest = {
        "manifest_id": "pending-thresholds",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "threshold_policy": {"status": "PENDING_APPROVAL"},
        "scenarios": [
            {
                "scenario": "soundcheck",
                "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
                "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
                "quality_axes": {
                    key: {"status": "passed"} for key in REQUIRED_QUALITY_AXES
                },
            }
        ],
    }
    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def complete_threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    pending_manifest["quality_threshold_policy_manifest"] = (
        complete_threshold_policy_manifest(
            threshold_policy={"status": "PENDING_APPROVAL"}
        )
    )
    pending_result = evaluate_offline_replay_evaluation_contract(pending_manifest)
    assert pending_result["ready_for_replay_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_quality_threshold_policy_required" in pending_result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in pending_result["blockers"]
    assert (
        "quality_threshold_policy:owner_approved_quality_threshold_policy_required"
        in pending_result["blockers"]
    )
    assert "missing_required_scenario:studio" in pending_result["blockers"]
    assert "missing_required_scenario:concert" in pending_result["blockers"]

    complete_result = evaluate_offline_replay_evaluation_contract(
        {
            "manifest_id": "complete-offline-replay",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": complete_threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }
    )
    assert complete_result["ready_for_replay_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["covered_scenarios"] == list(REQUIRED_SCENARIOS)
    assert complete_result["summary"]["threshold_policy_contract_status"] == (
        "ready_for_threshold_policy_use"
    )

    assert "offline, replay,\n   simulator, fault-injection" in completion
    assert "approved thresholds" in completion
    assert "M011-CP01" in plan
    assert "M011-CP02" in plan
    assert "Offline replay evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M011 offline replay evaluation contract is now covered" in current
    assert "M011 offline replay evaluation contract rail is now automated" in report


def test_m012_fault_injection_recovery_contract_stays_simulator_only_and_audit_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M012_PLAN.read_text(encoding="utf-8")
    adr = M012_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m012 = markdown_table_row(roadmap, "M012")
    assert m012[1] == "Fault-injection recovery contract"
    assert m012[2] == (
        "Simulator fault bundles fail closed until required failure modes, "
        "recovery outcomes and audit evidence exist"
    )
    assert m012[3] == "M011"
    assert m012[4] == "high offline"
    assert (
        m012[5]
        == "SPECIFIED to UNIT_VERIFIED for recovery gate; simulator evidence remains pending"
    )
    assert m012[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/fault_injection_recovery_contract.py`" in memory_eval[2]
    assert "tests/test_fault_injection_recovery_contract.py" in memory_eval[3]
    assert "fault-recovery simulator evidence cannot close live readiness" in memory_eval[4]
    assert memory_eval[-1] == "none"

    rate_limits = markdown_table_row(console, "Rate limits / bounds / allowlists")
    stale_state = markdown_table_row(console, "Stale state / disconnect fail-closed")
    audit_logging = markdown_table_row(console, "Audit logging")
    assert "`backend/fault_injection_recovery_contract.py`" in rate_limits[2]
    assert rate_limits[3] == "docs/tests only"
    assert rate_limits[4] == "required"
    assert rate_limits[5] == "fault-injection validation"
    assert stale_state[5] == "timeout/fail-closed proof"
    assert audit_logging[5] == "artifact review"
    assert rate_limits[6] == "none"
    assert stale_state[6] == "none"
    assert audit_logging[6] == "none"

    pending_manifest = {
        "manifest_id": "pending-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "recovery_policy_manifest": {
            "policy_id": "pending-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "recovery_controls": {
                key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
            },
            "recovery_evidence": {
                key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
            },
            "recovery_policy": {"status": "PENDING_APPROVAL"},
        },
        "fault_scenarios": [
            {
                "fault_scenario": "transport_timeout",
                "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
                "recovery_outcomes": {
                    key: True for key in REQUIRED_RECOVERY_OUTCOMES
                },
                "audit": {
                    key: f"transport-timeout-{key}"
                    for key in RECOVERY_REQUIRED_AUDIT_FIELDS
                },
            }
        ],
    }
    pending_result = evaluate_fault_injection_recovery_contract(pending_manifest)
    assert pending_result["ready_for_fault_recovery_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_recovery_policy_required" in pending_result["blockers"]
    assert "missing_required_fault_scenario:connection_loss" in pending_result["blockers"]
    assert "missing_required_fault_scenario:emergency_stop" in pending_result["blockers"]

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    complete_result = evaluate_fault_injection_recovery_contract(
        {
            "manifest_id": "complete-fault-recovery",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": {
                "policy_id": "approved-recovery-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "recovery_controls": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
                },
                "recovery_evidence": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
                },
                "recovery_policy": {
                    "status": "approved",
                    "approved_by": "owner",
                    "approved_decision_id": "adr:fault-recovery-policy",
                    "approved_at": "2026-06-19T18:00:10Z",
                },
            },
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }
    )
    assert complete_result["ready_for_fault_recovery_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["covered_fault_scenarios"] == list(
        RECOVERY_REQUIRED_FAULT_SCENARIOS
    )

    assert "Recovery, rollback, emergency-stop and audit behavior verified" in completion
    assert "fault-injection и integration evaluations" in completion
    assert "M012-CP01" in plan
    assert "M012-CP02" in plan
    assert "Simulator fault evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M012 fault-injection recovery contract is now covered" in current
    assert "M012 fault-injection recovery contract rail is now automated" in report


def test_m013_integration_evaluation_bundle_contract_stays_offline_and_child_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M013_PLAN.read_text(encoding="utf-8")
    adr = M013_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m013 = markdown_table_row(roadmap, "M013")
    assert m013[1] == "Integration evaluation bundle contract"
    assert m013[2] == (
        "Replay and fault-recovery manifests are composed into an offline "
        "integration bundle before promotion"
    )
    assert m013[3] == "M011-M012"
    assert m013[4] == "high offline"
    assert (
        m013[5]
        == "SPECIFIED to UNIT_VERIFIED for integration gate; integration evidence remains pending"
    )
    assert m013[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/evaluation_integration_bundle_contract.py`" in memory_eval[2]
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration bundle evidence cannot close live readiness" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def complete_threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def complete_replay_manifest(**overrides) -> dict:
        manifest = {
            "manifest_id": "complete-offline-replay",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": complete_threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }
        manifest.update(overrides)
        return manifest

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def complete_fault_manifest(**overrides) -> dict:
        manifest = {
            "manifest_id": "complete-fault-recovery",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": {
                "policy_id": "approved-recovery-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "recovery_controls": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
                },
                "recovery_evidence": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
                },
                "recovery_policy": {
                    "status": "approved",
                    "approved_by": "owner",
                    "approved_decision_id": "adr:fault-recovery-policy",
                    "approved_at": "2026-06-19T18:00:10Z",
                },
            },
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }
        manifest.update(overrides)
        return manifest

    def complete_integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "complete-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-evaluation-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    pending_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "pending-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "integration_policy_manifest": complete_integration_policy_manifest(
                integration_policy={"status": "PENDING_APPROVAL"}
            ),
            "integration_checks": {"analysis_trace_bound": True},
            "offline_replay_manifest": complete_replay_manifest(
                quality_threshold_policy_manifest=complete_threshold_policy_manifest(
                    threshold_policy={"status": "PENDING_APPROVAL"}
                ),
                scenarios=[complete_scenario("soundcheck")],
            ),
            "fault_recovery_manifest": complete_fault_manifest(
                fault_scenarios=[complete_fault("transport_timeout")]
            ),
        }
    )
    assert pending_result["ready_for_integration_evaluation_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_integration_policy_required" in pending_result["blockers"]
    assert "integration_policy_contract_not_ready" in pending_result["blockers"]
    assert (
        "integration_policy:owner_approved_integration_policy_required"
        in pending_result["blockers"]
    )
    assert "offline_replay_contract_not_ready" in pending_result["blockers"]
    assert "fault_recovery_contract_not_ready" in pending_result["blockers"]
    assert "offline_replay:approved_quality_threshold_policy_required" in pending_result[
        "blockers"
    ]
    assert "offline_replay:quality_threshold_policy_contract_not_ready" in pending_result[
        "blockers"
    ]
    assert (
        "offline_replay:quality_threshold_policy:"
        "owner_approved_quality_threshold_policy_required"
        in pending_result["blockers"]
    )
    assert "fault_recovery:missing_required_fault_scenario:connection_loss" in pending_result[
        "blockers"
    ]
    assert "missing_integration_check:fault_recovery_result_bound" in pending_result[
        "blockers"
    ]

    complete_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "complete-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": complete_integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": complete_replay_manifest(),
            "fault_recovery_manifest": complete_fault_manifest(),
        }
    )
    assert complete_result["ready_for_integration_evaluation_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["offline_replay_ready"] is True
    assert complete_result["summary"]["fault_recovery_ready"] is True
    assert complete_result["summary"]["missing_integration_checks"] == []

    assert "offline, replay,\n   simulator, fault-injection и integration evaluations" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M013-CP01" in plan
    assert "M013-CP02" in plan
    assert "Integration bundle evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M013 integration evaluation bundle contract is now covered" in current
    assert "M013 integration evaluation bundle contract rail is now automated" in report


def test_m014_quality_threshold_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M014_PLAN.read_text(encoding="utf-8")
    adr = M014_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m014 = markdown_table_row(roadmap, "M014")
    assert m014[1] == "Quality threshold policy contract"
    assert m014[2] == (
        "Candidate quality thresholds are complete, evidence-bound and "
        "owner-approved before replay or integration promotion"
    )
    assert m014[3] == "M011-M013"
    assert m014[4] == "high offline"
    assert (
        m014[5]
        == "SPECIFIED to UNIT_VERIFIED for threshold gate; owner approval remains required"
    )
    assert m014[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/quality_threshold_policy_contract.py`" in memory_eval[2]
    assert "tests/test_quality_threshold_policy_contract.py" in memory_eval[3]
    assert "quality threshold policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"artifacts/evaluation/{axis}.json",
            "rationale": f"Candidate threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold if regression appears",
        }

    candidate_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "candidate-quality-thresholds",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {"status": "candidate"},
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_threshold_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_quality_threshold_policy_required"
    ]

    approved_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "approved-quality-thresholds",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:quality-threshold-policy",
                "approved_at": "2026-06-19T17:20:04Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
    )
    assert approved_result["ready_for_threshold_policy_use"] is True
    assert approved_result["threshold_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["covered_threshold_axes"] == list(
        REQUIRED_THRESHOLD_AXES
    )

    blocked_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "blocked-quality-thresholds",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "policy_checks": {"baseline_dataset_bound": True},
            "threshold_policy": {"status": "PENDING_APPROVAL"},
            "thresholds": [complete_threshold("objective_audio_metrics")],
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "quality_threshold_policy_forbids_live_write" in blocked_result["blockers"]
    assert "quality_threshold_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "threshold_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_threshold_axis:reference_mix_comparison" in blocked_result[
        "blockers"
    ]

    assert REQUIRED_THRESHOLD_AXES == REQUIRED_QUALITY_AXES
    assert "Candidate threshold policies must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M014-CP01" in plan
    assert "M014-CP02" in plan
    assert "Candidate threshold policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M014 quality threshold policy contract is now covered" in current
    assert "M014 quality threshold policy contract rail is now automated" in report


def test_m015_replay_threshold_policy_binding_requires_m014_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M015_PLAN.read_text(encoding="utf-8")
    adr = M015_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m015 = markdown_table_row(roadmap, "M015")
    assert m015[1] == "Replay threshold policy binding"
    assert m015[2] == (
        "Offline replay promotion consumes the quality threshold policy contract "
        "instead of loose threshold status fields"
    )
    assert m015[3] == "M011-M014"
    assert m015[4] == "high offline"
    assert m015[5] == "UNIT_VERIFIED gate binding; replay evidence remains pending"
    assert m015[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/offline_replay_evaluation_contract.py`" in memory_eval[2]
    assert "`backend/quality_threshold_policy_contract.py`" in memory_eval[2]
    assert "tests/test_offline_replay_evaluation_contract.py" in memory_eval[3]
    assert "replay promotion now requires a bound quality threshold policy manifest" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    legacy_only_replay = {
        "manifest_id": "legacy-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "threshold_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:legacy-threshold-policy",
        },
        "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    legacy_result = evaluate_offline_replay_evaluation_contract(legacy_only_replay)
    assert legacy_result["ready_for_replay_promotion"] is False
    assert "approved_quality_threshold_policy_required" in legacy_result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_threshold_policy_without_contract_manifest" in legacy_result["blockers"]
    assert "quality_threshold_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_threshold_policy_status"] == "approved"
    assert legacy_result["summary"]["thresholds_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    complete_replay = {
        "manifest_id": "bound-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "quality_threshold_policy_manifest": threshold_policy_manifest(),
        "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    complete_result = evaluate_offline_replay_evaluation_contract(complete_replay)
    assert complete_result["ready_for_replay_promotion"] is True
    assert complete_result["summary"]["threshold_policy_contract_status"] == (
        "ready_for_threshold_policy_use"
    )
    assert complete_result["threshold_policy_result"]["ready_for_threshold_policy_use"] is True
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    integration_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "legacy-threshold-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": legacy_only_replay,
            "fault_recovery_manifest": {
                "manifest_id": "missing-fault-recovery-for-prefix-check",
                "offline_only": True,
            },
        }
    )
    assert integration_result["ready_for_integration_evaluation_promotion"] is False
    assert "offline_replay:quality_threshold_policy_contract_not_ready" in integration_result[
        "blockers"
    ]
    assert "offline_replay:legacy_threshold_policy_without_contract_manifest" in integration_result[
        "blockers"
    ]

    assert "Replay manifests must consume the quality threshold policy contract result" in completion
    assert "M015-CP01" in plan
    assert "M015-CP02" in plan
    assert "Legacy `threshold_policy` fields without the M014 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M015 replay threshold policy binding is now covered" in current
    assert "M015 replay threshold policy binding rail is now automated" in report


def test_m016_recovery_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M016_PLAN.read_text(encoding="utf-8")
    adr = M016_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m016 = markdown_table_row(roadmap, "M016")
    assert m016[1] == "Recovery policy contract"
    assert m016[2] == (
        "Candidate recovery policies are complete, evidence-bound and "
        "owner-approved before simulator fault-recovery promotion"
    )
    assert m016[3] == "M012-M015"
    assert m016[4] == "high offline"
    assert (
        m016[5]
        == "SPECIFIED to UNIT_VERIFIED for recovery-policy gate; owner approval remains required"
    )
    assert m016[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/recovery_policy_contract.py`" in memory_eval[2]
    assert "tests/test_recovery_policy_contract.py" in memory_eval[3]
    assert "recovery policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    candidate_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "candidate-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {"status": "candidate"},
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_recovery_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_recovery_policy_required"
    ]

    approved_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "approved-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:recovery-policy",
                "approved_at": "2026-06-19T17:51:02Z",
            },
        }
    )
    assert approved_result["ready_for_recovery_policy_use"] is True
    assert approved_result["recovery_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["missing_recovery_controls"] == []
    assert approved_result["summary"]["missing_recovery_evidence"] == []

    blocked_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "blocked-recovery-policy",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "recovery_controls": {"fault_detection_policy": True},
            "recovery_evidence": {"fault_scenario_matrix_bound": True},
            "recovery_policy": {"status": "PENDING_APPROVAL"},
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "recovery_policy_forbids_live_write" in blocked_result["blockers"]
    assert "recovery_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "recovery_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_recovery_control:fail_closed_policy" in blocked_result["blockers"]

    assert "Recovery policy candidates must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M016-CP01" in plan
    assert "M016-CP02" in plan
    assert "Candidate recovery policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M016 recovery policy contract is now covered" in current
    assert "M016 recovery policy contract rail is now automated" in report


def test_m017_fault_recovery_policy_binding_requires_m016_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M017_PLAN.read_text(encoding="utf-8")
    adr = M017_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m017 = markdown_table_row(roadmap, "M017")
    assert m017[1] == "Fault recovery policy binding"
    assert m017[2] == (
        "Simulator fault-recovery promotion consumes the recovery policy "
        "contract instead of loose recovery_policy status fields"
    )
    assert m017[3] == "M012-M016"
    assert m017[4] == "high offline"
    assert m017[5] == "UNIT_VERIFIED gate binding; simulator evidence remains pending"
    assert m017[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/fault_injection_recovery_contract.py`" in memory_eval[2]
    assert "`backend/recovery_policy_contract.py`" in memory_eval[2]
    assert "tests/test_fault_injection_recovery_contract.py" in memory_eval[3]
    assert "fault-recovery promotion now requires a bound recovery policy manifest" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def recovery_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "approved-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {
                key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
            },
            "recovery_evidence": {
                key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
            },
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:fault-recovery-policy",
                "approved_at": "2026-06-19T18:00:10Z",
            },
        }
        manifest.update(overrides)
        return manifest

    legacy_only_fault_recovery = {
        "manifest_id": "legacy-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:legacy-recovery-policy",
        },
        "fault_scenarios": [
            complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
        ],
    }
    legacy_result = evaluate_fault_injection_recovery_contract(legacy_only_fault_recovery)
    assert legacy_result["ready_for_fault_recovery_promotion"] is False
    assert "approved_recovery_policy_required" in legacy_result["blockers"]
    assert "recovery_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_recovery_policy_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "recovery_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_recovery_policy_status"] == "approved"
    assert legacy_result["summary"]["recovery_policy_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    complete_fault_recovery = {
        "manifest_id": "bound-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_policy_manifest": recovery_policy_manifest(),
        "fault_scenarios": [
            complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
        ],
    }
    complete_result = evaluate_fault_injection_recovery_contract(complete_fault_recovery)
    assert complete_result["ready_for_fault_recovery_promotion"] is True
    assert complete_result["summary"]["recovery_policy_contract_status"] == (
        "ready_for_recovery_policy_use"
    )
    assert complete_result["recovery_policy_result"]["ready_for_recovery_policy_use"] is True
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    integration_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "legacy-recovery-policy-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": {
                "manifest_id": "bound-threshold-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "quality_threshold_policy_manifest": threshold_policy_manifest(),
                "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
            },
            "fault_recovery_manifest": legacy_only_fault_recovery,
        }
    )
    assert integration_result["ready_for_integration_evaluation_promotion"] is False
    assert "fault_recovery:recovery_policy_contract_not_ready" in integration_result[
        "blockers"
    ]
    assert "fault_recovery:legacy_recovery_policy_without_contract_manifest" in integration_result[
        "blockers"
    ]

    assert "Fault-recovery manifests must consume the recovery policy contract result" in completion
    assert "M017-CP01" in plan
    assert "M017-CP02" in plan
    assert "Legacy `recovery_policy` fields without the M016 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M017 fault recovery policy binding is now covered" in current
    assert "M017 fault recovery policy binding rail is now automated" in report


def test_m018_integration_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M018_PLAN.read_text(encoding="utf-8")
    adr = M018_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m018 = markdown_table_row(roadmap, "M018")
    assert m018[1] == "Integration policy contract"
    assert m018[2] == (
        "Candidate integration policies are complete, evidence-bound and "
        "owner-approved before offline integration promotion"
    )
    assert m018[3] == "M013-M017"
    assert m018[4] == "high offline"
    assert (
        m018[5]
        == "SPECIFIED to UNIT_VERIFIED for integration-policy gate; owner approval remains required"
    )
    assert m018[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/integration_policy_contract.py`" in memory_eval[2]
    assert "tests/test_integration_policy_contract.py" in memory_eval[3]
    assert "integration policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    candidate_result = evaluate_integration_policy_contract(
        {
            "policy_id": "candidate-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {"status": "candidate"},
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_integration_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_integration_policy_required"
    ]

    approved_result = evaluate_integration_policy_contract(
        {
            "policy_id": "approved-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
    )
    assert approved_result["ready_for_integration_policy_use"] is True
    assert approved_result["integration_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["missing_integration_controls"] == []
    assert approved_result["summary"]["missing_integration_evidence"] == []

    blocked_result = evaluate_integration_policy_contract(
        {
            "policy_id": "blocked-integration-policy",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "integration_controls": {"analysis_trace_policy": True},
            "integration_evidence": {"analysis_trace_bound": True},
            "integration_policy": {"status": "PENDING_APPROVAL"},
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "integration_policy_forbids_live_write" in blocked_result["blockers"]
    assert "integration_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "integration_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_integration_control:offline_replay_binding_policy" in blocked_result[
        "blockers"
    ]

    assert "Integration policy candidates must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M018-CP01" in plan
    assert "M018-CP02" in plan
    assert "Candidate integration policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M018 integration policy contract is now covered" in current
    assert "M018 integration policy contract rail is now automated" in report


def test_m019_integration_policy_binding_requires_m018_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M019_PLAN.read_text(encoding="utf-8")
    adr = M019_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m019 = markdown_table_row(roadmap, "M019")
    assert m019[1] == "Integration policy binding"
    assert m019[2] == (
        "Offline integration promotion consumes the integration policy contract "
        "instead of loose integration_policy status fields"
    )
    assert m019[3] == "M013-M018"
    assert m019[4] == "high offline"
    assert m019[5] == "UNIT_VERIFIED gate binding; integration evidence remains pending"
    assert m019[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration promotion now requires a bound integration policy manifest" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def replay_manifest() -> dict:
        return {
            "manifest_id": "bound-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def recovery_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "fault-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:fault-recovery-policy",
                "approved_at": "2026-06-19T18:00:10Z",
            },
        }
        manifest.update(overrides)
        return manifest

    def fault_manifest() -> dict:
        return {
            "manifest_id": "bound-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": recovery_policy_manifest(),
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    def integration_bundle(**overrides) -> dict:
        bundle = {
            "manifest_id": "integration-policy-binding",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": replay_manifest(),
            "fault_recovery_manifest": fault_manifest(),
        }
        bundle.update(overrides)
        return bundle

    legacy_result = evaluate_integration_bundle_contract(
        integration_bundle(
            integration_policy_manifest={},
            integration_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:legacy-integration-policy",
            },
        )
    )
    assert legacy_result["ready_for_integration_evaluation_promotion"] is False
    assert "approved_integration_policy_required" in legacy_result["blockers"]
    assert "integration_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_integration_policy_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "integration_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_integration_policy_status"] == "approved"
    assert legacy_result["summary"]["integration_policy_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    pending_manifest_result = evaluate_integration_bundle_contract(
        integration_bundle(
            integration_policy_manifest=integration_policy_manifest(
                integration_policy={"status": "PENDING_APPROVAL"}
            )
        )
    )
    assert (
        "integration_policy:owner_approved_integration_policy_required"
        in pending_manifest_result["blockers"]
    )
    assert pending_manifest_result["child_results"]["integration_policy"][
        "ready_for_owner_review"
    ] is True
    assert pending_manifest_result["child_results"]["integration_policy"][
        "ready_for_integration_policy_use"
    ] is False

    bound_result = evaluate_integration_bundle_contract(integration_bundle())
    assert bound_result["ready_for_integration_evaluation_promotion"] is True
    assert bound_result["summary"]["integration_policy_contract_status"] == (
        "ready_for_integration_policy_use"
    )
    assert bound_result["child_results"]["integration_policy"][
        "ready_for_integration_policy_use"
    ] is True
    assert bound_result["authority"]["readiness_closure_allowed"] is False

    assert "Integration bundle manifests must consume the integration policy contract result" in completion
    assert "M019-CP01" in plan
    assert "M019-CP02" in plan
    assert "Legacy `integration_policy` fields without the M018 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M019 integration policy binding is now covered" in current
    assert "M019 integration policy binding rail is now automated" in report


def test_m020_safety_kernel_trace_contract_stays_offline_and_non_bypassable():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M020_PLAN.read_text(encoding="utf-8")
    adr = M020_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m020 = markdown_table_row(roadmap, "M020")
    assert m020[1] == "Safety kernel trace contract"
    assert m020[2] == (
        "Deterministic safety-kernel traces are complete, audited and "
        "non-bypassable before integration use"
    )
    assert m020[3] == "M013-M019"
    assert m020[4] == "high offline"
    assert (
        m020[5]
        == "SPECIFIED to UNIT_VERIFIED for safety-kernel trace gate; integration binding remains pending"
    )
    assert m020[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/safety_kernel_trace_contract.py`" in memory_eval[2]
    assert "tests/test_safety_kernel_trace_contract.py" in memory_eval[3]
    assert "deterministic safety-kernel trace evidence cannot grant live authority" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def trace_record(scenario: str, **overrides) -> dict:
        decision = (
            "bounded_allow"
            if scenario == "supervised_fader_gain_allowed"
            else "reject"
        )
        record = {
            "scenario": scenario,
            "safety_decision": decision,
            "guard_checks": {
                key: True for key in SAFETY_KERNEL_REQUIRED_GUARD_CHECKS
            },
            "audit": {
                "decision_id": f"{scenario}-decision",
                "checked_at": "2026-06-19T18:39:18Z",
                "candidate_action": f"{scenario}-candidate",
                "safety_decision": decision,
                "reason": f"{scenario} exercised deterministic safety kernel",
                "policy_version": "safety-kernel-offline-v1",
                "audit_reference": f"artifacts/safety-kernel/{scenario}.json",
            },
        }
        record.update(overrides)
        return record

    def trace_manifest(**overrides) -> dict:
        manifest = {
            "trace_id": "safety-kernel-trace",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "trace_records": [
                trace_record(scenario)
                for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
            ],
        }
        manifest.update(overrides)
        return manifest

    incomplete_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            trace_records=[
                trace_record(
                    "supervised_fader_gain_allowed",
                    guard_checks={"mode_permission_checked": True},
                )
            ]
        )
    )
    assert incomplete_result["ready_for_safety_kernel_trace_use"] is False
    assert "missing_required_trace_scenario:unsupported_live_write_rejected" in (
        incomplete_result["blockers"]
    )
    assert (
        "supervised_fader_gain_allowed:missing_guard_check:"
        "live_write_allowlist_checked"
        in incomplete_result["blockers"]
    )
    assert incomplete_result["authority"]["live_authority_granted"] is False

    complete_result = evaluate_safety_kernel_trace_contract(trace_manifest())
    assert complete_result["ready_for_safety_kernel_trace_use"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["summary"]["missing_trace_scenarios"] == []
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    blocked_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )
    assert blocked_result["ready_for_safety_kernel_trace_use"] is False
    assert "offline_only_required" in blocked_result["blockers"]
    assert "safety_kernel_trace_forbids_live_write" in blocked_result["blockers"]
    assert "safety_kernel_trace_forbids_wing_contact" in blocked_result["blockers"]
    assert "safety_kernel_trace_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert blocked_result["authority"]["live_authority_granted"] is False

    wrong_decision_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            trace_records=[
                trace_record(
                    scenario,
                    safety_decision="allow",
                    audit={
                        key: f"audit-{key}"
                        for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
                    }
                    | {"safety_decision": "allow"},
                )
                if scenario == "unsupported_live_write_rejected"
                else trace_record(scenario)
                for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
            ]
        )
    )
    assert wrong_decision_result["ready_for_safety_kernel_trace_use"] is False
    assert "unsupported_live_write_rejected:unexpected_safety_decision" in (
        wrong_decision_result["blockers"]
    )

    assert "Safety-kernel trace candidates must be offline, complete and audited" in (
        completion
    )
    assert "deterministic safety kernel is the non-bypassable authority" in safety
    assert "M020-CP01" in plan
    assert "M020-CP02" in plan
    assert "Safety-kernel trace evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M020 safety kernel trace contract is now covered" in current
    assert "M020 safety kernel trace contract rail is now automated" in report


def test_mvp_doc_declares_scope_and_stop_rules():
    text = MVP_DOC.read_text(encoding="utf-8")

    assert "Supervised WING Rack Proof" in text
    assert "Full autonomous live mixing" in text
    assert "Stop-The-Line Rules" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "tools/wing_readonly_preflight_runner.py" in text
    assert "tools/review_wing_readonly_preflight_artifacts.py" in text
    assert "tools/no_wing_release_check.py" in text
    assert "readiness candidates are not enough" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in text
    assert "backend_port_prestart_summary_sha256_available: true" in text
    assert (
        "backend_port_prestart_summary_sha256_available_matches_digest: true"
        in text
    )
    assert "expected_operator_sequence_count: 12" in text
    assert "operator_sequence_count_matches_contract: true" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in text
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in text
    )
    assert "backend_port_prestart_check_command" in text
    assert "backend_port_prestart_check_command_text" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "listener_confirmation_template_status: pending_operator_edit" in text
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in text
    assert "listener_confirmation_template_live_authority_granted: false" in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe packet values" in text
    assert "unsafe safety flags" in text
    assert "full_autonomous_ready" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "required_supervised_prestart_gate_marker_count: 9" in text
    assert "expected_required_supervised_prestart_gate_marker_count: 9" in text
    assert "supervised_prestart_gate_required_marker_count: 9" in text
    assert "expected_supervised_prestart_gate_required_marker_count: 9" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "valid accepted-candidate contact trace" in text
    assert "valid accepted-candidate expected-artifact trace" in text
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in text
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in text
    )
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in text
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in text
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in text
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_contact_trace: true"
        in text
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_contact_trace: true"
        in text
    )
    assert "grants_live_authority: true" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text


def test_blocker_table_has_exact_nine_rows_and_no_closed_baseline():
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)

    assert len(rows) == 9
    assert rows[0]["blocker"] == "WING read-only connection proof"
    assert rows[0]["status"] == "tooling_ready_waits_real_wing"
    missing_index_reports = [
        report_name
        for report_name in current_wing_mvp_20260614_index_report_names()
        if report_name not in rows[0]["evidence"]
    ]
    assert missing_index_reports == []
    assert "tools/no_wing_release_check.py" in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in rows[0]["evidence"]
    )
    for report_name in (
        "AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md",
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md",
        "AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md",
    ):
        assert report_name in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available_matches_digest: true" in rows[
        0
    ]["next action"]
    assert "expected_operator_sequence_count: 12" in rows[0]["next action"]
    assert "operator_sequence_count_matches_contract: true" in rows[0][
        "next action"
    ]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "accepted-candidate prestart trace" in rows[0]["next action"]
    assert "backend_port_prestart_gate" in rows[0]["next action"]
    assert "expected_backend_port_prestart_gate" in rows[0]["next action"]
    assert "backend_port_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate.must_pass_before_backend_wing_session: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in rows[
        0
    ]["next action"]
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in rows[0]["next action"]
    )
    assert (
        "backend_port_prestart_gate.runs_command: prestart_backend_port_check"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate.required_markers" in rows[0]["next action"]
    assert "Before starting the intended backend/WING session" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_command" in rows[0]["next action"]
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in rows[
        0
    ]["next action"]
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_failure_detail_fields" in rows[0][
        "next action"
    ]
    assert "failed busy-port no-WING summary" in rows[0]["next action"]
    assert "no-WING handoff summary" in rows[0]["next action"]
    assert "listener_processes" in rows[0]["next action"]
    assert "listener_process_error" in rows[0]["next action"]
    assert "safe_to_stop_listener_by_gate" in rows[0]["next action"]
    assert "operator_next_step" in rows[0]["next action"]
    assert "listener_confirmation_template_path" in rows[0]["next action"]
    assert "listener_confirmation_template_acceptance_status" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_status: pending_operator_edit" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in rows[
        0
    ]["next action"]
    assert "listener_confirmation_template_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_accepted_by_template: false" in rows[0][
        "next action"
    ]
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_required_edits" in rows[0]["next action"]
    assert "set template_status to operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "set accepted_by_template to true" in rows[0]["next action"]
    assert "REPLACE_WITH_*" in rows[0]["next action"]
    assert "template_status: operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "accepted_by_template: true" in rows[0]["next action"]
    assert "manual_review_required_before_readiness_update: true" in rows[0][
        "next action"
    ]
    assert "readiness_matrix_update_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "readiness_closure_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "read_only: true" in rows[0]["next action"]
    assert "live_write: false" in rows[0]["next action"]
    assert "mutates_console_state: false" in rows[0]["next action"]
    assert "queries_console: false" in rows[0]["next action"]
    assert "executes_tests: false" in rows[0]["next action"]
    assert "operator_required: true" in rows[0]["next action"]
    assert "expected_plan_required_for_real_run: true" in rows[0]["next action"]
    assert "backend_port_prestart_summary_required: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_type: no_wing_release_check" in rows[0][
        "next action"
    ]
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in rows[0]["next action"]
    assert "backend_port_prestart_summary_path_placeholder" in rows[0][
        "next action"
    ]
    assert "preview_command_prestart_summary" in rows[0]["next action"]
    assert "preview_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "capture_command_prestart_summary" in rows[0]["next action"]
    assert "capture_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_status: passed" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_read_only: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_live_write: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_wing: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_mutates_console_state: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_queries_console: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_executes_tests: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_status: passed" in rows[0]["next action"]
    assert (
        "physical_preflight_requires_existing_backend_connection: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "execution_allowed_by_packet: false" in rows[0]["next action"]
    assert "live_authority_granted: false" in rows[0]["next action"]
    assert "full_autonomous_ready: false" in rows[0]["next action"]
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in rows[0]["next action"]
    )
    assert "expected_artifacts" in rows[0]["next action"]
    assert "expected_artifact_globs" in rows[0]["next action"]
    assert "plan_expected_artifacts" in rows[0]["next action"]
    assert "expected_artifacts_match_plan: true" in rows[0]["next action"]
    assert "summary_glob_matches_plan: true" in rows[0]["next action"]
    assert "event_log_glob_matches_plan: true" in rows[0]["next action"]
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in rows[0]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_operator_sequence_step" in rows[0][
        "next action"
    ]
    assert "prestart_backend_port_check" in rows[0]["next action"]
    assert "unexpected listeners" in rows[0]["next action"]
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in rows[
        0
    ]["next action"]
    assert (
        "backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_has_backend_connection_confirmation: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in rows[0]["next action"]
    )
    assert "operator_sequence_backend_connection_confirmation_step" in rows[0][
        "next action"
    ]
    assert "existing backend connection" in rows[0]["next action"]
    assert "intended WING Rack" in rows[0]["next action"]
    assert "read-only inspection" in rows[0]["next action"]
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_index: 5"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_capture_readonly_preflight_step_index: 6"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor" in rows[0]["next action"]
    assert "mvp_blocker_anchor_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_count: 12" in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor_sha256" in rows[0]["next action"]
    assert "current first open blocker" in rows[0]["next action"]
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_supervised_prestart_gate: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in rows[0][
        "next action"
    ]
    assert "required_supervised_prestart_gate_markers" in rows[0]["next action"]
    assert "accepted_decision_ready" in rows[0]["next action"]
    assert "valid accepted-candidate prestart trace" in rows[0]["next action"]
    assert "valid accepted-candidate verifier WebSocket proof" in rows[0][
        "next action"
    ]
    assert "valid accepted-candidate artifact-root trace" in rows[0]["next action"]
    assert "valid accepted-candidate request-sequence trace" in rows[0][
        "next action"
    ]
    assert "expected-artifact-bound accepted source id" in rows[0]["next action"]
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in rows[0]["next action"]
    assert "supervised_prestart_gate" in rows[0]["next action"]
    assert "expected_supervised_prestart_gate" in rows[0]["next action"]
    assert "supervised_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in rows[0][
        "next action"
    ]
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_contact_trace: true"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert "valid accepted-candidate contact trace" in rows[0]["next action"]
    fader_row = next(row for row in rows if row["blocker"] == "Supervised fader write proof")
    gain_row = next(
        row for row in rows if row["blocker"] == "Supervised gain-down / trim proof"
    )
    for row in (fader_row, gain_row):
        assert row["status"] == "blocked_by_readonly_preflight"
        assert "accepted decision JSON" in row["next action"]
        assert "backend_port_prestart_gate" in row["next action"]
        assert "expected_backend_port_prestart_gate" in row["next action"]
        assert "without live authority" in row["next action"]
        assert "final handoff operator-sequence step" in row["next action"]
        assert "valid accepted-candidate prestart trace" in row["next action"]
        assert "valid accepted-candidate verifier WebSocket proof" in row[
            "next action"
        ]
        assert "valid accepted-candidate artifact-root trace" in row["next action"]
        assert "valid accepted-candidate request-sequence trace" in row["next action"]
        assert "valid accepted-candidate contact trace" in row["next action"]
        assert (
            "valid accepted-candidate expected-artifact trace" in row["next action"]
        )
        assert "expected-artifact-bound accepted source id" in row["next action"]
        assert "supervised_prestart_gate" in row["next action"]
        assert "expected_supervised_prestart_gate" in row["next action"]
    assert all(row["status"] != "closed" for row in rows)
    assert progress.first_open_blocker(rows)["blocker"] == "WING read-only connection proof"


def test_runbooks_keep_readonly_before_supervised_execute():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    supervised = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")
    missing_sidecar_markers = [
        marker
        for marker in (
            release_check.PROJECT_DIRECTOR_SELECTED_ACTION_MANIFEST_SIDECAR_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_sidecar_markers == []
    missing_value_gate_self_check_markers = [
        marker
        for marker in (
            release_check.PROJECT_DIRECTOR_SELECTED_ACTION_VALUE_GATE_SELF_CHECK_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_value_gate_self_check_markers == []
    missing_operator_sequence_markdown_markers = [
        marker
        for marker in (
            release_check.WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES
        )
        if marker not in readonly
    ]
    assert missing_operator_sequence_markdown_markers == []
    missing_artifact_manifest_contract_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_CONTRACT_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_contract_markers == []
    missing_artifact_manifest_cross_surface_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_CONTRACT_CROSS_SURFACE_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_cross_surface_markers == []
    missing_artifact_manifest_summary_surface_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_SUMMARY_SURFACE_REQUIRED_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_summary_surface_markers == []
    missing_artifact_manifest_summary_surface_field_lines = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_SUMMARY_SURFACE_REQUIRED_FIELD_LINES
        if marker not in readonly
    ]
    assert missing_artifact_manifest_summary_surface_field_lines == []

    assert "wing_readonly_preflight_runner.py" in readonly
    assert "review_wing_readonly_preflight_artifacts.py" in readonly
    assert "--require-physical-readback" in readonly
    assert "--operator-confirm-channels" in readonly
    assert "--url ws://localhost:8765" in readonly
    assert "--value-types fader,gain,pan" in readonly
    assert "--artifact-root artifacts/wing_readonly_preflight" in readonly
    assert "--value-types fader,gain \\" not in readonly
    assert "--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json" in readonly
    assert "--validate-run-inputs-only" in readonly
    assert "status: ready_for_websocket" in readonly
    assert "reason: run_inputs_validated_no_websocket" in readonly
    assert "validation_only: true" in readonly
    assert "websocket_opened: false" in readonly
    assert "contacts_backend: false" in readonly
    assert "contacts_wing: false" in readonly
    assert "expected_plan_match: true" in readonly
    assert "expected_plan_violations: []" in readonly
    assert "sent_request_types: []" in readonly
    assert "--execute" not in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "write_request_sent: false" in readonly
    assert "proposal_request_sent: false" in readonly
    assert "readiness_matrix_update_allowed: false" in readonly
    assert "readiness_closure_allowed: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "operator_confirmed_channels == channels" in readonly
    assert "strict `send`/`recv` pairs" in readonly
    assert "channel, value type, and address" in readonly
    assert "final handoff operator-sequence step" in readonly
    assert "operator_sequence_has_supervised_prestart_gate" in readonly
    assert "operator_sequence_supervised_prestart_gate_is_final" in readonly
    assert "supervised_prestart_gate" in readonly
    assert "expected_supervised_prestart_gate" in readonly
    missing_supervised_prestart_gate_markers = [
        marker
        for marker in (
            readonly_contract.WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_supervised_prestart_gate_markers == []
    assert "artifact_operator_next_steps" in readonly
    assert "before opening a backend/WING WebSocket" in readonly
    assert "they do not approve live writes or close readiness rows" in readonly
    assert "artifact_manifest_verifier_required: true" in readonly
    assert "artifact_manifest_verifier_command" in readonly
    assert "artifact_manifest_verifier_target_artifact_dir" in readonly
    assert "artifact_manifest_verifier_status_required: passed" in readonly
    assert "artifact_manifest_verifier_failed_checks_required: []" in readonly
    assert "artifact_manifest_verifier_required_fields_missing: []" in readonly
    assert "artifact_manifest_verifier_contacts_wing: false" in readonly
    assert "artifact_manifest_verifier_mutates_console_state: false" in readonly
    assert "artifact_manifest_verifier_queries_console: false" in readonly
    assert "artifact_manifest_verifier_readiness_closure_allowed: false" in readonly
    assert "artifact_manifest_verifier_live_authority_granted: false" in readonly
    assert "summary_json_artifact_manifest_verifier_fields_present: true" in readonly
    assert "manifest_artifact_manifest_verifier_fields_present: true" in readonly
    assert "summary_md_artifact_manifest_verifier_fields_rendered: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert "manifest_artifact_manifest_verifier_command_matches_expected: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    missing_artifact_manifest_verifier_operator_markers = [
        marker
        for marker in (
            readonly_contract.WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_artifact_manifest_verifier_operator_markers == []
    for marker in (
        "summary_json_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_present: true",
        "summary_md_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_rendered: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_present: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_rendered: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
    ):
        assert marker in readonly
    assert "verifier_report_read_only_matches_expected: true" in readonly
    assert "verifier_report_live_write_matches_expected: true" in readonly
    assert "verifier_report_contacts_wing_matches_expected: true" in readonly
    assert "verifier_report_contacts_backend_matches_expected: true" in readonly
    assert "verifier_report_mutates_console_state_matches_expected: true" in readonly
    assert "verifier_report_queries_console_matches_expected: true" in readonly
    assert "verifier_report_executes_tests_matches_expected: true" in readonly
    assert (
        "verifier_report_readiness_closure_allowed_matches_expected: true"
        in readonly
    )
    assert "verifier_report_live_authority_granted_matches_expected: true" in readonly
    assert "artifact_manifest_verifier_check_mismatches: []" in readonly
    assert "artifact_manifest_verifier_check_<check_name>: true" in readonly
    assert "artifact_manifest_verifier_check_manifest_type: true" in readonly
    assert (
        "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_check_summary_denies_live_authority: true"
        in readonly
    )
    assert "artifact_manifest_verifier_must_pass_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_passed_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_websocket_opened_after_pass: true" in readonly
    assert "artifact_manifest_verifier_operator_sequence_step_index: 1" in readonly
    assert (
        "expected_artifact_manifest_verifier_operator_sequence_step_index: 1"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_operator_sequence_step" in readonly
    assert "expected_artifact_manifest_verifier_operator_sequence_step" in readonly
    assert (
        "artifact_manifest_verifier_operator_sequence_step_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_required_markers" in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS:
        assert marker in readonly
    assert "artifact_manifest_verifier_operator_sequence_before_preview: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in readonly
    assert "backend_port_prestart_check_operator_sequence_step" in readonly
    assert "prestart_backend_port_check" in readonly
    assert "unexpected listeners" in readonly
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in readonly
    assert "backend_connection_confirmation_step_index: 3" in readonly
    assert "operator_sequence_has_backend_connection_confirmation: true" in readonly
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in readonly
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readonly
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readonly
    assert "existing backend connection" in readonly
    assert "intended WING Rack" in readonly
    assert "read-only inspection" in readonly
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in readonly
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in readonly
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in readonly
    assert "mvp_blocker_anchor" in readonly
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor_sha256" in readonly
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readonly
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readonly
    assert "operator_sequence_count: 12" in readonly
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in readonly
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in readonly
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in readonly
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in readonly
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in readonly
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readonly
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in readonly
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in readonly
    )
    assert "current first open blocker" in readonly
    assert "backend_port_prestart_gate" in readonly
    assert "expected_backend_port_prestart_gate" in readonly
    assert "backend_port_prestart_gate.grants_live_authority: false" in readonly
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in readonly
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in readonly
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in readonly
    )
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in readonly
    assert "backend_port_prestart_gate.required_markers" in readonly
    assert "Before starting the intended backend/WING session" in readonly
    assert "backend_port_prestart_check_command" in readonly
    assert "backend_port_prestart_check_command_text" in readonly
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in readonly
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary both show"
        in readonly
    )
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "backend_port_prestart_check_failure_detail_fields" in readonly
    assert "listener_processes" in readonly
    assert "listener_process_error" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "operator_next_step" in readonly
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in readonly
    assert "listener_confirmation_template_accepted_by_template: false" in readonly
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in readonly
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in readonly
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in readonly
    assert "REPLACE_WITH_*" in readonly
    assert "manual_review_required_before_readiness_update: true" in readonly
    assert "readiness_matrix_update_allowed_by_packet: false" in readonly
    assert "readiness_closure_allowed_by_packet: false" in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "mutates_console_state: false" in readonly
    assert "queries_console: false" in readonly
    assert "executes_tests: false" in readonly
    assert "operator_required: true" in readonly
    assert "expected_plan_required_for_real_run: true" in readonly
    assert "physical_preflight_requires_existing_backend_connection: true" in readonly
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "execution_allowed_by_packet: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "full_autonomous_ready: false" in readonly
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in readonly
    )
    assert "expected_artifacts" in readonly
    assert "expected_artifact_globs" in readonly
    assert "plan_expected_artifacts" in readonly
    assert "expected_artifacts_match_plan: true" in readonly
    assert "summary_glob_matches_plan: true" in readonly
    assert "event_log_glob_matches_plan: true" in readonly
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in readonly
    assert "supervised_prestart_gate.grants_live_authority: false" in readonly
    assert "required_supervised_prestart_gate_markers" in readonly
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in readonly
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in readonly
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in readonly
    )
    assert "supervised_prestart_gate.requires_operator_decision_status:" in readonly
    assert "accepted_decision_ready" in readonly
    assert "supervised_prestart_gate.unlocks_runbook:" in readonly
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in readonly
    assert "--backend-listener-confirmation" in readonly
    assert "wing_backend_port_listener_confirmation.v1" in readonly
    assert "operator_confirmed_intended_backend" in readonly
    assert "confirms_intended_backend" in readonly
    assert "reviewed_listener_pid" in readonly
    assert "reviewed_listener_command_line" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "live_authority_granted" in readonly
    assert '"artifact_root": "artifacts/wing_readonly_preflight"' in readonly
    assert "`review_id` field must remain present" in readonly
    assert (
        "keeps the record invalid even if the artifact hashes still match"
        in readonly
    )
    assert (
        "`template_status: operator_must_replace_placeholders_before_ledger_acceptance`"
        in readonly
    )
    assert "unedited pending template and fails closed" in readonly
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_sha256`, `events_sha256`"
        in readonly
    )
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_path`, `events_path`"
        in readonly
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readonly
    )
    assert "valid accepted-candidate prestart trace" in readonly
    assert "expected-artifact-bound accepted source id" in readonly
    assert "wing_operator_proposal_validation_runner.py" in supervised
    assert "--execute" in supervised
    assert "blocked until a clean read-only preflight" in supervised
    assert "valid accepted-candidate prestart trace" in supervised
    assert "valid accepted-candidate verifier WebSocket proof" in supervised
    assert "valid accepted-candidate artifact-root trace" in supervised
    assert "valid accepted-candidate request-sequence trace" in supervised
    assert "valid accepted-candidate contact trace" in supervised
    assert "valid accepted-candidate expected-artifact trace" in supervised
    assert "expected-artifact-bound accepted source id" in supervised
    assert "supervised_prestart_gate" in supervised
    assert "expected_supervised_prestart_gate" in supervised
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in supervised
    assert (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in supervised
    )
    assert "accepted_candidate_prestart_contract_violations_by_source" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in readonly
    )
    assert "accepted_candidate_prestart_contract_violations_by_source: {}" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in readonly
    )
    assert "negative delta" in supervised
    assert "Full autonomous" not in supervised


def test_readonly_runbook_requires_no_wing_manifest_summary_surface_sidecar_gate():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    normalized_readonly = " ".join(readonly.split())

    assert "Before opening a backend/WING WebSocket" in readonly
    assert "do not open the WING session" in normalized_readonly
    assert "do not close readiness" in readonly
    assert "approve live writes" in readonly
    assert "grant live authority" in readonly
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in readonly


def test_progress_tool_generates_latest_report_from_blocker_table(tmp_path):
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)
    current_command = progress.current_first_command(BLOCKERS_DOC)
    assert current_command is not None
    assert current_command.startswith("Use `PYTHONPATH=backend python3")
    assert "required_supervised_proof_plan_summary_fields" in current_command
    assert "expected_required_supervised_proof_plan_summary_fields" in current_command
    assert (
        "required_supervised_proof_plan_summary_fields_match_contract: true"
        in current_command
    )
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in current_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in current_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in current_command
    )
    for marker in (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
    ):
        assert marker in current_command
    assert "operator-readable no-WING `summary.md`" in current_command
    assert "operator_markdown_summary_required: true" in current_command
    assert "operator_markdown_summary_field_lines_rendered: true" in current_command
    assert "operator_markdown_summary_missing_field_lines: []" in current_command
    assert "operator_markdown_summary_artifact: summary.md" in current_command
    assert "operator_markdown_summary_sha256" in current_command
    assert "operator_markdown_summary_sha256_available: true" in current_command
    assert "operator_markdown_summary_actual_sha256" in current_command
    assert "operator_markdown_summary_sha256_matches_actual: true" in current_command
    assert "artifact_manifest_summary_surface_required: true" in current_command
    assert "artifact_manifest_summary_surface_missing_fields: []" in current_command
    assert "artifact_manifest_summary_surface_required_field_lines" in current_command
    assert (
        "artifact_manifest_summary_surface_missing_field_lines: []"
        in current_command
    )
    assert (
        "artifact_manifest_summary_surface_field_lines_rendered: true"
        in current_command
    )
    assert "artifact_manifest_contract_id" in current_command
    assert "artifact_manifest_contract_required: true" in current_command
    assert "artifact_manifest_contract_fields" in current_command
    assert "artifact_manifest_contract_sha256" in current_command
    assert "artifact_manifest_contract_sha256_available: true" in current_command
    assert (
        "artifact_manifest_contract_sha256_matches_expected: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_required: true" in current_command
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in current_command
    assert "artifact_manifest_contract_cross_surface_required_fields" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_expected_match: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in current_command
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in current_command
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in current_command
    assert "artifact_manifest_path: artifact-manifest.json" in current_command
    assert "artifact_manifest_artifact_count: 2" in current_command
    assert "artifact_manifest_required_artifact_count: 2" in current_command
    assert "artifact_manifest_artifact_count_matches_required: true" in current_command
    assert "artifact_manifest_required_artifact_ids" in current_command
    assert "artifact_manifest_artifact_ids" in current_command
    assert "artifact_manifest_required_artifacts_present: true" in current_command
    assert "artifact_manifest_required_artifact_paths" in current_command
    assert "artifact_manifest_artifact_paths" in current_command
    assert "artifact_manifest_required_artifact_paths_present: true" in current_command
    assert (
        "artifact_manifest_required_artifact_sha256_available: true"
        in current_command
    )
    assert "artifact_manifest_summary_json_sha256" in current_command
    assert "artifact_manifest_summary_json_sha256_available: true" in current_command
    assert "artifact_manifest_summary_md_sha256" in current_command
    assert "artifact_manifest_summary_md_sha256_available: true" in current_command
    assert "artifact_manifest_readiness_closure_allowed: false" in current_command
    assert "artifact_manifest_live_authority_granted: false" in current_command
    assert "Before opening a backend/WING WebSocket" in current_command
    assert "do not open the WING session" in current_command
    assert "do not close readiness" in current_command
    assert "approve live writes" in current_command
    assert "grant live authority" in current_command
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in current_command
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in current_command
    assert "backend_port_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_backend_port_prestart_gate_matches_expected: true"
        in current_command
    )
    assert "supervised_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_supervised_prestart_gate_matches_expected: true"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_request_sequence_by_source"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert "project_director_selected_action_anchor" in current_command
    assert "expected_project_director_selected_action_anchor" in current_command
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in current_command
    )
    assert "project_director_selected_action_sha256" in current_command
    assert "actual_project_director_selected_action_sha256" in current_command
    assert "expected_project_director_selected_action_sha256" in current_command
    assert "project_director_selected_action_sha256_available: true" in current_command
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_anchor: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_text: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_missing_markers: []"
        in current_command
    )
    assert progress.current_first_command_missing_baseline_markers(current_command) == []
    assert progress.current_first_command_covers_baseline(current_command)
    missing_final_supervised_step_contact_marker_command = current_command.replace(
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_final_supervised_step_contact_marker_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in progress.current_first_command_missing_baseline_markers(
            missing_final_supervised_step_contact_marker_command
        )
    )
    missing_final_supervised_step_expected_artifact_marker_command = (
        current_command.replace(
            "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
            "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_final_supervised_step_expected_artifact_marker_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in progress.current_first_command_missing_baseline_markers(
            missing_final_supervised_step_expected_artifact_marker_command
        )
    )
    missing_supervised_contact_gate_marker_command = current_command.replace(
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_contact_gate_marker_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_contact_gate_marker_command
        )
    )
    missing_supervised_expected_artifact_gate_marker_command = (
        current_command.replace(
            'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_expected_artifact_gate_marker_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_expected_artifact_gate_marker_command
        )
    )
    missing_supervised_gate_required_contact_marker_command = current_command.replace(
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_gate_required_contact_marker_command
    )
    assert (
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_gate_required_contact_marker_command
        )
    )
    missing_expected_supervised_gate_required_contact_marker_command = (
        current_command.replace(
            'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_supervised_gate_required_contact_marker_command
    )
    assert (
        'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_supervised_gate_required_contact_marker_command
        )
    )
    missing_expected_required_supervised_contact_marker_command = (
        current_command.replace(
            'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_required_supervised_contact_marker_command
    )
    assert (
        'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_required_supervised_contact_marker_command
        )
    )
    querying_console_contact_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
    )
    assert not progress.current_first_command_covers_baseline(
        querying_console_contact_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            querying_console_contact_trace_command
        )
    )
    missing_proof_plan_fields_command = current_command.replace(
        "required_supervised_proof_plan_summary_fields",
        "missing_supervised_proof_plan_summary_fields",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_proof_plan_fields_command
    )
    assert "required_supervised_proof_plan_summary_fields" in (
        progress.current_first_command_missing_baseline_markers(
            missing_proof_plan_fields_command
        )
    )
    failing_candidate_prestart_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true",
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_prestart_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_prestart_command
        )
    )
    failing_candidate_prestart_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_prestart_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_prestart_source_command
        )
    )
    failing_candidate_request_sequence_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true",
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_request_sequence_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_request_sequence_command
        )
    )
    failing_candidate_request_sequence_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_request_sequence_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_request_sequence_source_command
        )
    )
    failing_candidate_artifact_root_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true",
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_artifact_root_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_artifact_root_command
        )
    )
    failing_candidate_artifact_roots_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]',
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/stale_preflight"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_artifact_roots_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_artifact_roots_command
        )
    )
    failing_expected_artifact_root_command = current_command.replace(
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}",
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {'stale_source': ['missing artifact_root']}",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_root_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}"
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_root_command
        )
    )
    failing_verifier_ws_proof_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_verifier_ws_proof_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_verifier_ws_proof_command
        )
    )
    failing_verifier_ws_proof_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_verifier_ws_proof_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_verifier_ws_proof_source_command
        )
    )
    missing_contact_query_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source",
        "readonly_preflight_review_decision_accepted_candidate_queries_console_missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_query_trace_command
        )
    )
    missing_contact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_trace_command
        )
    )
    failing_contact_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_trace_source_command
        )
    )
    failing_expected_contact_backend_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_contact_backend_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_contact_backend_trace_command
        )
    )
    failing_expected_console_query_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_console_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_console_query_trace_command
        )
    )
    failing_contact_backend_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_backend_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_backend_trace_command
        )
    )
    failing_console_query_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_console_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            failing_console_query_trace_command
        )
    )
    missing_contact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_source_trace_command
        )
    )
    failing_contact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_value_gate_command
        )
    )
    granting_contact_value_gate_live_authority_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: true",
    )
    assert not progress.current_first_command_covers_baseline(
        granting_contact_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_contact_value_gate_live_authority_command
        )
    )
    closing_readiness_contact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_contact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_contact_value_gate_command
        )
    )
    granting_contact_summary_surface_live_authority_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: true",
    )
    assert not progress.current_first_command_covers_baseline(
        granting_contact_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_contact_summary_surface_live_authority_command
        )
    )
    closing_readiness_contact_summary_surface_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_contact_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_contact_summary_surface_command
        )
    )
    failing_request_sequence_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_value_gate_command
        )
    )
    granting_request_sequence_value_gate_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_request_sequence_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_request_sequence_value_gate_live_authority_command
        )
    )
    closing_readiness_request_sequence_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_request_sequence_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_request_sequence_value_gate_command
        )
    )
    granting_request_sequence_summary_surface_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_request_sequence_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_request_sequence_summary_surface_live_authority_command
        )
    )
    closing_readiness_request_sequence_summary_surface_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_request_sequence_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_request_sequence_summary_surface_command
        )
    )
    failing_request_sequence_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_trace_source_command
        )
    )
    failing_request_sequence_match_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_match_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_match_trace_command
        )
    )
    missing_request_sequence_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_source_trace_command
        )
    )
    failing_expected_artifact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_value_gate_command
        )
    )
    granting_expected_artifact_value_gate_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_expected_artifact_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_expected_artifact_value_gate_live_authority_command
        )
    )
    closing_readiness_expected_artifact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_expected_artifact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_expected_artifact_value_gate_command
        )
    )
    granting_expected_artifact_summary_surface_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_expected_artifact_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_expected_artifact_summary_surface_live_authority_command
        )
    )
    closing_readiness_expected_artifact_summary_surface_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_expected_artifact_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_expected_artifact_summary_surface_command
        )
    )
    failing_expected_artifact_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_trace_source_command
        )
    )
    missing_expected_artifact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_source_trace_command
        )
    )
    missing_expected_artifact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_trace_command
        )
    )
    missing_request_sequence_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_trace_command
        )
    )
    missing_markdown_summary_command = current_command.replace(
        "operator-readable no-WING `summary.md`",
        "machine-only no-WING summary",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_summary_command
    )
    assert "operator-readable no-WING `summary.md`" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_summary_command
        )
    )
    missing_markdown_rendered_command = current_command.replace(
        "operator_markdown_summary_field_lines_rendered: true",
        "operator_markdown_summary_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_rendered_command
    )
    assert "operator_markdown_summary_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_rendered_command
        )
    )
    missing_markdown_sha_command = current_command.replace(
        "operator_markdown_summary_sha256_available: true",
        "operator_markdown_summary_sha256_available: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_command
    )
    assert "operator_markdown_summary_sha256_available: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_command
        )
    )
    missing_markdown_sha_match_command = current_command.replace(
        "operator_markdown_summary_sha256_matches_actual: true",
        "operator_markdown_summary_sha256_matches_actual: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_match_command
    )
    assert "operator_markdown_summary_sha256_matches_actual: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_match_command
        )
    )
    missing_artifact_manifest_command = current_command.replace(
        "artifact_manifest_path: artifact-manifest.json",
        "artifact_manifest_path: missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_command
    )
    assert "artifact_manifest_path: artifact-manifest.json" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_command
        )
    )
    missing_artifact_manifest_surface_command = current_command.replace(
        "artifact_manifest_summary_surface_required: true",
        "artifact_manifest_summary_surface_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_command
    )
    assert "artifact_manifest_summary_surface_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_command
        )
    )
    missing_artifact_manifest_surface_rendered_command = current_command.replace(
        "artifact_manifest_summary_surface_field_lines_rendered: true",
        "artifact_manifest_summary_surface_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_rendered_command
    )
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_rendered_command
        )
    )
    missing_artifact_manifest_contract_digest_command = current_command.replace(
        "artifact_manifest_contract_sha256_matches_expected: true",
        "artifact_manifest_contract_sha256_matches_expected: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_contract_digest_command
    )
    assert "artifact_manifest_contract_sha256_matches_expected: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_contract_digest_command
        )
    )
    missing_artifact_manifest_cross_surface_command = current_command.replace(
        "artifact_manifest_contract_cross_surface_match: true",
        "artifact_manifest_contract_cross_surface_match: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_cross_surface_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_cross_surface_command
        )
    )
    missing_artifact_manifest_count_match_command = current_command.replace(
        "artifact_manifest_artifact_count_matches_required: true",
        "artifact_manifest_artifact_count_matches_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_count_match_command
    )
    assert "artifact_manifest_artifact_count_matches_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_count_match_command
        )
    )
    missing_artifact_manifest_closure_command = current_command.replace(
        "artifact_manifest_readiness_closure_allowed: false",
        "artifact_manifest_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_closure_command
    )
    assert "artifact_manifest_readiness_closure_allowed: false" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_closure_command
        )
    )
    missing_artifact_manifest_ids_command = current_command.replace(
        "artifact_manifest_required_artifacts_present: true",
        "artifact_manifest_required_artifacts_present: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_ids_command
    )
    assert "artifact_manifest_required_artifacts_present: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_ids_command
        )
    )
    missing_director_action_digest_command = current_command.replace(
        "project_director_selected_action_sha256",
        "missing_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_director_action_digest_command
    )
    assert "project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_director_action_digest_command
        )
    )
    missing_expected_digest_command = current_command.replace(
        "expected_project_director_selected_action_sha256",
        "missing_expected_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_digest_command
    )
    assert "expected_project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_expected_digest_command
        )
    )
    report = progress.build_progress_report(
        rows,
        generated_at="2026-06-06T00:00:00Z",
        current_first_command=current_command,
    )

    assert "Total blockers: 9" in report
    assert "Closed blockers: 0" in report
    assert "Current first open blocker: WING read-only connection proof" in report
    assert "Next required action source: Current First Command" in report
    assert "Current First Command baseline coverage required: true" in report
    assert "Current First Command baseline coverage passed: true" in report
    assert (
        "Current First Command baseline coverage marker count: "
        f"{len(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)}"
        in report
    )
    assert "Current First Command baseline missing markers: []" in report
    assert (
        f"Selected evidence file count: {len(progress.extract_evidence_files(rows[0]['evidence']))}"
        in report
    )
    assert "backend_port_prestart_gate" in report
    assert "expected_backend_port_prestart_gate" in report
    assert "backend_port_prestart_gate.grants_live_authority: false" in report
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in report
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in report
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in report
    assert "backend_port_prestart_gate.required_markers" in report
    assert "Before starting the intended backend/WING session" in report
    assert "backend_port_prestart_check_command" in report
    assert "backend_port_prestart_check_command_text" in report
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in report
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in report
    )
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "backend_port_prestart_check_failure_detail_fields" in report
    assert "listener_processes" in report
    assert "listener_process_error" in report
    assert "safe_to_stop_listener_by_gate" in report
    assert "operator_next_step" in report
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in report
    assert "listener_confirmation_template_path" in report
    assert "listener_confirmation_template_acceptance_status" in report
    assert "listener_confirmation_template_accepted_by_template: false" in report
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in report
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in report
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in report
    assert "listener_confirmation_template_required_edits" in report
    assert "set template_status to operator_reviewed_and_edited" in report
    assert "set accepted_by_template to true" in report
    assert "REPLACE_WITH_*" in report
    assert "template_status: operator_reviewed_and_edited" in report
    assert "accepted_by_template: true" in report
    assert "manual_review_required_before_readiness_update: true" in report
    assert "readiness_matrix_update_allowed_by_packet: false" in report
    assert "readiness_closure_allowed_by_packet: false" in report
    assert "read_only: true" in report
    assert "live_write: false" in report
    assert "mutates_console_state: false" in report
    assert "queries_console: false" in report
    assert "executes_tests: false" in report
    assert "operator_required: true" in report
    assert "expected_plan_required_for_real_run: true" in report
    assert "backend_port_prestart_summary_type: no_wing_release_check" in report
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in report
    assert "backend_port_prestart_summary_read_only: true" in report
    assert "backend_port_prestart_summary_live_write: false" in report
    assert "backend_port_prestart_summary_contacts_backend: false" in report
    assert "backend_port_prestart_summary_contacts_wing: false" in report
    assert "backend_port_prestart_summary_mutates_console_state: false" in report
    assert "backend_port_prestart_summary_queries_console: false" in report
    assert "backend_port_prestart_summary_executes_tests: false" in report
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in report
    assert "physical_preflight_requires_existing_backend_connection: true" in report
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "execution_allowed_by_packet: false" in report
    assert "live_authority_granted: false" in report
    assert "full_autonomous_ready: false" in report
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in report
    )
    assert "expected_artifacts" in report
    assert "expected_artifact_globs" in report
    assert "plan_expected_artifacts" in report
    assert "expected_artifacts_match_plan: true" in report
    assert "summary_glob_matches_plan: true" in report
    assert "event_log_glob_matches_plan: true" in report
    assert "accepted_candidate_prestart_contract_violations_by_source" in report
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in report
    )
    assert "Both maps must be `{}` before supervised proof steps are planned" in report
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in report
    assert "backend_port_prestart_check_operator_sequence_step" in report
    assert "prestart_backend_port_check" in report
    assert "unexpected listeners" in report
    assert "backend_connection_confirmation_step_index: 3" in report
    assert "operator_sequence_has_backend_connection_confirmation: true" in report
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in report
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in report
    )
    assert "operator_sequence_backend_connection_confirmation_step" in report
    assert "existing backend connection" in report
    assert "intended WING Rack" in report
    assert "read-only inspection" in report
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in report
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in report
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in report
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in report
    )
    assert "mvp_blocker_anchor" in report
    assert "mvp_blocker_anchor_matches_current_first_open: true" in report
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in report
    assert "project_director_selected_action_anchor" in report
    assert "expected_project_director_selected_action_anchor" in report
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_source: Current First Command" in report
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in report
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_sha256" in report
    assert "actual_project_director_selected_action_sha256" in report
    assert "expected_project_director_selected_action_sha256" in report
    assert "project_director_selected_action_sha256_available: true" in report
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in report
    )
    assert "project_director_selected_action_sha256_matches_anchor: true" in report
    assert "project_director_selected_action_sha256_matches_text: true" in report
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in report
    )
    assert "project_director_current_first_command_baseline_missing_markers: []" in report
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in report
    assert "operator_sequence_count: 12" in report
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in report
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in report
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in report
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in report
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in report
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in report
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in report
    )
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in report
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in report
    assert "operator_sequence_has_supervised_prestart_gate: true" in report
    assert "required_supervised_prestart_gate_markers" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in report
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in report
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in report
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in report
    )
    assert "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true" in report
    assert "accepted-candidate prestart trace" in report
    assert "expected-artifact-bound accepted source id" in report
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in report
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in report
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in report
    )
    assert "final handoff operator-sequence step" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "Offline tooling, shadow" in report

    output = tmp_path / "MVP_PROGRESS_LATEST.md"
    progress.write_report(output, report)
    assert output.read_text(encoding="utf-8") == report


def test_correction_methods_roadmap_preserves_full_console_goal_without_live_widening():
    roadmap = CORRECTION_ROADMAP.read_text(encoding="utf-8")
    backlog = CORRECTION_BACKLOG.read_text(encoding="utf-8")

    for family in (
        "Gain / Trim",
        "Pan / Stereo Placement",
        "EQ",
        "Compressor",
        "FX / Sends",
        "Fader / Balance",
    ):
        assert family in roadmap

    assert "offline analysis" in roadmap
    assert "proposal-only" in roadmap
    assert "No method may skip stages." in roadmap
    assert "must not widen live writes" in roadmap
    assert "separate ADR after WING MVP" in backlog
    assert "Do not widen live writes" in backlog


def test_codex_prompt_requires_mvp_rails_before_new_work():
    text = PROMPT.read_text(encoding="utf-8")

    assert "Docs/MVP_WING_SUPERVISED.md" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/reports/MVP_PROGRESS_LATEST.md" in text
    assert "first non-closed row" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_check_command" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "failed busy-port no-WING summary and no-WING handoff summary" in text
    assert "must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe `true` values" in text
    assert "failed handoff is auditable" in text
    assert "packet safety flags" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
    assert "supervised_prestart_gate.requires_operator_decision_status:" in text
    assert "supervised_prestart_gate.unlocks_runbook:" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "required_supervised_proof_plan_summary_fields" in text
    assert "expected_required_supervised_proof_plan_summary_fields" in text
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in text
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in text
    )
    for marker in (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
    ):
        assert marker in text
    assert "operator-readable no-WING `summary.md`" in text
    assert "operator_markdown_summary_required: true" in text
    assert "operator_markdown_summary_field_lines_rendered: true" in text
    assert "operator_markdown_summary_missing_field_lines: []" in text
    assert "operator_markdown_summary_artifact: summary.md" in text
    assert "operator_markdown_summary_sha256" in text
    assert "operator_markdown_summary_sha256_available: true" in text
    assert "operator_markdown_summary_actual_sha256" in text
    assert "operator_markdown_summary_sha256_matches_actual: true" in text
    assert "artifact_manifest_summary_surface_required: true" in text
    assert "artifact_manifest_summary_surface_missing_fields: []" in text
    assert "artifact_manifest_summary_surface_required_field_lines" in text
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in text
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in text
    assert "artifact_manifest_contract_id" in text
    assert "artifact_manifest_contract_required: true" in text
    assert "artifact_manifest_contract_fields" in text
    assert "artifact_manifest_contract_sha256" in text
    assert "artifact_manifest_contract_sha256_available: true" in text
    assert "artifact_manifest_contract_sha256_matches_expected: true" in text
    assert "artifact_manifest_contract_cross_surface_required: true" in text
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in text
    assert "artifact_manifest_contract_cross_surface_required_fields" in text
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in text
    assert "artifact_manifest_contract_cross_surface_match: true" in text
    assert "artifact_manifest_summary_json_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_summary_md_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in text
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in text
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in text
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in text
    assert "artifact_manifest_path: artifact-manifest.json" in text
    assert "artifact_manifest_artifact_count: 2" in text
    assert "artifact_manifest_required_artifact_count: 2" in text
    assert "artifact_manifest_artifact_count_matches_required: true" in text
    assert "artifact_manifest_required_artifact_ids" in text
    assert "artifact_manifest_artifact_ids" in text
    assert "artifact_manifest_required_artifacts_present: true" in text
    assert "artifact_manifest_required_artifact_paths" in text
    assert "artifact_manifest_artifact_paths" in text
    assert "artifact_manifest_required_artifact_paths_present: true" in text
    assert "artifact_manifest_required_artifact_sha256_available: true" in text
    assert "artifact_manifest_summary_json_sha256" in text
    assert "artifact_manifest_summary_json_sha256_available: true" in text
    assert "artifact_manifest_summary_md_sha256" in text
    assert "artifact_manifest_summary_md_sha256_available: true" in text
    assert "artifact_manifest_readiness_closure_allowed: false" in text
    assert "artifact_manifest_live_authority_granted: false" in text
    assert "backend_port_prestart_gate_matches_expected: true" in text
    assert "declared_backend_port_prestart_gate_matches_expected: true" in text
    assert "supervised_prestart_gate_matches_expected: true" in text
    assert "declared_supervised_prestart_gate_matches_expected: true" in text
    assert (
        'readonly_preflight_review_decision_expected_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in text
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}"
        in text
    )


def test_docs_index_points_to_current_wing_mvp_prestart_gate():
    text = DOCS_INDEX.read_text(encoding="utf-8")

    assert "Current WING MVP rails" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in text
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_REQUIREMENT_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_PENDING_SUMMARY_FIELDS_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_FINAL_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_DECISION_RUNBOOK_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_REQUIRED_MARKERS_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_INDEXES_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_PROJECT_GOAL_SUPERVISED_AUTOMIX_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert "final handoff operator-sequence step" in text
    assert "accepted-candidate prestart gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.runs_command" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "backend_port_prestart_check_command" in text
    assert "failed busy-port no-WING summary" in text
    assert "no-WING handoff summary must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "manual_review_required_before_readiness_update" in text
    assert "readiness_matrix_update_allowed_by_packet" in text
    assert "readiness_closure_allowed_by_packet" in text
    assert "live_write" in text
    assert "execution_allowed_by_packet" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "operator_sequence_count" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
