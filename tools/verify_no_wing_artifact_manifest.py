#!/usr/bin/env python3
"""Verify a no-WING release-check artifact bundle sidecar manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import no_wing_release_check

VERIFIER_REPORT_STATIC_SAFETY_FIELDS = {
    "read_only": True,
    "live_write": False,
    "contacts_wing": False,
    "contacts_backend": False,
    "mutates_console_state": False,
    "queries_console": False,
    "executes_tests": False,
    "readiness_matrix_update_allowed": False,
}
VERIFIER_REPORT_HANDOFF_PLAN_SUMMARY_SURFACE_EXPECTED_FIELDS = {
    "wing_readonly_handoff_plan_verifier_summary_surface_mismatches": [],
    "wing_readonly_handoff_plan_verifier_summary_surface_match": True,
    "wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed": False,
    "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed": False,
    "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted": False,
}
SUMMARY_MD_REQUIRED_FALSE_SAFETY_FIELDS = (
    "readiness_closure_allowed_by_packet",
    "live_authority_granted",
    "artifact_manifest_contract_cross_surface_readiness_closure_allowed",
    "artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed",
    "artifact_manifest_contract_cross_surface_live_authority_granted",
    "artifact_manifest_readiness_closure_allowed",
    "artifact_manifest_readiness_matrix_update_allowed",
    "artifact_manifest_live_authority_granted",
    "artifact_manifest_verifier_readiness_closure_allowed",
    "artifact_manifest_verifier_readiness_matrix_update_allowed",
    "artifact_manifest_verifier_live_authority_granted",
    "wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed",
    "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed",
    "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_rows_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return {
        str(row.get("artifact_id")): row
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("artifact_id"), str)
    }


def _record_check(
    checks: dict[str, bool],
    name: str,
    passed: bool,
) -> None:
    checks[name] = bool(passed)


def _compare_recomputed_fields(
    checks: dict[str, bool],
    *,
    prefix: str,
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    for field_name, expected_value in recomputed.items():
        _record_check(
            checks,
            f"{prefix}_{field_name}_matches_recomputed",
            recorded.get(field_name) == expected_value,
        )


def _expected_verifier_summary_surface() -> dict[str, object]:
    return {
        "artifact_manifest_verifier_required": True,
        "artifact_manifest_verifier_command": (
            no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_VERIFIER_COMMAND
        ),
        "artifact_manifest_verifier_target_artifact_dir": (
            no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_VERIFIER_TARGET_ARTIFACT_DIR
        ),
        "artifact_manifest_verifier_status_required": "passed",
        "artifact_manifest_verifier_failed_checks_required": [],
        "artifact_manifest_verifier_report_required_fields": list(
            no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_VERIFIER_REPORT_REQUIRED_FIELDS
        ),
        "artifact_manifest_verifier_read_only": True,
        "artifact_manifest_verifier_live_write": False,
        "artifact_manifest_verifier_contacts_wing": False,
        "artifact_manifest_verifier_contacts_backend": False,
        "artifact_manifest_verifier_mutates_console_state": False,
        "artifact_manifest_verifier_queries_console": False,
        "artifact_manifest_verifier_executes_tests": False,
        "artifact_manifest_verifier_readiness_matrix_update_allowed": False,
        "artifact_manifest_verifier_readiness_closure_allowed": False,
        "artifact_manifest_verifier_live_authority_granted": False,
    }


def _record_verifier_summary_surface_checks(
    checks: dict[str, bool],
    *,
    summary_json: Mapping[str, Any],
    summary_md_text: str,
) -> None:
    expected = _expected_verifier_summary_surface()
    _record_check(
        checks,
        "summary_json_artifact_manifest_verifier_fields_present",
        all(field_name in summary_json for field_name in expected),
    )
    _record_check(
        checks,
        "summary_md_artifact_manifest_verifier_fields_rendered",
        all(f"- {field_name}: `" in summary_md_text for field_name in expected),
    )
    for field_name, expected_value in expected.items():
        _record_check(
            checks,
            f"summary_json_{field_name}_matches_expected",
            summary_json.get(field_name) == expected_value,
        )
        expected_line = no_wing_release_check._format_markdown_summary_field_line(
            field_name,
            expected_value,
        )
        _record_check(
            checks,
            f"summary_md_{field_name}_matches_expected",
            expected_line in summary_md_text.splitlines(),
        )


def _record_verifier_manifest_surface_checks(
    checks: dict[str, bool],
    *,
    manifest: Mapping[str, Any],
) -> None:
    expected = _expected_verifier_summary_surface()
    _record_check(
        checks,
        "manifest_artifact_manifest_verifier_fields_present",
        all(field_name in manifest for field_name in expected),
    )
    for field_name, expected_value in expected.items():
        _record_check(
            checks,
            f"manifest_{field_name}_matches_expected",
            manifest.get(field_name) == expected_value,
        )


def _record_verifier_report_safety_checks(
    checks: dict[str, bool],
    *,
    report: Mapping[str, Any],
    readiness_closure_allowed: object,
    live_authority_granted: object,
) -> None:
    expected = {
        **VERIFIER_REPORT_STATIC_SAFETY_FIELDS,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
    }
    actual = {
        **{
            field_name: report.get(field_name)
            for field_name in VERIFIER_REPORT_STATIC_SAFETY_FIELDS
        },
        "readiness_closure_allowed": readiness_closure_allowed,
        "live_authority_granted": live_authority_granted,
    }
    for field_name, expected_value in expected.items():
        _record_check(
            checks,
            f"verifier_report_{field_name}_matches_expected",
            actual.get(field_name) == expected_value,
        )


def _record_verifier_report_handoff_plan_summary_surface_checks(
    checks: dict[str, bool],
    *,
    report: dict[str, object],
    summary_json: Mapping[str, Any],
) -> None:
    required_field = "wing_readonly_handoff_plan_verifier_summary_surface_required"
    required_value = summary_json.get(required_field)
    report[required_field] = required_value
    _record_check(
        checks,
        f"verifier_report_{required_field}_is_boolean",
        isinstance(required_value, bool),
    )
    for field_name, expected_value in (
        VERIFIER_REPORT_HANDOFF_PLAN_SUMMARY_SURFACE_EXPECTED_FIELDS.items()
    ):
        actual_value = summary_json.get(field_name)
        report[field_name] = actual_value
        _record_check(
            checks,
            f"verifier_report_{field_name}_matches_expected",
            actual_value == expected_value,
        )


def _record_handoff_validate_run_inputs_only_safety_checks(
    checks: dict[str, bool],
    *,
    summary_json: Mapping[str, Any],
    summary_md_text: str,
) -> None:
    handoff = summary_json.get("wing_readonly_preflight_handoff_packet_artifact")
    if not isinstance(handoff, Mapping):
        handoff = {}
    summary_md_lines = set(summary_md_text.splitlines())
    handoff_section_present = (
        "## WING Read-Only Preflight Handoff Packet Artifact" in summary_md_lines
    )
    expected = (
        no_wing_release_check.WING_READONLY_PREFLIGHT_VALIDATE_RUN_INPUTS_ONLY_EXPECTED_SAFETY_VALUES
    )
    if not handoff and not handoff_section_present:
        _record_check(
            checks,
            "summary_json_handoff_validate_run_inputs_only_surface_present",
            True,
        )
        _record_check(
            checks,
            "summary_md_handoff_validate_run_inputs_only_surface_rendered",
            True,
        )
        return
    _record_check(
        checks,
        "summary_json_handoff_validate_run_inputs_only_surface_present",
        bool(handoff) and all(field_name in handoff for field_name in expected),
    )
    _record_check(
        checks,
        "summary_md_handoff_validate_run_inputs_only_surface_rendered",
        all(
            no_wing_release_check._format_markdown_summary_field_line(
                field_name,
                expected_value,
            )
            in summary_md_lines
            for field_name, expected_value in expected.items()
        ),
    )
    for field_name, expected_value in expected.items():
        _record_check(
            checks,
            f"summary_json_handoff_{field_name}_matches_expected",
            handoff.get(field_name) == expected_value,
        )
        _record_check(
            checks,
            f"summary_json_handoff_{field_name}_match_flag_true",
            handoff.get(f"{field_name}_matches_expected") is True,
        )
        _record_check(
            checks,
            f"summary_md_handoff_{field_name}_matches_expected",
            no_wing_release_check._format_markdown_summary_field_line(
                field_name,
                expected_value,
            )
            in summary_md_lines,
        )
        _record_check(
            checks,
            f"summary_md_handoff_{field_name}_match_flag_true",
            no_wing_release_check._format_markdown_summary_field_line(
                f"{field_name}_matches_expected",
                True,
            )
            in summary_md_lines,
        )


def _record_manifest_handoff_plan_summary_surface_checks(
    checks: dict[str, bool],
    *,
    manifest: Mapping[str, Any],
    summary_json: Mapping[str, Any],
) -> None:
    required_fields = (
        no_wing_release_check.WING_READONLY_HANDOFF_PLAN_VERIFIER_SUMMARY_SURFACE_FIELDS
    )
    _record_check(
        checks,
        "manifest_wing_readonly_handoff_plan_verifier_summary_surface_fields_present",
        all(field_name in manifest for field_name in required_fields),
    )
    for field_name in required_fields:
        _record_check(
            checks,
            f"manifest_{field_name}_matches_summary",
            manifest.get(field_name) == summary_json.get(field_name),
        )

    required_field = "wing_readonly_handoff_plan_verifier_summary_surface_required"
    _record_check(
        checks,
        f"manifest_{required_field}_is_boolean",
        isinstance(manifest.get(required_field), bool),
    )
    for field_name, expected_value in (
        VERIFIER_REPORT_HANDOFF_PLAN_SUMMARY_SURFACE_EXPECTED_FIELDS.items()
    ):
        _record_check(
            checks,
            f"manifest_{field_name}_matches_expected",
            manifest.get(field_name) == expected_value,
        )


def _record_summary_md_safety_checks(
    checks: dict[str, bool],
    *,
    summary_md_text: str,
) -> None:
    summary_md_lines = set(summary_md_text.splitlines())
    for field_name in SUMMARY_MD_REQUIRED_FALSE_SAFETY_FIELDS:
        expected_line = no_wing_release_check._format_markdown_summary_field_line(
            field_name,
            False,
        )
        _record_check(
            checks,
            f"summary_md_{field_name}_denies_authority",
            expected_line in summary_md_lines,
        )


def _record_verifier_report_required_field_checks(
    checks: dict[str, bool],
    *,
    report: dict[str, object],
) -> None:
    required_fields = list(
        no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_VERIFIER_REPORT_REQUIRED_FIELDS
    )
    report["verifier_report_required_fields"] = required_fields
    report.setdefault("verifier_report_required_fields_missing", [])
    report.setdefault("verifier_report_required_fields_match_contract", False)
    missing_fields = [
        field_name for field_name in required_fields if field_name not in report
    ]
    report["verifier_report_required_fields_missing"] = missing_fields
    report["verifier_report_required_fields_match_contract"] = not missing_fields
    _record_check(
        checks,
        "verifier_report_required_fields_present",
        not missing_fields,
    )
    _record_check(
        checks,
        "verifier_report_required_fields_missing_empty",
        missing_fields == [],
    )
    _record_check(
        checks,
        "verifier_report_required_fields_match_contract",
        report["verifier_report_required_fields_match_contract"] is True,
    )


def _finalize_verifier_report(
    report: dict[str, object],
    checks: dict[str, bool],
    *,
    failed_checks: list[str] | None = None,
) -> dict[str, object]:
    _record_verifier_report_required_field_checks(checks, report=report)
    check_failures = [name for name, passed in sorted(checks.items()) if not passed]
    initial_failed_checks = list(dict.fromkeys((failed_checks or []) + check_failures))
    missing_check_entries = [
        name for name in initial_failed_checks if name not in checks
    ]
    report["failed_checks_missing_check_entries"] = missing_check_entries
    _record_check(
        checks,
        "failed_checks_have_check_entries",
        not missing_check_entries,
    )
    _record_check(
        checks,
        "failed_checks_missing_check_entries_empty",
        missing_check_entries == [],
    )
    check_failures = [name for name, passed in sorted(checks.items()) if not passed]
    all_failed_checks = list(dict.fromkeys((failed_checks or []) + check_failures))
    report["status"] = "failed" if all_failed_checks else "passed"
    report["failed_checks"] = all_failed_checks
    return report


def verify_no_wing_artifact_manifest(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    summary_json_path = run_dir / no_wing_release_check.preflight_contract.WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME
    summary_md_path = run_dir / no_wing_release_check.NEXT_WING_PROOF_PLAN_OPERATOR_MARKDOWN_ARTIFACT
    manifest_path = run_dir / no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_NAME
    required_paths = {
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "artifact_manifest": manifest_path,
    }
    missing_files = [
        artifact_id
        for artifact_id, path in required_paths.items()
        if not path.exists()
    ]

    checks: dict[str, bool] = {}
    report: dict[str, object] = {
        "type": "no_wing_release_check_artifact_manifest_verification",
        "run_dir": str(run_dir),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "artifact_manifest_path": str(manifest_path),
        "missing_files": missing_files,
        "status": "pending",
        "failed_checks": [],
        "failed_checks_missing_check_entries": [],
        **VERIFIER_REPORT_STATIC_SAFETY_FIELDS,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "wing_readonly_handoff_plan_verifier_summary_surface_required": False,
        "wing_readonly_handoff_plan_verifier_summary_surface_mismatches": [],
        "wing_readonly_handoff_plan_verifier_summary_surface_match": False,
        "wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed": False,
        "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed": False,
        "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted": False,
        "artifact_manifest_self_consistency_match": False,
        "artifact_manifest_contract_cross_surface_match": False,
        "artifact_manifest_self_consistency_summary_surface_match": False,
        "checks": checks,
    }
    for artifact_id, path in required_paths.items():
        _record_check(checks, f"{artifact_id}_exists", path.exists())
    if missing_files:
        return _finalize_verifier_report(
            report,
            checks,
            failed_checks=[
                f"{artifact_id}_exists" for artifact_id in missing_files
            ],
        )

    try:
        summary_json = json.loads(summary_json_path.read_text(encoding="utf-8"))
        summary_md_text = summary_md_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _record_check(checks, "artifact_bundle_loadable", False)
        report["load_error"] = str(exc)
        return _finalize_verifier_report(
            report,
            checks,
            failed_checks=["artifact_bundle_loadable"],
        )
    _record_check(checks, "artifact_bundle_loadable", True)

    artifacts_by_id = _artifact_rows_by_id(manifest)
    expected_specs = list(
        no_wing_release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_REQUIRED_ARTIFACT_SPECS
    )
    expected_ids = [artifact_id for artifact_id, _ in expected_specs]
    expected_paths = [relative_path for _, relative_path in expected_specs]
    actual_sha256_by_id: dict[str, str] = {}

    _record_check(
        checks,
        "manifest_type",
        manifest.get("type") == "no_wing_release_check_artifact_manifest",
    )
    _record_check(
        checks,
        "manifest_required_artifact_ids_match_contract",
        manifest.get("artifact_manifest_required_artifact_ids") == expected_ids,
    )
    _record_check(
        checks,
        "manifest_required_artifact_paths_match_contract",
        manifest.get("artifact_manifest_required_artifact_paths") == expected_paths,
    )
    _record_check(
        checks,
        "manifest_records_required_artifact_ids",
        sorted(artifacts_by_id) == sorted(expected_ids),
    )

    for artifact_id, relative_path in expected_specs:
        artifact = artifacts_by_id.get(artifact_id, {})
        artifact_path = run_dir / relative_path
        actual_sha256 = _sha256_file(artifact_path)
        actual_sha256_by_id[artifact_id] = actual_sha256
        _record_check(
            checks,
            f"{artifact_id}_path_matches_contract",
            artifact.get("path") == relative_path,
        )
        _record_check(
            checks,
            f"{artifact_id}_sha256_matches_actual",
            artifact.get("sha256") == actual_sha256,
        )

    contract_sha256 = no_wing_release_check.no_wing_artifact_manifest_contract_sha256()
    cross_surface_check = (
        no_wing_release_check.build_no_wing_artifact_manifest_contract_cross_surface_check_from_surfaces(
            summary_json,
            summary_md_text,
            contract_sha256=contract_sha256,
        )
    )
    self_consistency_check = (
        no_wing_release_check.build_no_wing_artifact_manifest_self_consistency_check(
            manifest
        )
    )
    summary_surface_check = (
        no_wing_release_check.build_no_wing_artifact_manifest_self_consistency_summary_surface_check(
            summary_json,
            summary_md_text,
        )
    )

    _compare_recomputed_fields(
        checks,
        prefix="manifest_cross_surface",
        recorded=manifest,
        recomputed=cross_surface_check,
    )
    _compare_recomputed_fields(
        checks,
        prefix="manifest_self_consistency",
        recorded=manifest,
        recomputed=self_consistency_check,
    )
    _compare_recomputed_fields(
        checks,
        prefix="manifest_summary_surface",
        recorded=manifest,
        recomputed=summary_surface_check,
    )
    _record_verifier_summary_surface_checks(
        checks,
        summary_json=summary_json,
        summary_md_text=summary_md_text,
    )
    _record_verifier_manifest_surface_checks(
        checks,
        manifest=manifest,
    )
    _record_manifest_handoff_plan_summary_surface_checks(
        checks,
        manifest=manifest,
        summary_json=summary_json,
    )

    _record_check(
        checks,
        "summary_json_sha256_matches_manifest",
        manifest.get("artifact_manifest_summary_json_sha256")
        == actual_sha256_by_id.get("summary_json"),
    )
    _record_check(
        checks,
        "summary_md_sha256_matches_manifest",
        manifest.get("artifact_manifest_summary_md_sha256")
        == actual_sha256_by_id.get("summary_md"),
    )
    _record_check(
        checks,
        "manifest_denies_readiness_closure",
        manifest.get("artifact_manifest_readiness_closure_allowed") is False,
    )
    _record_check(
        checks,
        "manifest_denies_readiness_matrix_update",
        manifest.get("artifact_manifest_readiness_matrix_update_allowed") is False,
    )
    _record_check(
        checks,
        "manifest_denies_live_authority",
        manifest.get("artifact_manifest_live_authority_granted") is False,
    )
    readiness_closure_allowed = summary_json.get(
        "readiness_closure_allowed_by_packet",
        manifest.get("artifact_manifest_readiness_closure_allowed"),
    )
    readiness_matrix_update_allowed = summary_json.get(
        "readiness_matrix_update_allowed_by_packet",
        summary_json.get(
            "readiness_matrix_update_allowed",
            manifest.get("artifact_manifest_readiness_matrix_update_allowed"),
        ),
    )
    live_authority_granted = summary_json.get(
        "live_authority_granted",
        manifest.get("artifact_manifest_live_authority_granted"),
    )
    if "readiness_closure_allowed_by_packet" in summary_json:
        _record_check(
            checks,
            "summary_denies_readiness_closure",
            readiness_closure_allowed is False,
        )
    if (
        "readiness_matrix_update_allowed_by_packet" in summary_json
        or "readiness_matrix_update_allowed" in summary_json
    ):
        _record_check(
            checks,
            "summary_denies_readiness_matrix_update",
            readiness_matrix_update_allowed is False,
        )
    if "live_authority_granted" in summary_json:
        _record_check(
            checks,
            "summary_denies_live_authority",
            live_authority_granted is False,
        )

    _record_verifier_report_safety_checks(
        checks,
        report=report,
        readiness_closure_allowed=readiness_closure_allowed,
        live_authority_granted=live_authority_granted,
    )
    _record_verifier_report_handoff_plan_summary_surface_checks(
        checks,
        report=report,
        summary_json=summary_json,
    )
    _record_handoff_validate_run_inputs_only_safety_checks(
        checks,
        summary_json=summary_json,
        summary_md_text=summary_md_text,
    )
    _record_summary_md_safety_checks(
        checks,
        summary_md_text=summary_md_text,
    )

    self_consistency_match = self_consistency_check[
        "artifact_manifest_self_consistency_match"
    ]
    cross_surface_match = cross_surface_check[
        "artifact_manifest_contract_cross_surface_match"
    ]
    summary_surface_match = summary_surface_check[
        "artifact_manifest_self_consistency_summary_surface_match"
    ]
    _record_check(
        checks,
        "verifier_report_artifact_manifest_self_consistency_match_matches_expected",
        self_consistency_match is True,
    )
    _record_check(
        checks,
        "verifier_report_artifact_manifest_contract_cross_surface_match_matches_expected",
        cross_surface_match is True,
    )
    _record_check(
        checks,
        "verifier_report_artifact_manifest_self_consistency_summary_surface_match_matches_expected",
        summary_surface_match is True,
    )

    report.update(
        {
            "summary_json_sha256": actual_sha256_by_id.get("summary_json"),
            "summary_md_sha256": actual_sha256_by_id.get("summary_md"),
            "readiness_matrix_update_allowed": readiness_matrix_update_allowed,
            "readiness_closure_allowed": readiness_closure_allowed,
            "live_authority_granted": live_authority_granted,
            "artifact_manifest_contract_sha256": contract_sha256,
            "artifact_manifest_self_consistency_match": self_consistency_match,
            "artifact_manifest_contract_cross_surface_match": cross_surface_match,
            "artifact_manifest_self_consistency_summary_surface_match": summary_surface_match,
        }
    )
    return _finalize_verifier_report(report, checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a no-WING release-check artifact-manifest.json sidecar.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a *_no_wing_release_check artifact directory.",
    )
    args = parser.parse_args(argv)
    report = verify_no_wing_artifact_manifest(args.run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
