#!/usr/bin/env python3
"""Review WING read-only preflight artifacts without contacting hardware."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.wing_readonly_preflight_evidence import (  # noqa: E402
    DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS,
    REQUEST_ID_POLICY,
    build_wing_readonly_preflight_evidence,
)
from backend.wing_readonly_preflight_contract import (  # noqa: E402
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
)
from backend.wing_readonly_preflight_review_decision import (  # noqa: E402
    REQUIRED_REVIEW_CHECKLIST_IDS,
    UNSAFE_DECISION_FLAGS,
)


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "wing_readonly_preflight_review" / "latest_review.md"
DEFAULT_DECISION_TEMPLATE_OUTPUT = (
    REPO_ROOT / WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
)
REVIEW_DECISION_TEMPLATE_STATUS = WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS = tuple(
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
)
MANUAL_REVIEW_CHECKLIST = (
    {
        "id": "confirm_artifact_paths",
        "label": "Confirm summary_path and events_path point to the intended real WING run.",
    },
    {
        "id": "confirm_operator_channel_scope",
        "label": "Confirm operator_confirmed_channels exactly match the queried channels.",
    },
    {
        "id": "confirm_operator_session_ready",
        "label": (
            "Confirm operator_confirmed_session_ready is true for the intended "
            "backend/WING session."
        ),
    },
    {
        "id": "confirm_physical_readbacks",
        "label": "Confirm every readback is verified_from_physical_readback for the declared value types.",
    },
    {
        "id": "confirm_event_log_readonly_sequence",
        "label": "Confirm events.jsonl contains only get_operator_mode, get_state, and query_mixer_parameter sends.",
    },
    {
        "id": "confirm_no_live_authority",
        "label": "Confirm the review does not grant live authority or close readiness rows.",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def manual_review_checklist() -> list[dict[str, str]]:
    return [dict(item) for item in MANUAL_REVIEW_CHECKLIST]


def review_decision_template_required_fields() -> list[str]:
    return list(REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS)


def review_decision_template_status() -> str:
    return REVIEW_DECISION_TEMPLATE_STATUS


def _artifact_review_rows(artifacts: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "source_id": item.get("source_id"),
                "status": item.get("status"),
                "run_id": item.get("run_id"),
                "captured_at": item.get("captured_at"),
                "summary_path": item.get("summary_path"),
                "summary_sha256": item.get("summary_sha256"),
                "events_path": item.get("events_path"),
                "events_sha256": item.get("events_sha256"),
                "expected_artifacts": (
                    item.get("expected_artifacts")
                    if isinstance(item.get("expected_artifacts"), list)
                    else []
                ),
                "expected_summary_artifact": item.get("expected_summary_artifact"),
                "expected_events_artifact": item.get("expected_events_artifact"),
                "summary_path_matches_expected_artifact": (
                    item.get("summary_path_matches_expected_artifact") is True
                ),
                "events_path_matches_expected_artifact": (
                    item.get("events_path_matches_expected_artifact") is True
                ),
                "channels": item.get("channels") if isinstance(item.get("channels"), list) else [],
                "operator_confirmed_channels": (
                    item.get("operator_confirmed_channels")
                    if isinstance(item.get("operator_confirmed_channels"), list)
                    else []
                ),
                "operator_session_confirmation_required": (
                    item.get("operator_session_confirmation_required") is True
                ),
                "operator_confirmed_session_ready": (
                    item.get("operator_confirmed_session_ready") is True
                ),
                "backend_port_prestart_check_required": (
                    item.get("backend_port_prestart_check_required") is True
                ),
                "backend_port_prestart_check_must_pass_before_starting_backend": (
                    item.get("backend_port_prestart_check_must_pass_before_starting_backend")
                    is True
                ),
                "backend_port_prestart_check_command": (
                    item.get("backend_port_prestart_check_command")
                    if isinstance(item.get("backend_port_prestart_check_command"), list)
                    else []
                ),
                "backend_port_prestart_check_failure_detail_fields": (
                    item.get("backend_port_prestart_check_failure_detail_fields")
                    if isinstance(
                        item.get("backend_port_prestart_check_failure_detail_fields"),
                        list,
                    )
                    else []
                ),
                "backend_port_prestart_summary_required": (
                    item.get("backend_port_prestart_summary_required") is True
                ),
                "backend_port_prestart_summary_path": item.get(
                    "backend_port_prestart_summary_path"
                ),
                "backend_port_prestart_summary_sha256": item.get(
                    "backend_port_prestart_summary_sha256"
                ),
                "backend_port_prestart_summary_type": item.get(
                    "backend_port_prestart_summary_type"
                ),
                "backend_port_prestart_summary_status": item.get(
                    "backend_port_prestart_summary_status"
                ),
                "backend_port_prestart_summary_connection_policy": item.get(
                    "backend_port_prestart_summary_connection_policy"
                ),
                "backend_port_prestart_summary_live_write": item.get(
                    "backend_port_prestart_summary_live_write"
                ),
                "backend_port_prestart_summary_contacts_backend": item.get(
                    "backend_port_prestart_summary_contacts_backend"
                ),
                "backend_port_prestart_summary_queries_console": item.get(
                    "backend_port_prestart_summary_queries_console"
                ),
                "backend_port_prestart_summary_executes_tests": item.get(
                    "backend_port_prestart_summary_executes_tests"
                ),
                "backend_port_prestart_check_status": item.get(
                    "backend_port_prestart_check_status"
                ),
                "operator_next_step": (
                    str(item.get("operator_next_step"))
                    if item.get("operator_next_step")
                    else None
                ),
                "value_types": (
                    item.get("value_types") if isinstance(item.get("value_types"), list) else []
                ),
                "readback_count": int(item.get("readback_count") or 0),
                "physical_readback_count": int(item.get("physical_readback_count") or 0),
                "violations": (
                    item.get("violations") if isinstance(item.get("violations"), list) else []
                ),
            }
        )
    return rows


def _valid_artifact_review_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = (
        report.get("artifact_reviews")
        if isinstance(report.get("artifact_reviews"), list)
        else []
    )
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("status") == "valid_readonly_preflight_artifact"
        and not row.get("violations")
    ]


def _artifact_operator_next_steps(artifacts: Iterable[Any]) -> list[str]:
    steps: list[str] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        step = str(item.get("operator_next_step") or "")
        if step and step not in steps:
            steps.append(step)
    return steps


def build_review_decision_template(
    report: Mapping[str, Any],
    *,
    artifact_source_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if report.get("readiness_update_candidate_available") is not True:
        raise ValueError("clean_readonly_preflight_candidate_required")
    candidates = _valid_artifact_review_rows(report)
    if artifact_source_id:
        candidates = [
            row
            for row in candidates
            if str(row.get("source_id") or "") == artifact_source_id
        ]
        if not candidates:
            raise ValueError("decision_template_artifact_source_not_found")
    elif len(candidates) != 1:
        raise ValueError("decision_template_requires_single_candidate_or_source_id")
    candidate = candidates[0]
    source_id = str(candidate.get("source_id") or "")
    run_id = str(candidate.get("run_id") or "")
    captured_at = str(candidate.get("captured_at") or "")
    summary_path = str(candidate.get("summary_path") or "")
    summary_sha256 = str(candidate.get("summary_sha256") or "")
    events_path = str(candidate.get("events_path") or "")
    events_sha256 = str(candidate.get("events_sha256") or "")
    if (
        not source_id
        or not run_id
        or not captured_at
        or not summary_path
        or not summary_sha256
        or not events_path
        or not events_sha256
    ):
        raise ValueError("decision_template_candidate_identity_incomplete")
    template = {
        "template_status": REVIEW_DECISION_TEMPLATE_STATUS,
        "template_generated_at": generated_at or utc_now(),
        "review_id": f"review-wing-readonly-{run_id}",
        "artifact_source_id": source_id,
        "run_id": run_id,
        "artifact_captured_at": captured_at,
        "summary_path": summary_path,
        "summary_sha256": summary_sha256,
        "events_path": events_path,
        "events_sha256": events_sha256,
        "decision": "pending_operator_decision",
        "reviewed_by": "",
        "reviewed_at": "",
        "checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
    }
    template.update({flag: False for flag in UNSAFE_DECISION_FLAGS})
    return template


def build_review_report(
    *,
    artifact_roots: Iterable[str | Path] | None = None,
    artifact_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=artifact_roots,
        artifact_paths=artifact_paths,
    )
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else {}
    safety = evidence.get("safety") if isinstance(evidence.get("safety"), Mapping) else {}
    artifacts = evidence.get("artifacts") if isinstance(evidence.get("artifacts"), list) else []
    artifact_operator_next_steps = _artifact_operator_next_steps(artifacts)
    valid_candidate_count = int(summary.get("valid_candidate_count") or 0)
    invalid_candidate_count = int(summary.get("invalid_candidate_count") or 0)
    candidate_available = (
        evidence.get("status") == "readonly_preflight_candidate_available"
        and valid_candidate_count > 0
        and invalid_candidate_count == 0
        and summary.get("readiness_matrix_update_allowed") is True
        and summary.get("readiness_closure_allowed_by_evidence") is not True
        and summary.get("live_authority_granted") is not True
        and summary.get("full_autonomous_ready") is not True
    )
    next_required_action = str(
        summary.get("next_global_step")
        or "Capture a clean WING read-only preflight artifact."
    )
    if artifact_operator_next_steps:
        next_required_action = artifact_operator_next_steps[0]
    if candidate_available:
        next_required_action = (
            "Operator manually reviews summary.json and events.jsonl before readiness update."
        )
    return {
        "type": "wing_readonly_preflight_artifact_review",
        "generated_at": utc_now(),
        "status": "candidate_ready_for_manual_review" if candidate_available else "blocked",
        "readiness_update_candidate_available": candidate_available,
        "read_only": safety.get("read_only") is True,
        "live_write": safety.get("live_write") is True,
        "queries_console": safety.get("queries_console") is True,
        "execution_allowed_by_review": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "manual_review_required_before_readiness_update": True,
        "manual_review_decision_required": candidate_available,
        "request_id_policy": REQUEST_ID_POLICY,
        "request_id_correlation_required": True,
        "manual_review_checklist": manual_review_checklist(),
        "review_decision_template_available": candidate_available,
        "review_decision_template_required_fields": review_decision_template_required_fields(),
        "review_decision_template_status": REVIEW_DECISION_TEMPLATE_STATUS,
        "artifact_operator_next_steps": artifact_operator_next_steps,
        "evidence_status": evidence.get("status"),
        "artifact_count": int(summary.get("artifact_count") or 0),
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": invalid_candidate_count,
        "latest_valid_candidate_run_id": summary.get("latest_valid_candidate_run_id"),
        "blocked_by": _string_list(summary.get("blocked_by")),
        "violations": _string_list(summary.get("violations")),
        "next_required_action": next_required_action,
        "evidence": evidence,
        "artifact_sources": [
            str(item.get("source_id"))
            for item in artifacts
            if isinstance(item, Mapping) and item.get("source_id")
        ],
        "artifact_reviews": _artifact_review_rows(artifacts),
    }


def exit_code_for_report(report: Mapping[str, Any]) -> int:
    if report.get("readiness_update_candidate_available") is True:
        return 0
    if int(report.get("artifact_count") or 0) <= 0:
        return 2
    return 3


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WING Read-Only Preflight Artifact Review",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- evidence_status: `{report['evidence_status']}`",
        f"- artifact_count: `{report['artifact_count']}`",
        f"- valid_candidate_count: `{report['valid_candidate_count']}`",
        f"- invalid_candidate_count: `{report['invalid_candidate_count']}`",
        f"- readiness_update_candidate_available: `{str(report['readiness_update_candidate_available']).lower()}`",
        f"- latest_valid_candidate_run_id: `{report['latest_valid_candidate_run_id']}`",
        f"- artifact_operator_next_steps: `{json.dumps(report.get('artifact_operator_next_steps') or [])}`",
        "",
        "## Safety",
        "",
        f"- read_only: `{str(report['read_only']).lower()}`",
        f"- live_write: `{str(report['live_write']).lower()}`",
        f"- queries_console: `{str(report['queries_console']).lower()}`",
        f"- execution_allowed_by_review: `{str(report['execution_allowed_by_review']).lower()}`",
        f"- live_authority_granted: `{str(report['live_authority_granted']).lower()}`",
        f"- full_autonomous_ready: `{str(report['full_autonomous_ready']).lower()}`",
        f"- manual_review_required_before_readiness_update: `{str(report['manual_review_required_before_readiness_update']).lower()}`",
        f"- manual_review_decision_required: `{str(report['manual_review_decision_required']).lower()}`",
        f"- request_id_policy: `{report['request_id_policy']}`",
        f"- request_id_correlation_required: `{str(report['request_id_correlation_required']).lower()}`",
        f"- review_decision_template_available: `{str(report['review_decision_template_available']).lower()}`",
        f"- review_decision_template_status: `{report['review_decision_template_status']}`",
        "",
        "## Manual Review Gate",
        "",
    ]
    checklist = (
        report.get("manual_review_checklist")
        if isinstance(report.get("manual_review_checklist"), list)
        else []
    )
    if checklist:
        for item in checklist:
            row = item if isinstance(item, Mapping) else {}
            lines.append(f"- `{row.get('id')}`: {row.get('label')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Review Decision Template", ""])
    lines.extend(
        [
            f"- available: `{str(report.get('review_decision_template_available') is True).lower()}`",
            "- required_fields: "
            f"`{json.dumps(report.get('review_decision_template_required_fields') or [])}`",
        ]
    )
    if report.get("decision_template_output"):
        lines.append(f"- output: `{report['decision_template_output']}`")
    if report.get("decision_template_error"):
        lines.append(f"- error: `{report['decision_template_error']}`")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    blocked_by = report.get("blocked_by") if isinstance(report.get("blocked_by"), list) else []
    lines.extend(f"- `{item}`" for item in blocked_by)
    if not blocked_by:
        lines.append("- none")
    lines.extend(["", "## Violations", ""])
    violations = report.get("violations") if isinstance(report.get("violations"), list) else []
    lines.extend(f"- `{item}`" for item in violations)
    if not violations:
        lines.append("- none")
    lines.extend(["", "## Artifact Sources", ""])
    sources = report.get("artifact_sources") if isinstance(report.get("artifact_sources"), list) else []
    lines.extend(f"- `{item}`" for item in sources)
    if not sources:
        lines.append("- none")
    lines.extend(["", "## Artifact Details", ""])
    artifact_reviews = (
        report.get("artifact_reviews")
        if isinstance(report.get("artifact_reviews"), list)
        else []
    )
    if artifact_reviews:
        for item in artifact_reviews:
            row = item if isinstance(item, Mapping) else {}
            lines.extend(
                [
                    f"- source: `{row.get('source_id')}`",
                    f"  - status: `{row.get('status')}`",
                    f"  - run_id: `{row.get('run_id')}`",
                    f"  - captured_at: `{row.get('captured_at')}`",
                    f"  - summary_path: `{row.get('summary_path')}`",
                    f"  - summary_sha256: `{row.get('summary_sha256')}`",
                    f"  - events_path: `{row.get('events_path')}`",
                    f"  - events_sha256: `{row.get('events_sha256')}`",
                    "  - expected_artifacts: "
                    f"`{json.dumps(row.get('expected_artifacts') or [])}`",
                    "  - expected_summary_artifact: "
                    f"`{row.get('expected_summary_artifact')}`",
                    "  - expected_events_artifact: "
                    f"`{row.get('expected_events_artifact')}`",
                    "  - summary_path_matches_expected_artifact: "
                    f"`{str(row.get('summary_path_matches_expected_artifact') is True).lower()}`",
                    "  - events_path_matches_expected_artifact: "
                    f"`{str(row.get('events_path_matches_expected_artifact') is True).lower()}`",
                    f"  - channels: `{json.dumps(row.get('channels') or [])}`",
                    "  - operator_confirmed_channels: "
                    f"`{json.dumps(row.get('operator_confirmed_channels') or [])}`",
                    "  - operator_session_confirmation_required: "
                    f"`{str(row.get('operator_session_confirmation_required') is True).lower()}`",
                    "  - operator_confirmed_session_ready: "
                    f"`{str(row.get('operator_confirmed_session_ready') is True).lower()}`",
                    "  - backend_port_prestart_check_required: "
                    f"`{str(row.get('backend_port_prestart_check_required') is True).lower()}`",
                    "  - backend_port_prestart_check_must_pass_before_starting_backend: "
                    f"`{str(row.get('backend_port_prestart_check_must_pass_before_starting_backend') is True).lower()}`",
                    "  - backend_port_prestart_check_command: "
                    f"`{json.dumps(row.get('backend_port_prestart_check_command') or [])}`",
                    "  - backend_port_prestart_check_failure_detail_fields: "
                    f"`{json.dumps(row.get('backend_port_prestart_check_failure_detail_fields') or [])}`",
                    "  - backend_port_prestart_summary_required: "
                    f"`{str(row.get('backend_port_prestart_summary_required') is True).lower()}`",
                    "  - backend_port_prestart_summary_path: "
                    f"`{row.get('backend_port_prestart_summary_path')}`",
                    "  - backend_port_prestart_summary_sha256: "
                    f"`{row.get('backend_port_prestart_summary_sha256')}`",
                    "  - backend_port_prestart_summary_type: "
                    f"`{row.get('backend_port_prestart_summary_type')}`",
                    "  - backend_port_prestart_summary_status: "
                    f"`{row.get('backend_port_prestart_summary_status')}`",
                    "  - backend_port_prestart_summary_connection_policy: "
                    f"`{row.get('backend_port_prestart_summary_connection_policy')}`",
                    "  - backend_port_prestart_summary_live_write: "
                    f"`{json.dumps(row.get('backend_port_prestart_summary_live_write'))}`",
                    "  - backend_port_prestart_summary_contacts_backend: "
                    f"`{json.dumps(row.get('backend_port_prestart_summary_contacts_backend'))}`",
                    "  - backend_port_prestart_summary_queries_console: "
                    f"`{json.dumps(row.get('backend_port_prestart_summary_queries_console'))}`",
                    "  - backend_port_prestart_summary_executes_tests: "
                    f"`{json.dumps(row.get('backend_port_prestart_summary_executes_tests'))}`",
                    "  - backend_port_prestart_check_status: "
                    f"`{row.get('backend_port_prestart_check_status')}`",
                    f"  - operator_next_step: `{row.get('operator_next_step')}`",
                    f"  - value_types: `{json.dumps(row.get('value_types') or [])}`",
                    f"  - readback_count: `{int(row.get('readback_count') or 0)}`",
                    "  - physical_readback_count: "
                    f"`{int(row.get('physical_readback_count') or 0)}`",
                    f"  - violations: `{json.dumps(row.get('violations') or [])}`",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Required Action",
            "",
            str(report.get("next_required_action") or ""),
            "",
            "## Boundary",
            "",
            "This review is offline and read-only. It does not contact WING,",
            "does not grant live authority, and does not close readiness rows",
            "without operator review of real artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review WING read-only preflight artifacts.")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Artifact root to scan for summary.json files. May be passed multiple times.",
    )
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Exact summary.json path to review. May be passed multiple times.",
    )
    parser.add_argument(
        "--decision-template-output",
        default=None,
        help=(
            "Optional JSON path for a pending operator decision template. The "
            "template is intentionally invalid until the operator fills decision, "
            "reviewed_by, and reviewed_at."
        ),
    )
    parser.add_argument(
        "--decision-artifact-source-id",
        default=None,
        help="Artifact source id to use when multiple clean preflight candidates exist.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(args.artifact_root) or DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS
    paths = tuple(args.artifact_path) or None
    report = build_review_report(artifact_roots=roots, artifact_paths=paths)
    template_error: str | None = None
    if args.decision_template_output:
        try:
            template = build_review_decision_template(
                report,
                artifact_source_id=args.decision_artifact_source_id,
            )
            template_output = Path(args.decision_template_output)
            write_output(
                template_output,
                json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            report["decision_template_output"] = str(template_output)
        except ValueError as exc:
            template_error = str(exc)
            report["decision_template_error"] = template_error
    text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    if template_error is not None:
        return 3
    return exit_code_for_report(report)


if __name__ == "__main__":
    sys.exit(main())
