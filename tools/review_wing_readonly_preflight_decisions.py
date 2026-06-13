#!/usr/bin/env python3
"""Review WING read-only preflight decision records without contacting hardware."""

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
    build_wing_readonly_preflight_evidence,
)
from backend.wing_readonly_preflight_review_decision import (  # noqa: E402
    DEFAULT_WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOTS,
    build_wing_readonly_preflight_review_decision_ledger,
)
from backend.wing_readonly_preflight_contract import (  # noqa: E402
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "wing_readonly_preflight_review_decisions"
    / "latest_decision_review.md"
)
ACCEPTED_DECISION_READY_STATUS = (
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS
)
DECISION_REVIEW_EXIT_CODES = {
    ACCEPTED_DECISION_READY_STATUS: 0,
    "preflight_candidate_missing": 2,
    "decision_missing_or_invalid": 3,
    "candidate_rejected": 4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def build_decision_review_report(
    *,
    artifact_roots: Iterable[str | Path] | None = None,
    artifact_paths: Iterable[str | Path] | None = None,
    decision_roots: Iterable[str | Path] | None = None,
    decision_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=artifact_roots,
        artifact_paths=artifact_paths,
    )
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_roots=decision_roots,
        decision_paths=decision_paths,
    )
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else {}
    safety = ledger.get("safety") if isinstance(ledger.get("safety"), Mapping) else {}
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    accepted_candidate_prestart_source_ids = _string_list(
        summary.get("accepted_candidate_prestart_source_ids")
    )
    accepted_candidate_prestart_contract_valid = (
        summary.get("accepted_candidate_prestart_contract_valid") is True
    )
    accepted_ready = (
        summary.get("manual_readiness_update_candidate_available") is True
        and int(summary.get("accepted_record_count") or 0) > 0
        and bool(accepted_source_ids)
        and accepted_candidate_prestart_contract_valid
        and accepted_candidate_prestart_source_ids == accepted_source_ids
        and int(summary.get("invalid_record_count") or 0) == 0
        and int(summary.get("unreviewed_candidate_count") or 0) == 0
    )
    rejected_only = (
        int(summary.get("rejected_record_count") or 0) > 0
        and int(summary.get("accepted_record_count") or 0) == 0
        and int(summary.get("invalid_record_count") or 0) == 0
        and int(summary.get("unreviewed_candidate_count") or 0) == 0
    )
    return {
        "type": "wing_readonly_preflight_review_decision_report",
        "generated_at": utc_now(),
        "status": (
            ACCEPTED_DECISION_READY_STATUS
            if accepted_ready
            else "candidate_rejected"
            if rejected_only
            else "blocked"
        ),
        "ledger_status": ledger.get("status"),
        "read_only": safety.get("read_only") is True,
        "live_write": safety.get("live_write") is True,
        "mutates_console_state": safety.get("mutates_console_state") is True,
        "queries_console": safety.get("queries_console") is True,
        "executes_tests": safety.get("executes_tests") is True,
        "operator_required": safety.get("operator_required") is True,
        "execution_allowed_by_decision": safety.get("execution_allowed_by_decision") is True,
        "readiness_matrix_update_allowed_by_decision": (
            safety.get("readiness_matrix_update_allowed_by_decision") is True
        ),
        "readiness_closure_allowed_by_decision": (
            safety.get("readiness_closure_allowed_by_decision") is True
        ),
        "live_authority_granted": safety.get("live_authority_granted") is True,
        "full_autonomous_ready": safety.get("full_autonomous_ready") is True,
        "preflight_evidence_status": summary.get("preflight_evidence_status"),
        "review_candidate_count": int(summary.get("review_candidate_count") or 0),
        "decision_record_count": int(summary.get("decision_record_count") or 0),
        "accepted_record_count": int(summary.get("accepted_record_count") or 0),
        "rejected_record_count": int(summary.get("rejected_record_count") or 0),
        "invalid_record_count": int(summary.get("invalid_record_count") or 0),
        "unreviewed_candidate_count": int(summary.get("unreviewed_candidate_count") or 0),
        "accepted_source_ids": accepted_source_ids,
        "accepted_candidate_prestart_contract_valid": (
            accepted_candidate_prestart_contract_valid
        ),
        "accepted_candidate_prestart_source_ids": accepted_candidate_prestart_source_ids,
        "rejected_source_ids": _string_list(summary.get("rejected_source_ids")),
        "invalid_record_ids": _string_list(summary.get("invalid_record_ids")),
        "duplicate_source_ids": _string_list(summary.get("duplicate_source_ids")),
        "conflicting_source_ids": _string_list(summary.get("conflicting_source_ids")),
        "unreviewed_candidate_source_ids": _string_list(
            summary.get("unreviewed_candidate_source_ids")
        ),
        "candidate_reviews": (
            summary.get("candidate_reviews")
            if isinstance(summary.get("candidate_reviews"), list)
            else []
        ),
        "required_checklist_ids": _string_list(summary.get("required_checklist_ids")),
        "manual_readiness_update_candidate_available": accepted_ready,
        "blocked_by": _string_list(summary.get("blocked_by")),
        "next_required_action": (
            "Accepted review decision is ready for the next manual WING MVP step."
            if accepted_ready
            else (
                "Accepted review decision remains blocked until accepted source ids "
                "have a valid accepted-candidate prestart trace."
            )
            if accepted_source_ids and not accepted_candidate_prestart_contract_valid
            else "Operator rejected the candidate; capture or review a replacement artifact."
            if rejected_only
            else str(summary.get("next_global_step") or "Capture and review WING preflight evidence.")
        ),
        "ledger": ledger,
    }


def exit_code_for_report(report: Mapping[str, Any]) -> int:
    if report.get("status") == ACCEPTED_DECISION_READY_STATUS:
        return DECISION_REVIEW_EXIT_CODES[ACCEPTED_DECISION_READY_STATUS]
    if int(report.get("review_candidate_count") or 0) <= 0:
        return DECISION_REVIEW_EXIT_CODES["preflight_candidate_missing"]
    if report.get("status") == "candidate_rejected":
        return DECISION_REVIEW_EXIT_CODES["candidate_rejected"]
    return DECISION_REVIEW_EXIT_CODES["decision_missing_or_invalid"]


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WING Read-Only Preflight Decision Review",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- ledger_status: `{report['ledger_status']}`",
        f"- preflight_evidence_status: `{report['preflight_evidence_status']}`",
        f"- review_candidate_count: `{report['review_candidate_count']}`",
        f"- decision_record_count: `{report['decision_record_count']}`",
        f"- accepted_record_count: `{report['accepted_record_count']}`",
        f"- rejected_record_count: `{report['rejected_record_count']}`",
        f"- invalid_record_count: `{report['invalid_record_count']}`",
        f"- unreviewed_candidate_count: `{report['unreviewed_candidate_count']}`",
        "- manual_readiness_update_candidate_available: "
        f"`{str(report['manual_readiness_update_candidate_available']).lower()}`",
        "- accepted_candidate_prestart_contract_valid: "
        f"`{str(report['accepted_candidate_prestart_contract_valid']).lower()}`",
        "- accepted_candidate_prestart_source_ids: "
        f"`{json.dumps(report['accepted_candidate_prestart_source_ids'])}`",
        "",
        "## Safety",
        "",
        f"- read_only: `{str(report['read_only']).lower()}`",
        f"- live_write: `{str(report['live_write']).lower()}`",
        f"- mutates_console_state: `{str(report['mutates_console_state']).lower()}`",
        f"- queries_console: `{str(report['queries_console']).lower()}`",
        f"- executes_tests: `{str(report['executes_tests']).lower()}`",
        f"- operator_required: `{str(report['operator_required']).lower()}`",
        "- execution_allowed_by_decision: "
        f"`{str(report['execution_allowed_by_decision']).lower()}`",
        "- readiness_matrix_update_allowed_by_decision: "
        f"`{str(report['readiness_matrix_update_allowed_by_decision']).lower()}`",
        "- readiness_closure_allowed_by_decision: "
        f"`{str(report['readiness_closure_allowed_by_decision']).lower()}`",
        f"- live_authority_granted: `{str(report['live_authority_granted']).lower()}`",
        f"- full_autonomous_ready: `{str(report['full_autonomous_ready']).lower()}`",
        "",
        "## Decisions",
        "",
        f"- accepted_source_ids: `{json.dumps(report['accepted_source_ids'])}`",
        f"- rejected_source_ids: `{json.dumps(report['rejected_source_ids'])}`",
        f"- invalid_record_ids: `{json.dumps(report['invalid_record_ids'])}`",
        f"- duplicate_source_ids: `{json.dumps(report['duplicate_source_ids'])}`",
        f"- conflicting_source_ids: `{json.dumps(report['conflicting_source_ids'])}`",
        "- unreviewed_candidate_source_ids: "
        f"`{json.dumps(report['unreviewed_candidate_source_ids'])}`",
        f"- required_checklist_ids: `{json.dumps(report['required_checklist_ids'])}`",
        "",
        "## Candidate Prestart Contract",
        "",
    ]
    candidate_reviews = (
        report.get("candidate_reviews")
        if isinstance(report.get("candidate_reviews"), list)
        else []
    )
    if candidate_reviews:
        for item in candidate_reviews:
            row = item if isinstance(item, Mapping) else {}
            lines.extend(
                [
                    f"- source: `{row.get('source_id')}`",
                    f"  - status: `{row.get('status')}`",
                    f"  - run_id: `{row.get('run_id')}`",
                    f"  - summary_path: `{row.get('summary_path')}`",
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
                    f"`{row.get('backend_port_prestart_summary_path') or ''}`",
                    "  - backend_port_prestart_summary_sha256: "
                    f"`{row.get('backend_port_prestart_summary_sha256') or ''}`",
                    "  - backend_port_prestart_summary_type: "
                    f"`{row.get('backend_port_prestart_summary_type') or ''}`",
                    "  - backend_port_prestart_summary_status: "
                    f"`{row.get('backend_port_prestart_summary_status') or ''}`",
                    "  - backend_port_prestart_summary_connection_policy: "
                    f"`{row.get('backend_port_prestart_summary_connection_policy') or ''}`",
                    "  - backend_port_prestart_summary_live_write: "
                    f"`{str(row.get('backend_port_prestart_summary_live_write') is True).lower()}`",
                    "  - backend_port_prestart_summary_contacts_backend: "
                    f"`{str(row.get('backend_port_prestart_summary_contacts_backend') is True).lower()}`",
                    "  - backend_port_prestart_summary_queries_console: "
                    f"`{str(row.get('backend_port_prestart_summary_queries_console') is True).lower()}`",
                    "  - backend_port_prestart_summary_executes_tests: "
                    f"`{str(row.get('backend_port_prestart_summary_executes_tests') is True).lower()}`",
                    "  - backend_port_prestart_check_status: "
                    f"`{row.get('backend_port_prestart_check_status') or ''}`",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    blocked_by = report.get("blocked_by") if isinstance(report.get("blocked_by"), list) else []
    rendered_blockers = list(blocked_by)
    if (
        report.get("accepted_source_ids")
        and report.get("accepted_candidate_prestart_contract_valid") is not True
        and WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER not in blocked_by
    ):
        rendered_blockers.append(WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER)
    lines.extend(f"- `{item}`" for item in rendered_blockers)
    if not rendered_blockers:
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
            "does not grant live authority, and does not close readiness rows.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review WING read-only preflight decision records."
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Preflight artifact root to scan for summary.json files. May be repeated.",
    )
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Exact preflight summary.json path. May be repeated.",
    )
    parser.add_argument(
        "--decision-root",
        action="append",
        default=[],
        help="Decision JSON root to scan. May be repeated.",
    )
    parser.add_argument(
        "--decision-path",
        action="append",
        default=[],
        help="Exact decision JSON path. May be repeated.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_roots = tuple(args.artifact_root) or DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS
    artifact_paths = tuple(args.artifact_path) or None
    decision_roots = (
        tuple(args.decision_root)
        or DEFAULT_WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOTS
    )
    decision_paths = tuple(args.decision_path) or None
    report = build_decision_review_report(
        artifact_roots=artifact_roots,
        artifact_paths=artifact_paths,
        decision_roots=decision_roots,
        decision_paths=decision_paths,
    )
    text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return exit_code_for_report(report)


if __name__ == "__main__":
    sys.exit(main())
