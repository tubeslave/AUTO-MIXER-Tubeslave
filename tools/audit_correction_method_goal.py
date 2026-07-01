#!/usr/bin/env python3
"""Audit offline correction-method goal coverage without mixer contact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from backend.processing_proposal_adapters import (  # noqa: E402
    CORRECTION_METHOD_GOAL_PRIORITY_GROUPS,
    CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES,
    CORRECTION_METHOD_GOAL_REQUIRED_OPERATOR_SURFACES,
    MAIN_PIPELINE_ID,
    build_correction_method_goal_operator_review_packet,
    build_proposal_only_method_contracts,
)


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "correction_method_goal_audit"
EXPECTED_PACKET_SCHEMA = "automixer.correction_method_goal_operator_review_packet.v1"
FALSE_AUTHORITY_FLAGS = (
    "contacts_wing",
    "contacts_backend",
    "mutates_mixer",
    "live_authority_granted",
    "readiness_closure_allowed",
    "live_apply_allowed",
    "queue_import_allowed",
    "can_apply_live",
    "can_import_to_queue",
    "can_authorize_live",
    "can_close_readiness",
)
TRUE_BOUNDARY_FLAGS = (
    "read_only_or_dry_run_only",
    "operator_review_required",
    "future_wing_readback_required",
    "hardware_proof_required_before_live",
    "blocks_live_promotion",
    "blocks_queue_import",
)
REQUIRED_METHOD_REVIEW_FIELDS = (
    "source_evidence",
    "source_evidence_details",
    "measured_or_audible_problem",
    "proposed_correction",
    "confidence",
    "bounded_delta",
    "safety_blockers",
    "headroom_feedback_true_peak_checks",
    "operator_readable_explanation",
    "rollback_readback_contract",
    "future_wing_readback_contract",
    "test_evidence",
)
REQUIRED_METHOD_DETAIL_LIST_FIELDS = (
    "goal_requirement_details",
    "method_specific_review_details",
    "operator_actionable_explanation_details",
    "evidence_problem_correction_chain_details",
    "safety_blocker_details",
    "headroom_feedback_true_peak_check_details",
    "operator_acceptance_gate_details",
    "operator_next_action_details",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_run_id() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_correction_method_goal_audit"
    )


def _failure(
    code: str,
    detail: str,
    *,
    actual: Any = None,
    expected: Any = None,
) -> Dict[str, Any]:
    failure: Dict[str, Any] = {"code": code, "detail": detail}
    if actual is not None:
        failure["actual"] = actual
    if expected is not None:
        failure["expected"] = expected
    return failure


def _sorted_str_items(values: Iterable[Any]) -> List[str]:
    return sorted(str(value) for value in values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 100.0
    return round((numerator / denominator) * 100.0, 2)


def _expected_method_contracts(pipeline_id: str) -> List[Mapping[str, Any]]:
    return list(build_proposal_only_method_contracts(pipeline_id=pipeline_id))


def _expected_method_ids(contracts: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(str(contract.get("method_id") or "") for contract in contracts)


def _expected_objective_families(
    contracts: Sequence[Mapping[str, Any]],
) -> List[str]:
    return sorted(
        {
            str(contract.get("objective_family") or "")
            for contract in contracts
            if contract.get("objective_family")
        }
    )


def _required_operator_surfaces() -> List[str]:
    return [
        str(surface["surface"])
        for surface in CORRECTION_METHOD_GOAL_REQUIRED_OPERATOR_SURFACES
    ]


def _check_flag_values(
    failures: List[Dict[str, Any]],
    *,
    payload: Mapping[str, Any],
    label: str,
    false_flags: Sequence[str],
    true_flags: Sequence[str],
) -> None:
    for flag in false_flags:
        actual = payload.get(flag)
        if actual is not False:
            failures.append(
                _failure(
                    f"{label}_{flag}_not_false",
                    f"{label}.{flag} must be false for offline-only audit",
                    actual=actual,
                    expected=False,
                )
            )
    for flag in true_flags:
        actual = payload.get(flag)
        if actual is not True:
            failures.append(
                _failure(
                    f"{label}_{flag}_not_true",
                    f"{label}.{flag} must be true for offline-only audit",
                    actual=actual,
                    expected=True,
                )
            )


def _check_method_card(
    failures: List[Dict[str, Any]],
    *,
    card: Mapping[str, Any],
    required_operator_surfaces: Sequence[str],
) -> None:
    method_id = str(card.get("method_id") or "missing_method_id")
    _check_flag_values(
        failures,
        payload=card,
        label=f"method_{method_id}",
        false_flags=FALSE_AUTHORITY_FLAGS,
        true_flags=TRUE_BOUNDARY_FLAGS,
    )
    if card.get("review_scope") != "offline_proposal_only":
        failures.append(
            _failure(
                f"method_{method_id}_review_scope_not_offline_proposal_only",
                "method review scope must remain offline proposal only",
                actual=card.get("review_scope"),
                expected="offline_proposal_only",
            )
        )
    actual_required_surfaces = _sorted_str_items(
        _as_list(card.get("required_operator_surfaces"))
    )
    expected_required_surfaces = sorted(required_operator_surfaces)
    if actual_required_surfaces != expected_required_surfaces:
        failures.append(
            _failure(
                f"method_{method_id}_required_operator_surfaces_mismatch",
                "method card must expose the canonical required operator surfaces",
                actual=actual_required_surfaces,
                expected=expected_required_surfaces,
            )
        )
    missing_required_surfaces = _as_list(
        card.get("missing_required_operator_surfaces")
    )
    if missing_required_surfaces:
        failures.append(
            _failure(
                f"method_{method_id}_missing_required_operator_surface_claims",
                "method card reports missing required operator surfaces",
                actual=missing_required_surfaces,
                expected=[],
            )
        )
    for field in REQUIRED_METHOD_REVIEW_FIELDS:
        if not card.get(field):
            failures.append(
                _failure(
                    f"method_{method_id}_missing_{field}",
                    f"method card must include {field}",
                    actual=card.get(field),
                    expected="non-empty",
                )
            )
    for field in REQUIRED_METHOD_DETAIL_LIST_FIELDS:
        if not isinstance(card.get(field), list) or not card.get(field):
            failures.append(
                _failure(
                    f"method_{method_id}_missing_{field}",
                    f"method card must include non-empty {field}",
                    actual=card.get(field),
                    expected="non-empty-list",
                )
            )


def _check_priority_section(
    failures: List[Dict[str, Any]],
    *,
    section: Mapping[str, Any],
    expected_group: Mapping[str, Any],
) -> None:
    priority = expected_group["priority"]
    label = expected_group["label"]
    families = list(expected_group["objective_families"])
    if section.get("priority") != priority:
        failures.append(
            _failure(
                f"priority_{priority}_priority_mismatch",
                "priority section has unexpected priority id",
                actual=section.get("priority"),
                expected=priority,
            )
        )
    if section.get("label") != label:
        failures.append(
            _failure(
                f"priority_{priority}_label_mismatch",
                "priority section has unexpected label",
                actual=section.get("label"),
                expected=label,
            )
        )
    if list(section.get("objective_families") or []) != families:
        failures.append(
            _failure(
                f"priority_{priority}_objective_families_mismatch",
                "priority section objective families changed",
                actual=section.get("objective_families"),
                expected=families,
            )
        )
    if not _as_list(section.get("methods")):
        failures.append(
            _failure(
                f"priority_{priority}_methods_missing",
                "priority section must include method summaries",
                actual=section.get("methods"),
                expected="non-empty-list",
            )
        )
    _check_flag_values(
        failures,
        payload=section,
        label=f"priority_{priority}",
        false_flags=FALSE_AUTHORITY_FLAGS,
        true_flags=TRUE_BOUNDARY_FLAGS,
    )
    for field in REQUIRED_METHOD_DETAIL_LIST_FIELDS:
        if not isinstance(section.get(field), list) or not section.get(field):
            failures.append(
                _failure(
                    f"priority_{priority}_missing_{field}",
                    f"priority section must include non-empty {field}",
                    actual=section.get(field),
                    expected="non-empty-list",
                )
            )


def audit_correction_method_goal_packet(
    packet: Optional[Mapping[str, Any]] = None,
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
    generated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit the offline correction-method goal packet.

    The audit proves proposal/review coverage only. A passed report still
    requires real WING readback before any live promotion.
    """

    packet = (
        build_correction_method_goal_operator_review_packet(
            pipeline_id=pipeline_id,
        )
        if packet is None
        else packet
    )
    packet = _as_mapping(packet)
    generated_at_utc = generated_at_utc or _utc_now()
    failures: List[Dict[str, Any]] = []
    contracts = _expected_method_contracts(pipeline_id)
    expected_method_ids = _expected_method_ids(contracts)
    expected_families = _expected_objective_families(contracts)
    required_families = list(CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES)
    required_operator_surfaces = _required_operator_surfaces()

    if packet.get("schema") != EXPECTED_PACKET_SCHEMA:
        failures.append(
            _failure(
                "packet_schema_mismatch",
                "packet schema is not the correction-method operator review schema",
                actual=packet.get("schema"),
                expected=EXPECTED_PACKET_SCHEMA,
            )
        )
    if packet.get("pipeline_id") != pipeline_id:
        failures.append(
            _failure(
                "packet_pipeline_id_mismatch",
                "packet pipeline_id does not match audit target",
                actual=packet.get("pipeline_id"),
                expected=pipeline_id,
            )
        )
    _check_flag_values(
        failures,
        payload=packet,
        label="packet",
        false_flags=FALSE_AUTHORITY_FLAGS,
        true_flags=TRUE_BOUNDARY_FLAGS,
    )
    if packet.get("queue_action") != "not_queueable_until_promoted":
        failures.append(
            _failure(
                "packet_queue_action_mismatch",
                "packet must not be queueable before live promotion",
                actual=packet.get("queue_action"),
                expected="not_queueable_until_promoted",
            )
        )

    cards = _as_list(packet.get("method_review_cards"))
    card_mappings = [card for card in cards if isinstance(card, Mapping)]
    if len(card_mappings) != len(cards):
        failures.append(
            _failure(
                "packet_method_review_cards_include_non_objects",
                "all method_review_cards must be objects",
                actual=len(cards) - len(card_mappings),
                expected=0,
            )
        )
    cards_by_method_id: Dict[str, Mapping[str, Any]] = {}
    duplicate_method_ids: List[str] = []
    for card in card_mappings:
        method_id = str(card.get("method_id") or "")
        if not method_id:
            failures.append(
                _failure(
                    "method_card_missing_method_id",
                    "every method card must have method_id",
                    actual=card.get("method_id"),
                    expected="non-empty-string",
                )
            )
            continue
        if method_id in cards_by_method_id:
            duplicate_method_ids.append(method_id)
            continue
        cards_by_method_id[method_id] = card
        _check_method_card(
            failures,
            card=card,
            required_operator_surfaces=required_operator_surfaces,
        )
    if duplicate_method_ids:
        failures.append(
            _failure(
                "packet_duplicate_method_ids",
                "method_review_cards must be unique by method_id",
                actual=sorted(duplicate_method_ids),
                expected=[],
            )
        )
    actual_method_ids = sorted(cards_by_method_id)
    missing_method_ids = sorted(set(expected_method_ids) - set(actual_method_ids))
    extra_method_ids = sorted(set(actual_method_ids) - set(expected_method_ids))
    for method_id in missing_method_ids:
        failures.append(
            _failure(
                f"missing_expected_method_{method_id}",
                "packet is missing an expected proposal-only method card",
                actual=None,
                expected=method_id,
            )
        )
    if extra_method_ids:
        failures.append(
            _failure(
                "packet_extra_method_ids",
                "packet has method cards not present in proposal-only contracts",
                actual=extra_method_ids,
                expected=[],
            )
        )

    actual_families = sorted(
        {
            str(card.get("objective_family") or "")
            for card in card_mappings
            if card.get("objective_family")
        }
    )
    missing_required_families = sorted(
        set(required_families) - set(actual_families)
    )
    missing_expected_families = sorted(
        set(expected_families) - set(actual_families)
    )
    for family in missing_required_families:
        failures.append(
            _failure(
                f"missing_objective_family_{family}",
                "packet is missing a required correction objective family",
                actual=actual_families,
                expected=family,
            )
        )
    for family in sorted(set(missing_expected_families) - set(missing_required_families)):
        failures.append(
            _failure(
                f"missing_supporting_objective_family_{family}",
                "packet is missing a supporting proposal-only objective family",
                actual=actual_families,
                expected=family,
            )
        )

    priority_sections = _as_list(packet.get("priority_sections"))
    section_by_priority = {
        int(section.get("priority")): section
        for section in priority_sections
        if isinstance(section, Mapping) and isinstance(section.get("priority"), int)
    }
    if len(priority_sections) != len(CORRECTION_METHOD_GOAL_PRIORITY_GROUPS):
        failures.append(
            _failure(
                "packet_priority_section_count_mismatch",
                "packet must expose one priority section per required group",
                actual=len(priority_sections),
                expected=len(CORRECTION_METHOD_GOAL_PRIORITY_GROUPS),
            )
        )
    for group in CORRECTION_METHOD_GOAL_PRIORITY_GROUPS:
        priority = int(group["priority"])
        section = section_by_priority.get(priority)
        if section is None:
            failures.append(
                _failure(
                    f"missing_priority_section_{priority}",
                    "packet is missing a required priority section",
                    actual=None,
                    expected=priority,
                )
            )
            continue
        _check_priority_section(failures, section=section, expected_group=group)

    required_family_coverage_percent = _percent(
        len(set(required_families) - set(missing_required_families)),
        len(required_families),
    )
    expected_family_coverage_percent = _percent(
        len(set(expected_families) - set(missing_expected_families)),
        len(expected_families),
    )
    expected_method_coverage_percent = _percent(
        len(set(expected_method_ids) - set(missing_method_ids)),
        len(expected_method_ids),
    )
    surface_checks_total = len(card_mappings) * (
        len(REQUIRED_METHOD_REVIEW_FIELDS)
        + len(REQUIRED_METHOD_DETAIL_LIST_FIELDS)
    )
    surface_checks_failed = sum(
        1
        for failure in failures
        if "_missing_" in failure["code"]
        and failure["code"].startswith("method_")
    )
    operator_surface_coverage_percent = _percent(
        surface_checks_total - surface_checks_failed,
        surface_checks_total,
    )
    no_live_authority = not any(
        failure["code"].endswith("_not_false")
        or failure["code"].endswith("_not_true")
        or failure["code"] == "packet_queue_action_mismatch"
        for failure in failures
    )

    priority_rows = []
    for group in CORRECTION_METHOD_GOAL_PRIORITY_GROUPS:
        section = section_by_priority.get(int(group["priority"]))
        priority_rows.append(
            {
                "priority": group["priority"],
                "label": group["label"],
                "objective_families": list(group["objective_families"]),
                "covered": section is not None
                and not (
                    set(group["objective_families"])
                    & set(missing_required_families)
                ),
                "method_count": len(_as_list(section.get("methods")))
                if section
                else 0,
            }
        )

    status = "passed" if not failures else "failed"
    return {
        "schema": "automixer.correction_method_goal_audit.v1",
        "generated_at_utc": generated_at_utc,
        "pipeline_id": pipeline_id,
        "packet_schema": packet.get("schema"),
        "status": status,
        "failed_checks": failures,
        "read_only": True,
        "contacts_wing": False,
        "contacts_backend": False,
        "mutates_mixer": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "hardware_proof_required_before_live": True,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "no_live_authority": no_live_authority,
        "method_review_card_count": len(card_mappings),
        "expected_method_count": len(expected_method_ids),
        "missing_expected_method_ids": missing_method_ids,
        "extra_method_ids": extra_method_ids,
        "priority_section_count": len(priority_sections),
        "expected_priority_section_count": len(CORRECTION_METHOD_GOAL_PRIORITY_GROUPS),
        "required_objective_families": required_families,
        "covered_required_objective_families": sorted(
            set(required_families) - set(missing_required_families)
        ),
        "missing_required_objective_families": missing_required_families,
        "expected_objective_families": expected_families,
        "covered_expected_objective_families": sorted(
            set(expected_families) - set(missing_expected_families)
        ),
        "missing_expected_objective_families": missing_expected_families,
        "supporting_objective_families": sorted(
            set(expected_families) - set(required_families)
        ),
        "priority_rows": priority_rows,
        "required_operator_surfaces": required_operator_surfaces,
        "required_method_review_fields": list(REQUIRED_METHOD_REVIEW_FIELDS),
        "required_method_detail_list_fields": list(
            REQUIRED_METHOD_DETAIL_LIST_FIELDS
        ),
        "coverage_percent": {
            "required_objective_families": required_family_coverage_percent,
            "expected_objective_families": expected_family_coverage_percent,
            "expected_method_cards": expected_method_coverage_percent,
            "operator_surfaces": operator_surface_coverage_percent,
            "no_live_authority": 100.0 if no_live_authority else 0.0,
        },
        "overall_offline_goal_percent": round(
            (
                required_family_coverage_percent
                + expected_method_coverage_percent
                + operator_surface_coverage_percent
                + (100.0 if no_live_authority else 0.0)
            )
            / 4.0,
            2,
        ),
        "denial_reason": "offline_goal_audit_cannot_replace_real_wing_proof",
        "operator_next_step": (
            "review_offline_method_cards_and_collect_real_wing_readback_"
            "before_live_promotion"
        ),
    }


def load_packet(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_correction_method_goal_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Automixer Correction Method Goal Audit",
        "",
        f"- status: `{report.get('status')}`",
        f"- pipeline_id: `{report.get('pipeline_id')}`",
        f"- read_only: `{str(report.get('read_only')).lower()}`",
        f"- contacts_wing: `{str(report.get('contacts_wing')).lower()}`",
        f"- live_authority_granted: `{str(report.get('live_authority_granted')).lower()}`",
        f"- readiness_closure_allowed: `{str(report.get('readiness_closure_allowed')).lower()}`",
        f"- method_review_card_count: `{report.get('method_review_card_count')}`",
        f"- priority_section_count: `{report.get('priority_section_count')}`",
        f"- overall_offline_goal_percent: `{report.get('overall_offline_goal_percent')}`",
        f"- denial_reason: `{report.get('denial_reason')}`",
        "",
        "## Coverage Percent",
    ]
    coverage = _as_mapping(report.get("coverage_percent"))
    for key in sorted(coverage):
        lines.append(f"- {key}: `{coverage[key]}`")
    lines.extend(["", "## Required Objective Families"])
    missing_required = set(_as_list(report.get("missing_required_objective_families")))
    for family in _as_list(report.get("required_objective_families")):
        state = "missing" if family in missing_required else "covered"
        lines.append(f"- `{family}`: `{state}`")
    lines.extend(["", "## Priority Sections"])
    for row in _as_list(report.get("priority_rows")):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"`{row.get('priority')}` {row.get('label')}: "
            f"`{'covered' if row.get('covered') else 'missing'}` "
            f"({row.get('method_count')} methods)"
        )
    lines.extend(["", "## Failed Checks"])
    failures = _as_list(report.get("failed_checks"))
    if failures:
        for failure in failures:
            if isinstance(failure, Mapping):
                lines.append(f"- `{failure.get('code')}`: {failure.get('detail')}")
            else:
                lines.append(f"- `{failure}`")
    else:
        lines.append("- `none`")
    lines.extend(
        [
            "",
            "## Operator Next Step",
            f"- `{report.get('operator_next_step')}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_correction_method_goal_audit(
    report: Mapping[str, Any],
    *,
    output: Optional[Path] = None,
    markdown_output: Optional[Path] = None,
) -> Dict[str, Any]:
    written: Dict[str, Any] = {}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written["json"] = str(output)
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(
            render_correction_method_goal_audit_markdown(report),
            encoding="utf-8",
        )
        written["markdown"] = str(markdown_output)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit offline correction-method goal coverage.",
    )
    parser.add_argument("--packet", type=Path, help="Existing packet.json to audit.")
    parser.add_argument("--pipeline-id", default=MAIN_PIPELINE_ID)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--generated-at-utc", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_id = args.run_id or _default_run_id()
    output = args.output
    markdown_output = args.markdown_output
    if output is None and markdown_output is None:
        run_dir = args.artifact_root / run_id
        output = run_dir / "audit.json"
        markdown_output = run_dir / "audit.md"

    packet = load_packet(args.packet) if args.packet else None
    report = audit_correction_method_goal_packet(
        packet,
        pipeline_id=args.pipeline_id,
        generated_at_utc=args.generated_at_utc,
    )
    written = write_correction_method_goal_audit(
        report,
        output=output,
        markdown_output=markdown_output,
    )
    report_for_stdout = dict(report)
    if written:
        report_for_stdout["written"] = written
    print(json.dumps(report_for_stdout, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
