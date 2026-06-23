#!/usr/bin/env python3
"""Export correction-method operator review artifacts without mixer contact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from backend.processing_proposal_adapters import (  # noqa: E402
    MAIN_PIPELINE_ID,
    build_correction_method_goal_operator_review_packet,
    render_correction_method_goal_operator_review_markdown,
)


DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "correction_method_operator_review"
)
ALLOWED_OFFLINE_REVIEW_DECISIONS = [
    "request_more_evidence",
    "accept_as_offline_proposal",
    "reject_proposal",
    "defer_until_real_wing_proof",
]
OPERATOR_REVIEW_CONTRACT_FIELDS = (
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
PENDING_OPERATOR_REVIEW_RECORD_FIELDS = {
    "reviewer": "",
    "reviewed_at_utc": "",
    "rationale": "",
    "evidence_packet_reference": "",
    "source_material_reference": "",
    "source_evidence_acknowledgement": "",
    "test_evidence_reference": "",
    "test_evidence_acknowledgement": "",
    "verification_report_reference": "",
    "verification_report_acknowledgement": "",
    "measured_problem_acknowledgement": "",
    "proposed_correction_acknowledgement": "",
    "bounded_delta_acknowledgement": "",
    "confidence_acknowledgement": "",
    "operator_explanation_acknowledgement": "",
    "future_wing_readback_acknowledgement": "",
    "headroom_feedback_true_peak_acknowledgement": "",
    "rollback_readback_acknowledgement": "",
    "safety_blockers_acknowledgement": "",
    "required_confirmations_acknowledgement": "",
    "offline_acceptance_boundary_acknowledgement": "",
    "operator_notes": "",
}
EXPECTED_VERIFICATION_REPORT_REFERENCE = "verification-report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_run_id() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_correction_method_operator_review"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> Dict[str, Any]:
    path.write_bytes(data)
    return {
        "path": path.name,
        "sha256": _sha256(data),
        "bytes": len(data),
    }


def build_operator_decision_template(
    *,
    packet: Dict[str, Any],
    run_id: str,
    pipeline_id: str,
    generated_at_utc: str,
) -> Dict[str, Any]:
    """Build a pending offline review template for operator decisions."""

    method_decisions = []
    for card in packet["method_review_cards"]:
        allowed_decisions = list(
            card.get("operator_review_decisions")
            or ALLOWED_OFFLINE_REVIEW_DECISIONS
        )
        method_decisions.append(
            {
                "method_id": card["method_id"],
                "method_label": card["method_label"],
                "objective_family": card["objective_family"],
                "priority": card.get("priority"),
                "priority_label": card.get("priority_label"),
                "priority_review_scope": card.get("priority_review_scope"),
                "review_scope": "offline_proposal_only",
                "review_status": "operator_review_pending",
                "operator_decision": None,
                **PENDING_OPERATOR_REVIEW_RECORD_FIELDS,
                "expected_evidence_packet_reference": (
                    f"packet.json#method_review_cards/{card['method_id']}"
                ),
                "expected_source_material_reference": (
                    "packet.json#method_review_cards/"
                    f"{card['method_id']}/source_evidence"
                ),
                "expected_test_evidence_references": list(
                    card["test_evidence_references"]
                ),
                "expected_verification_report_reference": (
                    EXPECTED_VERIFICATION_REPORT_REFERENCE
                ),
                "allowed_decisions": allowed_decisions,
                "required_confirmations": list(
                    card["what_operator_must_confirm"]
                ),
                "required_confirmation_details": list(
                    card["required_confirmation_details"]
                ),
                "operator_review_decision_option_details": list(
                    card["operator_review_decision_option_details"]
                ),
                "required_evidence": list(card["required_evidence"]),
                "required_safety_checks": list(
                    card["required_safety_checks"]
                ),
                "operator_problem_review_summary": list(
                    card["operator_problem_review_summary"]
                ),
                "operator_method_review_focus": list(
                    card["operator_method_review_focus"]
                ),
                "why_safe_summary": list(card["why_safe_summary"]),
                "why_blocked_summary": list(card["why_blocked_summary"]),
                "operator_visible_explanation_contract": dict(
                    card["operator_visible_explanation_contract"]
                ),
                **{
                    field: dict(card[field])
                    for field in OPERATOR_REVIEW_CONTRACT_FIELDS
                },
                "test_evidence_references": list(
                    card["test_evidence_references"]
                ),
                "bounded_delta_unit": card["bounded_delta_unit"],
                "bounded_delta_policy": card["bounded_delta_policy"],
                "rollback_expectation": card["rollback_expectation"],
                "operator_payload_required_fields": list(
                    card["operator_payload_required_fields"]
                ),
                "operator_evidence_required_fields": list(
                    card["operator_evidence_required_fields"]
                ),
                "future_wing_readback_targets": list(
                    card["future_wing_readback_targets"]
                ),
                "future_wing_readback_contract_summary": list(
                    card["future_wing_readback_contract_summary"]
                ),
                "future_live_required_proofs": list(
                    card["future_live_required_proofs"]
                ),
                "future_live_requirement_details": list(
                    card["future_live_requirement_details"]
                ),
                "blocked_live_actions": list(card["blocked_live_actions"]),
                "safety_blocker_details": list(card["safety_blocker_details"]),
                "headroom_feedback_true_peak_check_details": list(
                    card["headroom_feedback_true_peak_check_details"]
                ),
                "live_authority_boundary_summary": list(
                    card["live_authority_boundary_summary"]
                ),
                "operator_review_record_required_fields": list(
                    card["operator_review_record_required_fields"]
                ),
                "operator_review_checklist_details": list(
                    card["operator_review_checklist_details"]
                ),
                "operator_review_record_field_details": list(
                    card["operator_review_record_field_details"]
                ),
                "operator_next_action_details": list(
                    card["operator_next_action_details"]
                ),
                "can_authorize_live": False,
                "read_only_or_dry_run_only": True,
                "contacts_wing": False,
                "contacts_backend": False,
                "queries_console": False,
                "mutates_mixer": False,
                "mutates_console_state": False,
                "live_write": False,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "hardware_proof_required_before_live": True,
            }
        )

    method_decisions_by_id = {
        row["method_id"]: row for row in method_decisions
    }
    priority_decision_sections = []
    for section in packet["priority_sections"]:
        section_method_ids = [
            card["method_id"] for card in section.get("methods", [])
        ]
        section_decisions = [
            method_decisions_by_id[method_id]
            for method_id in section_method_ids
            if method_id in method_decisions_by_id
        ]
        def _section_items(field: str) -> list[str]:
            return sorted(
                {
                    str(item)
                    for row in section_decisions
                    for item in row[field]
                }
            )

        def _section_mapping_items(field: str) -> list[str]:
            values: list[str] = []
            for row in section_decisions:
                mapping = row.get(field)
                if not isinstance(mapping, dict):
                    continue
                values.extend(
                    f"{key}={str(value).lower()}"
                    for key, value in sorted(mapping.items())
                )
            return sorted(set(values))

        operator_payload_required_fields = _section_items(
            "operator_payload_required_fields"
        )
        operator_evidence_required_fields = _section_items(
            "operator_evidence_required_fields"
        )
        operator_problem_review_summary = _section_items(
            "operator_problem_review_summary"
        )
        operator_method_review_focus = _section_items(
            "operator_method_review_focus"
        )
        why_safe_summary = _section_items("why_safe_summary")
        why_blocked_summary = _section_items("why_blocked_summary")
        operator_visible_explanation_contract = _section_mapping_items(
            "operator_visible_explanation_contract"
        )
        required_evidence = list(section["required_evidence"])
        required_safety_checks = list(section["required_safety_checks"])
        test_evidence_references = _section_items("test_evidence_references")
        operator_review_contracts = {
            field: list(section[field])
            for field in OPERATOR_REVIEW_CONTRACT_FIELDS
        }
        bounded_delta_policies = list(section["bounded_delta_policies"])
        rollback_expectations = list(section["rollback_expectations"])
        future_wing_readback_targets = sorted(
            {
                str(target)
                for row in section_decisions
                for target in row["future_wing_readback_targets"]
            }
        )
        future_wing_readback_contract_summary = _section_items(
            "future_wing_readback_contract_summary"
        )
        future_live_required_proofs = sorted(
            {
                str(proof)
                for row in section_decisions
                for proof in row["future_live_required_proofs"]
            }
        )
        future_live_requirement_details = _section_items(
            "future_live_requirement_details"
        )
        blocked_live_actions = sorted(
            {
                str(action)
                for row in section_decisions
                for action in row["blocked_live_actions"]
            }
        )
        safety_blocker_details = _section_items("safety_blocker_details")
        headroom_feedback_true_peak_check_details = _section_items(
            "headroom_feedback_true_peak_check_details"
        )
        live_authority_boundary_summary = _section_items(
            "live_authority_boundary_summary"
        )
        operator_review_record_required_fields = _section_items(
            "operator_review_record_required_fields"
        )
        required_confirmation_details = _section_items(
            "required_confirmation_details"
        )
        operator_review_decision_option_details = _section_items(
            "operator_review_decision_option_details"
        )
        operator_review_checklist_details = _section_items(
            "operator_review_checklist_details"
        )
        operator_review_record_field_details = _section_items(
            "operator_review_record_field_details"
        )
        operator_next_action_details = _section_items(
            "operator_next_action_details"
        )
        priority_decision_sections.append(
            {
                "priority": section["priority"],
                "label": section["label"],
                "objective_families": list(section["objective_families"]),
                "review_scope": "offline_proposal_only",
                "review_status": "operator_review_pending",
                "operator_decision": None,
                **PENDING_OPERATOR_REVIEW_RECORD_FIELDS,
                "expected_evidence_packet_reference": (
                    f"packet.json#priority_sections/{section['priority']}"
                ),
                "expected_source_material_reference": (
                    "packet.json#priority_sections/"
                    f"{section['priority']}/source_evidence"
                ),
                "expected_test_evidence_references": test_evidence_references,
                "expected_verification_report_reference": (
                    EXPECTED_VERIFICATION_REPORT_REFERENCE
                ),
                "method_ids": section_method_ids,
                "method_decision_count": len(section_decisions),
                "required_priority_operator_decision": (
                    "one_offline_decision_per_priority_section"
                ),
                "required_operator_decisions": "one_offline_decision_per_method",
                "allowed_decisions": list(ALLOWED_OFFLINE_REVIEW_DECISIONS),
                "required_evidence": required_evidence,
                "required_safety_checks": required_safety_checks,
                "test_evidence_references": test_evidence_references,
                **operator_review_contracts,
                "bounded_delta_policies": bounded_delta_policies,
                "rollback_expectations": rollback_expectations,
                "operator_payload_required_fields": operator_payload_required_fields,
                "operator_evidence_required_fields": operator_evidence_required_fields,
                "operator_problem_review_summary": operator_problem_review_summary,
                "operator_method_review_focus": operator_method_review_focus,
                "why_safe_summary": why_safe_summary,
                "why_blocked_summary": why_blocked_summary,
                "operator_visible_explanation_contract": (
                    operator_visible_explanation_contract
                ),
                "future_wing_readback_targets": future_wing_readback_targets,
                "future_wing_readback_contract_summary": (
                    future_wing_readback_contract_summary
                ),
                "future_live_required_proofs": future_live_required_proofs,
                "future_live_requirement_details": (
                    future_live_requirement_details
                ),
                "blocked_live_actions": blocked_live_actions,
                "safety_blocker_details": safety_blocker_details,
                "headroom_feedback_true_peak_check_details": (
                    headroom_feedback_true_peak_check_details
                ),
                "live_authority_boundary_summary": live_authority_boundary_summary,
                "operator_review_record_required_fields": (
                    operator_review_record_required_fields
                ),
                "required_confirmation_details": (
                    required_confirmation_details
                ),
                "operator_review_decision_option_details": (
                    operator_review_decision_option_details
                ),
                "operator_review_checklist_details": (
                    operator_review_checklist_details
                ),
                "operator_review_record_field_details": (
                    operator_review_record_field_details
                ),
                "operator_next_action_details": (
                    operator_next_action_details
                ),
                "can_authorize_live": False,
                "read_only_or_dry_run_only": True,
                "read_only": True,
                "contacts_wing": False,
                "contacts_backend": False,
                "queries_console": False,
                "mutates_mixer": False,
                "mutates_console_state": False,
                "live_write": False,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "hardware_proof_required_before_live": True,
            }
        )

    return {
        "schema": "automixer.correction_method_operator_decision_template.v1",
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "pipeline_id": pipeline_id,
        "status": "operator_review_template_pending_proposal_only",
        "packet_schema": packet["schema"],
        "packet_status": packet["status"],
        "method_decision_count": len(method_decisions),
        "priority_decision_section_count": len(priority_decision_sections),
        "allowed_decisions": list(ALLOWED_OFFLINE_REVIEW_DECISIONS),
        "operator_completion_requires": [
            "review_packet_json",
            "operator_review_markdown",
            "verification_report_passed",
            "explicit_offline_decision_per_priority_section",
            "explicit_offline_decision_per_method",
            "future_real_wing_readback_before_live_promotion",
        ],
        "read_only_or_dry_run_only": True,
        "read_only": True,
        "contacts_wing": False,
        "contacts_backend": False,
        "queries_console": False,
        "mutates_mixer": False,
        "mutates_console_state": False,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_authorize_live": False,
        "hardware_proof_required_before_live": True,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "denial_reason": "operator_decision_template_cannot_authorize_live_or_close_readiness",
        "priority_decision_sections": priority_decision_sections,
        "method_decisions": method_decisions,
    }


def build_operator_decision_template_summary(
    decision_template: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the manifest-visible summary of the pending decision template."""

    priority_sections = decision_template["priority_decision_sections"]
    def _priority_field_counts(field: str) -> Dict[str, int]:
        return {
            str(section["priority"]): len(section.get(field) or [])
            for section in priority_sections
        }

    return {
        "method_decision_count": decision_template["method_decision_count"],
        "priority_decision_section_count": decision_template[
            "priority_decision_section_count"
        ],
        "priority_decision_labels": [
            section["label"] for section in priority_sections
        ],
        "priority_method_decision_counts": {
            str(section["priority"]): section["method_decision_count"]
            for section in priority_sections
        },
        "priority_required_evidence_counts": _priority_field_counts(
            "required_evidence"
        ),
        "priority_required_safety_check_counts": _priority_field_counts(
            "required_safety_checks"
        ),
        "priority_source_evidence_counts": _priority_field_counts(
            "source_evidence"
        ),
        "priority_source_evidence_detail_counts": _priority_field_counts(
            "source_evidence_details"
        ),
        "priority_measured_or_audible_problem_counts": _priority_field_counts(
            "measured_or_audible_problem"
        ),
        "priority_proposed_correction_counts": _priority_field_counts(
            "proposed_correction"
        ),
        "priority_confidence_counts": _priority_field_counts("confidence"),
        "priority_bounded_delta_counts": _priority_field_counts(
            "bounded_delta"
        ),
        "priority_safety_blocker_counts": _priority_field_counts(
            "safety_blockers"
        ),
        "priority_safety_blocker_detail_counts": _priority_field_counts(
            "safety_blocker_details"
        ),
        "priority_headroom_feedback_true_peak_check_counts": (
            _priority_field_counts("headroom_feedback_true_peak_checks")
        ),
        "priority_headroom_feedback_true_peak_check_detail_counts": (
            _priority_field_counts("headroom_feedback_true_peak_check_details")
        ),
        "priority_operator_readable_explanation_counts": _priority_field_counts(
            "operator_readable_explanation"
        ),
        "priority_rollback_readback_contract_counts": _priority_field_counts(
            "rollback_readback_contract"
        ),
        "priority_future_wing_readback_contract_counts": _priority_field_counts(
            "future_wing_readback_contract"
        ),
        "priority_test_evidence_counts": _priority_field_counts(
            "test_evidence"
        ),
        "priority_test_evidence_reference_counts": _priority_field_counts(
            "test_evidence_references"
        ),
        "priority_bounded_delta_policy_counts": _priority_field_counts(
            "bounded_delta_policies"
        ),
        "priority_rollback_expectation_counts": _priority_field_counts(
            "rollback_expectations"
        ),
        "priority_operator_payload_required_field_counts": _priority_field_counts(
            "operator_payload_required_fields"
        ),
        "priority_operator_evidence_required_field_counts": _priority_field_counts(
            "operator_evidence_required_fields"
        ),
        "priority_operator_problem_review_summary_counts": _priority_field_counts(
            "operator_problem_review_summary"
        ),
        "priority_operator_method_review_focus_counts": _priority_field_counts(
            "operator_method_review_focus"
        ),
        "priority_why_safe_summary_counts": _priority_field_counts(
            "why_safe_summary"
        ),
        "priority_why_blocked_summary_counts": _priority_field_counts(
            "why_blocked_summary"
        ),
        "priority_operator_visible_explanation_contract_counts": (
            _priority_field_counts("operator_visible_explanation_contract")
        ),
        "priority_future_wing_readback_contract_summary_counts": (
            _priority_field_counts("future_wing_readback_contract_summary")
        ),
        "priority_future_live_requirement_detail_counts": (
            _priority_field_counts("future_live_requirement_details")
        ),
        "priority_live_authority_boundary_summary_counts": _priority_field_counts(
            "live_authority_boundary_summary"
        ),
        "priority_operator_review_record_required_field_counts": (
            _priority_field_counts("operator_review_record_required_fields")
        ),
        "priority_required_confirmation_detail_counts": (
            _priority_field_counts("required_confirmation_details")
        ),
        "priority_operator_review_decision_option_detail_counts": (
            _priority_field_counts("operator_review_decision_option_details")
        ),
        "priority_operator_review_checklist_detail_counts": (
            _priority_field_counts("operator_review_checklist_details")
        ),
        "priority_operator_review_record_field_detail_counts": (
            _priority_field_counts("operator_review_record_field_details")
        ),
        "priority_operator_next_action_detail_counts": (
            _priority_field_counts("operator_next_action_details")
        ),
        "operator_completion_requires": list(
            decision_template["operator_completion_requires"]
        ),
        "read_only_or_dry_run_only": decision_template[
            "read_only_or_dry_run_only"
        ],
        "read_only": decision_template["read_only"],
        "contacts_wing": decision_template["contacts_wing"],
        "contacts_backend": decision_template["contacts_backend"],
        "queries_console": decision_template["queries_console"],
        "mutates_mixer": decision_template["mutates_mixer"],
        "mutates_console_state": decision_template["mutates_console_state"],
        "live_write": decision_template["live_write"],
        "live_authority_granted": decision_template[
            "live_authority_granted"
        ],
        "readiness_closure_allowed": decision_template[
            "readiness_closure_allowed"
        ],
        "live_apply_allowed": decision_template["live_apply_allowed"],
        "queue_import_allowed": decision_template["queue_import_allowed"],
        "can_authorize_live": decision_template["can_authorize_live"],
        "hardware_proof_required_before_live": decision_template[
            "hardware_proof_required_before_live"
        ],
        "operator_review_required": decision_template[
            "operator_review_required"
        ],
        "future_wing_readback_required": decision_template[
            "future_wing_readback_required"
        ],
        "denial_reason": decision_template["denial_reason"],
    }


def export_correction_method_operator_review_packet(
    *,
    artifact_root: Path,
    run_id: Optional[str] = None,
    pipeline_id: str = MAIN_PIPELINE_ID,
    generated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Write packet JSON, Markdown and manifest for offline operator review.

    The export is an artifact-only view of proposal contracts. It does not
    contact WING/backend, query a console, queue proposals or apply mixer
    changes.
    """

    resolved_run_id = run_id or _default_run_id()
    generated_at = generated_at_utc or _utc_now()
    run_dir = Path(artifact_root) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    packet = build_correction_method_goal_operator_review_packet(
        pipeline_id=pipeline_id
    )
    markdown = render_correction_method_goal_operator_review_markdown(
        pipeline_id=pipeline_id
    )
    decision_template = build_operator_decision_template(
        packet=packet,
        run_id=resolved_run_id,
        pipeline_id=pipeline_id,
        generated_at_utc=generated_at,
    )

    packet_bytes = (
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    markdown_bytes = markdown.encode("utf-8")
    decision_template_bytes = (
        json.dumps(
            decision_template,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    packet_artifact = _write_bytes(run_dir / "packet.json", packet_bytes)
    markdown_artifact = _write_bytes(
        run_dir / "operator-review.md", markdown_bytes
    )
    decision_template_artifact = _write_bytes(
        run_dir / "operator-decision-template.json",
        decision_template_bytes,
    )

    manifest = {
        "schema": "automixer.correction_method_operator_review_export_manifest.v1",
        "run_id": resolved_run_id,
        "generated_at_utc": generated_at,
        "pipeline_id": pipeline_id,
        "artifact_root": str(run_dir),
        "status": (
            "operator_review_export_ready_proposal_only"
            if packet["covered"]
            else "operator_review_export_incomplete"
        ),
        "packet_schema": packet["schema"],
        "packet_status": packet["status"],
        "operator_decision_template_schema": decision_template["schema"],
        "operator_decision_template_status": decision_template["status"],
        "operator_decision_template_summary": (
            build_operator_decision_template_summary(decision_template)
        ),
        "read_only_or_dry_run_only": True,
        "read_only": True,
        "contacts_wing": False,
        "contacts_backend": False,
        "queries_console": False,
        "mutates_mixer": False,
        "mutates_console_state": False,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "hardware_proof_required_before_live": True,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "denial_reason": packet["denial_reason"],
        "operator_next_step": packet["operator_next_step"],
        "summary": dict(packet["summary"]),
        "artifacts": [
            {"artifact_id": "packet_json", **packet_artifact},
            {"artifact_id": "operator_review_markdown", **markdown_artifact},
            {
                "artifact_id": "operator_decision_template",
                **decision_template_artifact,
            },
        ],
    }

    manifest_bytes = (
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes(run_dir / "artifact-manifest.json", manifest_bytes)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export correction-method operator review packet artifacts without "
            "WING/backend contact."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory where the run folder will be created.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run directory name. Defaults to a UTC timestamped run id.",
    )
    parser.add_argument(
        "--pipeline-id",
        default=MAIN_PIPELINE_ID,
        help="Processing pipeline id to export.",
    )
    args = parser.parse_args(argv)

    manifest = export_correction_method_operator_review_packet(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        pipeline_id=args.pipeline_id,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
