"""Shared read-only WING preflight contract constants.

The values here are report-only safety boundaries. They do not open a mixer
transport or grant live-write authority.
"""

from __future__ import annotations


WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA = "wing_readonly_preflight.v1"
WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME = "summary.json"
WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME = "events.jsonl"
WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT = (
    f"{WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA} "
    f"{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
)
WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED = (
    f"real {WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT}"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER = (
    "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
)
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER = (
    "artifacts/no_wing_release_check/<run_id>/summary.json"
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = (
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = (
    "listener_processes",
    "listener_process_error",
    "listener_confirmation_checklist",
    "preflight_blocked_until_listener_confirmed",
    "intended_backend_confirmation_required",
    "safe_to_stop_listener_by_gate",
    "listener_confirmation_template_status",
    "listener_confirmation_template_safe_to_stop_listener_by_gate",
    "listener_confirmation_template_live_authority_granted",
    "listener_confirmation_template_accepted_by_template",
    "listener_confirmation_template_operator_placeholder_fields",
    "listener_confirmation_template_execution_allowed_by_template",
    "listener_confirmation_template_readiness_closure_allowed_by_template",
    "listener_confirmation_template_required_edits",
    "fail_closed_reason",
    "operator_next_step",
)
