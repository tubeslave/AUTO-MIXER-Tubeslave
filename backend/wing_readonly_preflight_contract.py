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

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = (
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = (
    "listener_processes",
    "listener_confirmation_checklist",
    "preflight_blocked_until_listener_confirmed",
    "intended_backend_confirmation_required",
    "safe_to_stop_listener_by_gate",
    "fail_closed_reason",
    "operator_next_step",
)
