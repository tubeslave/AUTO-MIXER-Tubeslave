"""Shared read-only WING preflight contract constants.

The values here are report-only safety boundaries. They do not open a mixer
transport or grant live-write authority.
"""

from __future__ import annotations


WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = (
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = (
    "listener_processes",
    "preflight_blocked_until_listener_confirmed",
    "intended_backend_confirmation_required",
    "safe_to_stop_listener_by_gate",
    "fail_closed_reason",
    "operator_next_step",
)
