"""Fail-closed read-only boundary and redacted security audit."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


READ_ONLY_ENV_VAR = "INSPECTOR_READ_ONLY"
INSPECTOR_READ_ONLY = True
_TRUE_VALUES = {"1", "true", "yes", "on"}


class InspectorReadOnlyConfigurationError(RuntimeError):
    """Raised when v1 is asked to start with read-only disabled."""


class InspectorWriteProhibited(PermissionError):
    """Raised before any attempted Inspector mutation."""


def assert_read_only_mode(environ: Optional[dict[str, str]] = None) -> None:
    """Require the v1 invariant, including an explicitly false env override."""

    env = os.environ if environ is None else environ
    raw = env.get(READ_ONLY_ENV_VAR)
    if raw is not None and raw.strip().lower() not in _TRUE_VALUES:
        raise InspectorReadOnlyConfigurationError(
            "inspector_v1_read_only_cannot_be_disabled"
        )
    if INSPECTOR_READ_ONLY is not True:
        raise InspectorReadOnlyConfigurationError(
            "inspector_read_only_compile_time_invariant_failed"
        )


@dataclass(frozen=True)
class SecurityAuditEvent:
    event: str
    action: str
    blocked: bool
    timestamp: str
    detail: str


def prohibit_write(
    action: str,
    *,
    audit_path: Optional[Path] = None,
    now: Optional[Callable[[], datetime]] = None,
) -> None:
    """Record a redacted attempt and raise before the requested action runs."""

    assert_read_only_mode()
    clean_action = _redact_action(action)
    clock = now or (lambda: datetime.now(timezone.utc))
    event = SecurityAuditEvent(
        event="inspector_write_blocked",
        action=clean_action,
        blocked=True,
        timestamp=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        detail="No target, value, address, credential, or payload was logged.",
    )
    if audit_path is not None:
        _append_audit_event(audit_path, event)
    raise InspectorWriteProhibited(f"inspector_write_prohibited:{clean_action}")


def _redact_action(action: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in str(action).strip()
    )
    return (normalized or "unknown")[:96]


def _append_audit_event(path: Path, event: SecurityAuditEvent) -> None:
    resolved = Path(path)
    if resolved.exists() and resolved.is_symlink():
        raise InspectorWriteProhibited("inspector_security_log_symlink_rejected")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action": event.action,
        "blocked": event.blocked,
        "detail": event.detail,
        "event": event.event,
        "timestamp": event.timestamp,
    }
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

