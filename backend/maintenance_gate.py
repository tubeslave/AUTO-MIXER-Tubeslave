"""Maintenance-only gate for snapshot, restore, and reset operations."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable
import uuid


logger = logging.getLogger(__name__)


MIXER_MUTATION_OPERATIONS = {
    "bypass_mixer",
    "load_snapshot",
    "restore_snapshot",
    "reset_trim",
    "reset_all_eq",
    "reset_phase_delay",
    "reset_all_functions",
}


ALLOWED_OPERATOR_SOURCES = {
    "manual_ui",
    "maintenance_ui",
}


@dataclass(slots=True)
class MaintenanceActionRequest:
    source: str
    operation: str
    dry_run: bool | None = None
    confirm_maintenance: bool = False
    rollback_snapshot_path: str | None = None
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaintenanceActionPolicy:
    require_confirmation: bool = True
    allowed_sources: frozenset[str] = field(
        default_factory=lambda: frozenset(ALLOWED_OPERATOR_SOURCES)
    )


@dataclass(slots=True)
class MaintenanceActionResult:
    operation: str
    source: str
    accepted: bool
    dry_run: bool
    send_attempted: bool
    send_result: bool | None
    audit_id: str = ""
    send_status: str = "not_sent"
    blocked_reason: str | None = None
    failure_reason: str | None = None
    confirm_maintenance: bool = False
    rollback_plan_required: bool = False
    rollback_plan_present: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    audit_event: dict[str, Any] = field(default_factory=dict)


class MaintenanceGate:
    """Authorize maintenance mutations before the code reaches console writes."""

    def __init__(
        self,
        policy: MaintenanceActionPolicy | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy or MaintenanceActionPolicy()
        self.audit_sink = audit_sink or self._log_audit_event

    def evaluate(
        self,
        request: MaintenanceActionRequest,
        *,
        mixer_client: Any | None,
    ) -> MaintenanceActionResult:
        dry_run = bool(request.dry_run is True)
        rollback_required = request.operation in MIXER_MUTATION_OPERATIONS
        rollback_present = bool(request.rollback_snapshot_path)

        if request.source not in self.policy.allowed_sources:
            return self._blocked(
                request,
                dry_run=dry_run,
                reason="maintenance_source_not_allowed",
                rollback_required=rollback_required,
                rollback_present=rollback_present,
            )

        if dry_run:
            return self._dry_run_result(
                request,
                rollback_required=rollback_required,
                rollback_present=rollback_present,
            )

        if self.policy.require_confirmation and not request.confirm_maintenance:
            return self._blocked(
                request,
                dry_run=False,
                reason="maintenance_confirmation_required",
                rollback_required=rollback_required,
                rollback_present=rollback_present,
            )

        if rollback_required and not rollback_present:
            return self._blocked(
                request,
                dry_run=False,
                reason="rollback_plan_required",
                rollback_required=rollback_required,
                rollback_present=rollback_present,
            )

        if mixer_client is None or not getattr(mixer_client, "is_connected", False):
            return self._blocked(
                request,
                dry_run=False,
                reason="transport_disconnected",
                send_status="disconnected",
                failure_reason="transport_disconnected",
                rollback_required=rollback_required,
                rollback_present=rollback_present,
            )

        audit_event = self._build_audit_event(
            request,
            accepted=True,
            dry_run=False,
            send_attempted=False,
            send_result=None,
            send_status="approved_pending_execution",
            blocked_reason=None,
            failure_reason=None,
            rollback_required=rollback_required,
            rollback_present=rollback_present,
        )
        self.audit_sink(audit_event)
        return MaintenanceActionResult(
            operation=request.operation,
            source=request.source,
            accepted=True,
            dry_run=False,
            send_attempted=False,
            send_result=None,
            audit_id=str(audit_event["audit_id"]),
            send_status="approved_pending_execution",
            blocked_reason=None,
            failure_reason=None,
            confirm_maintenance=bool(request.confirm_maintenance),
            rollback_plan_required=rollback_required,
            rollback_plan_present=rollback_present,
            target=dict(request.target),
            metadata=dict(request.metadata),
            audit_event=audit_event,
        )

    @staticmethod
    def mark_execution(
        result: MaintenanceActionResult,
        *,
        send_result: bool,
        failure_reason: str | None = None,
    ) -> MaintenanceActionResult:
        result.send_attempted = True
        result.send_result = bool(send_result)
        result.send_status = "sent" if send_result else "failed"
        result.failure_reason = failure_reason if not send_result else None
        if not send_result and result.blocked_reason is None:
            result.blocked_reason = "maintenance_execution_failed"
            result.accepted = False
        return result

    def _dry_run_result(
        self,
        request: MaintenanceActionRequest,
        *,
        rollback_required: bool,
        rollback_present: bool,
    ) -> MaintenanceActionResult:
        audit_event = self._build_audit_event(
            request,
            accepted=True,
            dry_run=True,
            send_attempted=False,
            send_result=None,
            send_status="not_sent",
            blocked_reason=None,
            failure_reason=None,
            rollback_required=rollback_required,
            rollback_present=rollback_present,
        )
        self.audit_sink(audit_event)
        return MaintenanceActionResult(
            operation=request.operation,
            source=request.source,
            accepted=True,
            dry_run=True,
            send_attempted=False,
            send_result=None,
            audit_id=str(audit_event["audit_id"]),
            send_status="not_sent",
            blocked_reason=None,
            failure_reason=None,
            confirm_maintenance=bool(request.confirm_maintenance),
            rollback_plan_required=rollback_required,
            rollback_plan_present=rollback_present,
            target=dict(request.target),
            metadata=dict(request.metadata),
            audit_event=audit_event,
        )

    def _blocked(
        self,
        request: MaintenanceActionRequest,
        *,
        dry_run: bool,
        reason: str,
        send_status: str = "not_sent",
        failure_reason: str | None = None,
        rollback_required: bool,
        rollback_present: bool,
    ) -> MaintenanceActionResult:
        audit_event = self._build_audit_event(
            request,
            accepted=False,
            dry_run=dry_run,
            send_attempted=False,
            send_result=None,
            send_status=send_status,
            blocked_reason=reason,
            failure_reason=failure_reason or reason,
            rollback_required=rollback_required,
            rollback_present=rollback_present,
        )
        self.audit_sink(audit_event)
        return MaintenanceActionResult(
            operation=request.operation,
            source=request.source,
            accepted=False,
            dry_run=dry_run,
            send_attempted=False,
            send_result=None,
            audit_id=str(audit_event["audit_id"]),
            send_status=send_status,
            blocked_reason=reason,
            failure_reason=failure_reason or reason,
            confirm_maintenance=bool(request.confirm_maintenance),
            rollback_plan_required=rollback_required,
            rollback_plan_present=rollback_present,
            target=dict(request.target),
            metadata=dict(request.metadata),
            audit_event=audit_event,
        )

    @staticmethod
    def _log_audit_event(event: dict[str, Any]) -> None:
        logger.info("maintenance_gate_audit %s", event)

    def _build_audit_event(
        self,
        request: MaintenanceActionRequest,
        *,
        accepted: bool,
        dry_run: bool,
        send_attempted: bool,
        send_result: bool | None,
        send_status: str,
        blocked_reason: str | None,
        failure_reason: str | None,
        rollback_required: bool,
        rollback_present: bool,
    ) -> dict[str, Any]:
        return {
            "audit_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "source": request.source,
            "operation": request.operation,
            "accepted": accepted,
            "dry_run": dry_run,
            "send_attempted": send_attempted,
            "send_result": send_result,
            "send_status": send_status,
            "blocked_reason": blocked_reason,
            "failure_reason": failure_reason,
            "confirm_maintenance": bool(request.confirm_maintenance),
            "rollback_plan_required": rollback_required,
            "rollback_plan_present": rollback_present,
            "rollback_snapshot_path": request.rollback_snapshot_path,
            "target": dict(request.target),
            "metadata": dict(request.metadata),
        }


def maintenance_message_for_user(result: MaintenanceActionResult) -> str:
    if result.blocked_reason == "maintenance_source_not_allowed":
        return "Maintenance action blocked: operator-owned entrypoint required."
    if result.blocked_reason == "maintenance_confirmation_required":
        return "Maintenance action blocked: operator confirmation required."
    if result.blocked_reason == "rollback_plan_required":
        return "Maintenance action blocked: rollback snapshot path required."
    if result.blocked_reason == "transport_disconnected":
        return "Maintenance action blocked: mixer transport is disconnected."
    if result.blocked_reason == "maintenance_execution_failed":
        return "Maintenance action failed during execution."
    if result.dry_run:
        return "Dry-run only: no maintenance mutation was executed."
    if result.accepted and not result.send_attempted:
        return "Maintenance action approved and ready for execution."
    if result.send_result:
        return "Maintenance action executed."
    return "Maintenance action blocked."


def build_maintenance_operator_payload(
    result: MaintenanceActionResult,
    *,
    event_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": event_type,
        "operation": result.operation,
        "accepted": result.accepted,
        "blocked": not result.accepted,
        "blocked_reason": result.blocked_reason,
        "dry_run_only": result.dry_run,
        "send_attempted": result.send_attempted,
        "send_result": result.send_result,
        "send_status": result.send_status,
        "failure_reason": result.failure_reason,
        "audit_id": result.audit_id,
        "confirm_maintenance": result.confirm_maintenance,
        "rollback_plan_required": result.rollback_plan_required,
        "rollback_plan_present": result.rollback_plan_present,
        "rollback_snapshot_path": result.audit_event.get("rollback_snapshot_path"),
        "maintenance_source": result.source,
        "target": dict(result.target),
        "message_for_user": maintenance_message_for_user(result),
    }
    if extra:
        payload.update(extra)
    return payload
