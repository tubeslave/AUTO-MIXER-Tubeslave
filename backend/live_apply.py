"""Tier A live-apply safety gate for approved fader/gain operations.

Tier A is intentionally narrow in this wave:

- approved operations: ``set_fader`` and ``set_gain`` only;
- approved callers: manual UI and voice command paths that route through this gate;
- default behavior: dry-run unless a caller explicitly requests a real write;
- real write behavior: explicit confirmation plus confirmed readback;
- panic stop scope: ``manual_live_apply_only`` semantics only, not a system-wide stop;
- fail-closed behavior for unsupported operations, missing mixer access, send failures,
  and unconfirmed readback states.

This module is the reference contract for operator-visible dry-run, approval,
send-status, readback-status, and audit semantics in Tier A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable
import uuid


logger = logging.getLogger(__name__)


SUPPORTED_OPERATIONS = {"set_fader", "set_gain"}


def _resolve_replay_correlation_id(
    explicit: str | None,
    metadata: dict[str, Any] | None,
) -> str:
    if explicit:
        return str(explicit)
    payload = dict(metadata or {})
    value = payload.get("replay_correlation_id") or payload.get("trace_id")
    return str(value) if value else ""


@dataclass(slots=True)
class LiveApplyRequest:
    """A minimal mutation request entering the live-apply layer."""

    source: str
    operation: str
    channel: int
    value: float
    dry_run: bool | None = None
    confirm_live_apply: bool = False
    current_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    max_step: float | None = None
    replay_correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def target_method_name(self) -> str:
        return {
            "set_fader": "set_channel_fader",
            "set_gain": "set_channel_gain",
        }[self.operation]


@dataclass(slots=True)
class LiveApplyPolicy:
    """Policy defaults for the minimal first wave."""

    default_dry_run: bool = True
    require_confirm_live_apply: bool = True
    readback_timeout_ms: int = 300
    panic_stop_active: bool = False
    panic_stop_scope: str = "manual_live_apply_only"
    panic_stop_reason: str | None = None


@dataclass(slots=True)
class LiveApplyResult:
    """Outcome of a live-apply request."""

    operation: str
    channel: int
    requested_value: float
    source: str
    accepted: bool
    dry_run: bool
    send_attempted: bool
    send_result: bool | None
    audit_id: str = ""
    replay_correlation_id: str = ""
    send_status: str = "not_sent"
    readback_status: str = "not_requested"
    desired_value: float | None = None
    confirmed_value: float | None = None
    verification_id: str | None = None
    failure_reason: str | None = None
    timeout_ms: int | None = None
    blocked_reason: str | None = None
    panic_stop_active: bool = False
    panic_stop_scope: str | None = None
    panic_stop_reason: str | None = None
    guard_reasons: tuple[str, ...] = ()
    audit_event: dict[str, Any] = field(default_factory=dict)


class LiveApplyService:
    """Gate manual mixer operations behind dry-run and confirm checks."""

    def __init__(
        self,
        policy: LiveApplyPolicy | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy or LiveApplyPolicy()
        self.audit_sink = audit_sink or self._log_audit_event

    def apply(self, request: LiveApplyRequest, mixer_client: Any | None) -> LiveApplyResult:
        desired_value = float(request.value)
        dry_run = self.policy.default_dry_run if request.dry_run is None else bool(request.dry_run)
        replay_correlation_id = _resolve_replay_correlation_id(
            request.replay_correlation_id,
            request.metadata,
        )
        audit_id = str(uuid.uuid4())
        desired_value, guard_reasons = self._apply_request_guards(
            request,
            desired_value=desired_value,
        )

        if request.operation not in SUPPORTED_OPERATIONS:
            return self._blocked(
                request,
                dry_run=dry_run,
                reason="unsupported_operation",
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        if mixer_client is None:
            return self._blocked(
                request,
                dry_run=dry_run,
                reason="mixer_client_unavailable",
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        if not dry_run and self.policy.require_confirm_live_apply and not request.confirm_live_apply:
            return self._blocked(
                request,
                dry_run=dry_run,
                reason="confirm_live_apply_required",
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        if dry_run:
            return self._dry_run_result(
                request,
                dry_run=True,
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        if self.policy.panic_stop_active:
            return self._blocked(
                request,
                dry_run=False,
                reason="panic_stop_active",
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        sender = getattr(mixer_client, request.target_method_name(), None)
        if sender is None:
            return self._blocked(
                request,
                dry_run=False,
                reason="operation_not_supported_by_client",
                desired_value=desired_value,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        self._set_transport_context(
            mixer_client,
            audit_id=audit_id,
            replay_correlation_id=replay_correlation_id,
            request=request,
        )
        try:
            send_result = bool(sender(request.channel, desired_value))
            shadow_event = self._consume_shadow_transport_event(mixer_client)
        finally:
            self._clear_transport_context(mixer_client)
        send_status = self._resolve_send_status(mixer_client, send_result)
        if send_status == "shadow_blocked":
            return self._blocked(
                request,
                dry_run=False,
                reason="shadow_mode_active",
                send_attempted=True,
                send_result=False,
                desired_value=desired_value,
                send_status=send_status,
                failure_reason="shadow_mode_active",
                guard_reasons=guard_reasons,
                audit_id=audit_id,
                shadow_transport_event=shadow_event,
            )
        if not send_result:
            return self._blocked(
                request,
                dry_run=False,
                reason="send_failed",
                send_attempted=True,
                send_result=False,
                desired_value=desired_value,
                send_status=send_status,
                failure_reason=send_status,
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        verification = self._verify_manual_write(
            mixer_client,
            request,
            desired_value=desired_value,
        )
        if verification["readback_status"] != "confirmed":
            blocked_reason = self._resolve_readback_block_reason(verification)
            failure_reason = verification["failure_reason"] or blocked_reason
            return self._blocked(
                request,
                dry_run=False,
                reason=blocked_reason,
                send_attempted=True,
                send_result=True,
                desired_value=desired_value,
                send_status=send_status,
                readback_status=verification["readback_status"],
                confirmed_value=verification["confirmed_value"],
                verification_id=verification["verification_id"],
                failure_reason=failure_reason,
                timeout_ms=verification["timeout_ms"],
                guard_reasons=guard_reasons,
                audit_id=audit_id,
            )

        audit_event = self._build_audit_event(
            request,
            dry_run=False,
            accepted=True,
            send_attempted=True,
            send_result=True,
            blocked_reason=None,
            send_status=send_status,
            readback_status=verification["readback_status"],
            desired_value=desired_value,
            confirmed_value=verification["confirmed_value"],
            verification_id=verification["verification_id"],
            failure_reason=verification["failure_reason"],
            timeout_ms=verification["timeout_ms"],
            panic_stop_active=self.policy.panic_stop_active,
            panic_stop_scope=self.policy.panic_stop_scope,
            panic_stop_reason=self.policy.panic_stop_reason,
            guard_reasons=guard_reasons,
            audit_id=audit_id,
        )
        self.audit_sink(audit_event)
        return LiveApplyResult(
            operation=request.operation,
            channel=request.channel,
            requested_value=float(request.value),
            source=request.source,
            accepted=True,
            dry_run=False,
            send_attempted=True,
            send_result=True,
            audit_id=str(audit_event["audit_id"]),
            replay_correlation_id=replay_correlation_id,
            send_status=send_status,
            readback_status=verification["readback_status"],
            desired_value=desired_value,
            confirmed_value=verification["confirmed_value"],
            verification_id=verification["verification_id"],
            failure_reason=verification["failure_reason"],
            timeout_ms=verification["timeout_ms"],
            guard_reasons=tuple(guard_reasons),
            audit_event=audit_event,
        )

    def _dry_run_result(
        self,
        request: LiveApplyRequest,
        *,
        dry_run: bool,
        desired_value: float,
        guard_reasons: list[str],
        audit_id: str,
    ) -> LiveApplyResult:
        audit_event = self._build_audit_event(
            request,
            dry_run=dry_run,
            accepted=True,
            send_attempted=False,
            send_result=None,
            blocked_reason=None,
            send_status="not_sent",
            readback_status="not_requested",
            desired_value=desired_value,
            confirmed_value=None,
            verification_id=None,
            failure_reason=None,
            timeout_ms=None,
            panic_stop_active=self.policy.panic_stop_active,
            panic_stop_scope=self.policy.panic_stop_scope,
            panic_stop_reason=self.policy.panic_stop_reason,
            guard_reasons=guard_reasons,
            audit_id=audit_id,
        )
        self.audit_sink(audit_event)
        return LiveApplyResult(
            operation=request.operation,
            channel=request.channel,
            requested_value=float(request.value),
            source=request.source,
            accepted=True,
            dry_run=dry_run,
            send_attempted=False,
            send_result=None,
            audit_id=str(audit_event["audit_id"]),
            replay_correlation_id=_resolve_replay_correlation_id(
                request.replay_correlation_id,
                request.metadata,
            ),
            send_status="not_sent",
            readback_status="not_requested",
            desired_value=desired_value,
            confirmed_value=None,
            verification_id=None,
            failure_reason=None,
            timeout_ms=None,
            panic_stop_active=self.policy.panic_stop_active,
            panic_stop_scope=self.policy.panic_stop_scope,
            panic_stop_reason=self.policy.panic_stop_reason,
            guard_reasons=tuple(guard_reasons),
            audit_event=audit_event,
        )

    def _blocked(
        self,
        request: LiveApplyRequest,
        *,
        dry_run: bool,
        reason: str,
        send_attempted: bool = False,
        send_result: bool | None = None,
        desired_value: float,
        send_status: str = "not_sent",
        readback_status: str = "not_requested",
        confirmed_value: float | None = None,
        verification_id: str | None = None,
        failure_reason: str | None = None,
        timeout_ms: int | None = None,
        guard_reasons: list[str] | None = None,
        audit_id: str,
        shadow_transport_event: dict[str, Any] | None = None,
    ) -> LiveApplyResult:
        resolved_guard_reasons = list(guard_reasons or [])
        audit_event = self._build_audit_event(
            request,
            dry_run=dry_run,
            accepted=False,
            send_attempted=send_attempted,
            send_result=send_result,
            blocked_reason=reason,
            send_status=send_status,
            readback_status=readback_status,
            desired_value=desired_value,
            confirmed_value=confirmed_value,
            verification_id=verification_id,
            failure_reason=failure_reason or reason,
            timeout_ms=timeout_ms,
            panic_stop_active=self.policy.panic_stop_active,
            panic_stop_scope=self.policy.panic_stop_scope,
            panic_stop_reason=self.policy.panic_stop_reason,
            guard_reasons=resolved_guard_reasons,
            audit_id=audit_id,
            shadow_transport_event=shadow_transport_event,
        )
        self.audit_sink(audit_event)
        return LiveApplyResult(
            operation=request.operation,
            channel=request.channel,
            requested_value=float(request.value),
            source=request.source,
            accepted=False,
            dry_run=dry_run,
            send_attempted=send_attempted,
            send_result=send_result,
            audit_id=str(audit_event["audit_id"]),
            replay_correlation_id=_resolve_replay_correlation_id(
                request.replay_correlation_id,
                request.metadata,
            ),
            send_status=send_status,
            readback_status=readback_status,
            desired_value=desired_value,
            confirmed_value=confirmed_value,
            verification_id=verification_id,
            failure_reason=failure_reason or reason,
            timeout_ms=timeout_ms,
            blocked_reason=reason,
            panic_stop_active=self.policy.panic_stop_active,
            panic_stop_scope=self.policy.panic_stop_scope,
            panic_stop_reason=self.policy.panic_stop_reason,
            guard_reasons=tuple(resolved_guard_reasons),
            audit_event=audit_event,
        )

    def _apply_request_guards(
        self,
        request: LiveApplyRequest,
        *,
        desired_value: float,
    ) -> tuple[float, list[str]]:
        lower_bound, upper_bound = self._operation_bounds(request.operation)
        if request.min_value is not None:
            lower_bound = max(lower_bound, float(request.min_value))
        if request.max_value is not None:
            upper_bound = min(upper_bound, float(request.max_value))
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        guard_reasons: list[str] = []
        bounded_value = float(desired_value)
        if bounded_value < lower_bound:
            bounded_value = lower_bound
            guard_reasons.append("min_bound_clamped")
        elif bounded_value > upper_bound:
            bounded_value = upper_bound
            guard_reasons.append("max_bound_clamped")

        if request.max_step is not None and request.current_value is not None:
            step = abs(float(request.max_step))
            current = float(request.current_value)
            if step > 0.0:
                lower_step = current - step
                upper_step = current + step
                stepped_value = max(lower_step, min(upper_step, bounded_value))
                if stepped_value != bounded_value:
                    guard_reasons.append("max_step_clamped")
                bounded_value = stepped_value

        return float(bounded_value), guard_reasons

    @staticmethod
    def _operation_bounds(operation: str) -> tuple[float, float]:
        return {
            "set_fader": (-144.0, 10.0),
            "set_gain": (-18.0, 18.0),
        }.get(operation, (-1e9, 1e9))

    def _resolve_send_status(self, mixer_client: Any, send_result: bool) -> str:
        if send_result:
            return "sent"
        status_getter = getattr(mixer_client, "get_last_send_status", None)
        if callable(status_getter):
            status = status_getter()
            if status in {"failed", "throttled", "disconnected", "shadow_blocked"}:
                return status
        return "failed"

    @staticmethod
    def _set_transport_context(
        mixer_client: Any,
        *,
        audit_id: str,
        replay_correlation_id: str,
        request: LiveApplyRequest,
    ) -> None:
        setter = getattr(mixer_client, "set_transport_context", None)
        if callable(setter):
            setter(
                audit_id=audit_id,
                replay_correlation_id=replay_correlation_id,
                source=request.source,
                operation=request.operation,
                channel=request.channel,
                metadata=dict(request.metadata),
            )

    @staticmethod
    def _clear_transport_context(mixer_client: Any) -> None:
        clearer = getattr(mixer_client, "clear_transport_context", None)
        if callable(clearer):
            clearer()

    @staticmethod
    def _consume_shadow_transport_event(mixer_client: Any) -> dict[str, Any] | None:
        consumer = getattr(mixer_client, "consume_shadow_transport_event", None)
        if callable(consumer):
            return consumer()
        return None

    def _verify_manual_write(
        self,
        mixer_client: Any,
        request: LiveApplyRequest,
        *,
        desired_value: float,
    ) -> dict[str, Any]:
        verifier = getattr(mixer_client, "confirm_manual_write", None)
        if not callable(verifier):
            return {
                "readback_status": "unsupported",
                "confirmed_value": None,
                "verification_id": None,
                "failure_reason": "readback_unsupported",
                "timeout_ms": self.policy.readback_timeout_ms,
            }
        verification = verifier(
            request.operation,
            request.channel,
            desired_value,
            timeout_ms=self.policy.readback_timeout_ms,
        )
        return {
            "readback_status": verification.get("readback_status", "unsupported"),
            "confirmed_value": verification.get("confirmed_value"),
            "verification_id": verification.get("verification_id"),
            "failure_reason": verification.get("failure_reason"),
            "timeout_ms": verification.get("timeout_ms", self.policy.readback_timeout_ms),
        }

    @staticmethod
    def _resolve_readback_block_reason(verification: dict[str, Any]) -> str:
        readback_status = verification.get("readback_status")
        failure_reason = verification.get("failure_reason")

        if readback_status == "pending":
            return "readback_pending"
        if readback_status == "timeout":
            if failure_reason == "failed":
                return "readback_query_failed"
            if failure_reason == "throttled":
                return "readback_query_throttled"
            if failure_reason == "disconnected":
                return "readback_query_disconnected"
            return "readback_timeout"
        if readback_status == "mismatch":
            return "readback_mismatch"
        if readback_status == "unsupported":
            return "readback_unsupported"
        if readback_status == "disconnected":
            return "console_disconnected"
        return "readback_unconfirmed"

    @staticmethod
    def _log_audit_event(event: dict[str, Any]) -> None:
        logger.info("live_apply_audit %s", event)

    @staticmethod
    def _build_audit_event(
        request: LiveApplyRequest,
        *,
        dry_run: bool,
        accepted: bool,
        send_attempted: bool,
        send_result: bool | None,
        blocked_reason: str | None,
        send_status: str,
        readback_status: str,
        desired_value: float | None,
        confirmed_value: float | None,
        verification_id: str | None,
        failure_reason: str | None,
        timeout_ms: int | None,
        panic_stop_active: bool,
        panic_stop_scope: str | None,
        panic_stop_reason: str | None,
        guard_reasons: list[str],
        audit_id: str | None = None,
        shadow_transport_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "audit_id": audit_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "source": request.source,
            "operation": request.operation,
            "channel": int(request.channel),
            "requested_value": float(request.value),
            "replay_correlation_id": _resolve_replay_correlation_id(
                request.replay_correlation_id,
                request.metadata,
            ),
            "dry_run": bool(dry_run),
            "confirm_live_apply": bool(request.confirm_live_apply),
            "accepted": bool(accepted),
            "send_attempted": bool(send_attempted),
            "send_result": send_result,
            "send_status": send_status,
            "readback_status": readback_status,
            "desired_value": desired_value,
            "confirmed_value": confirmed_value,
            "verification_id": verification_id,
            "failure_reason": failure_reason,
            "timeout_ms": timeout_ms,
            "blocked_reason": blocked_reason,
            "shadow_mode_active": send_status == "shadow_blocked" or blocked_reason == "shadow_mode_active",
            "shadow_transport_event": dict(shadow_transport_event or {}),
            "panic_stop_active": panic_stop_active,
            "panic_stop_scope": panic_stop_scope,
            "panic_stop_reason": panic_stop_reason,
            "guard_reasons": list(guard_reasons),
            "metadata": dict(request.metadata),
        }


def live_apply_target_type(operation: str) -> str:
    return {
        "set_fader": "channel_fader",
        "set_gain": "channel_gain",
        "set_eq": "channel_eq",
        "set_compressor": "channel_compressor",
    }[operation]


def live_apply_message_for_user(result: LiveApplyResult) -> str:
    if result.blocked_reason == "confirm_live_apply_required":
        return "Live apply blocked: confirmation required."
    if result.blocked_reason == "mixer_client_unavailable":
        return "Mixer unavailable: command was not sent."
    if result.blocked_reason == "panic_stop_active":
        return "Live apply blocked: panic stop is active for manual live apply."
    if result.blocked_reason == "shadow_mode_active" or result.send_status == "shadow_blocked":
        return "Shadow mode blocked console write: no OSC command was sent."
    if result.blocked_reason == "unsupported_operation":
        return "Live apply blocked: operation is not gated for console writes."
    if result.send_status == "throttled":
        return "Live apply throttled: command was not sent."
    if result.send_status == "disconnected":
        return "Mixer disconnected: command was not sent."
    if result.blocked_reason == "send_failed":
        return "Live apply failed: mixer command was not sent."
    if result.dry_run:
        return "Dry-run only: no OSC command was sent."
    if result.readback_status == "confirmed":
        return "Live apply confirmed by mixer readback."
    if result.blocked_reason == "readback_query_failed":
        return "Live apply sent, but readback query failed."
    if result.blocked_reason == "readback_query_throttled":
        return "Live apply sent, but readback query was throttled."
    if result.blocked_reason == "readback_query_disconnected":
        return "Live apply sent, but mixer disconnected before readback query completed."
    if result.readback_status == "pending":
        return "Live apply sent, but not yet confirmed by mixer readback."
    if result.readback_status == "timeout":
        return "Live apply sent, but readback timed out before confirmation."
    if result.readback_status == "mismatch":
        return "Live apply sent, but mixer readback did not match the requested value."
    if result.readback_status == "disconnected":
        return "Live apply sent, but mixer disconnected before readback."
    return "Live apply sent, but readback is unsupported, so the command is not considered applied."


def build_live_apply_operator_payload(
    result: LiveApplyResult,
    *,
    event_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": event_type,
        "operation": result.operation,
        "channel": result.channel,
        "requested_value": result.requested_value,
        "accepted": result.accepted,
        "blocked": not result.accepted,
        "blocked_reason": result.blocked_reason,
        "dry_run_only": result.dry_run,
        "send_result": result.send_result,
        "send_status": result.send_status,
        "readback_status": result.readback_status,
        "desired_value": result.desired_value,
        "confirmed_value": result.confirmed_value,
        "verification_id": result.verification_id,
        "failure_reason": result.failure_reason,
        "timeout_ms": result.timeout_ms,
        "audit_id": result.audit_id,
        "replay_correlation_id": result.replay_correlation_id,
        "shadow_mode_active": result.send_status == "shadow_blocked" or result.blocked_reason == "shadow_mode_active",
        "guard_reasons": list(result.guard_reasons),
        "message_for_user": live_apply_message_for_user(result),
        "live_apply_source": result.source,
        "target_type": live_apply_target_type(result.operation),
        "panic_stop_active": result.panic_stop_active,
        "panic_stop_scope": result.panic_stop_scope,
        "panic_stop_reason": result.panic_stop_reason,
    }
    if extra:
        payload.update(extra)
    return payload
