"""Backend-native proposal queue for operator-directed automixing.

This queue is the product contract between analysis modules and the UI. It
does not apply mixer writes by itself; handlers route approved applies through
the server's supervised manual write gate.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from processing_registry import LIVE_ALLOWED_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_registry import LIVE_ALLOWED_WRITE_KINDS
from supervised_console_parameter_registry import get_supervised_console_family


FINAL_STATUSES = {"accepted", "dismissed", "applied", "apply_blocked", "apply_error"}
APPLYABLE_VALUE_TYPES = set(LIVE_ALLOWED_WRITE_KINDS)
DEFAULT_PROPOSAL_TTL_SECONDS = 120
MIN_PROPOSAL_TTL_SECONDS = 5
MAX_PROPOSAL_TTL_SECONDS = 300


def _utc_datetime_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now() -> str:
    return _format_utc(_utc_datetime_now())


def _parse_utc_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_ttl_seconds(value: Any, default: int = DEFAULT_PROPOSAL_TTL_SECONDS) -> int:
    ttl = _coerce_int(value)
    if ttl is None:
        ttl = default
    return max(MIN_PROPOSAL_TTL_SECONDS, min(MAX_PROPOSAL_TTL_SECONDS, ttl))


def _is_applyable_change(change: Optional[Dict[str, Any]]) -> bool:
    return not _apply_blockers_for_change(change)


def _apply_blockers_for_change(change: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(change, dict):
        return ["missing_requested_change"]

    blockers: List[str] = []
    value_type = str(change.get("value_type") or "").strip().lower()
    if not value_type:
        blockers.append("missing_value_type")
    elif value_type not in APPLYABLE_VALUE_TYPES:
        blockers.append(f"unsupported_value_type:{value_type}")
    if change.get("channel") is None:
        blockers.append("missing_channel")
    if change.get("value") is None:
        blockers.append("missing_value")
    return blockers


def _are_applyable_changes(changes: List[Dict[str, Any]]) -> bool:
    return bool(changes) and all(_is_applyable_change(change) for change in changes)


def _apply_blockers_for_changes(changes: List[Dict[str, Any]]) -> List[str]:
    if not changes:
        return ["missing_requested_change"]

    blockers: List[str] = []
    for change in changes:
        blockers.extend(_apply_blockers_for_change(change))
    return sorted(set(blockers))


def _apply_blockers_for_transaction_plan(plan: Any) -> List[str]:
    if not isinstance(plan, dict):
        return ["missing_supervised_transaction_plan"]
    if not str(plan.get("transaction_id") or "").strip():
        return ["missing_transaction_id"]
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        return ["missing_transaction_operations"]
    blockers = []
    separate_families = set()
    plan_families = set()
    for operation in operations:
        if not isinstance(operation, dict):
            blockers.append("invalid_transaction_operation")
            continue
        family_id = str(operation.get("family_id") or "").strip()
        write_kind = str(operation.get("write_kind") or "").strip()
        contract = get_supervised_console_family(family_id)
        if contract is None:
            blockers.append(f"unknown_transaction_family:{family_id or 'missing'}")
            continue
        plan_families.add(family_id)
        if write_kind not in contract.write_kinds:
            blockers.append(f"write_kind_family_mismatch:{family_id}:{write_kind}")
        if not contract.permits_execution(write_kind=write_kind, proof_run=False):
            blockers.append(f"write_kind_not_live_allowed:{family_id}:{write_kind}")
        if contract.separate_approval:
            separate_families.add(family_id)
    if separate_families and (
        len(separate_families) != 1 or len(plan_families) != 1
    ):
        blockers.append("high_risk_family_requires_separate_transaction")
    assessment = plan.get("preflight_assessment")
    if str(plan.get("run_id") or "").strip() and not isinstance(assessment, dict):
        blockers.append("soundcheck_physical_preflight_required")
    elif isinstance(assessment, dict):
        disposition = str(assessment.get("disposition") or "").strip()
        reason = str(assessment.get("reason") or "preflight_unresolved").strip()
        if disposition != "apply":
            blockers.append(f"soundcheck_preflight:{disposition or 'missing'}:{reason}")
    return sorted(set(blockers))


@dataclass
class OperatorProposal:
    id: str
    title: str
    target: Any = None
    kind: Optional[str] = None
    severity: str = "medium"
    confidence: Optional[float] = None
    status: str = "pending"
    source: str = "operator_proposal_queue"
    reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    ttl_seconds: int = DEFAULT_PROPOSAL_TTL_SECONDS
    expires_at: Optional[str] = None
    requested_change: Optional[Dict[str, Any]] = None
    requested_changes: List[Dict[str, Any]] = field(default_factory=list)
    mode_at_creation: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.status != "pending":
            return False
        expires = _parse_utc_timestamp(self.expires_at)
        if expires is None:
            return False
        current = now.astimezone(timezone.utc) if now else _utc_datetime_now()
        return current >= expires

    def expiration_epoch(self) -> Optional[float]:
        """Return the immutable proposal deadline even after it is claimed."""

        expires = _parse_utc_timestamp(self.expires_at)
        return expires.timestamp() if expires is not None else None

    def to_dict(self) -> Dict[str, Any]:
        transaction_plan = (
            self.raw.get("supervised_transaction_plan")
            if isinstance(self.raw, dict)
            else None
        )
        if transaction_plan is not None:
            apply_blockers = _apply_blockers_for_transaction_plan(transaction_plan)
        else:
            apply_blockers = (
                _apply_blockers_for_changes(self.requested_changes)
                if self.requested_changes
                else _apply_blockers_for_change(self.requested_change)
            )
        expires = _parse_utc_timestamp(self.expires_at)
        expired = bool(
            expires is not None
            and _utc_datetime_now() >= expires
        )
        if expired and "proposal_expired" not in apply_blockers:
            apply_blockers = [*apply_blockers, "proposal_expired"]
        if self.status != "pending":
            apply_blockers = [
                *apply_blockers,
                f"proposal_not_pending:{self.status}",
            ]
        can_apply = self.status == "pending" and not apply_blockers
        return {
            "id": self.id,
            "title": self.title,
            "target": copy.deepcopy(self.target),
            "kind": self.kind,
            "severity": self.severity,
            "confidence": self.confidence,
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "expired": expired,
            "requested_change": copy.deepcopy(self.requested_change) if self.requested_change else None,
            "requested_changes": copy.deepcopy(self.requested_changes) if self.requested_changes else None,
            "linked_change_count": len(self.requested_changes),
            "mode_at_creation": self.mode_at_creation,
            "can_apply": can_apply,
            "requires_approval": can_apply,
            "apply_blockers": apply_blockers,
            "raw": copy.deepcopy(self.raw),
            "result": copy.deepcopy(self.result) if self.result else None,
        }


class OperatorProposalQueue:
    def __init__(self, max_history: int = 200):
        self.max_history = int(max_history)
        self._pending: List[OperatorProposal] = []
        self._history: List[OperatorProposal] = []
        self._seen_proposal_ids: set[str] = set()

    def _trim_history(self) -> None:
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def _normalize_requested_change(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_change = payload.get("requested_change")
        if not isinstance(raw_change, dict):
            raw_change = {}
        value_type = raw_change.get("value_type") or payload.get("value_type") or payload.get("kind")
        channel = raw_change.get("channel", payload.get("channel"))
        value = raw_change.get("value", payload.get("value"))
        value_type = str(value_type or "").strip().lower()
        channel_int = _coerce_int(channel)
        value_float = _coerce_float(value)

        if (
            not value_type
            or channel_int is None
            or not 1 <= channel_int <= 48
            or value_float is None
        ):
            return None
        normalized = {
            "value_type": value_type,
            "channel": channel_int,
            "value": value_float,
            "raw": dict(raw_change) if raw_change else {},
        }
        current_value = _coerce_float(raw_change.get("current_value"))
        if current_value is not None:
            normalized["current_value"] = current_value
        delta_db = _coerce_float(raw_change.get("delta_db"))
        if delta_db is not None:
            normalized["delta_db"] = delta_db
        for key in ("old_value", "proposed_value", "rollback_value"):
            optional_value = _coerce_float(raw_change.get(key))
            if optional_value is not None:
                normalized[key] = optional_value
        if "readback_required" in raw_change:
            normalized["readback_required"] = bool(raw_change.get("readback_required"))
        for key in ("wing_surface", "wing_address", "proof_direction"):
            raw_value = raw_change.get(key, payload.get(key))
            if raw_value is not None:
                normalized[key] = str(raw_value).strip()
        safety_evidence = raw_change.get("safety_evidence")
        if isinstance(safety_evidence, list):
            normalized["safety_evidence"] = [dict(item) for item in safety_evidence if isinstance(item, dict)]
        cache_freshness = raw_change.get("cache_freshness")
        if isinstance(cache_freshness, dict):
            normalized["cache_freshness"] = dict(cache_freshness)
        return normalized

    def _normalize_requested_changes(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_changes = payload.get("requested_changes")
        if not isinstance(raw_changes, list):
            return []
        normalized = []
        for raw_change in raw_changes:
            if not isinstance(raw_change, dict):
                continue
            change = self._normalize_requested_change({"requested_change": raw_change})
            if change:
                normalized.append(change)
        return normalized

    def create(self, payload: Dict[str, Any], operator_mode: Dict[str, Any]) -> Dict[str, Any]:
        if not operator_mode["capabilities"]["can_create_proposals"]:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "operator_mode_blocks_proposal_creation",
                "operator_mode": operator_mode,
                "proposal": None,
            }

        requested_change = self._normalize_requested_change(payload)
        requested_changes = self._normalize_requested_changes(payload)
        target = payload.get("target")
        if target is None and requested_change and requested_change.get("channel"):
            target = requested_change["channel"]
        elif target is None and requested_changes:
            target = [change["channel"] for change in requested_changes]
        title = (
            payload.get("title")
            or payload.get("description")
            or payload.get("message")
            or payload.get("action")
            or "Operator proposal"
        )
        explicit_proposal_id = payload.get("id") or payload.get("proposal_id")
        if explicit_proposal_id:
            proposal_id = str(explicit_proposal_id)
        else:
            proposal_id = f"proposal-{uuid4().hex[:12]}"
            while proposal_id in self._seen_proposal_ids:
                proposal_id = f"proposal-{uuid4().hex[:12]}"
        existing = self.find(proposal_id=proposal_id)
        if existing and existing.status == "pending":
            if existing.is_expired():
                completed = self._complete_pending(
                    proposal_id=proposal_id,
                    status="apply_blocked",
                    result={
                        "reason": "proposal_expired",
                        "fresh_analysis_required": True,
                    },
                )
                return {
                    "type": "operator_proposal_created",
                    "status": "blocked",
                    "success": False,
                    "reason": "proposal_expired_fresh_analysis_required",
                    "proposal_id": proposal_id,
                    "proposal": completed.get("proposal"),
                    "operator_mode": operator_mode,
                }
            return {
                "type": "operator_proposal_created",
                "status": "exists",
                "success": True,
                "proposal": existing.to_dict(),
                "operator_mode": operator_mode,
            }
        if proposal_id in self._seen_proposal_ids:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "proposal_id_already_used",
                "proposal_id": proposal_id,
                "operator_mode": operator_mode,
                "proposal": existing.to_dict() if existing else None,
            }

        created_at = utc_now()
        ttl_seconds = _coerce_ttl_seconds(payload.get("ttl_seconds"))
        explicit_expires_at = _parse_utc_timestamp(payload.get("expires_at"))
        created_dt = _parse_utc_timestamp(created_at) or _utc_datetime_now()
        expires_at = (
            _format_utc(explicit_expires_at)
            if explicit_expires_at is not None
            else _format_utc(created_dt + timedelta(seconds=ttl_seconds))
        )
        proposal = OperatorProposal(
            id=proposal_id,
            title=str(title),
            target=target,
            kind=str(
                payload.get("kind")
                or (requested_change or {}).get("value_type")
                or (f"linked_{requested_changes[0]['value_type']}" if requested_changes else "analysis")
            ),
            severity=str(payload.get("severity") or payload.get("priority") or "medium"),
            confidence=_coerce_float(payload.get("confidence")),
            source=str(payload.get("source") or "operator_proposal_queue"),
            reason=str(payload.get("reason") or payload.get("rationale") or ""),
            created_at=created_at,
            updated_at=created_at,
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            requested_change=requested_change,
            requested_changes=requested_changes,
            mode_at_creation=operator_mode.get("mode"),
            raw=copy.deepcopy(payload),
        )
        self._pending.append(proposal)
        self._seen_proposal_ids.add(proposal_id)
        return {
            "type": "operator_proposal_created",
            "status": "created",
            "success": True,
            "proposal": proposal.to_dict(),
            "operator_mode": operator_mode,
        }

    def _find_pending_index(self, proposal_id: Optional[str] = None, index: Optional[int] = None) -> Optional[int]:
        if proposal_id:
            for idx, proposal in enumerate(self._pending):
                if proposal.id == proposal_id:
                    return idx
        index_int = _coerce_int(index)
        if index_int is not None and 0 <= index_int < len(self._pending):
            return index_int
        return None

    def find(self, proposal_id: Optional[str] = None, index: Optional[int] = None) -> Optional[OperatorProposal]:
        idx = self._find_pending_index(proposal_id, index)
        if idx is not None:
            return self._pending[idx]
        if proposal_id:
            for proposal in reversed(self._history):
                if proposal.id == proposal_id:
                    return proposal
        return None

    def claim_for_apply(self, proposal: OperatorProposal) -> Dict[str, Any]:
        """Atomically claim one pending row before the first async write await."""
        pending_idx = self._find_pending_index(proposal.id)
        if (
            pending_idx is None
            or self._pending[pending_idx] is not proposal
            or proposal.status != "pending"
            or proposal.is_expired()
        ):
            return {
                "claimed": False,
                "reason": (
                    "proposal_expired"
                    if proposal.status == "pending" and proposal.is_expired()
                    else "proposal_not_pending"
                ),
                "proposal_status": proposal.status,
            }
        proposal.status = "applying"
        proposal.updated_at = utc_now()
        return {
            "claimed": True,
            "reason": "claimed",
            "proposal_status": proposal.status,
        }

    def _complete_pending(
        self,
        *,
        proposal_id: Optional[str] = None,
        index: Optional[int] = None,
        status: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        idx = self._find_pending_index(proposal_id, index)
        if idx is None:
            return {
                "status": "not_found",
                "success": False,
                "reason": "proposal_not_found",
                "proposal": None,
            }
        proposal = self._pending[idx]
        if proposal.status != "pending":
            return {
                "status": "blocked",
                "success": False,
                "reason": "proposal_not_pending",
                "proposal": proposal.to_dict(),
            }
        proposal = self._pending.pop(idx)
        proposal.status = status
        proposal.updated_at = utc_now()
        proposal.result = copy.deepcopy(result) if result else None
        self._history.append(proposal)
        self._trim_history()
        return {
            "status": status,
            "success": True,
            "proposal": proposal.to_dict(),
        }

    def accept(self, proposal_id: Optional[str] = None, index: Optional[int] = None) -> Dict[str, Any]:
        return self._complete_pending(proposal_id=proposal_id, index=index, status="accepted")

    def dismiss(
        self,
        proposal_id: Optional[str] = None,
        index: Optional[int] = None,
        *,
        reason: str = "",
    ) -> Dict[str, Any]:
        return self._complete_pending(
            proposal_id=proposal_id,
            index=index,
            status="dismissed",
            result={"reason": str(reason or "dismissed_by_operator")},
        )

    def expire_pending(self, proposal: OperatorProposal) -> Dict[str, Any]:
        return self._complete_pending(
            proposal_id=proposal.id,
            status="apply_blocked",
            result={
                "reason": "proposal_expired",
                "expires_at": proposal.expires_at,
                "ttl_seconds": proposal.ttl_seconds,
                "fresh_analysis_required": True,
            },
        )

    def mark_apply_result(self, proposal: OperatorProposal, result: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(result)
        pending_idx = self._find_pending_index(proposal.id)
        exact_pending = (
            pending_idx is not None and self._pending[pending_idx] is proposal
        )
        claimed_without_no_send_proof = bool(
            exact_pending
            and proposal.status == "applying"
            and result.get("status") != "applied"
            and result.get("no_console_send_proven") is not True
        )
        terminal_expiry = bool(
            exact_pending
            and proposal.status == "applying"
            and (
                str(result.get("reason") or "") == "proposal_expired"
                or str(result.get("reason") or "").startswith(
                    "operator_approval_expired"
                )
            )
        )
        terminal_verified_rollback = bool(
            exact_pending
            and proposal.status == "applying"
            and result.get("no_console_send_proven") is False
            and result.get("physical_state_restored") is True
            and result.get("terminal_after_console_send") is True
        )
        status = "applied"
        possible_console_send = bool(
            claimed_without_no_send_proof
            or result.get("no_console_send_proven") is False
        )
        if possible_console_send:
            # A retry is unsafe once a transaction may have reached the
            # console, even when the coordinator's top-level status is
            # "blocked".  Keep the proposal terminal until an operator has
            # reconciled the physical state.
            status = (
                "apply_blocked"
                if terminal_verified_rollback
                else "apply_error"
            )
            if not terminal_verified_rollback:
                result.setdefault("manual_review_required", True)
                result.setdefault("fresh_analysis_required", True)
        elif result.get("status") == "blocked":
            status = "apply_blocked"
        elif result.get("status") != "applied":
            status = "apply_error"

        if status != "applied":
            proposal.updated_at = utc_now()
            proposal.result = copy.deepcopy(result)
            if terminal_expiry or terminal_verified_rollback:
                if exact_pending:
                    self._pending.pop(pending_idx)
                proposal.status = "apply_blocked"
                proposal.result = {
                    **copy.deepcopy(result),
                    "fresh_analysis_required": True,
                }
                if proposal not in self._history:
                    self._history.append(proposal)
                self._trim_history()
            elif (
                status == "apply_blocked"
                and exact_pending
                and proposal.status == "applying"
            ):
                proposal.status = "pending"
            elif status == "apply_error":
                if exact_pending:
                    self._pending.pop(pending_idx)
                proposal.status = "apply_error"
                if proposal not in self._history:
                    self._history.append(proposal)
                self._trim_history()
            return {
                "status": status,
                "success": False,
                "proposal": proposal.to_dict(),
                "apply_result": result,
            }

        if not exact_pending or proposal.status != "applying":
            return {
                "status": "apply_error",
                "success": False,
                "reason": "proposal_not_claimed",
                "proposal": proposal.to_dict(),
                "apply_result": result,
            }
        self._pending.pop(pending_idx)
        proposal.status = status
        proposal.updated_at = utc_now()
        proposal.result = copy.deepcopy(result)
        if proposal not in self._history:
            self._history.append(proposal)
        self._trim_history()
        return {
            "status": status,
            "success": status == "applied",
            "proposal": proposal.to_dict(),
            "apply_result": result,
        }

    def pending_actions(self) -> List[Dict[str, Any]]:
        return [proposal.to_dict() for proposal in self._pending]

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, _coerce_int(limit) or 50)
        return [proposal.to_dict() for proposal in self._history[-limit:]]

    def summary(self) -> Dict[str, int]:
        return {
            "pending_count": len(self._pending),
            "history_count": len(self._history),
            "applied_count": sum(1 for item in self._history if item.status == "applied"),
            "accepted_count": sum(1 for item in self._history if item.status == "accepted"),
            "dismissed_count": sum(1 for item in self._history if item.status == "dismissed"),
            "blocked_count": sum(1 for item in self._history if item.status == "apply_blocked"),
        }
