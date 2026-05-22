"""Backend-native proposal queue for operator-directed automixing.

This queue is the product contract between analysis modules and the UI. It
does not apply mixer writes by itself; handlers route approved applies through
the server's supervised manual write gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


FINAL_STATUSES = {"accepted", "dismissed", "applied", "apply_blocked", "apply_error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_applyable_change(change: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(change, dict):
        return False
    return bool(change.get("value_type")) and change.get("channel") is not None and change.get("value") is not None


def _are_applyable_changes(changes: List[Dict[str, Any]]) -> bool:
    return bool(changes) and all(_is_applyable_change(change) for change in changes)


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
    requested_change: Optional[Dict[str, Any]] = None
    requested_changes: List[Dict[str, Any]] = field(default_factory=list)
    mode_at_creation: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        can_apply = _are_applyable_changes(self.requested_changes) if self.requested_changes else _is_applyable_change(self.requested_change)
        return {
            "id": self.id,
            "title": self.title,
            "target": self.target,
            "kind": self.kind,
            "severity": self.severity,
            "confidence": self.confidence,
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "requested_change": dict(self.requested_change) if self.requested_change else None,
            "requested_changes": [dict(change) for change in self.requested_changes] if self.requested_changes else None,
            "linked_change_count": len(self.requested_changes),
            "mode_at_creation": self.mode_at_creation,
            "can_apply": can_apply,
            "requires_approval": can_apply,
            "raw": dict(self.raw),
            "result": dict(self.result) if self.result else None,
        }


class OperatorProposalQueue:
    def __init__(self, max_history: int = 200):
        self.max_history = int(max_history)
        self._pending: List[OperatorProposal] = []
        self._history: List[OperatorProposal] = []

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

        if not value_type or channel_int is None or value_float is None:
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
        safety_evidence = raw_change.get("safety_evidence")
        if isinstance(safety_evidence, list):
            normalized["safety_evidence"] = [dict(item) for item in safety_evidence if isinstance(item, dict)]
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
        proposal_id = str(payload.get("id") or payload.get("proposal_id") or f"proposal-{uuid4().hex[:12]}")
        existing = self.find(proposal_id=proposal_id)
        if existing and existing.status == "pending":
            return {
                "type": "operator_proposal_created",
                "status": "exists",
                "success": True,
                "proposal": existing.to_dict(),
                "operator_mode": operator_mode,
            }

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
            requested_change=requested_change,
            requested_changes=requested_changes,
            mode_at_creation=operator_mode.get("mode"),
            raw=dict(payload),
        )
        self._pending.append(proposal)
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
        proposal = self._pending.pop(idx)
        proposal.status = status
        proposal.updated_at = utc_now()
        proposal.result = dict(result) if result else None
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

    def mark_apply_result(self, proposal: OperatorProposal, result: Dict[str, Any]) -> Dict[str, Any]:
        status = "applied"
        if result.get("status") == "blocked":
            status = "apply_blocked"
        elif result.get("status") != "applied":
            status = "apply_error"

        if status != "applied":
            proposal.updated_at = utc_now()
            proposal.result = dict(result)
            return {
                "status": status,
                "success": False,
                "proposal": proposal.to_dict(),
                "apply_result": result,
            }

        pending_idx = self._find_pending_index(proposal.id)
        if pending_idx is not None:
            self._pending.pop(pending_idx)
        proposal.status = status
        proposal.updated_at = utc_now()
        proposal.result = dict(result)
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
