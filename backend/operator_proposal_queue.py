"""Backend-native proposal queue for operator-directed automixing.

This queue is the product contract between analysis modules and the UI. It
does not apply mixer writes by itself; handlers route approved applies through
the server's supervised manual write gate.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

try:
    from processing_registry import LIVE_ALLOWED_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_registry import LIVE_ALLOWED_WRITE_KINDS
from supervised_console_parameter_registry import get_supervised_console_family
from decision_queue_wire_contract import (
    REVIEW_CONTRACT_VERSION,
    ReviewContentError,
    build_review_identity,
    is_sha256_digest,
    terminal_result_proves_verified_apply,
    terminal_result_proves_verified_restored_proof,
)
from operator_proposal_result_store import (
    ResultEvidenceError,
    ResultEvidenceRecord,
    ResultEvidenceReservationBinding,
    ResultEvidenceStore,
)


FINAL_STATUSES = {
    "accepted",
    "dismissed",
    "applied",
    "proof_passed",
    "apply_blocked",
    "apply_error",
}
APPLYABLE_VALUE_TYPES = set(LIVE_ALLOWED_WRITE_KINDS)
DEFAULT_PROPOSAL_TTL_SECONDS = 120
MIN_PROPOSAL_TTL_SECONDS = 5
MAX_PROPOSAL_TTL_SECONDS = 300
SOUNDCHECK_RECOMMENDATION_SOURCE = "soundcheck_recommendation_bundle"
SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION = (
    "soundcheck_correction_requires_supervised_transaction"
)
MAX_PROPOSAL_REVIEW_BYTES = 2_097_152
MAX_PENDING_PROPOSALS = 128
MAX_PENDING_REVIEW_BYTES = 16_777_216
MAX_SEEN_PROPOSAL_IDS = 4_096
MAX_TERMINAL_RETAINED_BYTES = 16_777_216


@dataclass(frozen=True)
class VerifiedProposalExecution:
    """Immutable identity plus stored canonical bytes claimed for execution."""

    proposal_id: str
    review_contract_version: int
    review_content_digest: str
    expires_at_epoch: Optional[float]
    canonical_review_bytes: bytes

    def content_copy(self) -> Dict[str, Any]:
        """Return a private execution copy parsed only from verified bytes."""

        return json.loads(self.canonical_review_bytes.decode("utf-8"))


def _utc_datetime_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now() -> str:
    return _format_utc(_utc_datetime_now())


def _result_proves_verified_rollback(result: Dict[str, Any]) -> bool:
    return bool(
        result.get("status") in {"blocked", "rolled_back"}
        and result.get("no_console_send_proven") is False
        and result.get("physical_state_restored") is True
        and result.get("terminal_after_console_send") is True
        and result.get("wal_terminal") == "rolled_back"
    )


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


def soundcheck_proposal_apply_blocker(proposal: Any) -> Optional[str]:
    """Keep soundcheck corrections off the generic manual-write apply path."""

    if str(getattr(proposal, "source", "") or "") != SOUNDCHECK_RECOMMENDATION_SOURCE:
        return None
    raw = getattr(proposal, "raw", None)
    transaction_plan = (
        raw.get("supervised_transaction_plan")
        if isinstance(raw, dict)
        else None
    )
    if isinstance(transaction_plan, dict):
        return None
    return SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION


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
    _review_contract_version: Optional[int] = field(default=None, init=False, repr=False)
    _review_content_digest: Optional[str] = field(default=None, init=False, repr=False)
    _review_content_bytes: Optional[bytes] = field(default=None, init=False, repr=False)
    _result_digest: Optional[str] = field(default=None, init=False, repr=False)
    _result_bytes: int = field(default=0, init=False, repr=False)
    _result_detail_available: bool = field(default=False, init=False, repr=False)
    _result_evidence_pinned: bool = field(default=False, init=False, repr=False)
    _terminal_tombstone: bool = field(default=False, init=False, repr=False)

    @property
    def review_contract_version(self) -> Optional[int]:
        return self._review_contract_version

    @property
    def review_content_digest(self) -> Optional[str]:
        return self._review_content_digest

    @property
    def review_content_bytes(self) -> bytes:
        return self._review_content_bytes or b""

    def bind_review_identity(self, content: bytes, digest: str) -> None:
        """Store the v2 identity once; rebinding is always an integrity error."""

        if self._review_contract_version is not None:
            raise RuntimeError("proposal_review_identity_already_bound")
        if not isinstance(content, bytes) or not is_sha256_digest(digest):
            raise ValueError("proposal_review_identity_invalid")
        self._review_contract_version = REVIEW_CONTRACT_VERSION
        self._review_content_digest = digest
        self._review_content_bytes = bytes(content)

    @property
    def result_digest(self) -> Optional[str]:
        return self._result_digest

    @property
    def result_bytes(self) -> int:
        return self._result_bytes

    @property
    def result_detail_available(self) -> bool:
        return self._result_detail_available

    @property
    def result_evidence_pinned(self) -> bool:
        return self._result_evidence_pinned

    @property
    def terminal_tombstone(self) -> bool:
        return self._terminal_tombstone

    def bind_result_evidence(self, record: ResultEvidenceRecord) -> None:
        """Attach a bounded digest reference after durable persistence."""

        self._result_digest = record.digest
        self._result_bytes = int(record.byte_count)
        self._result_detail_available = True
        self._result_evidence_pinned = bool(record.pinned)

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

    def apply_state(self) -> Dict[str, Any]:
        """Compute current applyability without copying raw/result evidence."""

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
        source_blocker = soundcheck_proposal_apply_blocker(self)
        if source_blocker and source_blocker not in apply_blockers:
            apply_blockers = [source_blocker, *apply_blockers]
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
            "expired": expired,
            "can_apply": can_apply,
            "requires_approval": can_apply,
            "apply_blockers": apply_blockers,
        }

    def to_dict(self) -> Dict[str, Any]:
        apply_state = self.apply_state()
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
            "expired": apply_state["expired"],
            "requested_change": copy.deepcopy(self.requested_change) if self.requested_change else None,
            "requested_changes": copy.deepcopy(self.requested_changes) if self.requested_changes else None,
            "linked_change_count": len(self.requested_changes),
            "mode_at_creation": self.mode_at_creation,
            "can_apply": apply_state["can_apply"],
            "requires_approval": apply_state["requires_approval"],
            "apply_blockers": apply_state["apply_blockers"],
            "review_contract_version": self.review_contract_version,
            "review_content_digest": self.review_content_digest,
            "result_digest": self.result_digest,
            "result_bytes": self.result_bytes,
            "result_detail_available": self.result_detail_available,
            "result_evidence_pinned": self.result_evidence_pinned,
            "terminal_tombstone": self.terminal_tombstone,
            "raw": copy.deepcopy(self.raw),
            "result": copy.deepcopy(self.result) if self.result else None,
        }


class OperatorProposalQueue:
    def __init__(
        self,
        max_history: int = 200,
        *,
        max_history_bytes: int = MAX_TERMINAL_RETAINED_BYTES,
        result_store: Optional[ResultEvidenceStore] = None,
        result_evidence_lifecycle_ready: bool = False,
        terminal_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.max_history = int(max_history)
        self.max_history_bytes = int(max_history_bytes)
        self._pending: List[OperatorProposal] = []
        self._history: List[OperatorProposal] = []
        self._tombstones: Dict[str, OperatorProposal] = {}
        self._seen_proposal_ids: set[str] = set()
        self._review_claim_lock = threading.RLock()
        self._terminal_callback = terminal_callback
        self._result_persistence_failed = False
        self._result_store = result_store
        self._result_recovery_hazards: tuple[Dict[str, Any], ...] = ()
        if self._result_store is not None:
            try:
                self._result_recovery_hazards = (
                    self._result_store.recovery_hazards()
                )
            except ResultEvidenceError:
                self._result_persistence_failed = True
        # A configured store is only capacity.  Live apply stays fail-closed
        # until the handler lifecycle proves that every possible send owns and
        # consumes one pre-send reservation on every terminal path.
        self._result_evidence_lifecycle_ready = bool(
            result_evidence_lifecycle_ready
            and self._result_store is not None
            and not self._result_persistence_failed
            and not self._result_recovery_hazards
        )

    def _notify_terminal(self, proposal: OperatorProposal) -> None:
        callback = self._terminal_callback
        if callback is not None:
            callback(proposal.id, proposal.status)

    @property
    def result_persistence_failed(self) -> bool:
        return self._result_persistence_failed

    @property
    def result_evidence_contract_ready(self) -> bool:
        return (
            self._result_store is not None
            and self._result_evidence_lifecycle_ready
            and not self._result_persistence_failed
            and not self._result_recovery_hazards
        )

    @property
    def result_store(self) -> Optional[ResultEvidenceStore]:
        return self._result_store

    @property
    def result_recovery_hazards(self) -> tuple[Dict[str, Any], ...]:
        return self._result_recovery_hazards

    def _merge_result_recovery_hazards(
        self,
        hazards: tuple[Dict[str, Any], ...],
    ) -> None:
        merged = []
        seen = set()
        for item in (*self._result_recovery_hazards, *hazards):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("code") or ""),
                str(item.get("proposal_id") or ""),
                str(item.get("review_content_digest") or ""),
                str(item.get("result_digest") or ""),
                str(item.get("phase") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
        self._result_recovery_hazards = tuple(merged)
        if self._result_recovery_hazards:
            self._result_evidence_lifecycle_ready = False

    def _refresh_result_apply_fence(self) -> Optional[str]:
        """Refresh cross-process artifact/send truth before apply steps."""

        if self._result_recovery_hazards:
            return "operator_proposal_result_recovery_required"
        if self._result_persistence_failed:
            return "operator_proposal_result_persistence_failed"
        if self._result_store is None:
            # Claim is also used by internal/offline flows.  The canonical
            # live handler separately requires the lifecycle contract before
            # reserve/claim; absence of a store is not itself a recovered
            # cross-process hazard.
            return None
        try:
            hazards = self._result_store.current_apply_hazards()
        except ResultEvidenceError as exc:
            if exc.hazards:
                self._merge_result_recovery_hazards(exc.hazards)
            else:
                self._result_persistence_failed = True
                self._result_evidence_lifecycle_ready = False
            return exc.code
        if hazards:
            self._merge_result_recovery_hazards(hazards)
            return "operator_proposal_result_recovery_required"
        return None

    def latch_result_persistence_failure(self) -> None:
        """Fail closed after an unexpected result lifecycle exception."""

        self._result_persistence_failed = True
        self._result_evidence_lifecycle_ready = False

    def iter_pending(self) -> tuple[OperatorProposal, ...]:
        """Return proposal objects for bounded scalar projection only."""

        return tuple(self._pending)

    def iter_history(self) -> tuple[OperatorProposal, ...]:
        """Return retained terminal proposal objects without materializing evidence."""

        return tuple(self._history)

    def iter_terminal_hazards(self) -> tuple[OperatorProposal, ...]:
        """Return full rows and tombstones that preserve possible-write truth."""

        rows = [*self._history, *self._tombstones.values()]
        return tuple(
            item
            for item in rows
            if isinstance(item.result, dict)
            and (
                item.result.get("write_state_unknown") is True
                or item.result.get("rollback_state_unknown") is True
                or (
                    item.result.get("physical_state_restored") is False
                    and not terminal_result_proves_verified_apply(item.result)
                )
                or item.result.get("manual_review_required") is True
            )
        )

    def proposal_id_was_seen(self, proposal_id: str) -> bool:
        return str(proposal_id) in self._seen_proposal_ids

    def pending_review_bytes(self) -> int:
        return sum(len(item.review_content_bytes) for item in self._pending)

    def terminal_retained_bytes(self) -> int:
        return sum(
            len(item.review_content_bytes) + int(item.result_bytes)
            for item in self._history
        )

    def reserve_result_evidence(
        self,
        proposal: OperatorProposal,
    ) -> Dict[str, Any]:
        """Reserve durable capacity before any possible console send."""

        fence_reason = self._refresh_result_apply_fence()
        if fence_reason:
            return {
                "reserved": False,
                "reason": fence_reason,
                "reservation_token": None,
                "reservation_binding": None,
            }
        if self._result_store is None:
            return {
                "reserved": False,
                "reason": "operator_proposal_result_evidence_contract_incomplete",
                "reservation_token": None,
                "reservation_binding": None,
            }
        try:
            token = self._result_store.reserve(
                proposal_id=proposal.id,
                review_content_digest=str(proposal.review_content_digest),
            )
        except ResultEvidenceError as exc:
            if exc.hazards:
                self._merge_result_recovery_hazards(exc.hazards)
            return {
                "reserved": False,
                "reason": exc.code,
                "reservation_token": None,
                "reservation_binding": None,
            }
        return {
            "reserved": True,
            "reason": None,
            "reservation_token": token,
            "reservation_binding": ResultEvidenceReservationBinding(
                token=token,
                proposal_id=proposal.id,
                review_content_digest=str(proposal.review_content_digest),
            ),
        }

    def mark_result_evidence_send_may_have_occurred(
        self,
        reservation_binding: Any,
    ) -> Dict[str, Any]:
        """Durably cross the result-evidence no-send boundary."""

        if self._result_store is None:
            self._result_persistence_failed = True
            return {
                "transitioned": False,
                "reason": "operator_proposal_result_evidence_contract_incomplete",
            }
        try:
            self._result_store.mark_send_may_have_occurred(
                reservation_binding
            )
        except ResultEvidenceError as exc:
            if exc.hazards:
                self._merge_result_recovery_hazards(exc.hazards)
            else:
                self._result_persistence_failed = True
                self._result_evidence_lifecycle_ready = False
            return {"transitioned": False, "reason": exc.code}
        return {"transitioned": True, "reason": None}

    def release_result_evidence(self, reservation_binding: Any) -> None:
        if self._result_store is not None:
            self._result_store.release(reservation_binding)

    def _persist_result(
        self,
        proposal: OperatorProposal,
        result: Dict[str, Any],
        *,
        reservation_binding: Optional[ResultEvidenceReservationBinding] = None,
        proven_no_send: bool = False,
    ) -> Optional[ResultEvidenceRecord]:
        if self._result_store is None:
            self._result_persistence_failed = True
            return None
        try:
            if reservation_binding:
                record = self._result_store.commit(reservation_binding, result)
            elif proven_no_send:
                safe_result = {
                    **copy.deepcopy(result),
                    "no_console_send_proven": True,
                }
                safe_result.setdefault("write_state_unknown", False)
                safe_result.setdefault("rollback_state_unknown", False)
                safe_result.setdefault("manual_review_required", False)
                record = self._result_store.persist_no_send_result(safe_result)
            else:
                self._result_persistence_failed = True
                return None
        except ResultEvidenceError:
            self._result_persistence_failed = True
            return None
        proposal.bind_result_evidence(record)
        return record

    @staticmethod
    def _bounded_tombstone_text(value: Any, max_bytes: int) -> str:
        raw = str(value or "").encode("utf-8")[:max_bytes]
        while raw:
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                raw = raw[:-1]
        return ""

    def _compact_terminal_to_tombstone(self, proposal: OperatorProposal) -> bool:
        result = copy.deepcopy(proposal.result) if isinstance(proposal.result, dict) else {}
        explicitly_safe_no_send = bool(
            proposal.status in {"accepted", "dismissed", "apply_blocked"}
            and result.get("no_console_send_proven") is True
            and result.get("write_state_unknown") is not True
            and result.get("rollback_state_unknown") is not True
            and result.get("physical_state_restored") is not False
            and result.get("manual_review_required") is not True
            and result.get("terminal_after_console_send") is not True
            and result.get("result_persistence_failed") is not True
        )
        if not proposal.result_digest and explicitly_safe_no_send:
            if self._persist_result(proposal, result, proven_no_send=True) is None:
                return False
        elif not proposal.result_digest:
            # Never compact away the only full copy of possible-send truth.
            # Capacity pressure remains fail-closed until durable evidence is
            # available for the tombstone to reference.
            return False
        safety_truth = {
            key: result.get(key)
            for key in (
                "status",
                "reason",
                "no_console_send_proven",
                "write_state_unknown",
                "physical_state_restored",
                "manual_review_required",
                "rollback_state_unknown",
                "rollback_status",
                "wal_status",
                "wal_terminal",
                "terminal_after_console_send",
                "result_persistence_failed",
                "fresh_analysis_required",
                "emergency_stop_activated",
            )
            if key in result
        }
        safety_truth.update({
            "result_digest": proposal.result_digest,
            "result_detail_available": proposal.result_detail_available,
        })
        proposal.title = self._bounded_tombstone_text(proposal.title, 256)
        proposal.reason = self._bounded_tombstone_text(proposal.reason, 512)
        proposal.target = None
        proposal.requested_change = None
        proposal.requested_changes = []
        proposal.raw = {}
        proposal.result = safety_truth
        proposal._review_content_bytes = b""
        proposal._terminal_tombstone = True
        return True

    def _trim_history(self) -> None:
        while (
            len(self._history) > self.max_history
            or self.terminal_retained_bytes() > self.max_history_bytes
        ):
            oldest = self._history[0]
            if not self._compact_terminal_to_tombstone(oldest):
                return
            self._history.pop(0)
            self._tombstones[oldest.id] = oldest

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
        if len(proposal_id.encode("utf-8")) > 256:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "proposal_id_too_large",
                "proposal_id": None,
                "operator_mode": operator_mode,
                "proposal": None,
            }
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
        if len(self._seen_proposal_ids) >= MAX_SEEN_PROPOSAL_IDS:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "operator_proposal_id_ledger_capacity_exceeded",
                "proposal_id": proposal_id,
                "operator_mode": operator_mode,
                "proposal": None,
            }
        if len(self._pending) >= MAX_PENDING_PROPOSALS:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "operator_proposal_pending_capacity_exceeded",
                "proposal_id": proposal_id,
                "operator_mode": operator_mode,
                "proposal": None,
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
        try:
            review_content, review_digest = build_review_identity(proposal)
        except ReviewContentError as exc:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": str(exc) or "proposal_review_content_not_canonical_json",
                "proposal_id": proposal_id,
                "operator_mode": operator_mode,
                "proposal": None,
            }
        if len(review_content) > MAX_PROPOSAL_REVIEW_BYTES:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "proposal_review_content_too_large",
                "proposal_id": proposal_id,
                "review_content_bytes": len(review_content),
                "max_review_content_bytes": MAX_PROPOSAL_REVIEW_BYTES,
                "operator_mode": operator_mode,
                "proposal": None,
            }
        if self.pending_review_bytes() + len(review_content) > MAX_PENDING_REVIEW_BYTES:
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "operator_proposal_pending_byte_capacity_exceeded",
                "proposal_id": proposal_id,
                "operator_mode": operator_mode,
                "proposal": None,
            }
        proposal.bind_review_identity(review_content, review_digest)
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
            return self._tombstones.get(str(proposal_id))
        return None

    def load_result_evidence(
        self,
        proposal: OperatorProposal,
        *,
        expected_result_digest: Any,
    ) -> bytes:
        """Load terminal evidence only through the proposal's bound digest."""

        expected = str(expected_result_digest or "")
        if not expected:
            raise ResultEvidenceError("operator_proposal_result_digest_required")
        if not is_sha256_digest(expected):
            raise ResultEvidenceError("operator_proposal_result_digest_invalid")
        if expected != proposal.result_digest:
            raise ResultEvidenceError("operator_proposal_result_digest_mismatch")
        if self._result_store is None or not proposal.result_detail_available:
            raise ResultEvidenceError("operator_proposal_result_detail_not_retained")
        return self._result_store.load(expected)

    def validate_stored_review_integrity(self, proposal: OperatorProposal) -> Dict[str, Any]:
        """Recompute authoritative content and compare it with write-once identity."""

        if proposal.review_contract_version != REVIEW_CONTRACT_VERSION:
            return {
                "valid": False,
                "reason": "proposal_review_contract_upgrade_required",
            }
        stored_digest = proposal.review_content_digest
        stored_bytes = proposal.review_content_bytes
        if not stored_bytes or not is_sha256_digest(stored_digest):
            return {
                "valid": False,
                "reason": "proposal_review_content_integrity_error",
            }
        try:
            current_bytes, current_digest = build_review_identity(proposal)
        except ReviewContentError:
            return {
                "valid": False,
                "reason": "proposal_review_content_integrity_error",
            }
        if current_bytes != stored_bytes or current_digest != stored_digest:
            return {
                "valid": False,
                "reason": "proposal_review_content_integrity_error",
            }
        return {
            "valid": True,
            "reason": None,
            "review_contract_version": REVIEW_CONTRACT_VERSION,
            "review_content_digest": stored_digest,
        }

    def validate_review_request(
        self,
        proposal: OperatorProposal,
        *,
        expected_review_contract_version: Any,
        expected_review_content_digest: Any,
    ) -> Dict[str, Any]:
        """Validate request/stored/current identity without mutating queue state."""

        if proposal.review_contract_version != REVIEW_CONTRACT_VERSION:
            return {"valid": False, "reason": "proposal_review_contract_upgrade_required"}
        if expected_review_contract_version is None:
            return {"valid": False, "reason": "proposal_review_contract_version_required"}
        if (
            isinstance(expected_review_contract_version, bool)
            or not isinstance(expected_review_contract_version, int)
        ):
            return {"valid": False, "reason": "proposal_review_contract_version_invalid"}
        if expected_review_contract_version != REVIEW_CONTRACT_VERSION:
            return {"valid": False, "reason": "proposal_review_contract_version_mismatch"}
        expected_digest = str(expected_review_content_digest or "")
        if not expected_digest:
            return {"valid": False, "reason": "proposal_review_digest_required"}
        if not is_sha256_digest(expected_digest):
            return {"valid": False, "reason": "proposal_review_digest_invalid"}
        if expected_digest != proposal.review_content_digest:
            return {"valid": False, "reason": "proposal_review_digest_mismatch"}
        integrity = self.validate_stored_review_integrity(proposal)
        if not integrity["valid"]:
            return integrity
        return integrity

    def claim_verified_for_apply(
        self,
        proposal: OperatorProposal,
        *,
        expected_review_contract_version: Any,
        expected_review_content_digest: Any,
    ) -> Dict[str, Any]:
        """Atomically revalidate, claim and freeze the exact execution snapshot."""

        with self._review_claim_lock:
            fence_reason = self._refresh_result_apply_fence()
            if fence_reason:
                return {
                    "claimed": False,
                    "reason": fence_reason,
                    "proposal_status": proposal.status,
                }
            validation = self.validate_review_request(
                proposal,
                expected_review_contract_version=expected_review_contract_version,
                expected_review_content_digest=expected_review_content_digest,
            )
            if not validation.get("valid"):
                return {
                    "claimed": False,
                    "reason": validation.get("reason"),
                    "proposal_status": proposal.status,
                    "review_validation": validation,
                }
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
            execution = VerifiedProposalExecution(
                proposal_id=proposal.id,
                review_contract_version=proposal.review_contract_version,
                review_content_digest=str(proposal.review_content_digest),
                expires_at_epoch=proposal.expiration_epoch(),
                canonical_review_bytes=proposal.review_content_bytes,
            )
            proposal.status = "applying"
            proposal.updated_at = utc_now()
            return {
                "claimed": True,
                "reason": "claimed",
                "proposal_status": proposal.status,
                "execution": execution,
            }

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

    def abort_verified_apply_claim(
        self,
        proposal: OperatorProposal,
        *,
        expected_proposal_id: Any,
        expected_review_contract_version: Any,
        expected_review_content_digest: Any,
    ) -> Dict[str, Any]:
        """Return an exact no-send claim to pending after local setup fails.

        This is deliberately narrower than result finalization.  It may only
        validate an unchanged pending row or undo the exact in-memory
        ``pending -> applying`` transition while the same reviewed proposal is
        still present in the pending queue.  Callers must release their durable
        result reservation before invoking it.
        """

        with self._review_claim_lock:
            pending_idx = self._find_pending_index(str(expected_proposal_id or ""))
            exact_row = bool(
                pending_idx is not None
                and self._pending[pending_idx] is proposal
                and proposal.id == expected_proposal_id
                and proposal.review_contract_version
                == expected_review_contract_version
                and proposal.review_content_digest
                == expected_review_content_digest
            )
            if not exact_row or proposal.status not in {"pending", "applying"}:
                return {
                    "aborted": False,
                    "reason": "verified_apply_claim_identity_mismatch",
                    "proposal_status": proposal.status,
                }
            if proposal.status == "pending":
                return {
                    "aborted": True,
                    "reason": "verified_apply_claim_already_pending",
                    "proposal_status": proposal.status,
                }
            proposal.status = "pending"
            proposal.updated_at = utc_now()
            return {
                "aborted": True,
                "reason": "verified_apply_claim_aborted_before_send",
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
        self._notify_terminal(proposal)
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

    def mark_apply_result(
        self,
        proposal: OperatorProposal,
        result: Dict[str, Any],
        *,
        evidence_reservation_binding: Optional[
            ResultEvidenceReservationBinding
        ] = None,
        expected_proposal_id: Any = None,
        expected_review_contract_version: Any = None,
        expected_review_content_digest: Any = None,
    ) -> Dict[str, Any]:
        with self._review_claim_lock:
            if any(
                value is not None
                for value in (
                    expected_proposal_id,
                    expected_review_contract_version,
                    expected_review_content_digest,
                )
            ):
                pending_idx = self._find_pending_index(str(expected_proposal_id or ""))
                if (
                    pending_idx is None
                    or self._pending[pending_idx] is not proposal
                    or proposal.status != "applying"
                    or proposal.id != expected_proposal_id
                    or proposal.review_contract_version
                    != expected_review_contract_version
                    or proposal.review_content_digest
                    != expected_review_content_digest
                ):
                    raise RuntimeError(
                        "proposal_identity_changed_after_claim"
                    )
            return self._mark_apply_result_locked(
                proposal,
                result,
                evidence_reservation_binding=evidence_reservation_binding,
            )

    def _mark_apply_result_locked(
        self,
        proposal: OperatorProposal,
        result: Dict[str, Any],
        *,
        evidence_reservation_binding: Optional[
            ResultEvidenceReservationBinding
        ] = None,
    ) -> Dict[str, Any]:
        result = copy.deepcopy(result)
        if (
            result.get("status") == "applied"
            and not terminal_result_proves_verified_apply(result)
        ):
            result.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "apply_terminal_evidence_incomplete",
                    "manual_review_required": True,
                    "fresh_analysis_required": True,
                }
            )
        if (
            result.get("status") == "proof_passed"
            and not terminal_result_proves_verified_restored_proof(result)
        ):
            result.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "proof_terminal_evidence_incomplete",
                    "manual_review_required": True,
                    "fresh_analysis_required": True,
                }
            )
        pending_idx = self._find_pending_index(proposal.id)
        exact_pending = (
            pending_idx is not None and self._pending[pending_idx] is proposal
        )
        claimed = bool(exact_pending and proposal.status == "applying")
        pre_persist_verified_rollback = bool(
            claimed
            and _result_proves_verified_rollback(result)
        )
        pre_persist_restored_proof = bool(
            claimed
            and terminal_result_proves_verified_restored_proof(result)
        )
        pre_persist_possible_send = bool(
            claimed
            and result.get("status") not in {"applied", "proof_passed"}
            and result.get("no_console_send_proven") is not True
        )
        pre_persist_safe_no_send = bool(
            claimed
            and result.get("no_console_send_proven") is True
            and result.get("write_state_unknown") is not True
            and result.get("rollback_state_unknown") is not True
            and result.get("physical_state_restored") is not False
            and result.get("manual_review_required") is not True
            and result.get("terminal_after_console_send") is not True
        )
        if pre_persist_verified_rollback:
            # A durable, physically verified rollback is safe to evict only
            # when its exact unknown/manual flags are explicit in the bytes
            # that receive the result digest.  Contradictory explicit True
            # values remain untouched and therefore pinned.
            result.setdefault("write_state_unknown", False)
            result.setdefault("rollback_state_unknown", False)
            result.setdefault("manual_review_required", False)
        if pre_persist_safe_no_send:
            result.setdefault("write_state_unknown", False)
            result.setdefault("rollback_state_unknown", False)
            result.setdefault("manual_review_required", False)
        if (
            pre_persist_possible_send
            and not pre_persist_verified_rollback
            and not pre_persist_restored_proof
        ):
            # Freeze the exact terminal safety truth before digest-bound
            # persistence.  The artifact, in-memory row, tombstone and wire
            # receipt must never disagree about an unverified possible send.
            result.setdefault("no_console_send_proven", False)
            result.setdefault("manual_review_required", True)
            result.setdefault("fresh_analysis_required", True)
            if (
                result.get("physical_verified") is not True
                and result.get("physical_state_restored") is not True
            ):
                result.setdefault("write_state_unknown", True)
                result.setdefault("physical_state_restored", False)
        if claimed:
            evidence = self._persist_result(
                proposal,
                result,
                reservation_binding=evidence_reservation_binding,
            )
            if evidence is None:
                original_reason = result.get("reason")
                possible_send = result.get("no_console_send_proven") is not True
                result = {
                    **result,
                    "status": "error",
                    "reason": "operator_proposal_result_persistence_failed",
                    "original_reason": original_reason,
                    "result_persistence_failed": True,
                }
                if possible_send:
                    result.update({
                        "no_console_send_proven": False,
                        "write_state_unknown": True,
                        "physical_state_restored": result.get(
                            "physical_state_restored"
                        ),
                        "manual_review_required": True,
                        "fresh_analysis_required": True,
                    })
        claimed_without_no_send_proof = bool(
            exact_pending
            and proposal.status == "applying"
            and result.get("status") not in {"applied", "proof_passed"}
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
            and _result_proves_verified_rollback(result)
        )
        terminal_restored_proof = bool(
            exact_pending
            and proposal.status == "applying"
            and terminal_result_proves_verified_restored_proof(result)
        )
        status = "applied"
        possible_console_send = bool(
            claimed_without_no_send_proof
            or result.get("no_console_send_proven") is False
        )
        if terminal_result_proves_verified_apply(result):
            # Only the complete durable receipt can become green terminal
            # apply truth.  Physical readback alone is insufficient.
            status = "applied"
        elif terminal_restored_proof:
            status = "proof_passed"
        elif possible_console_send:
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

        if status not in {"applied", "proof_passed"}:
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
            if proposal.status in FINAL_STATUSES:
                self._notify_terminal(proposal)
            return {
                "status": status,
                "success": False,
                "proposal": proposal.to_dict(),
                "apply_result": result,
                "result_digest": proposal.result_digest,
                "result_detail_available": proposal.result_detail_available,
            }

        if not exact_pending or proposal.status != "applying":
            return {
                "status": "apply_error",
                "success": False,
                "reason": "proposal_not_claimed",
                "proposal": proposal.to_dict(),
                "apply_result": result,
                "result_digest": proposal.result_digest,
                "result_detail_available": proposal.result_detail_available,
            }
        self._pending.pop(pending_idx)
        proposal.status = status
        proposal.updated_at = utc_now()
        proposal.result = copy.deepcopy(result)
        if proposal not in self._history:
            self._history.append(proposal)
        self._trim_history()
        self._notify_terminal(proposal)
        return {
            "status": status,
            "success": status in {"applied", "proof_passed"},
            "accepted": status == "applied",
            "proposal": proposal.to_dict(),
            "apply_result": result,
            "result_digest": proposal.result_digest,
            "result_detail_available": proposal.result_detail_available,
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
            "proof_passed_count": sum(
                1 for item in self._history if item.status == "proof_passed"
            ),
            "accepted_count": sum(1 for item in self._history if item.status == "accepted"),
            "dismissed_count": sum(1 for item in self._history if item.status == "dismissed"),
            "blocked_count": sum(1 for item in self._history if item.status == "apply_blocked"),
        }
