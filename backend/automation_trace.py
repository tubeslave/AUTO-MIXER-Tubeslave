"""Append-only automation decision traces for safe operator workflows.

The trace layer is intentionally transport-agnostic: it records proposal,
blocked, apply, readback and rollback evidence, but never sends OSC/MIDI writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


TRACE_PROVENANCE_SCHEMA = "automixer.automation_trace_provenance.v1"
TRACE_TAIL_CHUNK_BYTES = 64 * 1024
TRACE_TAIL_MAX_BYTES = 8 * 1024 * 1024
_TERMINAL_SUCCESS_STATUSES = {"applied", "passed", "success", "verified"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _trace_text(value: Any) -> str:
    return str(value or "").strip()


def _is_offline_trace(payload: Dict[str, Any]) -> bool:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    haystack = " ".join(
        [
            *(
                _trace_text(payload.get(key)).lower()
                for key in ("source", "mode", "status", "proof_status", "proof_mode")
            ),
            _trace_text(metadata.get("proof_mode")).lower(),
        ]
    )
    return any(marker in haystack for marker in ("offline", "replay", "dry_run", "dry-run", "fake"))


def _has_bound_runtime_provenance(payload: Dict[str, Any]) -> bool:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    provenance = (
        metadata.get("trace_provenance")
        if isinstance(metadata.get("trace_provenance"), dict)
        else {}
    )
    nonce = _trace_text(provenance.get("runtime_instance_nonce"))
    return (
        provenance.get("schema") == TRACE_PROVENANCE_SCHEMA
        and provenance.get("origin") == "server_runtime"
        and provenance.get("runtime_kind") in {"source", "packaged"}
        and len(nonce) == 32
        and all(character in "0123456789abcdef" for character in nonce)
    )


def _nested_evidence_claims_physical_or_verified(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = _trace_text(key).lower()
            normalized_value = _trace_text(item).lower()
            if normalized_key == "physical_wing" and item is True:
                return True
            if normalized_key in {
                "source",
                "readback_source",
                "rollback_readback_source",
            } and normalized_value == "physical_readback":
                return True
            if normalized_key in {
                "status",
                "proof_status",
                "readback_status",
                "rollback_readback_status",
            } and (
                normalized_value == "verified"
                or normalized_value.startswith("verified_")
                or normalized_value.endswith("_verified")
            ):
                return True
            if _nested_evidence_claims_physical_or_verified(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_nested_evidence_claims_physical_or_verified(item) for item in value)
    return False


def _has_nested_physical_or_verified_evidence(payload: Dict[str, Any]) -> bool:
    return any(
        _nested_evidence_claims_physical_or_verified(payload.get(key))
        for key in ("pre_read", "readback", "rollback")
    )


def _quarantine_nested_evidence(
    value: Any,
    *,
    trust_status: str,
    proof_status: str,
) -> Any:
    if isinstance(value, list):
        return [
            _quarantine_nested_evidence(
                item,
                trust_status=trust_status,
                proof_status=proof_status,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    sanitized = dict(value)
    for key, item in list(value.items()):
        normalized_key = _trace_text(key).lower()
        normalized_value = _trace_text(item).lower()
        if normalized_key == "physical_wing" and item is True:
            sanitized.setdefault("original_physical_wing", True)
            sanitized[key] = False
            continue
        if normalized_key in {
            "source",
            "readback_source",
            "rollback_readback_source",
        } and normalized_value == "physical_readback":
            sanitized.setdefault(f"original_{key}", item)
            sanitized[key] = proof_status
            continue
        if normalized_key in {
            "status",
            "proof_status",
            "readback_status",
            "rollback_readback_status",
        } and (
            normalized_value == "verified"
            or normalized_value.startswith("verified_")
            or normalized_value.endswith("_verified")
        ):
            sanitized.setdefault(f"original_{key}", item)
            sanitized[key] = proof_status
            continue
        sanitized[key] = _quarantine_nested_evidence(
            item,
            trust_status=trust_status,
            proof_status=proof_status,
        )
    sanitized["trust_status"] = trust_status
    sanitized["physical_wing"] = False
    return sanitized


def _quarantine_untrusted_trace(
    payload: Dict[str, Any],
    *,
    trust_status: str,
    proof_status: str,
    terminal_status: str,
) -> Dict[str, Any]:
    normalized = dict(payload)
    metadata = normalized.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.setdefault("original_proof_status", normalized.get("proof_status"))
    metadata.setdefault("original_status", normalized.get("status"))
    metadata["trust_status"] = trust_status
    metadata["physical_wing"] = False
    normalized["metadata"] = metadata
    normalized["physical_wing"] = False
    normalized["proof_status"] = proof_status
    if _trace_text(normalized.get("status")).lower() in _TERMINAL_SUCCESS_STATUSES:
        normalized["status"] = terminal_status

    for key in ("pre_read", "readback", "rollback"):
        raw_evidence = normalized.get(key)
        if not isinstance(raw_evidence, dict):
            continue
        normalized[key] = _quarantine_nested_evidence(
            raw_evidence,
            trust_status=trust_status,
            proof_status=proof_status,
        )
    return normalized


def normalize_automation_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fail-closed read model without rewriting the append-only store.

    Historical dry-run proof runners could persist synthetic rows whose raw
    status looked like a physical, verified apply.  Their bytes remain intact
    for provenance, but operator-facing readers must never count or colour them
    as live proof.
    """

    normalized = dict(payload)
    if _is_offline_trace(normalized):
        normalized = _quarantine_untrusted_trace(
            normalized,
            trust_status="offline_non_physical",
            proof_status="offline",
            terminal_status="offline_simulated",
        )
        metadata = dict(normalized.get("metadata") or {})
        source = _trace_text(normalized.get("source")).lower()
        if not _trace_text(metadata.get("proof_mode")):
            metadata["proof_mode"] = (
                "dry_run"
                if any(marker in source for marker in ("dry_run", "dry-run", "fake"))
                else "offline"
            )
        normalized["metadata"] = metadata
        normalized["proof_mode"] = metadata["proof_mode"]
        for key in ("pre_read", "readback", "rollback"):
            if isinstance(normalized.get(key), dict) and normalized[key].get("status") == "offline":
                normalized[key]["status"] = "offline_unverified"
        return normalized

    raw_status = _trace_text(normalized.get("status")).lower()
    raw_proof_status = _trace_text(normalized.get("proof_status")).lower()
    claims_success = raw_status in _TERMINAL_SUCCESS_STATUSES or raw_proof_status == "verified"
    claims_trusted_evidence = claims_success or _has_nested_physical_or_verified_evidence(
        normalized
    )
    if claims_trusted_evidence and not _has_bound_runtime_provenance(normalized):
        return _quarantine_untrusted_trace(
            normalized,
            trust_status="legacy_unbound",
            proof_status="legacy_unbound",
            terminal_status="legacy_unbound_apply",
        )
    if raw_status == "applied" and raw_proof_status != "verified":
        return _quarantine_untrusted_trace(
            normalized,
            trust_status="runtime_bound_but_unverified",
            proof_status=raw_proof_status or "unverified",
            terminal_status="applied_unverified",
        )
    return normalized


def _read_recent_json_objects(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Read a bounded EOF snapshot without scanning the full append-only file."""

    newest_first: List[Dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        remaining = position
        carry = b""
        scanned = 0
        while remaining > 0 and len(newest_first) < limit and scanned < TRACE_TAIL_MAX_BYTES:
            chunk_size = min(TRACE_TAIL_CHUNK_BYTES, remaining, TRACE_TAIL_MAX_BYTES - scanned)
            remaining -= chunk_size
            handle.seek(remaining)
            chunk = handle.read(chunk_size)
            scanned += len(chunk)
            parts = (chunk + carry).split(b"\n")
            if remaining > 0:
                carry = parts[0]
                complete_lines = parts[1:]
            else:
                carry = b""
                complete_lines = parts
            for raw_line in reversed(complete_lines):
                if len(newest_first) >= limit:
                    break
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    newest_first.append(value)
    newest_first.reverse()
    return newest_first


@dataclass
class AutomationDecisionTrace:
    decision_id: str = field(default_factory=lambda: f"decision-{uuid4().hex[:12]}")
    idempotency_key: Optional[str] = None
    proposal_id: Optional[str] = None
    source: str = "unknown"
    mode: Optional[str] = None
    target: Any = None
    action_type: str = "unknown"
    status: str = "observed"
    current_value: Any = None
    proposed_value: Any = None
    delta: Any = None
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    pre_read: Optional[Dict[str, Any]] = None
    readback: Optional[Dict[str, Any]] = None
    rollback: Optional[Dict[str, Any]] = None
    proof_status: str = "unverified"
    blocked_reason: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "idempotency_key": self.idempotency_key,
            "proposal_id": self.proposal_id,
            "source": self.source,
            "mode": self.mode,
            "target": self.target,
            "action_type": self.action_type,
            "status": self.status,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "delta": self.delta,
            "safety_limits": self.safety_limits,
            "pre_read": self.pre_read,
            "readback": self.readback,
            "rollback": self.rollback,
            "proof_status": self.proof_status,
            "blocked_reason": self.blocked_reason,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class AutomationTraceStore:
    def __init__(self, root_dir: str | Path, filename: str = "decision_traces.jsonl"):
        self.root_dir = Path(root_dir)
        self.filename = str(filename)

    @property
    def path(self) -> Path:
        return self.root_dir / self.filename

    def append(self, trace: AutomationDecisionTrace | Dict[str, Any]) -> Dict[str, Any]:
        payload = trace.to_dict() if isinstance(trace, AutomationDecisionTrace) else dict(trace)
        payload.setdefault("decision_id", f"decision-{uuid4().hex[:12]}")
        payload.setdefault("created_at", utc_now())
        payload = json_safe(payload)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        if not self.path.exists():
            return []
        return [
            normalize_automation_trace(value)
            for value in _read_recent_json_objects(self.path, limit)
        ]
