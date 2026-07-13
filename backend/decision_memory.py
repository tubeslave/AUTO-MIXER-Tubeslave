"""Append-only decision memory derived from verified/offline evidence.

Decision memory is context for future proposals, not an authority to apply
changes. It is intentionally derived from existing decision traces and has no
mixer transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from automation_trace import json_safe, utc_now


VERIFIED_PROOF_STATUSES = {"verified"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_offline_evidence(trace: Dict[str, Any]) -> bool:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    haystack = " ".join(
        [
            *(
                _text(trace.get(key)).lower()
                for key in ("source", "mode", "status", "proof_status", "proof_mode")
            ),
            _text(metadata.get("proof_mode")).lower(),
        ]
    )
    return any(marker in haystack for marker in ("offline", "replay", "dry_run", "dry-run", "fake"))


def evidence_quality_for_trace(trace: Dict[str, Any]) -> Optional[str]:
    proof_status = _text(trace.get("proof_status")).lower()
    if _is_offline_evidence(trace):
        return "offline"
    if proof_status in VERIFIED_PROOF_STATUSES:
        return "verified"
    return None


def decision_memory_entry_from_trace(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    quality = evidence_quality_for_trace(trace)
    if not quality:
        return None

    action_type = _text(trace.get("action_type")) or "decision"
    target = trace.get("target")
    status = _text(trace.get("status")) or "observed"
    blocked_reason = _text(trace.get("blocked_reason"))
    summary = f"{action_type} target {target} {status}".strip()
    if blocked_reason:
        summary = f"{summary} blocked: {blocked_reason}"

    confidence = 0.95 if quality == "verified" else 0.82
    trace_metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    trace_source = trace.get("source")
    proof_mode = trace.get("proof_mode") or trace_metadata.get("proof_mode")
    physical_wing = trace.get("physical_wing")
    if physical_wing is None:
        physical_wing = trace_metadata.get("physical_wing")
    if _is_offline_evidence(trace):
        if not proof_mode and any(
            marker in _text(trace_source).lower()
            for marker in ("dry_run", "dry-run", "fake")
        ):
            proof_mode = "dry_run"
        if physical_wing is None:
            physical_wing = False
    return {
        "memory_id": f"decision-memory-{uuid4().hex[:12]}",
        "source_decision_id": trace.get("decision_id"),
        "source": "automation_decision_trace",
        "evidence_quality": quality,
        "summary": summary,
        "target": target,
        "action_type": action_type,
        "status": status,
        "proof_status": trace.get("proof_status"),
        "blocked_reason": trace.get("blocked_reason"),
        "confidence": confidence,
        "created_at": utc_now(),
        "trace_created_at": trace.get("created_at"),
        "metadata": {
            "mode": trace.get("mode"),
            "trace_source": trace_source,
            "proof_mode": proof_mode,
            "physical_wing": physical_wing,
            "delta": trace.get("delta"),
            "readback": trace.get("readback"),
            "rollback": trace.get("rollback"),
        },
    }


def normalize_decision_memory_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed when legacy verified rows have no trace provenance.

    The append-only store is never rewritten. Older rows predate preservation of
    ``metadata.trace_source`` and can include synthetic dry-run evidence that was
    labelled ``verified``. Readers therefore downgrade those rows in memory until
    an explicit provenance-aware replacement is recorded.
    """

    payload = dict(entry)
    metadata = payload.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if (
        payload.get("evidence_quality") == "verified"
        and not _text(metadata.get("trace_source"))
    ):
        payload["evidence_quality"] = "legacy_unbound"
        try:
            payload["confidence"] = min(float(payload.get("confidence", 0.5)), 0.5)
        except (TypeError, ValueError):
            payload["confidence"] = 0.5
        metadata["trust_status"] = "downgraded_missing_trace_provenance"
        metadata.setdefault("original_evidence_quality", "verified")
        payload["metadata"] = metadata
    return payload


class DecisionMemoryStore:
    def __init__(self, root_dir: str | Path, filename: str = "decision_memory.jsonl"):
        self.root_dir = Path(root_dir)
        self.filename = str(filename)

    @property
    def path(self) -> Path:
        return self.root_dir / self.filename

    def append_from_trace(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entry = decision_memory_entry_from_trace(trace)
        if not entry:
            return None
        self.root_dir.mkdir(parents=True, exist_ok=True)
        payload = json_safe(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(normalize_decision_memory_entry(value))
        return rows[-limit:]
