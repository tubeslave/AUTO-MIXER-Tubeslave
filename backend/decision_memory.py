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
    haystack = " ".join(
        _text(trace.get(key)).lower()
        for key in ("source", "mode", "status", "proof_status")
    )
    return "offline" in haystack or "replay" in haystack


def evidence_quality_for_trace(trace: Dict[str, Any]) -> Optional[str]:
    proof_status = _text(trace.get("proof_status")).lower()
    if proof_status in VERIFIED_PROOF_STATUSES:
        return "verified"
    if _is_offline_evidence(trace):
        return "offline"
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
            "delta": trace.get("delta"),
            "readback": trace.get("readback"),
            "rollback": trace.get("rollback"),
        },
    }


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
                    rows.append(value)
        return rows[-limit:]
