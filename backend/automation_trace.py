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

