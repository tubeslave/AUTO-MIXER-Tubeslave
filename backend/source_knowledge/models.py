"""Data models for source-grounded mixing rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

try:
    from output_paths import ai_logs_path
except ImportError:  # pragma: no cover - package import fallback
    from backend.output_paths import ai_logs_path


@dataclass
class SourceGroundedConfig:
    """Configuration for the source-grounded learning layer."""

    enabled: bool = False
    mode: str = "shadow"
    sources_path: str = ""
    rules_path: str = ""
    log_path: str = str(ai_logs_path("source_grounded_decisions.jsonl"))
    queue_maxsize: int = 256
    min_rule_confidence: float = 0.55
    allow_unsourced_rules: bool = False
    log_retrievals: bool = True
    log_feedback: bool = True
    default_limit: int = 8
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Optional[Dict[str, Any]] = None) -> "SourceGroundedConfig":
        payload = dict(config or {})
        if isinstance(payload.get("source_knowledge"), dict):
            payload = dict(payload["source_knowledge"])
        known = {name for name in cls.__dataclass_fields__ if name != "extra"}
        values = {key: payload[key] for key in known if key in payload}
        result = cls(**values)
        result.extra = {key: value for key, value in payload.items() if key not in known}
        return result


@dataclass
class SourceReference:
    """Authoritative source metadata."""

    source_id: str
    title: str
    source_type: str
    authors: List[str] = field(default_factory=list)
    publisher: str = ""
    year: Optional[int] = None
    url: str = ""
    authority_tier: str = "secondary"
    status: str = "active"
    allowed_uses: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "SourceReference":
        return cls(
            source_id=str(payload["source_id"]),
            title=str(payload["title"]),
            source_type=str(payload.get("source_type", "unknown")),
            authors=[str(item) for item in payload.get("authors", [])],
            publisher=str(payload.get("publisher", "")),
            year=int(payload["year"]) if payload.get("year") is not None else None,
            url=str(payload.get("url", "")),
            authority_tier=str(payload.get("authority_tier", "secondary")),
            status=str(payload.get("status", "active")),
            allowed_uses=[str(item) for item in payload.get("allowed_uses", [])],
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "authors": list(self.authors),
            "publisher": self.publisher,
            "year": self.year,
            "url": self.url,
            "authority_tier": self.authority_tier,
            "status": self.status,
            "allowed_uses": list(self.allowed_uses),
            "notes": self.notes,
        }


@dataclass
class SourceRule:
    """Paraphrased atomic rule with source provenance."""

    rule_id: str
    summary: str
    rationale: str
    domains: List[str]
    instruments: List[str]
    problems: List[str]
    action_templates: List[Dict[str, Any]]
    bounds: Dict[str, Any]
    source_ids: List[str]
    confidence: float = 0.7
    status: str = "active"
    tags: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    mode: str = "advisory"

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "SourceRule":
        return cls(
            rule_id=str(payload["rule_id"]),
            summary=str(payload["summary"]),
            rationale=str(payload.get("rationale", "")),
            domains=[str(item) for item in payload.get("domains", [])],
            instruments=[str(item) for item in payload.get("instruments", [])],
            problems=[str(item) for item in payload.get("problems", [])],
            action_templates=list(payload.get("action_templates", [])),
            bounds=dict(payload.get("bounds", {})),
            source_ids=[str(item) for item in payload.get("source_ids", [])],
            confidence=float(payload.get("confidence", 0.7)),
            status=str(payload.get("status", "active")),
            tags=[str(item) for item in payload.get("tags", [])],
            safety_notes=[str(item) for item in payload.get("safety_notes", [])],
            mode=str(payload.get("mode", "advisory")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "domains": list(self.domains),
            "instruments": list(self.instruments),
            "problems": list(self.problems),
            "action_templates": list(self.action_templates),
            "bounds": dict(self.bounds),
            "source_ids": list(self.source_ids),
            "confidence": float(self.confidence),
            "status": self.status,
            "tags": list(self.tags),
            "safety_notes": list(self.safety_notes),
            "mode": self.mode,
        }


@dataclass
class RuleMatch:
    """Rule search result with score and matched terms."""

    rule: SourceRule
    relevance_score: float
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "relevance_score": float(self.relevance_score),
            "matched_terms": list(self.matched_terms),
        }


@dataclass
class DecisionTrace:
    """Structured record for a source-grounded candidate decision."""

    session_id: str
    decision_id: str
    timestamp: float = field(default_factory=time.time)
    channel: Optional[str] = None
    instrument: Optional[str] = None
    problem: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    candidate_rule_ids: List[str] = field(default_factory=list)
    selected_rule_ids: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    selected_action: Dict[str, Any] = field(default_factory=dict)
    before_metrics: Dict[str, Any] = field(default_factory=dict)
    after_metrics: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "pending"
    confidence: float = 0.0
    osc_sent: bool = False
    safety_state: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "timestamp": float(self.timestamp),
            "channel": self.channel,
            "instrument": self.instrument,
            "problem": self.problem,
            "context": dict(self.context),
            "candidate_rule_ids": list(self.candidate_rule_ids),
            "selected_rule_ids": list(self.selected_rule_ids),
            "source_ids": list(self.source_ids),
            "candidate_actions": list(self.candidate_actions),
            "selected_action": dict(self.selected_action),
            "before_metrics": dict(self.before_metrics),
            "after_metrics": dict(self.after_metrics),
            "outcome": self.outcome,
            "confidence": float(self.confidence),
            "osc_sent": bool(self.osc_sent),
            "safety_state": dict(self.safety_state),
            "notes": self.notes,
        }


@dataclass
class FeedbackRecord:
    """Human or evaluator feedback attached to a decision trace."""

    session_id: str
    decision_id: str
    rating: str
    timestamp: float = field(default_factory=time.time)
    listener: str = "operator"
    comment: str = ""
    preferred_action: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "timestamp": float(self.timestamp),
            "listener": self.listener,
            "rating": self.rating,
            "comment": self.comment,
            "preferred_action": dict(self.preferred_action),
            "tags": list(self.tags),
        }
