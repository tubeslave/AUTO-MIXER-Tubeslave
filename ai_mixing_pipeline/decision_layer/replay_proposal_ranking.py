"""Replay-safe decision proposal model and deterministic ranking helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


REPLAY_TRACE_SCHEMA_VERSION = "replay_trace/v1"
PROPOSAL_FAMILIES = frozenset(
    {
        "gain_adjustment",
        "fader_adjustment",
        "speech_arbitration",
        "room_compensation",
        "feedback_mitigation",
        "operator_assistance",
        "performer_stage_interaction",
        "eq_adjustment",
        "compression",
    }
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _replay_correlation_id_from_payload(payload: dict[str, Any]) -> str | None:
    return payload.get("replay_correlation_id") or payload.get("rec_id")


@dataclass(frozen=True)
class ProposalScoreBreakdown:
    total_score: float
    weighted_scores: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    critic_confidence: float = 0.0
    family_priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayDecisionProposal:
    proposal_id: str
    family: str
    source_system: str
    action_type: str
    target: dict[str, Any]
    current_state: dict[str, Any]
    requested_state: dict[str, Any]
    confidence: float
    safety: dict[str, Any] = field(default_factory=dict)
    auto_apply_blocked: bool = True
    replay_correlation_id: str | None = None
    critic_scores: dict[str, Any] = field(default_factory=dict)
    replay_signature: str = ""
    transport_policy: str = "dry_run_only"
    live_mixer_writes: bool = False

    def __post_init__(self) -> None:
        if self.family not in PROPOSAL_FAMILIES:
            self.family = str(self.family)
        if not self.replay_correlation_id:
            self.replay_correlation_id = self.proposal_id
        if not self.replay_signature:
            self.replay_signature = _stable_hash(
                {
                    "proposal_id": self.proposal_id,
                    "family": self.family,
                    "source_system": self.source_system,
                    "action_type": self.action_type,
                    "target": self.target,
                    "current_state": self.current_state,
                    "requested_state": self.requested_state,
                    "confidence": round(_coerce_float(self.confidence), 6),
                    "safety": self.safety,
                    "auto_apply_blocked": self.auto_apply_blocked,
                    "replay_correlation_id": self.replay_correlation_id,
                    "critic_scores": self.critic_scores,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "family": self.family,
            "source_system": self.source_system,
            "action_type": self.action_type,
            "target": self.target,
            "current_state": self.current_state,
            "requested_state": self.requested_state,
            "confidence": self.confidence,
            "safety": self.safety,
            "auto_apply_blocked": self.auto_apply_blocked,
            "replay_correlation_id": self.replay_correlation_id,
            "critic_scores": self.critic_scores,
            "replay_signature": self.replay_signature,
            "transport_policy": self.transport_policy,
            "live_mixer_writes": self.live_mixer_writes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayDecisionProposal":
        return cls(
            proposal_id=str(payload["proposal_id"]),
            family=str(payload["family"]),
            source_system=str(payload.get("source_system", "unknown")),
            action_type=str(payload.get("action_type", "unknown")),
            target=dict(payload.get("target") or {}),
            current_state=dict(payload.get("current_state") or {}),
            requested_state=dict(payload.get("requested_state") or {}),
            confidence=_coerce_float(payload.get("confidence")),
            safety=dict(payload.get("safety") or {}),
            auto_apply_blocked=bool(payload.get("auto_apply_blocked", True)),
            replay_correlation_id=payload.get("replay_correlation_id"),
            critic_scores=dict(payload.get("critic_scores") or {}),
            replay_signature=str(payload.get("replay_signature") or ""),
            transport_policy=str(payload.get("transport_policy") or "dry_run_only"),
            live_mixer_writes=bool(payload.get("live_mixer_writes", False)),
        )


@dataclass(frozen=True)
class ProposalRankingTraceItem:
    proposal_id: str
    rank: int
    family: str
    total_score: float
    confidence: float
    replay_correlation_id: str | None
    replay_signature: str
    family_priority: float
    auto_apply_blocked: bool
    weighted_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalRankResult:
    schema_version: str
    selected_proposal_id: str
    ranked_proposal_ids: list[str]
    trace: list[ProposalRankingTraceItem]
    score_map: dict[str, float]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_proposal_id": self.selected_proposal_id,
            "ranked_proposal_ids": self.ranked_proposal_ids,
            "trace": [item.to_dict() for item in self.trace],
            "score_map": self.score_map,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProposalRankResult":
        trace = [ProposalRankingTraceItem(**item) for item in payload.get("trace", [])]
        return cls(
            schema_version=str(payload.get("schema_version", REPLAY_TRACE_SCHEMA_VERSION)),
            selected_proposal_id=str(payload["selected_proposal_id"]),
            ranked_proposal_ids=[str(item) for item in payload.get("ranked_proposal_ids", [])],
            trace=trace,
            score_map={str(k): _coerce_float(v) for k, v in (payload.get("score_map") or {}).items()},
            signature=str(payload.get("signature") or ""),
        )


@dataclass(frozen=True)
class ReplayGraphCheckpoint:
    checkpoint_id: str
    schema_version: str
    selected_proposal_id: str
    ranked_proposal_ids: list[str]
    proposals: list[dict[str, Any]]
    ranking: dict[str, Any]
    timeline_artifact: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayGraphCheckpoint":
        return cls(
            checkpoint_id=str(payload["checkpoint_id"]),
            schema_version=str(payload.get("schema_version", REPLAY_TRACE_SCHEMA_VERSION)),
            selected_proposal_id=str(payload["selected_proposal_id"]),
            ranked_proposal_ids=[str(item) for item in payload.get("ranked_proposal_ids", [])],
            proposals=list(payload.get("proposals", [])),
            ranking=dict(payload.get("ranking") or {}),
            timeline_artifact=dict(payload.get("timeline_artifact") or {}),
            metadata=dict(payload.get("metadata") or {}),
            checkpoint_signature=str(payload.get("checkpoint_signature") or ""),
        )


def _sanitize_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_stable_json(payload))


def _timeline_event_signature(event: dict[str, Any]) -> str:
    return _stable_hash(event)


def _proposal_id(family: str, payload: dict[str, Any]) -> str:
    correlation_id = _replay_correlation_id_from_payload(payload) or _stable_hash(payload)[:12]
    return f"{family}::{correlation_id}"


def _build_generic_proposals(
    *,
    family: str,
    source_system_default: str,
    action_type: str,
    payloads: list[dict[str, Any]],
    target_builder,
    current_state_builder,
    requested_state_builder,
    default_auto_apply_blocked: bool,
) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for raw in payloads:
        payload = dict(raw)
        proposal = ReplayDecisionProposal(
            proposal_id=_proposal_id(family, payload),
            family=family,
            source_system=str(payload.get("source", source_system_default)),
            action_type=action_type,
            target=target_builder(payload),
            current_state=current_state_builder(payload),
            requested_state=requested_state_builder(payload),
            confidence=_coerce_float(payload.get("confidence"), 0.0),
            safety=dict(payload.get("safety") or {}),
            auto_apply_blocked=bool(payload.get("auto_apply_blocked", default_auto_apply_blocked)),
            replay_correlation_id=_replay_correlation_id_from_payload(payload),
            critic_scores=dict(payload.get("critic_scores") or {}),
            transport_policy="dry_run_only",
            live_mixer_writes=False,
        )
        proposals.append(proposal)
    return proposals


def build_gain_proposals(*, recommendations: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="gain_adjustment",
        source_system_default="auto_soundcheck_engine",
        action_type="set_input_trim_db",
        payloads=recommendations,
        target_builder=lambda payload: {"mixer_channel": payload.get("mixer_channel")},
        current_state_builder=lambda payload: {"trim_db": payload.get("current_trim_db")},
        requested_state_builder=lambda payload: {"trim_db": payload.get("recommended_target_trim_db")},
        default_auto_apply_blocked=False,
    )


def build_fader_proposals(*, fader_updates: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="fader_adjustment",
        source_system_default="auto_soundcheck_engine",
        action_type="set_fader_db",
        payloads=fader_updates,
        target_builder=lambda payload: {"mixer_channel": payload.get("mixer_channel")},
        current_state_builder=lambda payload: {"fader_db": payload.get("current_fader_db")},
        requested_state_builder=lambda payload: {"fader_db": payload.get("recommended_target_fader_db")},
        default_auto_apply_blocked=False,
    )


def build_speech_priority_arbitration_proposals(
    *, arbitration_payloads: list[dict[str, Any]]
) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="speech_arbitration",
        source_system_default="speech_priority_engine",
        action_type="speech_priority_update",
        payloads=arbitration_payloads,
        target_builder=lambda payload: {"bus": payload.get("target_bus", "main")},
        current_state_builder=lambda payload: dict(payload.get("current_state") or {}),
        requested_state_builder=lambda payload: dict(payload.get("requested_state") or payload),
        default_auto_apply_blocked=False,
    )


def build_room_compensation_proposals(*, room_plans: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="room_compensation",
        source_system_default="room_compensation",
        action_type="room_compensation_plan",
        payloads=room_plans,
        target_builder=lambda payload: {"target": payload.get("target", "master_bus")},
        current_state_builder=lambda payload: dict(payload.get("current_state") or {}),
        requested_state_builder=lambda payload: dict(payload.get("requested_state") or payload),
        default_auto_apply_blocked=False,
    )


def build_feedback_mitigation_proposals(
    *, feedback_events: list[dict[str, Any]]
) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="feedback_mitigation",
        source_system_default="feedback_detector",
        action_type="feedback_mitigation",
        payloads=feedback_events,
        target_builder=lambda payload: {"mixer_channel": payload.get("mixer_channel")},
        current_state_builder=lambda payload: dict(payload.get("current_state") or {}),
        requested_state_builder=lambda payload: dict(payload.get("requested_state") or payload),
        default_auto_apply_blocked=False,
    )


def build_operator_assistance_proposals(
    *, operator_recommendations: list[dict[str, Any]]
) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="operator_assistance",
        source_system_default="operator_console",
        action_type="operator_prompt",
        payloads=operator_recommendations,
        target_builder=lambda payload: {"target": payload.get("target", "mix")},
        current_state_builder=lambda payload: dict(payload.get("current_state") or {}),
        requested_state_builder=lambda payload: {"message": payload.get("message")},
        default_auto_apply_blocked=True,
    )


def build_performer_stage_interaction_proposals(
    *, interaction_events: list[dict[str, Any]]
) -> list[ReplayDecisionProposal]:
    return _build_generic_proposals(
        family="performer_stage_interaction",
        source_system_default="performer_stage",
        action_type="stage_interaction",
        payloads=interaction_events,
        target_builder=lambda payload: {"performer": payload.get("performer")},
        current_state_builder=lambda payload: dict(payload.get("current_state") or {}),
        requested_state_builder=lambda payload: dict(payload.get("requested_state") or payload),
        default_auto_apply_blocked=True,
    )


def build_replay_scores(
    proposals: list[ReplayDecisionProposal],
    config: dict[str, Any] | None,
) -> dict[str, ProposalScoreBreakdown]:
    weights = ((config or {}).get("critics") or {})
    scores: dict[str, ProposalScoreBreakdown] = {}
    for proposal in proposals:
        weighted_scores: dict[str, float] = {}
        raw_scores: dict[str, float] = {}
        for critic_name, critic_payload in proposal.critic_scores.items():
            if isinstance(critic_payload, dict):
                raw_value = critic_payload.get("scores", {}).get("overall", critic_payload.get("overall", 0.0))
                critic_confidence = _coerce_float(critic_payload.get("confidence"), proposal.confidence)
            else:
                raw_value = critic_payload
                critic_confidence = proposal.confidence
            critic_config = weights.get(critic_name, {"enabled": True, "weight": 1.0})
            if critic_config.get("enabled", True):
                weighted_scores[critic_name] = _coerce_float(raw_value) * _coerce_float(
                    critic_config.get("weight"), 1.0
                )
                raw_scores[critic_name] = _coerce_float(raw_value)
            else:
                raw_scores[critic_name] = _coerce_float(raw_value)
        critic_total = sum(weighted_scores.values()) * 100.0
        if not weighted_scores:
            critic_confidence = proposal.confidence
            critic_total = proposal.confidence * 100.0
        total_score = critic_total + proposal.confidence * 10.0
        scores[proposal.proposal_id] = ProposalScoreBreakdown(
            total_score=round(total_score, 6),
            weighted_scores=weighted_scores,
            raw_scores=raw_scores,
            critic_confidence=round(_coerce_float(critic_confidence), 6),
            family_priority=0.0,
        )
    return scores


def rank_replay_proposals(
    proposals: list[ReplayDecisionProposal],
    *,
    config: dict[str, Any] | None = None,
    family_priority: dict[str, float] | None = None,
) -> ProposalRankResult:
    resolved = [
        proposal if isinstance(proposal, ReplayDecisionProposal) else ReplayDecisionProposal.from_dict(proposal)
        for proposal in proposals
    ]
    score_breakdowns = build_replay_scores(resolved, config)
    priority = {str(key): _coerce_float(value) for key, value in (family_priority or {}).items()}
    for proposal in resolved:
        base = score_breakdowns[proposal.proposal_id]
        family_boost = priority.get(proposal.family, 0.0)
        score_breakdowns[proposal.proposal_id] = ProposalScoreBreakdown(
            total_score=round(base.total_score + family_boost * 100.0, 6),
            weighted_scores=base.weighted_scores,
            raw_scores=base.raw_scores,
            critic_confidence=base.critic_confidence,
            family_priority=family_boost,
        )

    ranked = sorted(
        resolved,
        key=lambda proposal: (
            -score_breakdowns[proposal.proposal_id].total_score,
            -proposal.confidence,
            proposal.family,
            proposal.replay_signature,
        ),
    )
    trace: list[ProposalRankingTraceItem] = []
    for index, proposal in enumerate(ranked, start=1):
        score = score_breakdowns[proposal.proposal_id]
        trace.append(
            ProposalRankingTraceItem(
                proposal_id=proposal.proposal_id,
                rank=index,
                family=proposal.family,
                total_score=score.total_score,
                confidence=proposal.confidence,
                replay_correlation_id=proposal.replay_correlation_id,
                replay_signature=proposal.replay_signature,
                family_priority=score.family_priority,
                auto_apply_blocked=proposal.auto_apply_blocked,
                weighted_scores=dict(score.weighted_scores),
            )
        )

    signature = _stable_hash(
        {
            "ranked_proposal_ids": [proposal.proposal_id for proposal in ranked],
            "score_map": {proposal.proposal_id: score_breakdowns[proposal.proposal_id].total_score for proposal in ranked},
            "family_priority": priority,
        }
    )
    return ProposalRankResult(
        schema_version=REPLAY_TRACE_SCHEMA_VERSION,
        selected_proposal_id=ranked[0].proposal_id,
        ranked_proposal_ids=[proposal.proposal_id for proposal in ranked],
        trace=trace,
        score_map={proposal.proposal_id: score_breakdowns[proposal.proposal_id].total_score for proposal in ranked},
        signature=signature,
    )


def build_replay_timeline_artifact(
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_map = {proposal.proposal_id: proposal for proposal in proposals}
    events: list[dict[str, Any]] = []
    for item in ranking.trace:
        proposal = proposal_map[item.proposal_id]
        event = {
            "rank": item.rank,
            "proposal_id": proposal.proposal_id,
            "replay_correlation_id": proposal.replay_correlation_id,
            "family": proposal.family,
            "confidence": proposal.confidence,
            "replay_signature": proposal.replay_signature,
            "auto_apply_blocked": proposal.auto_apply_blocked,
            "transport_policy": proposal.transport_policy,
            "live_mixer_writes": proposal.live_mixer_writes,
            "target": proposal.target,
            "requested_state": proposal.requested_state,
            "selected": proposal.proposal_id == ranking.selected_proposal_id,
        }
        event["event_signature"] = _timeline_event_signature(event)
        events.append(event)
    artifact = {
        "schema_version": REPLAY_TRACE_SCHEMA_VERSION,
        "selected_proposal_id": ranking.selected_proposal_id,
        "events": events,
        "dry_run_only": all(event["transport_policy"] == "dry_run_only" for event in events),
        "live_mixer_writes": any(bool(event["live_mixer_writes"]) for event in events),
        "metadata": _sanitize_replay_payload(metadata or {}),
    }
    artifact["timeline_signature"] = _stable_hash(artifact)
    artifact["timeline_id"] = f"replay_timeline::{artifact['timeline_signature'][:16]}"
    return artifact


def write_replay_timeline_artifact(
    output_dir: str | Path,
    name: str,
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    path = Path(output_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_replay_timeline_artifact(proposals, ranking, metadata=metadata)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def build_replay_graph_checkpoint(
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> ReplayGraphCheckpoint:
    timeline_artifact = build_replay_timeline_artifact(proposals, ranking, metadata=metadata)
    proposal_rows = [proposal.to_dict() for proposal in proposals]
    ranking_payload = ranking.to_dict()
    checkpoint_payload = {
        "schema_version": REPLAY_TRACE_SCHEMA_VERSION,
        "selected_proposal_id": ranking.selected_proposal_id,
        "ranked_proposal_ids": ranking.ranked_proposal_ids,
        "proposals": proposal_rows,
        "ranking": ranking_payload,
        "timeline_artifact": timeline_artifact,
        "metadata": _sanitize_replay_payload(metadata or {}),
    }
    checkpoint_signature = _stable_hash(checkpoint_payload)
    return ReplayGraphCheckpoint(
        checkpoint_id=f"replay_graph::{checkpoint_signature[:16]}",
        schema_version=REPLAY_TRACE_SCHEMA_VERSION,
        selected_proposal_id=ranking.selected_proposal_id,
        ranked_proposal_ids=list(ranking.ranked_proposal_ids),
        proposals=proposal_rows,
        ranking=ranking_payload,
        timeline_artifact=timeline_artifact,
        metadata=_sanitize_replay_payload(metadata or {}),
        checkpoint_signature=checkpoint_signature,
    )


def compare_replay_graph_checkpoints(
    baseline: ReplayGraphCheckpoint | dict[str, Any],
    candidate: ReplayGraphCheckpoint | dict[str, Any],
) -> dict[str, Any]:
    left = baseline if isinstance(baseline, ReplayGraphCheckpoint) else ReplayGraphCheckpoint.from_dict(baseline)
    right = candidate if isinstance(candidate, ReplayGraphCheckpoint) else ReplayGraphCheckpoint.from_dict(candidate)
    left_ids = set(left.ranked_proposal_ids)
    right_ids = set(right.ranked_proposal_ids)
    return {
        "same_selected_proposal": left.selected_proposal_id == right.selected_proposal_id,
        "selected_proposal_changed": left.selected_proposal_id != right.selected_proposal_id,
        "baseline_selected_proposal_id": left.selected_proposal_id,
        "candidate_selected_proposal_id": right.selected_proposal_id,
        "proposal_ids_added": sorted(right_ids - left_ids),
        "proposal_ids_removed": sorted(left_ids - right_ids),
        "same_ranking_signature": left.checkpoint_signature == right.checkpoint_signature,
    }
