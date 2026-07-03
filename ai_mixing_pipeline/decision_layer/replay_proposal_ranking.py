"""Replay-safe decision proposal model and deterministic ranking helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .score_normalizer import aggregate_scores, configured_weights


REPLAY_TRACE_SCHEMA_VERSION = "replay_trace/v1"

PROPOSAL_FAMILIES = {
    "gain_adjustment",
    "fader_adjustment",
    "compression",
    "eq_adjustment",
    "speech_arbitration",
    "room_compensation",
    "feedback_mitigation",
    "operator_assistance",
    "performer_stage_interaction",
}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _replay_correlation_id_from_payload(payload: dict[str, Any] | None) -> str:
    data = dict(payload or {})
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("replay_correlation_id")
        if value:
            return str(value)
    for key in ("replay_correlation_id", "trace_id"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


@dataclass(frozen=True)
class ProposalScoreBreakdown:
    final_score: float
    confidence: float
    confidence_weighted_score: float
    normalized_weights: dict[str, float]
    critic_values: dict[str, float]
    critic_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": float(self.final_score),
            "confidence": float(self.confidence),
            "confidence_weighted_score": float(self.confidence_weighted_score),
            "normalized_weights": dict(self.normalized_weights),
            "critic_values": dict(self.critic_values),
            "critic_breakdown": dict(self.critic_breakdown),
        }


@dataclass(frozen=True)
class ReplayDecisionProposal:
    proposal_id: str
    family: str
    source_system: str
    action_type: str
    target: dict[str, Any]
    current_state: dict[str, Any]
    requested_state: dict[str, Any]
    confidence: float
    safety: dict[str, Any]
    dry_run_only: bool = True
    auto_apply_blocked: bool = False
    replay_correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    analyst_observation: dict[str, Any] = field(default_factory=dict)
    critic_scores: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in PROPOSAL_FAMILIES:
            raise ValueError(f"Unsupported proposal family: {self.family}")
        if not self.replay_correlation_id:
            object.__setattr__(self, "replay_correlation_id", self._default_replay_correlation_id())

    def _default_replay_correlation_id(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "proposal_id": self.proposal_id,
                    "family": self.family,
                    "source_system": self.source_system,
                    "target": self.target,
                    "requested_state": self.requested_state,
                    "current_state": self.current_state,
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        return f"replay::{self.family}::{digest[:16]}"

    @property
    def replay_signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "id": self.proposal_id,
                    "family": self.family,
                    "source_system": self.source_system,
                    "action_type": self.action_type,
                    "target": self.target,
                    "requested_state": self.requested_state,
                    "current_state": self.current_state,
                    "metadata": self.metadata,
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "family": self.family,
            "source_system": self.source_system,
            "action_type": self.action_type,
            "target": dict(self.target),
            "current_state": dict(self.current_state),
            "requested_state": dict(self.requested_state),
            "confidence": float(self.confidence),
            "safety": dict(self.safety),
            "dry_run_only": bool(self.dry_run_only),
            "auto_apply_blocked": bool(self.auto_apply_blocked),
            "replay_correlation_id": self.replay_correlation_id,
            "metadata": dict(self.metadata),
            "analyst_observation": dict(self.analyst_observation),
            "critic_scores": {name: dict(payload) for name, payload in self.critic_scores.items()},
            "replay_signature": self.replay_signature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayDecisionProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            family=str(payload.get("family", "")),
            source_system=str(payload.get("source_system", "")),
            action_type=str(payload.get("action_type", "")),
            target=dict(payload.get("target") or {}),
            current_state=dict(payload.get("current_state") or {}),
            requested_state=dict(payload.get("requested_state") or {}),
            confidence=float(payload.get("confidence", 0.0)),
            safety=dict(payload.get("safety") or {}),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            auto_apply_blocked=bool(payload.get("auto_apply_blocked", False)),
            replay_correlation_id=str(payload.get("replay_correlation_id", "")),
            metadata=dict(payload.get("metadata") or {}),
            analyst_observation=dict(payload.get("analyst_observation") or {}),
            critic_scores={
                str(name): dict(result)
                for name, result in (payload.get("critic_scores") or {}).items()
            },
        )


@dataclass(frozen=True)
class ProposalRankingTraceItem:
    proposal_id: str
    family: str
    score_breakdown: ProposalScoreBreakdown
    final_rank: int
    selected: bool = False


@dataclass(frozen=True)
class ProposalRankResult:
    run_signature: str
    ranked_proposals: list[dict[str, Any]]
    ranking_trace: list[dict[str, Any]]
    selected_proposal_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_signature": self.run_signature,
            "ranked_proposals": [dict(item) for item in self.ranked_proposals],
            "ranking_trace": [dict(item) for item in self.ranking_trace],
            "selected_proposal_id": self.selected_proposal_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProposalRankResult":
        return cls(
            run_signature=str(payload.get("run_signature", "")),
            ranked_proposals=[dict(item) for item in payload.get("ranked_proposals") or ()],
            ranking_trace=[dict(item) for item in payload.get("ranking_trace") or ()],
            selected_proposal_id=payload.get("selected_proposal_id"),
        )


@dataclass(frozen=True)
class ReplayGraphCheckpoint:
    version: str
    checkpoint_id: str
    proposals: list[ReplayDecisionProposal]
    ranking: ProposalRankResult
    timeline_artifact: dict[str, Any]
    replay_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "checkpoint_id": self.checkpoint_id,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "ranking": self.ranking.to_dict(),
            "timeline_artifact": _sanitize_replay_payload(self.timeline_artifact),
            "replay_metadata": _sanitize_replay_payload(self.replay_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayGraphCheckpoint":
        return cls(
            version=str(payload.get("version", "replay_graph_checkpoint/v1")),
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            proposals=[
                ReplayDecisionProposal.from_dict(dict(item))
                for item in payload.get("proposals") or ()
            ],
            ranking=ProposalRankResult.from_dict(dict(payload.get("ranking") or {})),
            timeline_artifact=dict(payload.get("timeline_artifact") or {}),
            replay_metadata=dict(payload.get("replay_metadata") or {}),
        )


def _sanitize_replay_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key in sorted(value):
            key_str = str(key)
            if key_str in {
                "path",
                "file",
                "filename",
                "source_file",
                "output_dir",
                "input_dir",
                "host",
                "ip",
                "port",
                "url",
                "socket",
            }:
                continue
            sanitized[key_str] = _sanitize_replay_payload(value[key])
        return sanitized
    if isinstance(value, list):
        return [_sanitize_replay_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_replay_payload(item) for item in value]
    return value


def _timeline_event_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_replay_timeline_artifact(
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a replay-safe deterministic Analyzer -> Critic -> Decision timeline."""

    proposal_index = {proposal.proposal_id: proposal for proposal in proposals}
    timeline: list[dict[str, Any]] = []

    for proposal in sorted(proposals, key=lambda item: item.proposal_id):
        analyzer_payload = {
            "stage": "analyzer",
            "proposal_id": proposal.proposal_id,
            "family": proposal.family,
            "source_system": proposal.source_system,
            "action_type": proposal.action_type,
            "replay_correlation_id": proposal.replay_correlation_id,
            "replay_signature": proposal.replay_signature,
            "dry_run_only": bool(proposal.dry_run_only),
            "auto_apply_blocked": bool(proposal.auto_apply_blocked),
            "target": _sanitize_replay_payload(proposal.target),
            "current_state": _sanitize_replay_payload(proposal.current_state),
            "requested_state": _sanitize_replay_payload(proposal.requested_state),
            "safety": _sanitize_replay_payload(proposal.safety),
            "metadata": _sanitize_replay_payload(proposal.metadata),
            "analyst_observation": _sanitize_replay_payload(proposal.analyst_observation),
        }
        analyzer_event = dict(analyzer_payload)
        analyzer_event["event_signature"] = _timeline_event_signature(analyzer_payload)
        timeline.append(analyzer_event)

        for critic_name in sorted(proposal.critic_scores):
            critic_payload = {
                "stage": "critic",
                "proposal_id": proposal.proposal_id,
                "family": proposal.family,
                "replay_correlation_id": proposal.replay_correlation_id,
                "critic_name": critic_name,
                "replay_signature": proposal.replay_signature,
                "critic_result": _sanitize_replay_payload(proposal.critic_scores[critic_name]),
            }
            critic_event = dict(critic_payload)
            critic_event["event_signature"] = _timeline_event_signature(critic_payload)
            timeline.append(critic_event)

    for ranked in ranking.ranked_proposals:
        proposal_id = str(ranked.get("proposal_id", ""))
        proposal = proposal_index.get(proposal_id)
        decision_payload = {
            "stage": "decision",
            "proposal_id": proposal_id,
            "family": str(ranked.get("family", proposal.family if proposal else "unknown")),
            "rank": int(ranked.get("rank", 0)),
            "selected": bool(ranked.get("proposal_id") == ranking.selected_proposal_id),
            "replay_correlation_id": str(
                ranked.get(
                    "replay_correlation_id",
                    proposal.replay_correlation_id if proposal else "",
                )
            ),
            "replay_signature": str(ranked.get("replay_signature", proposal.replay_signature if proposal else "")),
            "confidence_weighted_score": float(ranked.get("confidence_weighted_score", 0.0)),
            "score_breakdown": _sanitize_replay_payload(ranked.get("score_breakdown", {})),
        }
        decision_event = dict(decision_payload)
        decision_event["event_signature"] = _timeline_event_signature(decision_payload)
        timeline.append(decision_event)

    numbered_timeline = [
        {
            "seq": index,
            **event,
        }
        for index, event in enumerate(timeline, start=1)
    ]

    artifact_core = {
        "schema_version": REPLAY_TRACE_SCHEMA_VERSION,
        "artifact_type": "deterministic_replay_timeline",
        "dry_run_only": True,
        "live_mixer_writes": False,
        "run_signature": ranking.run_signature,
        "selected_proposal_id": ranking.selected_proposal_id,
        "proposal_count": len(proposals),
        "timeline_event_count": len(numbered_timeline),
        "metadata": _sanitize_replay_payload(metadata or {}),
        "timeline": numbered_timeline,
    }
    artifact = dict(artifact_core)
    artifact["artifact_signature"] = _timeline_event_signature(artifact_core)
    return artifact


def write_replay_timeline_artifact(
    target: str | Path,
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    artifact = build_replay_timeline_artifact(proposals, ranking, metadata=metadata)
    output_path = Path(target).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


def build_replay_graph_checkpoint(
    proposals: list[ReplayDecisionProposal],
    ranking: ProposalRankResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> ReplayGraphCheckpoint:
    sorted_proposals = sorted(proposals, key=lambda item: item.proposal_id)
    timeline_artifact = build_replay_timeline_artifact(
        sorted_proposals,
        ranking,
        metadata=metadata,
    )
    core_payload = {
        "version": "replay_graph_checkpoint/v1",
        "proposals": [proposal.to_dict() for proposal in sorted_proposals],
        "ranking": ranking.to_dict(),
        "timeline_artifact": timeline_artifact,
        "replay_metadata": _sanitize_replay_payload(metadata or {}),
    }
    checkpoint_digest = hashlib.sha256(
        json.dumps(
            core_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReplayGraphCheckpoint(
        version="replay_graph_checkpoint/v1",
        checkpoint_id=f"replay_graph::{checkpoint_digest[:16]}",
        proposals=sorted_proposals,
        ranking=ranking,
        timeline_artifact=timeline_artifact,
        replay_metadata=_sanitize_replay_payload(metadata or {}),
    )


def compare_replay_graph_checkpoints(
    baseline: ReplayGraphCheckpoint | dict[str, Any],
    candidate: ReplayGraphCheckpoint | dict[str, Any],
) -> dict[str, Any]:
    left = (
        baseline
        if isinstance(baseline, ReplayGraphCheckpoint)
        else ReplayGraphCheckpoint.from_dict(dict(baseline))
    )
    right = (
        candidate
        if isinstance(candidate, ReplayGraphCheckpoint)
        else ReplayGraphCheckpoint.from_dict(dict(candidate))
    )
    left_ids = [proposal.proposal_id for proposal in left.proposals]
    right_ids = [proposal.proposal_id for proposal in right.proposals]
    return {
        "baseline_checkpoint_id": left.checkpoint_id,
        "candidate_checkpoint_id": right.checkpoint_id,
        "same_selected_proposal": left.ranking.selected_proposal_id == right.ranking.selected_proposal_id,
        "selected_proposal_changed": left.ranking.selected_proposal_id != right.ranking.selected_proposal_id,
        "baseline_selected_proposal_id": left.ranking.selected_proposal_id,
        "candidate_selected_proposal_id": right.ranking.selected_proposal_id,
        "same_run_signature": left.ranking.run_signature == right.ranking.run_signature,
        "same_timeline_signature": left.timeline_artifact.get("artifact_signature")
        == right.timeline_artifact.get("artifact_signature"),
        "proposal_ids_added": sorted(set(right_ids) - set(left_ids)),
        "proposal_ids_removed": sorted(set(left_ids) - set(right_ids)),
        "shared_proposal_ids": sorted(set(left_ids) & set(right_ids)),
    }


def build_gain_proposals(*, recommendations: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for rec in recommendations:
        source = str(rec.get("source", "autogain"))
        channel = int(_coerce_float(rec.get("mixer_channel", rec.get("channel", 0)), 0))
        current_db = _coerce_float(rec.get("current_trim_db", 0.0))
        target_db = _coerce_float(rec.get("recommended_target_trim_db", current_db))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"gain::{source}::{channel}::{rec.get('role', 'unknown')}::v1",
                family="gain_adjustment",
                source_system=source,
                action_type="set_gain_db",
                target={"channel": channel, "kind": "gain"},
                current_state={"gain_db": current_db},
                requested_state={"gain_db": target_db, "delta_db": target_db - current_db},
                confidence=float(rec.get("confidence", 0.5)),
                safety={
                    "max_step_db": float(rec.get("metadata", {}).get("max_step_db", 3.0)),
                    "bounded_by": str(rec.get("limited_by", "none")),
                },
                auto_apply_blocked=not bool(rec.get("dry_run_only", True)),
                replay_correlation_id=_replay_correlation_id_from_payload(rec),
                metadata={
                    "reason": rec.get("reason", "auto_gain_recommendation"),
                    "analysis_state": rec.get("analysis_state", "idle"),
                    "role": rec.get("role", "unknown"),
                },
                analyst_observation={"source_payload": dict(rec)},
            )
        )
    return proposals


def build_fader_proposals(*, fader_updates: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for rec in fader_updates:
        channel = int(_coerce_float(rec.get("mixer_channel", rec.get("channel", 0)), 0))
        current = _coerce_float(rec.get("current_fader_db", 0.0))
        target = _coerce_float(rec.get("target_fader_db", current))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"fader::{rec.get('source', 'auto_fader')}::{channel}::v1",
                family="fader_adjustment",
                source_system=str(rec.get("source", "auto_fader")),
                action_type="set_fader_db",
                target={"channel": channel, "kind": "fader"},
                current_state={"fader_db": current},
                requested_state={"fader_db": target, "delta_db": target - current},
                confidence=float(rec.get("confidence", 0.55)),
                safety={
                    "max_step_db": float(rec.get("max_step_db", 2.0)),
                    "deadband_db": float(rec.get("deadband_db", 0.0)),
                },
                auto_apply_blocked=not bool(rec.get("live_apply_safe", False)),
                replay_correlation_id=_replay_correlation_id_from_payload(rec),
                metadata={
                    "reason": rec.get("reason", "fader_level_correction"),
                    "analysis_type": rec.get("analysis_type", "auto_fader"),
                },
                analyst_observation={"source_payload": dict(rec)},
            )
        )
    return proposals


def build_speech_priority_arbitration_proposals(*, arbitration_payloads: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for payload in arbitration_payloads:
        leader = int(_coerce_float(payload.get("leader_channel", 0), 0))
        follower = int(_coerce_float(payload.get("suppressed_channel", 0), 0))
        leader_gain = _coerce_float(payload.get("leader_boost_db", 0.0))
        follower_gain = _coerce_float(payload.get("suppressed_cut_db", 0.0))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"speech_arbitration::{leader}->{follower}::{payload.get('event_id', 'anon')}",
                family="speech_arbitration",
                source_system=str(payload.get("source", "speech_priority")),
                action_type="adjust_speech_priority_balance",
                target={"leader_channel": leader, "suppressed_channel": follower},
                current_state={"priority_boost_db": 0.0, "priority_cut_db": 0.0},
                requested_state={"priority_boost_db": leader_gain, "priority_cut_db": follower_gain},
                confidence=float(payload.get("confidence", 0.6)),
                safety={
                    "max_boost_db": float(payload.get("max_boost_db", 2.0)),
                    "max_cut_db": float(payload.get("max_cut_db", 3.0)),
                },
                auto_apply_blocked=not bool(payload.get("live_apply_safe", False)),
                replay_correlation_id=_replay_correlation_id_from_payload(payload),
                metadata={"dominant_speaker": str(payload.get("dominant_speaker", "unknown"))},
                analyst_observation={"reason": payload.get("reason", "speech_clarity_preservation")},
            )
        )
    return proposals


def build_room_compensation_proposals(*, room_plans: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for plan in room_plans:
        target = str(plan.get("recommended_target", "master"))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"room_compensation::{target}::{plan.get('room_classification', 'balanced')}::{plan.get('event_id', 'anon')}",
                family="room_compensation",
                source_system="room_compensation_planner",
                action_type="plan_room_eq",
                target={"target": target, "classification": str(plan.get("room_classification", "balanced"))},
                current_state={"filters": list(plan.get("planned_filters", []) or [])},
                requested_state={
                    "filters": list(plan.get("planned_filters", []) or []),
                    "filter_count": len(plan.get("planned_filters", []) or []),
                },
                confidence=float(plan.get("confidence_score", 0.5)),
                safety={
                    "dry_run_only": bool(plan.get("dry_run_only", True)),
                    "blocked": bool(plan.get("blocked", False)),
                    "max_cut_db": float(plan.get("max_cut_db", 0.0)),
                },
                auto_apply_blocked=True,
                replay_correlation_id=_replay_correlation_id_from_payload(plan),
                metadata={
                    "quality_indicator": str(plan.get("room_quality_indicator", "fair")),
                    "quality_score": _coerce_float(plan.get("room_quality_score", 0.0)),
                    "recommendation": str(plan.get("recommended_target", "master")),
                    "notes": list(plan.get("notes", []) or []),
                },
                analyst_observation={"source_payload": dict(plan)},
            )
        )
    return proposals


def build_feedback_mitigation_proposals(*, feedback_events: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for event in feedback_events:
        channel = int(_coerce_float(event.get("channel", 0), 0))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"feedback::{event.get('action', 'notch')}::{channel}::{event.get('event_id', 'anon')}",
                family="feedback_mitigation",
                source_system=str(event.get("source", "feedback_detector")),
                action_type=str(event.get("action", "notch")),
                target={"channel": channel, "frequency_hz": _coerce_float(event.get("frequency_hz", 0.0))},
                current_state={"magnitude_db": 0.0},
                requested_state={
                    "gain_db": -abs(_coerce_float(event.get("magnitude_db", -6.0), 0.0)),
                    "q": _coerce_float(event.get("q", 4.0)),
                },
                confidence=float(event.get("confidence", 0.7)),
                safety={
                    "operator_override_required": bool(event.get("operator_override_required", False)),
                    "max_q": float(event.get("max_q", 8.0)),
                },
                auto_apply_blocked=not bool(event.get("live_apply_allowed", False)),
                replay_correlation_id=_replay_correlation_id_from_payload(event),
                metadata={"action": str(event.get("action", "notch")), "channel": channel},
                analyst_observation={"frequency_hz": _coerce_float(event.get("frequency_hz", 0.0))},
            )
        )
    return proposals


def build_operator_assistance_proposals(*, operator_recommendations: list[dict[str, Any]]) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for rec in operator_recommendations:
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"operator::{rec.get('type', 'assist')}::{rec.get('rec_id', 'anon')}",
                family="operator_assistance",
                source_system=str(rec.get("source", "operator_advice")),
                action_type="operator_prompt",
                target={"target": str(rec.get("target", "mix"))},
                current_state={},
                requested_state={"message": str(rec.get("message", ""))},
                confidence=float(rec.get("confidence", 0.4)),
                safety={"requires_operator": True, "requires_confirmation": True},
                auto_apply_blocked=True,
                replay_correlation_id=_replay_correlation_id_from_payload(rec),
                metadata={"rec_type": str(rec.get("type", "operator_assistance")), "urgency": rec.get("urgency", "low")},
                critic_scores={},
            )
        )
    return proposals


def build_performer_stage_interaction_proposals(
    *,
    interaction_events: list[dict[str, Any]],
) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    for event in interaction_events:
        performer_id = str(event.get("performer_id", "performer_unknown"))
        target_channel = int(_coerce_float(event.get("mixer_channel", event.get("channel", 0)), 0))
        zone_from = str(event.get("zone_from", "center"))
        zone_to = str(event.get("zone_to", zone_from))
        orientation_deg = float(event.get("orientation_deg", 0.0))
        distance_m = float(event.get("distance_m", 2.5))
        proximity_delta_m = float(event.get("proximity_delta_m", 0.0))
        level_delta_db = float(event.get("movement_level_delta_db", 0.0))
        spill_delta_db = float(event.get("movement_spill_delta_db", 0.0))
        proposals.append(
            ReplayDecisionProposal(
                proposal_id=f"performer_stage_interaction::{performer_id}::{target_channel}::{event.get('event_id', 'anon')}",
                family="performer_stage_interaction",
                source_system=str(event.get("source", "stage_interaction_model")),
                action_type="simulate_stage_interaction",
                target={
                    "performer_id": performer_id,
                    "channel": target_channel,
                    "zone_from": zone_from,
                    "zone_to": zone_to,
                },
                current_state={
                    "zone": zone_from,
                    "orientation_deg": orientation_deg,
                    "distance_m": distance_m,
                    "proximity_delta_m": proximity_delta_m,
                    "level_delta_db": 0.0,
                    "spill_delta_db": 0.0,
                },
                requested_state={
                    "zone": zone_to,
                    "orientation_deg": orientation_deg + float(event.get("orientation_change_deg", 0.0)),
                    "distance_m": distance_m + proximity_delta_m,
                    "proximity_delta_m": proximity_delta_m,
                    "level_delta_db": level_delta_db,
                    "spill_delta_db": spill_delta_db,
                    "risk": str(event.get("risk_profile", "moderate")),
                },
                confidence=float(event.get("confidence", 0.6)),
                safety={
                    "max_level_delta_db": abs(float(event.get("max_level_delta_db", 2.5))),
                    "max_spill_delta_db": abs(float(event.get("max_spill_delta_db", 2.0))),
                    "requires_operator_review": bool(event.get("requires_operator_review", True)),
                },
                auto_apply_blocked=True,
                replay_correlation_id=_replay_correlation_id_from_payload(event),
                metadata={
                    "event_type": str(event.get("event_type", "movement")),
                    "song_context": str(event.get("song_context", "live")),
                    "audience_risk": str(event.get("audience_risk", "unknown")),
                },
                analyst_observation={
                    "source_payload": dict(event),
                    "movement_model": "stage_activity_simulation",
                },
            )
        )
    return proposals


def build_replay_scores(proposal: ReplayDecisionProposal, config: dict[str, Any] | None = None) -> ProposalScoreBreakdown:
    score_config = configured_weights(config)
    aggregated = aggregate_scores(proposal.critic_scores, score_config)
    base_score = float(aggregated.get("final_score", 0.0))
    confidence = float(proposal.confidence)
    safety_penalty = 1.0
    if proposal.auto_apply_blocked or bool(proposal.safety.get("blocked", False)):
        safety_penalty *= 0.9
    if proposal.safety.get("operator_override_required"):
        safety_penalty *= 0.95
    confidence_weighted = base_score * confidence * safety_penalty
    return ProposalScoreBreakdown(
        final_score=base_score,
        confidence=confidence,
        confidence_weighted_score=confidence_weighted,
        normalized_weights=dict(aggregated.get("normalized_weights", {})),
        critic_values=dict(aggregated.get("values", {})),
        critic_breakdown=aggregated,
    )


def rank_replay_proposals(
    proposals: list[ReplayDecisionProposal],
    *,
    config: dict[str, Any] | None = None,
    family_priority: dict[str, float] | None = None,
) -> ProposalRankResult:
    family_priority = dict(family_priority or {})
    for family in PROPOSAL_FAMILIES:
        family_priority.setdefault(family, 0.0)

    scored_rows: list[tuple[ReplayDecisionProposal, ProposalScoreBreakdown]] = []
    for proposal in proposals:
        scored_rows.append((proposal, build_replay_scores(proposal, config=config)))

    def sort_key(item: tuple[ReplayDecisionProposal, ProposalScoreBreakdown]) -> tuple[float, float, float, str]:
        proposal, score = item
        # Deterministic score priority: confidence-weighted score + family priority + ID.
        return (
            score.confidence_weighted_score + family_priority.get(proposal.family, 0.0),
            score.confidence,
            score.final_score,
            proposal.proposal_id,
        )

    ranked = sorted(scored_rows, key=sort_key, reverse=True)

    ranked_payload: list[dict[str, Any]] = []
    ranking_trace: list[dict[str, Any]] = []
    for index, (proposal, score) in enumerate(ranked, start=1):
        proposal_payload = proposal.to_dict()
        proposal_payload.update(
            {
                "rank": index,
                "confidence_weighted_score": score.confidence_weighted_score,
                "score_breakdown": score.to_dict(),
            }
        )
        ranked_payload.append(proposal_payload)
        ranking_trace.append(
            {
                "proposal_id": proposal.proposal_id,
                "family": proposal.family,
                "replay_correlation_id": proposal.replay_correlation_id,
                "selected": index == 1,
                "final_rank": index,
                "score_breakdown": score.to_dict(),
            }
        )

    selected = ranked_payload[0]["proposal_id"] if ranked_payload else None

    run_signature = hashlib.sha256(
        json.dumps(
            {
                "count": len(ranked_payload),
                "run_input_signature": _coerce_float(len(ranked_payload)),
                "ranked": [(item["proposal_id"], item["confidence_weighted_score"]) for item in ranked_payload],
            },
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    return ProposalRankResult(
        run_signature=run_signature,
        ranked_proposals=ranked_payload,
        ranking_trace=ranking_trace,
        selected_proposal_id=selected,
    )
