"""Dry-run-only soundcheck recommendation workflow extracted from AutoSoundcheckEngine.

This module intentionally does not accept a mixer client and cannot write to a
console. It converts soundcheck findings into replay-safe recommendation
artifacts that can be reviewed, ranked, and archived without touching WING.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Dict, Iterable, Optional

from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    ReplayDecisionProposal,
    build_replay_graph_checkpoint,
    build_replay_timeline_artifact,
    rank_replay_proposals,
)
from soundcheck_autonomy_contract import (
    AutonomyMode,
    ReviewMode as RecommendationMode,
    build_action_envelope,
)


def _compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}

KIND_TO_MODE: Dict[str, RecommendationMode] = {
    "input_gain": RecommendationMode.APPROVAL_REQUIRED,
    "hpf": RecommendationMode.ASSISTED,
    "eq": RecommendationMode.ASSISTED,
    "compressor": RecommendationMode.ASSISTED,
    "pan": RecommendationMode.ASSISTED,
    "fader": RecommendationMode.APPROVAL_REQUIRED,
    "fx_send": RecommendationMode.ASSISTED,
    "feedback_notch": RecommendationMode.OBSERVE_ONLY,
    "feedback_fader_reduce": RecommendationMode.OBSERVE_ONLY,
}

KIND_TO_FAMILY: Dict[str, str] = {
    "input_gain": "gain_adjustment",
    "hpf": "eq_adjustment",
    "eq": "eq_adjustment",
    "compressor": "compression",
    "pan": "pan_adjustment",
    "fader": "fader_adjustment",
    "fx_send": "fx_send_adjustment",
    "feedback_notch": "feedback_mitigation",
    "feedback_fader_reduce": "feedback_mitigation",
}

KIND_TO_ACTION: Dict[str, str] = {
    "input_gain": "set_input_trim_db",
    "hpf": "set_hpf",
    "eq": "set_eq_bands",
    "compressor": "set_compressor",
    "pan": "set_pan",
    "fader": "set_fader_db",
    "fx_send": "set_send_level",
    "feedback_notch": "feedback_notch_candidate",
    "feedback_fader_reduce": "feedback_fader_reduce_candidate",
}


@dataclass(frozen=True)
class SoundcheckRecommendation:
    recommendation_id: str
    kind: str
    family: str
    action_type: str
    mode: RecommendationMode
    autonomy_mode: AutonomyMode
    channel: int
    target: Dict[str, Any]
    current_state: Dict[str, Any]
    requested_state: Dict[str, Any]
    source_payload: Dict[str, Any]
    action_envelope: Dict[str, Any]
    safety: Dict[str, Any]
    confidence: float
    replay_correlation_id: str

    def to_proposal(self, *, source_system: str) -> ReplayDecisionProposal:
        return ReplayDecisionProposal(
            proposal_id=self.recommendation_id,
            family=self.family,
            source_system=source_system,
            action_type=self.action_type,
            target=dict(self.target),
            current_state=dict(self.current_state),
            requested_state=dict(self.requested_state),
            source_payload=dict(self.source_payload),
            confidence=float(self.confidence),
            safety=dict(self.safety),
            auto_apply_blocked=True,
            replay_correlation_id=self.replay_correlation_id,
            transport_policy="dry_run_only",
            live_mixer_writes=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "kind": self.kind,
            "family": self.family,
            "action_type": self.action_type,
            "mode": self.mode.value,
            "autonomy_mode": self.autonomy_mode.value,
            "channel": self.channel,
            "target": dict(self.target),
            "current_state": dict(self.current_state),
            "requested_state": dict(self.requested_state),
            "source_payload": dict(self.source_payload),
            "action_envelope": dict(self.action_envelope),
            "safety": dict(self.safety),
            "confidence": self.confidence,
            "replay_correlation_id": self.replay_correlation_id,
        }


class NoWriteSoundcheckRecommendationWorkflow:
    """Build replay-safe soundcheck recommendations with no live write path."""

    schema_version = "soundcheck_recommendation_bundle/v1"

    def __init__(
        self,
        *,
        source_system: str = "auto_soundcheck_engine",
        context_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source_system = str(source_system or "auto_soundcheck_engine")
        self.context_metadata = dict(context_metadata or {})
        self._sequence = count(1)
        self._recommendations: list[SoundcheckRecommendation] = []

    def list_recommendations(self) -> list[Dict[str, Any]]:
        return [recommendation.to_dict() for recommendation in self._recommendations]

    def recommend_input_gain(
        self,
        *,
        channel: int,
        current_trim_db: float,
        target_trim_db: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="input_gain",
            channel=channel,
            target={"mixer_channel": int(channel)},
            current_state={"trim_db": float(current_trim_db)},
            requested_state={"trim_db": float(target_trim_db)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_hpf(
        self,
        *,
        channel: int,
        current_freq_hz: Optional[float],
        current_enabled: Optional[bool],
        target_freq_hz: float,
        target_enabled: bool = True,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="hpf",
            channel=channel,
            target={"mixer_channel": int(channel), "filter_type": "hpf"},
            current_state=_compact(
                {
                    "enabled": current_enabled,
                    "freq_hz": float(current_freq_hz) if current_freq_hz is not None else None,
                }
            ),
            requested_state={
                "enabled": bool(target_enabled),
                "freq_hz": float(target_freq_hz),
            },
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_eq(
        self,
        *,
        channel: int,
        current_bands: Iterable[Dict[str, Any]],
        target_bands: Iterable[Dict[str, Any]],
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="eq",
            channel=channel,
            target={"mixer_channel": int(channel), "filter_type": "peq"},
            current_state={"bands": [dict(band) for band in current_bands]},
            requested_state={"bands": [dict(band) for band in target_bands]},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_compressor(
        self,
        *,
        channel: int,
        current_settings: Dict[str, Any],
        target_settings: Dict[str, Any],
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="compressor",
            channel=channel,
            target={"mixer_channel": int(channel), "processor": "compressor"},
            current_state=dict(current_settings),
            requested_state=dict(target_settings),
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_pan(
        self,
        *,
        channel: int,
        current_pan: float,
        target_pan: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="pan",
            channel=channel,
            target={"mixer_channel": int(channel)},
            current_state={"pan": float(current_pan)},
            requested_state={"pan": float(target_pan)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_fader(
        self,
        *,
        channel: int,
        current_fader_db: float,
        target_fader_db: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="fader",
            channel=channel,
            target={"mixer_channel": int(channel)},
            current_state={"fader_db": float(current_fader_db)},
            requested_state={"fader_db": float(target_fader_db)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_fx_send(
        self,
        *,
        channel: int,
        bus: int,
        current_level_db: Optional[float],
        target_level_db: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="fx_send",
            channel=channel,
            target={"mixer_channel": int(channel), "bus": int(bus)},
            current_state=_compact(
                {
                    "send_level_db": (
                        float(current_level_db) if current_level_db is not None else None
                    )
                }
            ),
            requested_state={"send_level_db": float(target_level_db)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_feedback_notch(
        self,
        *,
        channel: int,
        frequency_hz: float,
        gain_db: float = -6.0,
        q: float = 10.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="feedback_notch",
            channel=channel,
            target={"mixer_channel": int(channel), "processor": "eq", "band_role": "notch"},
            current_state={},
            requested_state={
                "frequency_hz": float(frequency_hz),
                "gain_db": float(gain_db),
                "q": float(q),
            },
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def recommend_feedback_fader_reduce(
        self,
        *,
        channel: int,
        current_fader_db: float,
        target_fader_db: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        rationale: Optional[Dict[str, Any]] = None,
    ) -> SoundcheckRecommendation:
        return self._add_recommendation(
            kind="feedback_fader_reduce",
            channel=channel,
            target={"mixer_channel": int(channel)},
            current_state={"fader_db": float(current_fader_db)},
            requested_state={"fader_db": float(target_fader_db)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def build_bundle(self) -> Dict[str, Any]:
        recommendations = list(self._recommendations)
        proposals = [
            recommendation.to_proposal(source_system=self.source_system)
            for recommendation in recommendations
        ]
        metadata = _compact(
            {
                "workflow": "soundcheck_no_write",
                "source_system": self.source_system,
                "recommendation_count": len(recommendations),
                **self.context_metadata,
            }
        )
        mode_summary = {
            mode.value: sum(
                1 for recommendation in recommendations if recommendation.mode == mode
            )
            for mode in RecommendationMode
        }
        autonomy_summary = {
            mode.value: sum(
                1 for recommendation in recommendations if recommendation.autonomy_mode == mode
            )
            for mode in AutonomyMode
        }
        mode_by_kind = {
            recommendation.kind: recommendation.mode.value
            for recommendation in recommendations
        }
        autonomy_by_kind = {
            recommendation.kind: recommendation.autonomy_mode.value
            for recommendation in recommendations
        }

        ranking = None
        timeline_artifact = None
        graph_checkpoint = None
        if proposals:
            ranking = rank_replay_proposals(proposals)
            timeline_artifact = build_replay_timeline_artifact(
                proposals,
                ranking,
                metadata=metadata,
            )
            graph_checkpoint = build_replay_graph_checkpoint(
                proposals,
                ranking,
                metadata=metadata,
            )

        return {
            "schema_version": self.schema_version,
            "workflow": "soundcheck_no_write",
            "source_system": self.source_system,
            "dry_run_only": True,
            "live_mixer_writes": False,
            "context_metadata": metadata,
            "recommendations": [recommendation.to_dict() for recommendation in recommendations],
            "proposals": [proposal.to_dict() for proposal in proposals],
            "ranking": ranking.to_dict() if ranking else None,
            "timeline_artifact": timeline_artifact,
            "graph_checkpoint": graph_checkpoint.to_dict() if graph_checkpoint else None,
            "mode_summary": mode_summary,
            "autonomy_summary": autonomy_summary,
            "mode_by_kind": mode_by_kind,
            "autonomy_by_kind": autonomy_by_kind,
        }

    def _add_recommendation(
        self,
        *,
        kind: str,
        channel: int,
        target: Dict[str, Any],
        current_state: Dict[str, Any],
        requested_state: Dict[str, Any],
        confidence: float,
        metadata: Optional[Dict[str, Any]],
        rationale: Optional[Dict[str, Any]],
    ) -> SoundcheckRecommendation:
        if kind not in KIND_TO_MODE:
            raise ValueError(f"Unsupported soundcheck recommendation kind: {kind}")

        mode = KIND_TO_MODE[kind]
        family = KIND_TO_FAMILY[kind]
        action_type = KIND_TO_ACTION[kind]
        replay_correlation_id = (
            f"corr::{self.source_system}::{kind}::ch{int(channel)}::{next(self._sequence)}"
        )
        requires_confirmation = mode == RecommendationMode.APPROVAL_REQUIRED
        envelope_id = f"{kind}::{replay_correlation_id}"
        action_envelope = build_action_envelope(
            envelope_id=envelope_id,
            source_system=self.source_system,
            kind=kind,
            family=family,
            action_type=action_type,
            target=target,
            current_state=current_state,
            requested_state=requested_state,
            confidence=confidence,
            replay_correlation_id=replay_correlation_id,
            context_metadata=self.context_metadata,
            metadata=metadata,
            rationale=rationale,
            audit={"workflow": "soundcheck_no_write", **dict(self.context_metadata)},
        )
        safety = {
            "console_family": "wing",
            "workflow": "soundcheck_no_write",
            "recommendation_mode": mode.value,
            "autonomy_mode": action_envelope["autonomy_mode"],
            "dry_run_only": True,
            "live_mixer_writes": False,
            "write_intent_only": True,
            "requires_operator": True,
            "requires_confirmation": requires_confirmation,
            "required_approval_level": action_envelope["required_approval_level"],
            "manual_review_only": mode == RecommendationMode.OBSERVE_ONLY,
        }
        payload = _compact(
            {
                "source": self.source_system,
                "recommendation_kind": kind,
                "mode": mode.value,
                "autonomy_mode": action_envelope["autonomy_mode"],
                "mixer_channel": int(channel),
                "confidence": float(confidence),
                "replay_correlation_id": replay_correlation_id,
                "requires_confirmation": requires_confirmation,
                "requested_live_write": False,
                "live_apply": False,
                "auto_apply_blocked": True,
                "transport_policy": "dry_run_only",
                "current_state": dict(current_state),
                "requested_state": dict(requested_state),
                "rationale": dict(rationale or {}),
                "metadata": dict(metadata or {}),
                "risk": dict(action_envelope["risk"]),
                "trust_gates": dict(action_envelope["trust_gates"]),
                "required_approval_level": action_envelope["required_approval_level"],
                "rollback_handle": dict(action_envelope["rollback_handle"]),
                "operator_summary": action_envelope["operator_summary"],
                "operator_explanation": action_envelope["operator_explanation"],
                "action_envelope": action_envelope,
                "safety": safety,
            }
        )
        recommendation = SoundcheckRecommendation(
            recommendation_id=envelope_id,
            kind=kind,
            family=family,
            action_type=action_type,
            mode=mode,
            autonomy_mode=AutonomyMode(action_envelope["autonomy_mode"]),
            channel=int(channel),
            target=dict(target),
            current_state=dict(current_state),
            requested_state=dict(requested_state),
            source_payload=payload,
            action_envelope=action_envelope,
            safety=safety,
            confidence=float(confidence),
            replay_correlation_id=replay_correlation_id,
        )
        self._recommendations.append(recommendation)
        return recommendation
