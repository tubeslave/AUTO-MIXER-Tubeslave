"""Dry-run-only soundcheck recommendation workflow for operator proposals.

This compatibility module builds reviewable recommendation bundles without a
mixer client and without any live write path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SoundcheckRecommendation:
    recommendation_id: str
    kind: str
    action_type: str
    mode: str
    channel: int
    target: Dict[str, Any]
    current_state: Dict[str, Any]
    requested_state: Dict[str, Any]
    confidence: float
    rationale: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        action_envelope = {
            "operator_summary": self._operator_summary(),
            "live_mixer_writes": False,
            "transport_policy": "dry_run_only",
        }
        return {
            "recommendation_id": self.recommendation_id,
            "kind": self.kind,
            "action_type": self.action_type,
            "mode": self.mode,
            "channel": self.channel,
            "target": dict(self.target),
            "current_state": dict(self.current_state),
            "requested_state": dict(self.requested_state),
            "confidence": self.confidence,
            "rationale": dict(self.rationale or {}),
            "metadata": dict(self.metadata or {}),
            "action_envelope": action_envelope,
            "safety": {
                "auto_apply_blocked": True,
                "live_mixer_writes": False,
                "transport_policy": "dry_run_only",
            },
        }

    def _operator_summary(self) -> str:
        if self.kind == "input_gain":
            target = self.requested_state.get("trim_db")
            return f"Review input trim recommendation for channel {self.channel}: {target} dB."
        if self.kind == "fader":
            target = self.requested_state.get("fader_db")
            return f"Review fader recommendation for channel {self.channel}: {target} dB."
        return f"Review soundcheck recommendation for channel {self.channel}."


class NoWriteSoundcheckRecommendationWorkflow:
    """Build replay-safe soundcheck recommendations with no live write path."""

    schema_version = "soundcheck_recommendation_bundle/v1"

    def __init__(
        self,
        *,
        source_system: str = "auto_soundcheck_engine",
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source_system = str(source_system or "auto_soundcheck_engine")
        self.context_metadata = dict(context_metadata or {})
        self._sequence = count(1)
        self._recommendations: List[SoundcheckRecommendation] = []

    def list_recommendations(self) -> List[Dict[str, Any]]:
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
            action_type="set_input_trim_db",
            channel=channel,
            current_state={"trim_db": float(current_trim_db)},
            requested_state={"trim_db": float(target_trim_db)},
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
            action_type="set_fader_db",
            channel=channel,
            current_state={"fader_db": float(current_fader_db)},
            requested_state={"fader_db": float(target_fader_db)},
            confidence=confidence,
            metadata=metadata,
            rationale=rationale,
        )

    def build_bundle(self) -> Dict[str, Any]:
        recommendations = self.list_recommendations()
        return {
            "schema_version": self.schema_version,
            "generated_at": _utc_now(),
            "source_system": self.source_system,
            "context_metadata": dict(self.context_metadata),
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "proposals": recommendations,
            "live_mixer_writes": False,
            "transport_policy": "dry_run_only",
        }

    def _add_recommendation(
        self,
        *,
        kind: str,
        action_type: str,
        channel: int,
        current_state: Dict[str, Any],
        requested_state: Dict[str, Any],
        confidence: float,
        metadata: Optional[Dict[str, Any]],
        rationale: Optional[Dict[str, Any]],
    ) -> SoundcheckRecommendation:
        recommendation = SoundcheckRecommendation(
            recommendation_id=f"soundcheck:{next(self._sequence)}",
            kind=kind,
            action_type=action_type,
            mode="approval_required",
            channel=int(channel),
            target={"mixer_channel": int(channel)},
            current_state=dict(current_state),
            requested_state=dict(requested_state),
            confidence=float(confidence),
            metadata=metadata,
            rationale=rationale,
        )
        self._recommendations.append(recommendation)
        return recommendation
