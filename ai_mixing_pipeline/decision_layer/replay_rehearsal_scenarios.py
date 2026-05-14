"""Replay-only rehearsal scenario schema and deterministic timeline fixtures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


REPLAY_REHEARSAL_SCHEMA_VERSION = "replay_rehearsal_scenario/v1"
REPLAY_REHEARSAL_TIMELINE_SCHEMA_VERSION = "replay_rehearsal_timeline/v1"


def _sanitize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _sanitize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _sanitize(payload),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class RehearsalStage(str, Enum):
    PRE_SHOW = "pre_show"
    SOUNDCHECK = "soundcheck"
    REPLAY_VALIDATION = "replay_validation"
    SHADOW_MODE = "shadow_mode"
    OPERATOR_REVIEW = "operator_review"
    ROLLBACK = "rollback"
    ESCALATION = "escalation"
    FALLBACK = "fallback"
    POST_SHOW_REVIEW = "post_show_review"


@dataclass(frozen=True)
class TelemetryExpectation:
    metric: str
    comparator: str
    expected: float | str | bool
    tolerance: float = 0.0
    unit: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(
            {
                "metric": self.metric,
                "comparator": self.comparator,
                "expected": self.expected,
                "tolerance": self.tolerance,
                "unit": self.unit,
                "notes": self.notes,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TelemetryExpectation":
        return cls(
            metric=str(payload.get("metric", "")),
            comparator=str(payload.get("comparator", "eq")),
            expected=payload.get("expected"),
            tolerance=float(payload.get("tolerance", 0.0)),
            unit=str(payload.get("unit", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class VenueContext:
    venue_id: str
    venue_name: str
    room_profile: str
    capacity: int
    timezone: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(
            {
                "venue_id": self.venue_id,
                "venue_name": self.venue_name,
                "room_profile": self.room_profile,
                "capacity": self.capacity,
                "timezone": self.timezone,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VenueContext":
        return cls(
            venue_id=str(payload.get("venue_id", "")),
            venue_name=str(payload.get("venue_name", "")),
            room_profile=str(payload.get("room_profile", "")),
            capacity=int(payload.get("capacity", 0)),
            timezone=str(payload.get("timezone", "UTC")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    show_id: str
    artist: str
    rehearsal_date: str
    dry_run_only: bool = True
    live_mixer_writes: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(
            {
                "session_id": self.session_id,
                "show_id": self.show_id,
                "artist": self.artist,
                "rehearsal_date": self.rehearsal_date,
                "dry_run_only": self.dry_run_only,
                "live_mixer_writes": self.live_mixer_writes,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionContext":
        return cls(
            session_id=str(payload.get("session_id", "")),
            show_id=str(payload.get("show_id", "")),
            artist=str(payload.get("artist", "")),
            rehearsal_date=str(payload.get("rehearsal_date", "")),
            dry_run_only=bool(payload.get("dry_run_only", True)),
            live_mixer_writes=bool(payload.get("live_mixer_writes", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class OperatorContext:
    operator_id: str
    display_name: str
    role: str
    approval_scope: str
    experience_tier: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(
            {
                "operator_id": self.operator_id,
                "display_name": self.display_name,
                "role": self.role,
                "approval_scope": self.approval_scope,
                "experience_tier": self.experience_tier,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorContext":
        return cls(
            operator_id=str(payload.get("operator_id", "")),
            display_name=str(payload.get("display_name", "")),
            role=str(payload.get("role", "")),
            approval_scope=str(payload.get("approval_scope", "")),
            experience_tier=str(payload.get("experience_tier", "")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RehearsalStageSpec:
    stage: RehearsalStage
    ordinal: int
    label: str
    goals: list[str] = field(default_factory=list)
    replay_correlation_ids: list[str] = field(default_factory=list)
    telemetry_expectations: list[TelemetryExpectation] = field(default_factory=list)
    required_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def base_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ordinal": self.ordinal,
            "label": self.label,
            "goals": list(self.goals),
            "replay_correlation_ids": sorted(
                {str(item).strip() for item in self.replay_correlation_ids if str(item).strip()}
            ),
            "telemetry_expectations": [item.to_dict() for item in self.telemetry_expectations],
            "required_artifacts": sorted(str(item) for item in self.required_artifacts),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreShowStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.PRE_SHOW, init=False)
    safety_brief_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["safety_brief_version"] = self.safety_brief_version
        return _sanitize(payload)


@dataclass(frozen=True)
class SoundcheckStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.SOUNDCHECK, init=False)
    channel_inventory_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["channel_inventory_ref"] = self.channel_inventory_ref
        return _sanitize(payload)


@dataclass(frozen=True)
class ReplayValidationStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.REPLAY_VALIDATION, init=False)
    validator_schema_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["validator_schema_version"] = self.validator_schema_version
        return _sanitize(payload)


@dataclass(frozen=True)
class ShadowModeStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.SHADOW_MODE, init=False)
    comparison_window_sec: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["comparison_window_sec"] = int(self.comparison_window_sec)
        return _sanitize(payload)


@dataclass(frozen=True)
class OperatorReviewStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.OPERATOR_REVIEW, init=False)
    approval_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["approval_required"] = bool(self.approval_required)
        return _sanitize(payload)


@dataclass(frozen=True)
class RollbackStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.ROLLBACK, init=False)
    rollback_target_manifest_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["rollback_target_manifest_id"] = self.rollback_target_manifest_id
        return _sanitize(payload)


@dataclass(frozen=True)
class EscalationStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.ESCALATION, init=False)
    escalation_policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["escalation_policy"] = self.escalation_policy
        return _sanitize(payload)


@dataclass(frozen=True)
class FallbackStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.FALLBACK, init=False)
    fallback_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["fallback_mode"] = self.fallback_mode
        return _sanitize(payload)


@dataclass(frozen=True)
class PostShowReviewStage(RehearsalStageSpec):
    stage: RehearsalStage = field(default=RehearsalStage.POST_SHOW_REVIEW, init=False)
    report_schema_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_dict()
        payload["report_schema_version"] = self.report_schema_version
        return _sanitize(payload)


_STAGE_TYPES: dict[RehearsalStage, type[RehearsalStageSpec]] = {
    RehearsalStage.PRE_SHOW: PreShowStage,
    RehearsalStage.SOUNDCHECK: SoundcheckStage,
    RehearsalStage.REPLAY_VALIDATION: ReplayValidationStage,
    RehearsalStage.SHADOW_MODE: ShadowModeStage,
    RehearsalStage.OPERATOR_REVIEW: OperatorReviewStage,
    RehearsalStage.ROLLBACK: RollbackStage,
    RehearsalStage.ESCALATION: EscalationStage,
    RehearsalStage.FALLBACK: FallbackStage,
    RehearsalStage.POST_SHOW_REVIEW: PostShowReviewStage,
}


def _stage_from_dict(payload: dict[str, Any]) -> RehearsalStageSpec:
    stage = RehearsalStage(str(payload.get("stage", "")))
    common = {
        "ordinal": int(payload.get("ordinal", 0)),
        "label": str(payload.get("label", stage.value)),
        "goals": [str(item) for item in payload.get("goals") or ()],
        "replay_correlation_ids": [str(item) for item in payload.get("replay_correlation_ids") or ()],
        "telemetry_expectations": [
            TelemetryExpectation.from_dict(dict(item))
            for item in payload.get("telemetry_expectations") or ()
        ],
        "required_artifacts": [str(item) for item in payload.get("required_artifacts") or ()],
        "metadata": dict(payload.get("metadata") or {}),
    }
    if stage is RehearsalStage.PRE_SHOW:
        return PreShowStage(
            **common,
            safety_brief_version=str(payload.get("safety_brief_version", "")),
        )
    if stage is RehearsalStage.SOUNDCHECK:
        return SoundcheckStage(
            **common,
            channel_inventory_ref=str(payload.get("channel_inventory_ref", "")),
        )
    if stage is RehearsalStage.REPLAY_VALIDATION:
        return ReplayValidationStage(
            **common,
            validator_schema_version=str(payload.get("validator_schema_version", "")),
        )
    if stage is RehearsalStage.SHADOW_MODE:
        return ShadowModeStage(
            **common,
            comparison_window_sec=int(payload.get("comparison_window_sec", 0)),
        )
    if stage is RehearsalStage.OPERATOR_REVIEW:
        return OperatorReviewStage(
            **common,
            approval_required=bool(payload.get("approval_required", True)),
        )
    if stage is RehearsalStage.ROLLBACK:
        return RollbackStage(
            **common,
            rollback_target_manifest_id=str(payload.get("rollback_target_manifest_id", "")),
        )
    if stage is RehearsalStage.ESCALATION:
        return EscalationStage(
            **common,
            escalation_policy=str(payload.get("escalation_policy", "")),
        )
    if stage is RehearsalStage.FALLBACK:
        return FallbackStage(
            **common,
            fallback_mode=str(payload.get("fallback_mode", "")),
        )
    return PostShowReviewStage(
        **common,
        report_schema_version=str(payload.get("report_schema_version", "")),
    )


@dataclass(frozen=True)
class ReplayRehearsalScenario:
    scenario_id: str
    venue: VenueContext
    session: SessionContext
    operator: OperatorContext
    stages: list[RehearsalStageSpec]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = REPLAY_REHEARSAL_SCHEMA_VERSION

    @property
    def replay_correlation_ids(self) -> list[str]:
        correlation_ids: set[str] = set()
        for stage in self.stages:
            correlation_ids.update(stage.base_dict()["replay_correlation_ids"])
        return sorted(correlation_ids)

    @property
    def scenario_signature(self) -> str:
        payload = self.to_dict()
        payload.pop("scenario_signature", None)
        return _hash_payload(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "venue": self.venue.to_dict(),
            "session": self.session.to_dict(),
            "operator": self.operator.to_dict(),
            "stages": [stage.to_dict() for stage in sorted(self.stages, key=lambda item: item.ordinal)],
            "metadata": _sanitize(self.metadata),
            "replay_correlation_ids": self.replay_correlation_ids,
        }
        payload["scenario_signature"] = _hash_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayRehearsalScenario":
        stages = [_stage_from_dict(dict(item)) for item in payload.get("stages") or ()]
        stages.sort(key=lambda item: item.ordinal)
        return cls(
            scenario_id=str(payload.get("scenario_id", "")),
            venue=VenueContext.from_dict(dict(payload.get("venue") or {})),
            session=SessionContext.from_dict(dict(payload.get("session") or {})),
            operator=OperatorContext.from_dict(dict(payload.get("operator") or {})),
            stages=stages,
            metadata=dict(payload.get("metadata") or {}),
            schema_version=str(payload.get("schema_version", REPLAY_REHEARSAL_SCHEMA_VERSION)),
        )


def normalize_replay_rehearsal_scenario(
    payload: ReplayRehearsalScenario | dict[str, Any],
) -> ReplayRehearsalScenario:
    scenario = (
        payload
        if isinstance(payload, ReplayRehearsalScenario)
        else ReplayRehearsalScenario.from_dict(dict(payload))
    )
    normalized_stages = [
        _stage_from_dict(stage.to_dict())
        for stage in sorted(scenario.stages, key=lambda item: item.ordinal)
    ]
    return ReplayRehearsalScenario(
        scenario_id=scenario.scenario_id,
        venue=VenueContext.from_dict(scenario.venue.to_dict()),
        session=SessionContext.from_dict(scenario.session.to_dict()),
        operator=OperatorContext.from_dict(scenario.operator.to_dict()),
        stages=normalized_stages,
        metadata=_sanitize(scenario.metadata),
        schema_version=scenario.schema_version,
    )


def build_replay_rehearsal_timeline(
    payload: ReplayRehearsalScenario | dict[str, Any],
) -> dict[str, Any]:
    scenario = normalize_replay_rehearsal_scenario(payload)
    events: list[dict[str, Any]] = []
    for stage in scenario.stages:
        event_core = {
            "stage": stage.stage.value,
            "ordinal": int(stage.ordinal),
            "label": stage.label,
            "replay_correlation_ids": stage.base_dict()["replay_correlation_ids"],
            "required_artifacts": stage.base_dict()["required_artifacts"],
            "telemetry_expectations": [item.to_dict() for item in stage.telemetry_expectations],
            "metadata": _sanitize(stage.metadata),
            "scenario_id": scenario.scenario_id,
            "scenario_signature": scenario.scenario_signature,
            "dry_run_only": True,
            "live_mixer_writes": False,
        }
        event = dict(event_core)
        event["event_signature"] = _hash_payload(event_core)
        events.append(event)

    numbered = [
        {
            "seq": index,
            **event,
        }
        for index, event in enumerate(events, start=1)
    ]
    artifact_core = {
        "schema_version": REPLAY_REHEARSAL_TIMELINE_SCHEMA_VERSION,
        "artifact_type": "replay_rehearsal_timeline",
        "scenario_id": scenario.scenario_id,
        "scenario_signature": scenario.scenario_signature,
        "dry_run_only": True,
        "live_mixer_writes": False,
        "replay_correlation_ids": scenario.replay_correlation_ids,
        "timeline_event_count": len(numbered),
        "timeline": numbered,
    }
    artifact = dict(artifact_core)
    artifact["artifact_signature"] = _hash_payload(artifact_core)
    return artifact
