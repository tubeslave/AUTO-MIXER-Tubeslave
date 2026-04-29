"""
Evaluation and rollback helpers for AutoFOH actions.

The current system observes raw input channels, not the post-console mix bus,
so acoustic-effect evaluation is only available in explicit proxy/testing mode.
Default behavior verifies control-state changes and logs the observability gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from autofoh_detectors import INDEX_ATTR_LOOKUP
from autofoh_models import AnalysisFeatures, RuntimeState
from autofoh_safety import (
    ChannelEQMove,
    ChannelFaderMove,
    TypedCorrectionAction,
)


@dataclass
class AutoFOHEvaluationPolicy:
    enabled: bool = True
    evaluation_window_sec: float = 2.0
    allow_proxy_audio_evaluation_for_testing: bool = False
    allow_proxy_audio_rollback_for_testing: bool = False
    min_band_improvement_db: float = 0.25
    min_rms_response_db: float = 0.1
    worsening_tolerance_db: float = 0.5

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "AutoFOHEvaluationPolicy":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", True)),
            evaluation_window_sec=float(config.get("evaluation_window_sec", 2.0)),
            allow_proxy_audio_evaluation_for_testing=bool(
                config.get("allow_proxy_audio_evaluation_for_testing", False)
            ),
            allow_proxy_audio_rollback_for_testing=bool(
                config.get("allow_proxy_audio_rollback_for_testing", False)
            ),
            min_band_improvement_db=float(config.get("min_band_improvement_db", 0.25)),
            min_rms_response_db=float(config.get("min_rms_response_db", 0.1)),
            worsening_tolerance_db=float(config.get("worsening_tolerance_db", 0.5)),
        )


@dataclass
class PendingActionEvaluation:
    evaluation_id: int
    action: TypedCorrectionAction
    registered_at: float
    due_at: float
    runtime_state: RuntimeState
    channel_id: Optional[int]
    pre_features: Optional[AnalysisFeatures] = None
    rollback_action: Optional[TypedCorrectionAction] = None
    evaluation_band_name: Optional[str] = None
    expected_effect: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionEvaluationOutcome:
    evaluation_id: int
    evaluated: bool
    observable: bool
    control_state_applied: bool
    improved: bool = False
    worsened: bool = False
    should_rollback: bool = False
    measured_effect: str = ""
    rollback_action: Optional[TypedCorrectionAction] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    note: str = ""


def infer_evaluation_band(action: TypedCorrectionAction) -> Optional[str]:
    if not isinstance(action, ChannelEQMove):
        return None
    frequency_hz = float(action.freq_hz)
    named_bands = [
        ("SUB", 30.0, 60.0),
        ("BASS", 60.0, 120.0),
        ("BODY", 120.0, 250.0),
        ("MUD", 250.0, 500.0),
        ("LOW_MID", 500.0, 1000.0),
        ("PRESENCE", 1500.0, 4000.0),
        ("HARSHNESS", 3000.0, 6000.0),
        ("SIBILANCE", 6000.0, 10000.0),
        ("AIR", 10000.0, 16000.0),
    ]
    for band_name, low_hz, high_hz in named_bands:
        if low_hz <= frequency_hz < high_hz:
            return band_name
    return min(
        named_bands,
        key=lambda item: abs(((item[1] * item[2]) ** 0.5) - frequency_hz),
    )[0]


def build_rollback_action(
    action: TypedCorrectionAction,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Optional[TypedCorrectionAction]:
    previous_state = previous_state or {}
    if isinstance(action, ChannelEQMove) and "gain_db" in previous_state:
        return ChannelEQMove(
            channel_id=action.channel_id,
            band=action.band,
            freq_hz=float(previous_state.get("freq_hz", action.freq_hz)),
            gain_db=float(previous_state["gain_db"]),
            q=float(previous_state.get("q", action.q)),
            reason=f"Rollback {action.reason}",
        )
    if isinstance(action, ChannelFaderMove) and "target_db" in previous_state:
        return ChannelFaderMove(
            channel_id=action.channel_id,
            target_db=float(previous_state["target_db"]),
            delta_db=0.0,
            is_lead=action.is_lead,
            reason=f"Rollback {action.reason}",
        )
    return None


def evaluate_pending_action(
    pending: PendingActionEvaluation,
    *,
    policy: AutoFOHEvaluationPolicy,
    control_state_applied: bool,
    post_features: Optional[AnalysisFeatures] = None,
) -> ActionEvaluationOutcome:
    outcome = ActionEvaluationOutcome(
        evaluation_id=pending.evaluation_id,
        evaluated=True,
        observable=False,
        control_state_applied=control_state_applied,
        rollback_action=pending.rollback_action,
    )

    if not control_state_applied:
        outcome.measured_effect = "Control-state verification failed"
        outcome.note = "requested parameter was not observed at evaluation time"
        return outcome

    if (
        not policy.allow_proxy_audio_evaluation_for_testing
        or pending.pre_features is None
        or post_features is None
    ):
        outcome.measured_effect = (
            "No post-action acoustic measurement path available; "
            "control-state change verified only"
        )
        outcome.note = "raw input channels do not reflect post-console processing"
        return outcome

    outcome.observable = True

    if isinstance(pending.action, ChannelEQMove) and pending.evaluation_band_name:
        band_name = pending.evaluation_band_name
        index_name = INDEX_ATTR_LOOKUP[band_name]
        pre_error = abs(float(getattr(pending.pre_features.mix_indexes, index_name)))
        post_error = abs(float(getattr(post_features.mix_indexes, index_name)))
        delta_db = pre_error - post_error
        pre_band_level = float(
            pending.pre_features.slope_compensated_band_levels_db.get(
                band_name,
                pending.pre_features.named_band_levels_db.get(band_name, -100.0),
            )
        )
        post_band_level = float(
            post_features.slope_compensated_band_levels_db.get(
                band_name,
                post_features.named_band_levels_db.get(band_name, -100.0),
            )
        )
        band_level_delta_db = post_band_level - pre_band_level
        expected_direction = 0.0
        if pending.action.gain_db > 0.0:
            expected_direction = 1.0
        elif pending.action.gain_db < 0.0:
            expected_direction = -1.0
        directional_response_db = expected_direction * band_level_delta_db
        outcome.metrics = {
            "pre_error_db": pre_error,
            "post_error_db": post_error,
            "error_delta_db": delta_db,
            "pre_band_level_db": pre_band_level,
            "post_band_level_db": post_band_level,
            "band_level_delta_db": band_level_delta_db,
            "directional_response_db": directional_response_db,
        }
        outcome.improved = (
            delta_db >= policy.min_band_improvement_db
            or directional_response_db >= policy.min_band_improvement_db
        )
        outcome.worsened = (
            delta_db <= -policy.worsening_tolerance_db
            or directional_response_db <= -policy.worsening_tolerance_db
        )
        outcome.should_rollback = (
            outcome.worsened
            and pending.rollback_action is not None
            and policy.allow_proxy_audio_rollback_for_testing
        )
        outcome.measured_effect = (
            f"{band_name} proxy level {pre_band_level:.2f}dB -> "
            f"{post_band_level:.2f}dB; error {pre_error:.2f}dB -> {post_error:.2f}dB"
        )
        return outcome

    if isinstance(pending.action, ChannelFaderMove):
        expected_delta = float(
            pending.metadata.get(
                "target_delta_db",
                pending.action.delta_db or 0.0,
            )
        )
        measured_delta = float(post_features.rms_db - pending.pre_features.rms_db)
        same_direction = (
            (expected_delta > 0.0 and measured_delta > 0.0)
            or (expected_delta < 0.0 and measured_delta < 0.0)
        )
        opposite_direction = (
            (expected_delta > 0.0 and measured_delta < 0.0)
            or (expected_delta < 0.0 and measured_delta > 0.0)
        )
        outcome.metrics = {
            "expected_delta_db": expected_delta,
            "measured_rms_delta_db": measured_delta,
        }
        outcome.improved = same_direction and abs(measured_delta) >= policy.min_rms_response_db
        outcome.worsened = opposite_direction and abs(measured_delta) >= policy.worsening_tolerance_db
        outcome.should_rollback = (
            outcome.worsened
            and pending.rollback_action is not None
            and policy.allow_proxy_audio_rollback_for_testing
        )
        outcome.measured_effect = (
            f"Channel RMS {pending.pre_features.rms_db:.2f}dB -> "
            f"{post_features.rms_db:.2f}dB"
        )
        return outcome

    outcome.measured_effect = "Evaluation not implemented for action type"
    outcome.note = pending.action.action_type
    return outcome
