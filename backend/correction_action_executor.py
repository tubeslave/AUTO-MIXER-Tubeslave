"""Proof-gated executor for AutoFOH correction actions.

The low-level AutoFOH safety controller bounds and rate-limits typed actions.
This module adds the live-show contract around that controller: action
promotion, pre-read, readback proof, rollback metadata, and optional rollback on
failed proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyController,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    ChannelMuteMove,
    CompressorAdjust,
    CompressorMakeupAdjust,
    DCAFaderMove,
    DelayAdjust,
    EmergencyFeedbackNotch,
    GateAdjust,
    HighPassAdjust,
    MasterFaderMove,
    PanAdjust,
    PolarityAdjust,
    SafetyDecision,
    SendLevelAdjust,
    TypedCorrectionAction,
)


DEFAULT_PROMOTED_ACTION_TYPES = {
    "ChannelGainMove": True,
    "ChannelFaderMove": True,
    "EmergencyFeedbackNotch": True,
    "BusFaderMove": False,
    "DCAFaderMove": False,
    "MasterFaderMove": False,
    "ChannelMuteMove": False,
    "ChannelEQMove": False,
    "BusEQMove": False,
    "HighPassAdjust": False,
    "CompressorAdjust": False,
    "BusCompressorAdjust": False,
    "CompressorMakeupAdjust": False,
    "BusCompressorMakeupAdjust": False,
    "GateAdjust": False,
    "PanAdjust": False,
    "DelayAdjust": False,
    "PolarityAdjust": False,
    "SendLevelAdjust": False,
}


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in value.__dict__.items()}
    if hasattr(value, "__dict__"):
        return {key: _normalize(item) for key, item in value.__dict__.items()}
    return repr(value)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CorrectionActionExecutorConfig:
    enabled: bool = True
    profile_id: str = "show_assist_auto_v1"
    require_pre_read: bool = True
    require_readback: bool = True
    rollback_required: bool = True
    auto_rollback_on_readback_failure: bool = True
    demote_on_first_failure: bool = True
    readback_tolerance_db: float = 0.25
    readback_tolerance_ms: float = 0.05
    artifact_path: str = ""
    promoted_action_types: Dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_PROMOTED_ACTION_TYPES)
    )

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "CorrectionActionExecutorConfig":
        config = dict(config or {})
        promoted = dict(DEFAULT_PROMOTED_ACTION_TYPES)
        raw_promoted = config.get("promoted_action_types")
        if isinstance(raw_promoted, dict):
            for action_type, enabled in raw_promoted.items():
                promoted[str(action_type)] = bool(enabled)

        return cls(
            enabled=bool(config.get("enabled", True)),
            profile_id=str(config.get("profile_id") or "show_assist_auto_v1"),
            require_pre_read=bool(config.get("require_pre_read", True)),
            require_readback=bool(config.get("require_readback", True)),
            rollback_required=bool(config.get("rollback_required", True)),
            auto_rollback_on_readback_failure=bool(
                config.get("auto_rollback_on_readback_failure", True)
            ),
            demote_on_first_failure=bool(config.get("demote_on_first_failure", True)),
            readback_tolerance_db=float(config.get("readback_tolerance_db", 0.25)),
            readback_tolerance_ms=float(config.get("readback_tolerance_ms", 0.05)),
            artifact_path=str(config.get("artifact_path") or ""),
            promoted_action_types=promoted,
        )


@dataclass(frozen=True)
class ActionReadPlan:
    value_type: str
    target_value: Any
    current_value: Any
    current_source: str
    tolerance: float
    rollback_action: Optional[TypedCorrectionAction]
    target: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_available(self) -> bool:
        return self.current_value is not None

    def value_matches_target(self, value: Any = None) -> bool:
        actual = self.current_value if value is None else value
        if isinstance(self.target_value, bool):
            return bool(actual) == bool(self.target_value)
        target_float = _coerce_float(self.target_value)
        actual_float = _coerce_float(actual)
        if target_float is None or actual_float is None:
            return actual == self.target_value
        return abs(actual_float - target_float) <= self.tolerance


@dataclass
class GatedCorrectionResult:
    action: TypedCorrectionAction
    runtime_state: RuntimeState
    profile_id: str
    eligible: bool = False
    allowed: bool = False
    sent: bool = False
    write_sent: bool = False
    dry_run: bool = False
    pre_read_ok: bool = False
    readback_ok: bool = False
    rollback_available: bool = False
    rollback_sent: bool = False
    demoted: bool = False
    message: str = ""
    pre_read: Dict[str, Any] = field(default_factory=dict)
    readback: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    safety_decision: Optional[SafetyDecision] = None
    rollback_action: Optional[TypedCorrectionAction] = None
    rollback_decision: Optional[SafetyDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return _normalize(
            {
                "profile_id": self.profile_id,
                "action": self.action,
                "runtime_state": self.runtime_state,
                "eligible": self.eligible,
                "allowed": self.allowed,
                "sent": self.sent,
                "write_sent": self.write_sent,
                "dry_run": self.dry_run,
                "pre_read_ok": self.pre_read_ok,
                "readback_ok": self.readback_ok,
                "rollback_available": self.rollback_available,
                "rollback_sent": self.rollback_sent,
                "demoted": self.demoted,
                "message": self.message,
                "pre_read": self.pre_read,
                "readback": self.readback,
                "target": self.target,
                "safety_decision": self.safety_decision,
                "rollback_action": self.rollback_action,
                "rollback_decision": self.rollback_decision,
            }
        )

    def to_safety_decision(self) -> SafetyDecision:
        source = self.safety_decision
        if source is None:
            decision = SafetyDecision(
                action=self.action,
                runtime_state=self.runtime_state,
                allowed=self.allowed,
                sent=self.sent,
                message=self.message,
            )
        else:
            decision = SafetyDecision(
                action=source.action,
                runtime_state=source.runtime_state,
                supported=source.supported,
                allowed=source.allowed,
                bounded=source.bounded,
                rate_limited=source.rate_limited,
                sent=source.sent,
                message=source.message,
                payload=dict(source.payload or {}),
            )

        decision.allowed = self.allowed
        decision.sent = self.sent
        decision.message = self.message
        decision.payload = dict(decision.payload or {})
        decision.payload["gated_executor"] = self.to_dict()
        return decision


class GatedCorrectionActionExecutor:
    """Apply promoted typed actions only after proof gates pass."""

    def __init__(
        self,
        safety_controller: AutoFOHSafetyController,
        *,
        config: Optional[CorrectionActionExecutorConfig] = None,
        event_logger: Optional[Callable[..., bool]] = None,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.safety_controller = safety_controller
        self.config = config or CorrectionActionExecutorConfig()
        self.event_logger = event_logger
        self.time_provider = time_provider or time.time
        self._demoted_action_types: Set[str] = set()

    @property
    def mixer_client(self) -> Any:
        return self.safety_controller.mixer_client

    def is_promoted(self, action_type: str) -> bool:
        return bool(self.config.promoted_action_types.get(action_type, False)) and (
            action_type not in self._demoted_action_types
        )

    def execute(
        self,
        action: TypedCorrectionAction,
        runtime_state: RuntimeState,
        *,
        dry_run: bool = False,
        proof_run: bool = False,
        idempotency_key: str = "",
    ) -> GatedCorrectionResult:
        result = GatedCorrectionResult(
            action=action,
            runtime_state=runtime_state,
            profile_id=self.config.profile_id,
            dry_run=bool(dry_run),
        )

        if not self.config.enabled:
            result.message = "gated_executor_disabled"
            self._record_event("gated_action_blocked", result, idempotency_key)
            return result

        if not self.is_promoted(action.action_type) and not proof_run:
            result.message = "action_not_promoted_for_live_write"
            result.target = self._target_for_action(action)
            self._record_event("gated_action_blocked", result, idempotency_key)
            return result

        result.eligible = True
        bounded_action, bounded = self._preview_bounds(action, runtime_state)
        result.action = bounded_action
        read_plan = self._build_read_plan(bounded_action)
        result.target = dict(read_plan.target)
        result.pre_read = {
            "value": read_plan.current_value,
            "source": read_plan.current_source,
            "value_type": read_plan.value_type,
            "target_value": read_plan.target_value,
            "bounded": bounded,
        }
        result.pre_read_ok = read_plan.current_available or not self.config.require_pre_read
        result.rollback_action = read_plan.rollback_action
        result.rollback_available = read_plan.rollback_action is not None

        if self.config.require_pre_read and not read_plan.current_available:
            result.message = "pre_read_unavailable"
            self._demote_after_failure(action.action_type, result)
            self._record_event("gated_action_blocked", result, idempotency_key)
            return result

        if self.config.rollback_required and not result.rollback_available and not dry_run:
            result.message = "rollback_unavailable"
            self._demote_after_failure(action.action_type, result)
            self._record_event("gated_action_blocked", result, idempotency_key)
            return result

        if dry_run:
            result.allowed = True
            result.message = "dry_run_only"
            self._record_event("gated_action_dry_run", result, idempotency_key)
            return result

        decision = self.safety_controller.execute(action, runtime_state)
        result.safety_decision = decision
        result.action = decision.action
        result.write_sent = bool(decision.sent)

        if not decision.allowed or not decision.sent:
            result.allowed = bool(decision.allowed)
            result.message = decision.message or "safety_controller_blocked"
            self._record_event("gated_action_blocked", result, idempotency_key)
            return result

        post_plan = self._build_read_plan(decision.action)
        result.target = dict(post_plan.target)
        result.rollback_action = self._rollback_action_for(decision.action, read_plan.current_value)
        result.rollback_available = result.rollback_action is not None

        if self.config.require_readback:
            result.readback = {
                "value": post_plan.current_value,
                "source": post_plan.current_source,
                "value_type": post_plan.value_type,
                "target_value": post_plan.target_value,
                "tolerance": post_plan.tolerance,
            }
            result.readback_ok = post_plan.current_available and post_plan.value_matches_target()
            if not result.readback_ok:
                result.allowed = False
                result.sent = False
                result.message = (
                    "readback_mismatch"
                    if post_plan.current_available
                    else "readback_unavailable_after_write"
                )
                self._rollback_after_failed_readback(result, runtime_state)
                self._demote_after_failure(action.action_type, result)
                self._record_event("gated_action_failed_proof", result, idempotency_key)
                return result
        else:
            result.readback_ok = True

        result.allowed = True
        result.sent = True
        result.message = "sent_and_verified"
        self._record_event("gated_action_verified", result, idempotency_key)
        return result

    def _preview_bounds(
        self,
        action: TypedCorrectionAction,
        runtime_state: Optional[RuntimeState] = None,
    ) -> Tuple[TypedCorrectionAction, bool]:
        apply_bounds = getattr(self.safety_controller, "_apply_bounds", None)
        if not callable(apply_bounds):
            return action, False
        try:
            return apply_bounds(action, runtime_state)
        except TypeError:
            try:
                return apply_bounds(action)
            except Exception:
                return action, False
        except Exception:
            return action, False

    def _build_read_plan(self, action: TypedCorrectionAction) -> ActionReadPlan:
        target = self._target_for_action(action)
        if isinstance(action, ChannelGainMove):
            current, source = self._read_channel_gain(action.channel_id)
            return ActionReadPlan(
                value_type="gain",
                target_value=action.target_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, ChannelFaderMove):
            current, source = self._read_method_or_state(
                methods=(("get_fader", (action.channel_id,)), ("get_channel_fader", (action.channel_id,))),
                state_keys=(f"/ch/{action.channel_id}/fdr", f"/ch/{action.channel_id:02d}/fdr"),
            )
            return ActionReadPlan(
                value_type="fader",
                target_value=action.target_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, BusFaderMove):
            current, source = self._read_method_or_state(
                methods=(("get_bus_fader", (action.bus_id,)),),
                state_keys=(f"/bus/{action.bus_id}/fdr", f"/bus/{action.bus_id:02d}/fdr"),
            )
            return ActionReadPlan(
                value_type="bus_fader",
                target_value=action.target_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, DCAFaderMove):
            current, source = self._read_method_or_state(
                methods=(("get_dca_fader", (action.dca_id,)),),
                state_keys=(f"/dca/{action.dca_id}/fdr", f"/dca/{action.dca_id:02d}/fdr"),
            )
            return ActionReadPlan(
                value_type="dca_fader",
                target_value=action.target_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, MasterFaderMove):
            current, source = self._read_method_or_state(
                methods=(("get_main_fader", (action.main_id,)),),
                state_keys=(f"/main/{action.main_id}/fdr", f"/main/{action.main_id:02d}/fdr"),
            )
            return ActionReadPlan(
                value_type="main_fader",
                target_value=action.target_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, (ChannelEQMove, EmergencyFeedbackNotch)):
            current, source = self._read_method_or_state(
                methods=(
                    ("get_eq_band_gain", (action.channel_id, f"{action.band}g")),
                    ("get_eq_band_gain", (action.channel_id, action.band)),
                ),
                state_keys=(
                    f"/ch/{action.channel_id}/eq/{action.band}/g",
                    f"/ch/{action.channel_id:02d}/eq/{action.band}/g",
                ),
            )
            return ActionReadPlan(
                value_type="eq_gain",
                target_value=action.gain_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, BusEQMove):
            current, source = self._read_method_or_state(
                methods=(("get_bus_eq_band_gain", (action.bus_id, action.band)),),
                state_keys=(f"/bus/{action.bus_id}/eq/{action.band}/g",),
            )
            return ActionReadPlan(
                value_type="bus_eq_gain",
                target_value=action.gain_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, HighPassAdjust):
            current, source = self._read_hpf(action.channel_id)
            current_freq = current.get("freq_hz") if isinstance(current, dict) else None
            return ActionReadPlan(
                value_type="hpf_freq",
                target_value=action.freq_hz,
                current_value=current_freq,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, DelayAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_delay", (action.channel_id,)),),
                state_keys=(f"/ch/{action.channel_id}/delay",),
            )
            return ActionReadPlan(
                value_type="delay_ms",
                target_value=action.delay_ms,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_ms,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, PolarityAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_polarity", (action.channel_id,)),),
                state_keys=(f"/ch/{action.channel_id}/in/set/invert",),
            )
            return ActionReadPlan(
                value_type="polarity",
                target_value=bool(action.inverted),
                current_value=bool(current) if current is not None else None,
                current_source=source,
                tolerance=0.0,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, PanAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_pan", (action.channel_id,)), ("get_channel_pan", (action.channel_id,))),
                state_keys=(f"/ch/{action.channel_id}/pan", f"/ch/{action.channel_id:02d}/pan"),
            )
            return ActionReadPlan(
                value_type="pan",
                target_value=action.pan,
                current_value=current,
                current_source=source,
                tolerance=1.0,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, ChannelMuteMove):
            current, source = self._read_method_or_state(
                methods=(("get_channel_mute", (action.channel_id,)), ("get_mute", (action.channel_id,))),
                state_keys=(f"/ch/{action.channel_id}/mute",),
            )
            return ActionReadPlan(
                value_type="mute",
                target_value=action.muted,
                current_value=bool(current) if current is not None else None,
                current_source=source,
                tolerance=0.0,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, SendLevelAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_send_level", (action.channel_id, action.send_bus)),),
                state_keys=(
                    f"/ch/{action.channel_id}/send/{action.send_bus}/lvl",
                    f"/ch/{action.channel_id}/send/{action.send_bus}/level",
                ),
            )
            return ActionReadPlan(
                value_type="send_level",
                target_value=action.level_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, CompressorAdjust):
            # pre-read/readback по порогу; rollback — полный снимок динамики
            current, source = self._read_method_or_state(
                methods=(("get_compressor_threshold", (action.channel_id,)),),
                state_keys=(
                    f"/ch/{action.channel_id}/dyn/thr",
                    f"/ch/{action.channel_id}/dyn/thr26",
                ),
            )
            return ActionReadPlan(
                value_type="comp_threshold",
                target_value=action.threshold_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(
                    action, self._read_compressor_snapshot(action.channel_id)),
                target=target,
            )
        if isinstance(action, CompressorMakeupAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_compressor_gain", (action.channel_id,)),),
                state_keys=(f"/ch/{action.channel_id}/dyn/gain",),
            )
            return ActionReadPlan(
                value_type="comp_makeup",
                target_value=action.makeup_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(action, current),
                target=target,
            )
        if isinstance(action, GateAdjust):
            current, source = self._read_method_or_state(
                methods=(("get_gate_threshold", (action.channel_id,)),),
                state_keys=(f"/ch/{action.channel_id}/gate/thr",),
            )
            return ActionReadPlan(
                value_type="gate_threshold",
                target_value=action.threshold_db,
                current_value=current,
                current_source=source,
                tolerance=self.config.readback_tolerance_db,
                rollback_action=self._rollback_action_for(
                    action, self._read_gate_snapshot(action.channel_id)),
                target=target,
            )

        return ActionReadPlan(
            value_type=action.action_type,
            target_value=None,
            current_value=None,
            current_source="unsupported",
            tolerance=0.0,
            rollback_action=None,
            target=target,
        )

    def _rollback_action_for(self, action: TypedCorrectionAction, old_value: Any) -> Optional[TypedCorrectionAction]:
        old_float = _coerce_float(old_value)
        if isinstance(action, ChannelGainMove) and old_float is not None:
            return ChannelGainMove(
                channel_id=action.channel_id,
                target_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, ChannelFaderMove) and old_float is not None:
            return ChannelFaderMove(
                channel_id=action.channel_id,
                target_db=old_float,
                is_lead=action.is_lead,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, BusFaderMove) and old_float is not None:
            return BusFaderMove(
                bus_id=action.bus_id,
                target_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, DCAFaderMove) and old_float is not None:
            return DCAFaderMove(
                dca_id=action.dca_id,
                target_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, MasterFaderMove) and old_float is not None:
            return MasterFaderMove(
                main_id=action.main_id,
                target_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, ChannelEQMove) and old_float is not None:
            return ChannelEQMove(
                channel_id=action.channel_id,
                band=action.band,
                freq_hz=action.freq_hz,
                gain_db=old_float,
                q=action.q,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, EmergencyFeedbackNotch) and old_float is not None:
            return ChannelEQMove(
                channel_id=action.channel_id,
                band=action.band,
                freq_hz=action.freq_hz,
                gain_db=old_float,
                q=action.q,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, BusEQMove) and old_float is not None:
            return BusEQMove(
                bus_id=action.bus_id,
                band=action.band,
                freq_hz=action.freq_hz,
                gain_db=old_float,
                q=action.q,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, HighPassAdjust) and isinstance(old_value, dict):
            freq = _coerce_float(old_value.get("freq_hz"))
            if freq is not None:
                return HighPassAdjust(
                    channel_id=action.channel_id,
                    freq_hz=freq,
                    enabled=bool(old_value.get("enabled", True)),
                    reason=f"rollback {action.reason}",
                )
        if isinstance(action, DelayAdjust) and old_float is not None:
            return DelayAdjust(
                channel_id=action.channel_id,
                delay_ms=old_float,
                enabled=action.enabled,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, PolarityAdjust) and old_value is not None:
            return PolarityAdjust(
                channel_id=action.channel_id,
                inverted=bool(old_value),
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, PanAdjust) and old_float is not None:
            return PanAdjust(
                channel_id=action.channel_id,
                pan=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, ChannelMuteMove) and old_value is not None:
            return ChannelMuteMove(
                channel_id=action.channel_id,
                muted=bool(old_value),
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, SendLevelAdjust) and old_float is not None:
            return SendLevelAdjust(
                channel_id=action.channel_id,
                send_bus=action.send_bus,
                level_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, CompressorAdjust) and isinstance(old_value, dict):
            old_thr = _coerce_float(old_value.get("threshold_db"))
            if old_thr is None:
                return None
            return CompressorAdjust(
                channel_id=action.channel_id,
                threshold_db=old_thr,
                ratio=_coerce_float(old_value.get("ratio")) or action.ratio,
                attack_ms=_coerce_float(old_value.get("attack_ms")) or action.attack_ms,
                release_ms=_coerce_float(old_value.get("release_ms")) or action.release_ms,
                makeup_db=_coerce_float(old_value.get("makeup_db")) or 0.0,
                enabled=bool(old_value.get("enabled") or 0),
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, CompressorMakeupAdjust) and old_float is not None:
            return CompressorMakeupAdjust(
                channel_id=action.channel_id,
                makeup_db=old_float,
                reason=f"rollback {action.reason}",
            )
        if isinstance(action, GateAdjust) and isinstance(old_value, dict):
            old_thr = _coerce_float(old_value.get("threshold_db"))
            if old_thr is None:
                return None
            return GateAdjust(
                channel_id=action.channel_id,
                threshold_db=old_thr,
                range_db=_coerce_float(old_value.get("range_db")) or action.range_db,
                attack_ms=_coerce_float(old_value.get("attack_ms")) or action.attack_ms,
                hold_ms=_coerce_float(old_value.get("hold_ms")) or action.hold_ms,
                release_ms=_coerce_float(old_value.get("release_ms")) or action.release_ms,
                enabled=bool(old_value.get("enabled") or 0),
                reason=f"rollback {action.reason}",
            )
        return None

    def _rollback_after_failed_readback(
        self,
        result: GatedCorrectionResult,
        runtime_state: RuntimeState,
    ) -> None:
        if not self.config.auto_rollback_on_readback_failure or result.rollback_action is None:
            return
        rollback_decision = self.safety_controller.execute(
            result.rollback_action,
            RuntimeState.ROLLBACK if runtime_state != RuntimeState.ROLLBACK else runtime_state,
        )
        result.rollback_decision = rollback_decision
        result.rollback_sent = bool(rollback_decision.sent)

    def _demote_after_failure(self, action_type: str, result: GatedCorrectionResult) -> None:
        if not self.config.demote_on_first_failure:
            return
        self._demoted_action_types.add(action_type)
        result.demoted = True

    def _read_channel_gain(self, channel_id: int) -> Tuple[Any, str]:
        return self._read_method_or_state(
            methods=(("get_channel_gain", (channel_id,)), ("get_gain", (channel_id,))),
            state_keys=(
                f"/ch/{channel_id}/in/set/trim",
                f"/ch/{channel_id:02d}/in/set/trim",
                f"/ch/{channel_id}/gain",
                f"gain/{channel_id}",
            ),
        )

    def _read_compressor_snapshot(self, channel_id: int) -> Optional[Dict[str, Any]]:
        """Снимок динамики для отката CompressorAdjust (None = нет данных)."""
        state = getattr(self.mixer_client, "state", None)
        if not isinstance(state, dict):
            return None
        base = f"/ch/{channel_id}/dyn"
        thr = state.get(f"{base}/thr")
        if thr is None:
            thr = state.get(f"{base}/thr26")
        snapshot = {
            "threshold_db": thr,
            "ratio": state.get(f"{base}/ratio"),
            "attack_ms": state.get(f"{base}/att"),
            "release_ms": state.get(f"{base}/rel"),
            "makeup_db": state.get(f"{base}/gain"),
            "enabled": state.get(f"{base}/on"),
        }
        if snapshot["threshold_db"] is None:
            return None
        return snapshot

    def _read_gate_snapshot(self, channel_id: int) -> Optional[Dict[str, Any]]:
        state = getattr(self.mixer_client, "state", None)
        if not isinstance(state, dict):
            return None
        base = f"/ch/{channel_id}/gate"
        thr = state.get(f"{base}/thr")
        if thr is None:
            return None
        return {
            "threshold_db": thr,
            "range_db": state.get(f"{base}/range"),
            "attack_ms": state.get(f"{base}/att"),
            "hold_ms": state.get(f"{base}/hld"),
            "release_ms": state.get(f"{base}/rel"),
            "enabled": state.get(f"{base}/on"),
        }

    def _read_hpf(self, channel_id: int) -> Tuple[Any, str]:
        method = getattr(self.mixer_client, "get_hpf", None)
        if callable(method):
            try:
                value = method(channel_id)
                if isinstance(value, dict):
                    return value, "method:get_hpf"
                if isinstance(value, (list, tuple)) and value:
                    return {
                        "freq_hz": value[0],
                        "enabled": bool(value[1]) if len(value) > 1 else True,
                    }, "method:get_hpf"
                if value is not None:
                    return {"freq_hz": value, "enabled": True}, "method:get_hpf"
            except Exception:
                pass
        state = getattr(self.mixer_client, "state", None)
        if isinstance(state, dict):
            freq = state.get(f"/ch/{channel_id}/eq/hpf/f")
            enabled = state.get(f"/ch/{channel_id}/eq/hpf/on", True)
            if freq is None:
                # WING: low-cut живёт в /ch/N/flt/{lcf,lc}
                freq = state.get(f"/ch/{channel_id}/flt/lcf")
                enabled = state.get(f"/ch/{channel_id}/flt/lc", True)
            if freq is not None:
                return {"freq_hz": freq, "enabled": bool(enabled)}, "state"
        return None, "unavailable"

    def _read_method_or_state(
        self,
        *,
        methods: Tuple[Tuple[str, Tuple[Any, ...]], ...],
        state_keys: Tuple[str, ...],
    ) -> Tuple[Any, str]:
        for method_name, args in methods:
            method = getattr(self.mixer_client, method_name, None)
            if not callable(method):
                continue
            try:
                value = method(*args)
                if value is not None:
                    return value, f"method:{method_name}"
            except Exception:
                continue

        state = getattr(self.mixer_client, "state", None)
        if isinstance(state, dict):
            for key in state_keys:
                if key in state:
                    return state.get(key), f"state:{key}"
        return None, "unavailable"

    def _target_for_action(self, action: TypedCorrectionAction) -> Dict[str, Any]:
        target = {"action_type": action.action_type}
        for key in ("channel_id", "bus_id", "dca_id", "main_id", "band", "send_bus"):
            if hasattr(action, key):
                target[key] = getattr(action, key)
        return target

    def _record_event(
        self,
        event_type: str,
        result: GatedCorrectionResult,
        idempotency_key: str,
    ) -> None:
        payload = {
            "idempotency_key": idempotency_key,
            "gated_executor": result.to_dict(),
        }
        if self.event_logger is not None:
            try:
                self.event_logger(event_type, **payload)
            except Exception:
                pass

        if not self.config.artifact_path:
            return
        path = Path(self.config.artifact_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "timestamp": self.time_provider(),
                "event_type": event_type,
                **_normalize(payload),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception:
            pass
