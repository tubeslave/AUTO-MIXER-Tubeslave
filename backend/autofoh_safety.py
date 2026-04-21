"""
Typed correction actions and bounded/rate-limited application layer.

This module wraps outgoing console writes so the existing automatic workflow
still works, but only through explicit permissions and safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from autofoh_models import RuntimeState
from autofoh_runtime import (
    ACTION_FAMILY_COMPRESSOR,
    ACTION_FAMILY_EMERGENCY_FADER,
    ACTION_FAMILY_FADER,
    ACTION_FAMILY_FEEDBACK,
    ACTION_FAMILY_FX,
    ACTION_FAMILY_GAIN,
    ACTION_FAMILY_HPF,
    ACTION_FAMILY_PAN,
    ACTION_FAMILY_TONAL_EQ,
    RuntimeStatePolicy,
)


@dataclass
class TypedCorrectionAction:
    reason: str

    @property
    def action_type(self) -> str:
        return self.__class__.__name__

    @property
    def family(self) -> str:
        raise NotImplementedError

    @property
    def target_key(self) -> Tuple[Any, ...]:
        raise NotImplementedError


@dataclass
class ChannelGainMove(TypedCorrectionAction):
    channel_id: int
    target_db: float

    @property
    def family(self) -> str:
        return ACTION_FAMILY_GAIN

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id)


@dataclass
class ChannelFaderMove(TypedCorrectionAction):
    channel_id: int
    target_db: float
    delta_db: float = 0.0
    is_lead: bool = False

    @property
    def family(self) -> str:
        return ACTION_FAMILY_FADER

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id)


@dataclass
class ChannelEQMove(TypedCorrectionAction):
    channel_id: int
    band: int
    freq_hz: float
    gain_db: float
    q: float

    @property
    def family(self) -> str:
        return ACTION_FAMILY_TONAL_EQ

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id, self.band)


@dataclass
class HighPassAdjust(TypedCorrectionAction):
    channel_id: int
    freq_hz: float
    enabled: bool = True

    @property
    def family(self) -> str:
        return ACTION_FAMILY_HPF

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id)


@dataclass
class CompressorAdjust(TypedCorrectionAction):
    channel_id: int
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float = 0.0
    enabled: bool = True

    @property
    def family(self) -> str:
        return ACTION_FAMILY_COMPRESSOR

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id)


@dataclass
class PanAdjust(TypedCorrectionAction):
    channel_id: int
    pan: float

    @property
    def family(self) -> str:
        return ACTION_FAMILY_PAN

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id)


@dataclass
class SendLevelAdjust(TypedCorrectionAction):
    channel_id: int
    send_bus: int
    level_db: float

    @property
    def family(self) -> str:
        return ACTION_FAMILY_FX

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id, self.send_bus)


@dataclass
class EmergencyFeedbackNotch(TypedCorrectionAction):
    channel_id: int
    band: int
    freq_hz: float
    q: float
    gain_db: float
    ttl_seconds: float = 120.0

    @property
    def family(self) -> str:
        return ACTION_FAMILY_FEEDBACK

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.channel_id, self.band)


@dataclass
class NoOp(TypedCorrectionAction):
    message: str = ""

    @property
    def family(self) -> str:
        return ACTION_FAMILY_FADER

    @property
    def target_key(self) -> Tuple[Any, ...]:
        return (self.action_type, self.reason, self.message)


@dataclass
class SafetyDecision:
    action: TypedCorrectionAction
    runtime_state: RuntimeState
    supported: bool = True
    allowed: bool = True
    bounded: bool = False
    rate_limited: bool = False
    sent: bool = False
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoFOHSafetyConfig:
    channel_fader_max_step_db: float = 1.0
    channel_fader_min_interval_sec: float = 3.0
    lead_fader_max_step_db: float = 0.5
    lead_fader_min_interval_sec: float = 2.0
    gain_max_abs_db: float = 12.0
    gain_min_interval_sec: float = 3.0
    broad_eq_max_step_db: float = 1.0
    broad_eq_max_total_db_from_snapshot: float = 3.0
    broad_eq_min_interval_sec: float = 5.0
    hpf_min_hz: float = 20.0
    hpf_max_hz: float = 2000.0
    compressor_min_interval_sec: float = 5.0
    fx_send_min_db: float = -40.0
    fx_send_max_db: float = -5.0
    fx_return_min_interval_sec: float = 3.0
    pan_min: float = -100.0
    pan_max: float = 100.0
    feedback_notch_max_cut_db: float = -6.0
    feedback_notch_min_q: float = 8.0
    feedback_notch_ttl_sec: float = 120.0

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "AutoFOHSafetyConfig":
        config = config or {}
        return cls(
            channel_fader_max_step_db=float(config.get("channel_fader_max_step_db", 1.0)),
            channel_fader_min_interval_sec=float(config.get("channel_fader_min_interval_sec", 3.0)),
            lead_fader_max_step_db=float(config.get("lead_fader_max_step_db", 0.5)),
            lead_fader_min_interval_sec=float(config.get("lead_fader_min_interval_sec", 2.0)),
            gain_max_abs_db=float(config.get("gain_max_abs_db", 12.0)),
            gain_min_interval_sec=float(config.get("gain_min_interval_sec", 3.0)),
            broad_eq_max_step_db=float(config.get("broad_eq_max_step_db", 1.0)),
            broad_eq_max_total_db_from_snapshot=float(config.get("broad_eq_max_total_db_from_snapshot", 3.0)),
            broad_eq_min_interval_sec=float(config.get("broad_eq_min_interval_sec", 5.0)),
            hpf_min_hz=float(config.get("hpf_min_hz", 20.0)),
            hpf_max_hz=float(config.get("hpf_max_hz", 2000.0)),
            compressor_min_interval_sec=float(config.get("compressor_min_interval_sec", 5.0)),
            fx_send_min_db=float(config.get("fx_send_min_db", -40.0)),
            fx_send_max_db=float(config.get("fx_send_max_db", -5.0)),
            fx_return_min_interval_sec=float(config.get("fx_return_min_interval_sec", 3.0)),
            pan_min=float(config.get("pan_min", -100.0)),
            pan_max=float(config.get("pan_max", 100.0)),
            feedback_notch_max_cut_db=float(config.get("feedback_notch_max_cut_db", -6.0)),
            feedback_notch_min_q=float(config.get("feedback_notch_min_q", 8.0)),
            feedback_notch_ttl_sec=float(config.get("feedback_notch_ttl_sec", 120.0)),
        )


class AutoFOHSafetyController:
    def __init__(
        self,
        mixer_client: Any,
        config: AutoFOHSafetyConfig | None = None,
        runtime_policy: RuntimeStatePolicy | None = None,
        time_provider: Callable[[], float] | None = None,
    ):
        self.mixer_client = mixer_client
        self.config = config or AutoFOHSafetyConfig()
        self.runtime_policy = runtime_policy or RuntimeStatePolicy()
        self.time_provider = time_provider or time.monotonic
        self._last_sent_at: Dict[Tuple[Any, ...], float] = {}
        self._eq_baselines: Dict[Tuple[int, int], float] = {}
        self.history: List[SafetyDecision] = []

    def execute(
        self,
        action: TypedCorrectionAction,
        runtime_state: RuntimeState,
    ) -> SafetyDecision:
        decision = SafetyDecision(action=action, runtime_state=runtime_state)

        if not self.runtime_policy.is_action_allowed(runtime_state, action.family):
            decision.allowed = False
            decision.message = f"{action.family} not allowed in {runtime_state.value}"
            self.history.append(decision)
            return decision

        bounded_action, bounded = self._apply_bounds(action)
        decision.action = bounded_action
        decision.bounded = bounded

        if runtime_state != RuntimeState.ROLLBACK and self._is_rate_limited(bounded_action):
            decision.allowed = False
            decision.rate_limited = True
            decision.message = "rate limited"
            self.history.append(decision)
            return decision

        translator = self._translator_for(bounded_action)
        if translator is None:
            decision.supported = False
            decision.allowed = False
            decision.message = "unsupported action for mixer"
            self.history.append(decision)
            return decision

        try:
            result = translator(bounded_action)
            decision.sent = bool(result is not False)
            self._last_sent_at[bounded_action.target_key] = self.time_provider()
            decision.payload = self._decision_payload(bounded_action)
            decision.message = "sent" if decision.sent else "send returned false"
        except Exception as exc:
            decision.allowed = False
            decision.message = str(exc)

        self.history.append(decision)
        return decision

    def _decision_payload(self, action: TypedCorrectionAction) -> Dict[str, Any]:
        payload = {"action_type": action.action_type}
        for key, value in action.__dict__.items():
            payload[key] = value
        return payload

    def _apply_bounds(self, action: TypedCorrectionAction) -> Tuple[TypedCorrectionAction, bool]:
        bounded = False

        if isinstance(action, ChannelGainMove):
            target = max(-self.config.gain_max_abs_db, min(self.config.gain_max_abs_db, action.target_db))
            bounded = target != action.target_db
            return ChannelGainMove(channel_id=action.channel_id, target_db=target, reason=action.reason), bounded

        if isinstance(action, ChannelFaderMove):
            current = self._safe_get(lambda: self.mixer_client.get_fader(action.channel_id), default=-144.0)
            max_step = self.config.lead_fader_max_step_db if action.is_lead else self.config.channel_fader_max_step_db
            desired_delta = action.target_db - current
            bounded_delta = max(-max_step, min(max_step, desired_delta))
            target = max(-30.0, min(0.0, current + bounded_delta))
            bounded = abs(target - action.target_db) > 1e-6
            return ChannelFaderMove(
                channel_id=action.channel_id,
                target_db=target,
                delta_db=target - current,
                is_lead=action.is_lead,
                reason=action.reason,
            ), bounded

        if isinstance(action, ChannelEQMove):
            key = (action.channel_id, action.band)
            current_gain = self._safe_get(
                lambda: self.mixer_client.get_eq_band_gain(action.channel_id, f"{action.band}g"),
                default=0.0,
            )
            baseline = self._eq_baselines.setdefault(key, float(current_gain or 0.0))
            desired_step = action.gain_db - float(current_gain or 0.0)
            bounded_step = max(-self.config.broad_eq_max_step_db, min(self.config.broad_eq_max_step_db, desired_step))
            target_gain = float(current_gain or 0.0) + bounded_step
            target_gain = max(
                baseline - self.config.broad_eq_max_total_db_from_snapshot,
                min(baseline + self.config.broad_eq_max_total_db_from_snapshot, target_gain),
            )
            target_gain = max(-15.0, min(15.0, target_gain))
            bounded = abs(target_gain - action.gain_db) > 1e-6
            return ChannelEQMove(
                channel_id=action.channel_id,
                band=action.band,
                freq_hz=max(20.0, min(20000.0, action.freq_hz)),
                gain_db=target_gain,
                q=max(0.44, min(12.0, action.q)),
                reason=action.reason,
            ), bounded

        if isinstance(action, HighPassAdjust):
            freq = max(self.config.hpf_min_hz, min(self.config.hpf_max_hz, action.freq_hz))
            bounded = freq != action.freq_hz
            return HighPassAdjust(channel_id=action.channel_id, freq_hz=freq, enabled=action.enabled, reason=action.reason), bounded

        if isinstance(action, CompressorAdjust):
            bounded_action = CompressorAdjust(
                channel_id=action.channel_id,
                threshold_db=max(-50.0, min(-5.0, action.threshold_db)),
                ratio=max(1.0, min(20.0, action.ratio)),
                attack_ms=max(1.0, min(120.0, action.attack_ms)),
                release_ms=max(30.0, min(600.0, action.release_ms)),
                makeup_db=max(-6.0, min(12.0, action.makeup_db)),
                enabled=action.enabled,
                reason=action.reason,
            )
            bounded = bounded_action != action
            return bounded_action, bounded

        if isinstance(action, PanAdjust):
            pan = max(self.config.pan_min, min(self.config.pan_max, action.pan))
            bounded = pan != action.pan
            return PanAdjust(channel_id=action.channel_id, pan=pan, reason=action.reason), bounded

        if isinstance(action, SendLevelAdjust):
            level = max(self.config.fx_send_min_db, min(self.config.fx_send_max_db, action.level_db))
            bounded = level != action.level_db
            return SendLevelAdjust(channel_id=action.channel_id, send_bus=action.send_bus, level_db=level, reason=action.reason), bounded

        if isinstance(action, EmergencyFeedbackNotch):
            gain_db = min(0.0, max(self.config.feedback_notch_max_cut_db, action.gain_db))
            q = max(self.config.feedback_notch_min_q, action.q)
            ttl = max(1.0, min(self.config.feedback_notch_ttl_sec, action.ttl_seconds))
            bounded_action = EmergencyFeedbackNotch(
                channel_id=action.channel_id,
                band=action.band,
                freq_hz=max(20.0, min(20000.0, action.freq_hz)),
                q=q,
                gain_db=gain_db,
                ttl_seconds=ttl,
                reason=action.reason,
            )
            bounded = bounded_action != action
            return bounded_action, bounded

        return action, False

    def _is_rate_limited(self, action: TypedCorrectionAction) -> bool:
        min_interval = 0.0
        if isinstance(action, ChannelFaderMove):
            min_interval = self.config.lead_fader_min_interval_sec if action.is_lead else self.config.channel_fader_min_interval_sec
        elif isinstance(action, ChannelEQMove):
            min_interval = self.config.broad_eq_min_interval_sec
        elif isinstance(action, ChannelGainMove):
            min_interval = self.config.gain_min_interval_sec
        elif isinstance(action, CompressorAdjust):
            min_interval = self.config.compressor_min_interval_sec
        elif isinstance(action, SendLevelAdjust):
            min_interval = self.config.fx_return_min_interval_sec
        elif isinstance(action, EmergencyFeedbackNotch):
            min_interval = 0.0

        last = self._last_sent_at.get(action.target_key)
        if last is None:
            return False
        return (self.time_provider() - last) < min_interval

    def _translator_for(self, action: TypedCorrectionAction):
        if isinstance(action, ChannelGainMove):
            return self._require_method("set_gain", lambda a: self.mixer_client.set_gain(a.channel_id, a.target_db))
        if isinstance(action, ChannelFaderMove):
            return self._require_method("set_fader", lambda a: self.mixer_client.set_fader(a.channel_id, a.target_db))
        if isinstance(action, ChannelEQMove):
            return self._require_method("set_eq_band", lambda a: self.mixer_client.set_eq_band(a.channel_id, a.band, a.freq_hz, a.gain_db, a.q))
        if isinstance(action, HighPassAdjust):
            return self._require_method("set_hpf", lambda a: self.mixer_client.set_hpf(a.channel_id, a.freq_hz, enabled=a.enabled))
        if isinstance(action, CompressorAdjust):
            return self._require_method(
                "set_compressor",
                lambda a: self.mixer_client.set_compressor(
                    a.channel_id,
                    threshold_db=a.threshold_db,
                    ratio=a.ratio,
                    attack_ms=a.attack_ms,
                    release_ms=a.release_ms,
                    makeup_db=a.makeup_db,
                    enabled=a.enabled,
                ),
            )
        if isinstance(action, PanAdjust):
            return self._require_method("set_pan", lambda a: self.mixer_client.set_pan(a.channel_id, a.pan))
        if isinstance(action, SendLevelAdjust):
            return self._require_method("set_send_level", lambda a: self.mixer_client.set_send_level(a.channel_id, a.send_bus, a.level_db))
        if isinstance(action, EmergencyFeedbackNotch):
            return self._require_method("set_eq_band", lambda a: self.mixer_client.set_eq_band(a.channel_id, a.band, a.freq_hz, a.gain_db, a.q))
        return None

    def _require_method(self, method_name: str, callback):
        if hasattr(self.mixer_client, method_name):
            return callback
        return None

    @staticmethod
    def _safe_get(callback, default):
        try:
            value = callback()
            if value is None:
                return default
            return value
        except Exception:
            return default
