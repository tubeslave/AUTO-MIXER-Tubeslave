"""Realtime arrangement-aware input trim controller.

This controller is intentionally separate from fader automation.  It adjusts
only the console input trim after a role-aware, bleed-aware confidence gate has
seen enough of the channel's own signal.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

try:
    from autogain_contract import AutoGainRecommendation
except ImportError:  # pragma: no cover - used when importing through backend package.
    from backend.autogain_contract import AutoGainRecommendation
try:
    from live_apply import LiveApplyRequest, LiveApplyService
except ImportError:  # pragma: no cover - used when importing through backend package.
    from backend.live_apply import LiveApplyRequest, LiveApplyService
from .activity_detector import ActivityDetector
from .channel_role_classifier import ChannelRoleClassifier
from .wing_meter_reader import WingMeterReading

try:
    from auto_fader_v2.core.bleed_compensator import get_compensated_level
    from auto_fader_v2.core.bleed_detector import BleedDetector, BleedInfo
except ImportError:  # pragma: no cover - used when importing through backend package.
    from backend.auto_fader_v2.core.bleed_compensator import get_compensated_level
    from backend.auto_fader_v2.core.bleed_detector import BleedDetector, BleedInfo

logger = logging.getLogger(__name__)


BAND_RANGES = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "high_mid": (2000.0, 4000.0),
    "high": (4000.0, 8000.0),
    "air": (8000.0, 14000.0),
}


ROLE_TARGETS = {
    "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 1.2, "windows": 4},
    "backing_vocal": {"peak_dbfs": -11.0, "main_sec": 1.5, "windows": 4},
    "kick": {"peak_dbfs": -8.0, "main_sec": 0.12, "windows": 2},
    "snare": {"peak_dbfs": -8.0, "main_sec": 0.12, "windows": 2},
    "tom": {"peak_dbfs": -8.5, "main_sec": 0.18, "windows": 2},
    "overhead": {"peak_dbfs": -14.0, "main_sec": 2.0, "windows": 5},
    "bass": {"peak_dbfs": -10.0, "main_sec": 1.0, "windows": 3},
    "rhythm_guitar": {"peak_dbfs": -12.0, "main_sec": 1.2, "windows": 4},
    "lead_guitar": {"peak_dbfs": -10.0, "main_sec": 0.8, "windows": 3},
    "acoustic_guitar": {"peak_dbfs": -12.0, "main_sec": 1.2, "windows": 4},
    "keys": {"peak_dbfs": -12.0, "main_sec": 1.2, "windows": 4},
    "pad": {"peak_dbfs": -13.0, "main_sec": 1.5, "windows": 4},
    "solo_instrument": {"peak_dbfs": -10.0, "main_sec": 0.8, "windows": 3},
    "fx": {"peak_dbfs": -14.0, "main_sec": 1.5, "windows": 4},
    "unknown": {"peak_dbfs": -14.0, "main_sec": 2.0, "windows": 5},
}

PERCUSSIVE_ROLES = {"kick", "snare", "tom"}


ROLE_TRIM_LIMITS = {
    "lead_vocal": {"min_db": -12.0, "max_db": 6.0},
    "backing_vocal": {"min_db": -10.0, "max_db": 5.0},
    "kick": {"min_db": -12.0, "max_db": 6.0},
    "snare": {"min_db": -12.0, "max_db": 6.0},
    "tom": {"min_db": -12.0, "max_db": 6.0},
    "overhead": {"min_db": -8.0, "max_db": 4.0},
    "bass": {"min_db": -10.0, "max_db": 5.0},
    "rhythm_guitar": {"min_db": -18.0, "max_db": 5.0},
    "lead_guitar": {"min_db": -18.0, "max_db": 6.0},
    "acoustic_guitar": {"min_db": -18.0, "max_db": 5.0},
    "keys": {"min_db": -18.0, "max_db": 5.0},
    "pad": {"min_db": -18.0, "max_db": 4.0},
    "solo_instrument": {"min_db": -10.0, "max_db": 6.0},
    "fx": {"min_db": -8.0, "max_db": 3.0},
    "unknown": {"min_db": -4.0, "max_db": 2.0},
}


@dataclass
class LiveInputTrimChannelState:
    """State accumulated for one live input channel."""

    audio_channel: int
    mixer_channel: int
    channel_name: str
    role: str
    role_confidence: float
    base_trim_db: float = 0.0
    current_trim_db: float = 0.0
    target_trim_db: float = 0.0
    state: str = "waiting_for_signal"
    noise_floor_db: float = -80.0
    main_signal_confidence: float = 0.0
    main_signal_sec: float = 0.0
    main_signal_windows: int = 0
    waiting_reason: str = "waiting_for_main_signal"
    last_rms_db: float = -120.0
    last_peak_dbfs: float = -120.0
    last_true_peak_dbtp: float = -120.0
    usb_rms_db: float = -120.0
    usb_peak_dbfs: float = -120.0
    usb_true_peak_dbtp: float = -120.0
    level_source: str = "usb_audio"
    meter_status: str = "not_configured"
    meter_trusted: bool = False
    meter_confidence: float = 0.0
    meter_peak_dbfs: float = -120.0
    meter_rms_db: float = -120.0
    meter_true_peak_dbtp: float = -120.0
    meter_age_sec: Optional[float] = None
    meter_activity_fallback: bool = False
    peak_hold_dbfs: float = -120.0
    compensated_level_db: float = -120.0
    bleed_ratio: float = 0.0
    bleed_confidence: float = 0.0
    bleed_source_channel: Optional[int] = None
    bleed_method: str = "none"
    last_apply_time: Optional[float] = None
    last_update_time: Optional[float] = None
    last_applied_delta_db: float = 0.0
    blocked_reason: str = ""
    sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LiveInputTrimChannelState":
        return cls(
            audio_channel=int(payload.get("audio_channel", 0)),
            mixer_channel=int(payload.get("mixer_channel", 0)),
            channel_name=str(payload.get("channel_name", "")),
            role=str(payload.get("role", "unknown")),
            role_confidence=float(payload.get("role_confidence", 0.0)),
            base_trim_db=float(payload.get("base_trim_db", 0.0)),
            current_trim_db=float(payload.get("current_trim_db", 0.0)),
            target_trim_db=float(payload.get("target_trim_db", 0.0)),
            state=str(payload.get("state", "waiting_for_signal")),
            noise_floor_db=float(payload.get("noise_floor_db", -80.0)),
            main_signal_confidence=float(payload.get("main_signal_confidence", 0.0)),
            main_signal_sec=float(payload.get("main_signal_sec", 0.0)),
            main_signal_windows=int(payload.get("main_signal_windows", 0)),
            waiting_reason=str(payload.get("waiting_reason", "waiting_for_main_signal")),
            last_rms_db=float(payload.get("last_rms_db", -120.0)),
            last_peak_dbfs=float(payload.get("last_peak_dbfs", -120.0)),
            last_true_peak_dbtp=float(payload.get("last_true_peak_dbtp", -120.0)),
            usb_rms_db=float(payload.get("usb_rms_db", -120.0)),
            usb_peak_dbfs=float(payload.get("usb_peak_dbfs", -120.0)),
            usb_true_peak_dbtp=float(payload.get("usb_true_peak_dbtp", -120.0)),
            level_source=str(payload.get("level_source", "usb_audio")),
            meter_status=str(payload.get("meter_status", "not_configured")),
            meter_trusted=bool(payload.get("meter_trusted", False)),
            meter_confidence=float(payload.get("meter_confidence", 0.0)),
            meter_peak_dbfs=float(payload.get("meter_peak_dbfs", -120.0)),
            meter_rms_db=float(payload.get("meter_rms_db", -120.0)),
            meter_true_peak_dbtp=float(payload.get("meter_true_peak_dbtp", -120.0)),
            meter_age_sec=_maybe_float(payload.get("meter_age_sec")),
            meter_activity_fallback=bool(payload.get("meter_activity_fallback", False)),
            peak_hold_dbfs=float(payload.get("peak_hold_dbfs", -120.0)),
            compensated_level_db=float(payload.get("compensated_level_db", -120.0)),
            bleed_ratio=float(payload.get("bleed_ratio", 0.0)),
            bleed_confidence=float(payload.get("bleed_confidence", 0.0)),
            bleed_source_channel=(
                int(payload["bleed_source_channel"])
                if payload.get("bleed_source_channel") is not None
                else None
            ),
            bleed_method=str(payload.get("bleed_method", "none")),
            last_apply_time=_maybe_float(payload.get("last_apply_time")),
            last_update_time=_maybe_float(payload.get("last_update_time")),
            last_applied_delta_db=float(payload.get("last_applied_delta_db", 0.0)),
            blocked_reason=str(payload.get("blocked_reason", "")),
            sent=bool(payload.get("sent", False)),
        )


@dataclass
class LiveInputTrimReplaySnapshot:
    """Versioned replay checkpoint for restoring controller state deterministically."""

    version: str
    snapshot_id: str
    channels: list[int]
    channel_mapping: Dict[int, int]
    states: Dict[int, LiveInputTrimChannelState]
    transport_policy: Dict[str, Any]
    controller_state: Dict[str, Any]
    replay_metadata: Dict[str, Any] = field(default_factory=dict)
    last_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "channels": [int(channel) for channel in self.channels],
            "channel_mapping": {
                str(int(audio_channel)): int(mixer_channel)
                for audio_channel, mixer_channel in sorted(self.channel_mapping.items())
            },
            "states": {
                str(int(channel)): state.to_dict()
                for channel, state in sorted(self.states.items())
            },
            "transport_policy": dict(self.transport_policy),
            "controller_state": dict(self.controller_state),
            "replay_metadata": dict(self.replay_metadata),
            "last_state": dict(self.last_state),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LiveInputTrimReplaySnapshot":
        return cls(
            version=str(payload.get("version", "replay_state_snapshot/v1")),
            snapshot_id=str(payload.get("snapshot_id", "")),
            channels=[int(channel) for channel in payload.get("channels") or ()],
            channel_mapping={
                int(audio_channel): int(mixer_channel)
                for audio_channel, mixer_channel in (payload.get("channel_mapping") or {}).items()
            },
            states={
                int(channel): LiveInputTrimChannelState.from_dict(dict(state_payload))
                for channel, state_payload in (payload.get("states") or {}).items()
            },
            transport_policy=dict(payload.get("transport_policy") or {}),
            controller_state=dict(payload.get("controller_state") or {}),
            replay_metadata=dict(payload.get("replay_metadata") or {}),
            last_state=dict(payload.get("last_state") or {}),
        )

    @staticmethod
    def build_snapshot_id(payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"replay_checkpoint::live_input_trim::{digest[:16]}"


@dataclass
class LiveInputTrimWritePlan:
    """Replayable write plan for a trim decision."""

    source: str
    operation: str
    channel: int
    value: float
    confirm_live_apply: bool = False
    current_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_step: Optional[float] = None
    replay_correlation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_live_apply_request(self, *, dry_run: bool) -> LiveApplyRequest:
        return LiveApplyRequest(
            source=self.source,
            operation=self.operation,
            channel=self.channel,
            value=self.value,
            dry_run=dry_run,
            confirm_live_apply=self.confirm_live_apply,
            current_value=self.current_value,
            min_value=self.min_value,
            max_value=self.max_value,
            max_step=self.max_step,
            replay_correlation_id=self.replay_correlation_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "operation": self.operation,
            "channel": int(self.channel),
            "value": float(self.value),
            "confirm_live_apply": bool(self.confirm_live_apply),
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "max_step": self.max_step,
            "replay_correlation_id": self.replay_correlation_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["LiveInputTrimWritePlan"]:
        if not payload:
            return None
        return cls(
            source=str(payload.get("source", "auto_gain_live_input_trim")),
            operation=str(payload.get("operation", "set_gain")),
            channel=int(payload.get("channel", 0)),
            value=float(payload.get("value", 0.0)),
            confirm_live_apply=bool(payload.get("confirm_live_apply", False)),
            current_value=_maybe_float(payload.get("current_value")),
            min_value=_maybe_float(payload.get("min_value")),
            max_value=_maybe_float(payload.get("max_value")),
            max_step=_maybe_float(payload.get("max_step")),
            replay_correlation_id=str(payload.get("replay_correlation_id", "")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class LiveInputTrimDecision:
    """Serializable trim decision before or after execution."""

    channel: int
    name: str
    role: str
    state: str
    reason: str
    blocked_reason: str
    current_trim_db: float
    target_trim_db: float
    delta_db: float
    main_signal_confidence: float
    main_signal_sec: float
    main_signal_windows: int
    bleed_ratio: float
    bleed_confidence: float
    bleed_source_channel: Optional[int]
    level_source: str
    rms_db: float
    peak_dbfs: float
    true_peak_dbtp: float
    usb_rms_db: float
    usb_peak_dbfs: float
    usb_true_peak_dbtp: float
    meter_status: str
    meter_trusted: bool
    meter_peak_dbfs: float
    meter_rms_db: float
    meter_true_peak_dbtp: float
    meter_age_sec: Optional[float]
    meter_activity_fallback: bool
    write_plan: Optional[LiveInputTrimWritePlan] = None
    audit_id: str = ""
    replay_correlation_id: str = ""
    send_status: str = "not_sent"
    readback_status: str = "not_requested"
    confirmed_value: Optional[float] = None
    guard_reasons: tuple[str, ...] = ()
    sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": int(self.channel),
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "reason": self.reason,
            "blocked_reason": self.blocked_reason,
            "current_trim_db": round(self.current_trim_db, 3),
            "target_trim_db": round(self.target_trim_db, 3),
            "delta_db": round(self.delta_db, 3),
            "main_signal_confidence": round(self.main_signal_confidence, 4),
            "main_signal_sec": round(self.main_signal_sec, 3),
            "main_signal_windows": int(self.main_signal_windows),
            "bleed_ratio": round(self.bleed_ratio, 4),
            "bleed_confidence": round(self.bleed_confidence, 4),
            "bleed_source_channel": self.bleed_source_channel,
            "level_source": self.level_source,
            "rms_db": round(self.rms_db, 3),
            "peak_dbfs": round(self.peak_dbfs, 3),
            "true_peak_dbtp": round(self.true_peak_dbtp, 3),
            "usb_rms_db": round(self.usb_rms_db, 3),
            "usb_peak_dbfs": round(self.usb_peak_dbfs, 3),
            "usb_true_peak_dbtp": round(self.usb_true_peak_dbtp, 3),
            "meter_status": self.meter_status,
            "meter_trusted": bool(self.meter_trusted),
            "meter_peak_dbfs": round(self.meter_peak_dbfs, 3),
            "meter_rms_db": round(self.meter_rms_db, 3),
            "meter_true_peak_dbtp": round(self.meter_true_peak_dbtp, 3),
            "meter_age_sec": (
                round(self.meter_age_sec, 3)
                if self.meter_age_sec is not None and math.isfinite(self.meter_age_sec)
                else None
            ),
            "meter_activity_fallback": bool(self.meter_activity_fallback),
            "write_plan": self.write_plan.to_dict() if self.write_plan else None,
            "audit_id": self.audit_id,
            "replay_correlation_id": self.replay_correlation_id,
            "send_status": self.send_status,
            "readback_status": self.readback_status,
            "confirmed_value": (
                round(float(self.confirmed_value), 3)
                if self.confirmed_value is not None and math.isfinite(float(self.confirmed_value))
                else self.confirmed_value
            ),
            "guard_reasons": list(self.guard_reasons),
            "sent": bool(self.sent),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LiveInputTrimDecision":
        return cls(
            channel=int(payload.get("channel", 0)),
            name=str(payload.get("name", "")),
            role=str(payload.get("role", "unknown")),
            state=str(payload.get("state", "idle")),
            reason=str(payload.get("reason", "")),
            blocked_reason=str(payload.get("blocked_reason", "")),
            current_trim_db=float(payload.get("current_trim_db", 0.0)),
            target_trim_db=float(payload.get("target_trim_db", 0.0)),
            delta_db=float(payload.get("delta_db", 0.0)),
            main_signal_confidence=float(payload.get("main_signal_confidence", 0.0)),
            main_signal_sec=float(payload.get("main_signal_sec", 0.0)),
            main_signal_windows=int(payload.get("main_signal_windows", 0)),
            bleed_ratio=float(payload.get("bleed_ratio", 0.0)),
            bleed_confidence=float(payload.get("bleed_confidence", 0.0)),
            bleed_source_channel=payload.get("bleed_source_channel"),
            level_source=str(payload.get("level_source", "usb_audio")),
            rms_db=float(payload.get("rms_db", -120.0)),
            peak_dbfs=float(payload.get("peak_dbfs", -120.0)),
            true_peak_dbtp=float(payload.get("true_peak_dbtp", -120.0)),
            usb_rms_db=float(payload.get("usb_rms_db", -120.0)),
            usb_peak_dbfs=float(payload.get("usb_peak_dbfs", -120.0)),
            usb_true_peak_dbtp=float(payload.get("usb_true_peak_dbtp", -120.0)),
            meter_status=str(payload.get("meter_status", "unavailable")),
            meter_trusted=bool(payload.get("meter_trusted", False)),
            meter_peak_dbfs=float(payload.get("meter_peak_dbfs", -120.0)),
            meter_rms_db=float(payload.get("meter_rms_db", -120.0)),
            meter_true_peak_dbtp=float(payload.get("meter_true_peak_dbtp", -120.0)),
            meter_age_sec=_maybe_float(payload.get("meter_age_sec")),
            meter_activity_fallback=bool(payload.get("meter_activity_fallback", False)),
            write_plan=LiveInputTrimWritePlan.from_dict(payload.get("write_plan")),
            audit_id=str(payload.get("audit_id", "")),
            replay_correlation_id=str(payload.get("replay_correlation_id", "")),
            send_status=str(payload.get("send_status", "not_sent")),
            readback_status=str(payload.get("readback_status", "not_requested")),
            confirmed_value=_maybe_float(payload.get("confirmed_value")),
            guard_reasons=tuple(payload.get("guard_reasons") or ()),
            sent=bool(payload.get("sent", False)),
        )


class LiveInputTrimController:
    """Realtime input trim staging with main-signal wait and bleed rejection."""

    def __init__(
        self,
        mixer_client=None,
        config: Optional[Dict[str, Any]] = None,
        audio_capture=None,
        meter_reader=None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        live_apply_service: Optional[LiveApplyService] = None,
    ):
        self.mixer_client = mixer_client
        self.config = config or {}
        self.audio_capture = audio_capture
        self.meter_reader = meter_reader
        self.status_callback = status_callback
        self.live_apply_service = live_apply_service or LiveApplyService()

        self.cfg = self._get_cfg(self.config)
        role_limits = self.cfg.get("role_trim_limits", {})
        self.role_trim_limits = _merge_role_map(ROLE_TRIM_LIMITS, role_limits)
        self.role_targets = _merge_role_map(ROLE_TARGETS, self.cfg.get("role_targets", {}))
        self.role_classifier = ChannelRoleClassifier(
            role_limits=self.cfg.get("role_limits", {})
        )
        self.activity_detector = ActivityDetector(
            min_active_db=float(self.cfg.get("activity_threshold_db", -50.0)),
            noise_margin_db=float(self.cfg.get("noise_margin_db", 10.0)),
            attack_hold_sec=float(self.cfg.get("activity_attack_hold_sec", 0.2)),
            release_hold_sec=float(self.cfg.get("activity_release_hold_sec", 1.0)),
        )
        bleed_cfg = dict(self.cfg.get("bleed_protection", {}))
        if not bleed_cfg and self.config:
            bleed_cfg = dict(self.config.get("bleed_protection", {}))
        self.bleed_detector = BleedDetector(bleed_cfg)

        self.enabled = bool(self.cfg.get("enabled", True))
        self.live_apply_enabled = bool(self.cfg.get("live_apply_enabled", False))
        self.analysis_only_mode = bool(self.cfg.get("analysis_only_mode", True))
        self.confirm_live_apply = bool(self.cfg.get("confirm_live_apply", False))
        self.run_background_loop = bool(self.cfg.get("run_background_loop", True))
        self.update_interval_sec = float(self.cfg.get("update_interval_ms", 250.0)) / 1000.0
        self.window_sec = float(self.cfg.get("analysis_window_sec", 0.5))
        self.noise_floor_alpha = float(self.cfg.get("noise_floor_alpha", 0.03))
        self.min_main_confidence = float(self.cfg.get("min_main_signal_confidence", 0.65))
        self.min_negative_confidence = float(self.cfg.get("min_negative_signal_confidence", 0.35))
        self.high_bleed_ratio = float(self.cfg.get("high_bleed_ratio", 0.45))
        self.high_bleed_confidence = float(self.cfg.get("high_bleed_confidence", 0.25))
        self.max_step_db_per_tick = float(self.cfg.get("max_step_db_per_tick", 0.25))
        self.max_boost_step_db = float(self.cfg.get("max_boost_step_db", 0.15))
        self.deadband_db = float(self.cfg.get("deadband_db", 0.1))
        self.cooldown_sec = float(self.cfg.get("apply_cooldown_sec", 1.0))
        self.trim_min_db = float(self.cfg.get("trim_min_db", -18.0))
        self.trim_max_db = float(self.cfg.get("trim_max_db", 18.0))
        self.hot_peak_threshold_dbfs = float(self.cfg.get("hot_peak_threshold_dbfs", -3.0))
        self.hot_peak_target_dbfs = float(self.cfg.get("hot_peak_target_dbfs", -6.0))
        self.true_peak_ceiling_dbtp = float(self.cfg.get("true_peak_ceiling_dbtp", -1.0))
        self.cut_requires_ready = bool(self.cfg.get("regular_cut_requires_ready", True))
        self.block_bleed_regular_moves = bool(self.cfg.get("block_regular_trim_on_bleed", True))
        self.percussive_peak_mode = bool(self.cfg.get("percussive_peak_mode", True))
        self.percussive_boost_margin_db = float(self.cfg.get("percussive_boost_margin_db", 1.0))
        self.limit_release_margin_db = float(self.cfg.get("limit_release_margin_db", 1.0))
        self.level_source = str(self.cfg.get("level_source", "usb_audio")).lower()
        if self.level_source not in {"usb_audio", "wing_meter", "hybrid"}:
            self.level_source = "usb_audio"
        self.require_meter_for_live_apply = bool(
            self.cfg.get("require_meter_for_live_apply", self.level_source != "usb_audio")
        )
        self.require_meter_for_analysis = bool(
            self.cfg.get("require_meter_for_analysis", self.level_source == "wing_meter")
        )
        self.meter_activity_fallback = bool(self.cfg.get("meter_activity_fallback", True))
        self.usb_silence_dbfs = float(self.cfg.get("usb_silence_dbfs", -90.0))

        self.channels: list[int] = []
        self.channel_mapping: Dict[int, int] = {}
        self.states: Dict[int, LiveInputTrimChannelState] = {}
        self.last_state: Dict[str, Any] = self._empty_status()
        self.started_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        channels: Iterable[int],
        channel_mapping: Optional[Dict[int, int]] = None,
        channel_names: Optional[Dict[int, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start realtime input-trim analysis for selected channels."""
        if settings:
            self.enabled = bool(settings.get("enabled", self.enabled))
            self.live_apply_enabled = bool(
                settings.get("live_apply_enabled", self.live_apply_enabled)
            )
            self.analysis_only_mode = bool(
                settings.get("analysis_only_mode", self.analysis_only_mode)
            )
            if "confirm_live_apply" in settings:
                self.confirm_live_apply = bool(settings.get("confirm_live_apply"))
            if settings.get("level_source"):
                requested_source = str(settings.get("level_source")).lower()
                if requested_source in {"usb_audio", "wing_meter", "hybrid"}:
                    self.level_source = requested_source
            if "require_meter_for_live_apply" in settings:
                self.require_meter_for_live_apply = bool(
                    settings["require_meter_for_live_apply"]
                )
            if "require_meter_for_analysis" in settings:
                self.require_meter_for_analysis = bool(settings["require_meter_for_analysis"])

        self.channels = [int(ch) for ch in channels]
        raw_mapping = channel_mapping or {}
        self.channel_mapping = (
            {int(k): int(v) for k, v in raw_mapping.items()}
            if raw_mapping
            else {ch: ch for ch in self.channels}
        )
        names = channel_names or self._read_channel_names(self.channel_mapping.values())
        instrument_types = {}
        self.states = {}
        for audio_ch, mixer_ch in self.channel_mapping.items():
            role_info = self.role_classifier.classify(
                mixer_ch,
                names.get(mixer_ch, f"Ch {mixer_ch}"),
            )
            current_trim = self._read_trim(mixer_ch)
            self.states[mixer_ch] = LiveInputTrimChannelState(
                audio_channel=audio_ch,
                mixer_channel=mixer_ch,
                channel_name=role_info.channel_name,
                role=role_info.role,
                role_confidence=role_info.confidence,
                base_trim_db=current_trim,
                current_trim_db=current_trim,
                target_trim_db=current_trim,
            )
            instrument_types[mixer_ch] = self._role_to_bleed_instrument(
                role_info.role,
                role_info.channel_name,
            )
        self.bleed_detector.configure(instrument_types)
        if self.meter_reader and self.level_source in {"wing_meter", "hybrid"}:
            try:
                self.meter_reader.start(self.channel_mapping.values())
            except Exception as exc:
                logger.warning("Could not start WING meter reader: %s", exc)
        self.started_at = time.time()
        self.last_state = self._build_status([], {})
        if self.run_background_loop:
            self.start_background_loop()
        return self.last_state

    def start_background_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.meter_reader:
            try:
                self.meter_reader.stop()
            except Exception as exc:
                logger.debug("Could not stop WING meter reader: %s", exc)
        self.channels = []
        self.channel_mapping = {}
        self.states = {}
        self.started_at = None
        self.activity_detector = ActivityDetector(
            min_active_db=float(self.cfg.get("activity_threshold_db", -50.0)),
            noise_margin_db=float(self.cfg.get("noise_margin_db", 10.0)),
            attack_hold_sec=float(self.cfg.get("activity_attack_hold_sec", 0.2)),
            release_hold_sec=float(self.cfg.get("activity_release_hold_sec", 1.0)),
        )
        self.bleed_detector.reset()
        self.last_state = self._empty_status()

    def set_live_apply(
        self,
        enabled: bool,
        analysis_only_mode: Optional[bool] = None,
        confirm_live_apply: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.live_apply_enabled = bool(enabled)
        if analysis_only_mode is not None:
            self.analysis_only_mode = bool(analysis_only_mode)
        if confirm_live_apply is not None:
            self.confirm_live_apply = bool(confirm_live_apply)
        self.last_state["live_apply_enabled"] = self.live_apply_enabled
        self.last_state["analysis_only_mode"] = self.analysis_only_mode
        self.last_state["confirm_live_apply"] = self.confirm_live_apply
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        return dict(self.last_state)

    def get_recommendations(self) -> Dict[int, Dict[str, Any]]:
        """Return shared AutoGain recommendations for the latest channel state."""
        recommendations: Dict[int, Dict[str, Any]] = {}
        for mixer_ch, state in self.states.items():
            delta_db = float(state.target_trim_db) - float(state.current_trim_db)
            live_apply_safe = bool(
                not state.blocked_reason
                and self._can_apply(state)
                and abs(delta_db) >= self.deadband_db
            )
            lo, hi = self._role_trim_bounds(state.role)
            recommendation = AutoGainRecommendation(
                channel=mixer_ch,
                audio_channel=state.audio_channel,
                mixer_channel=mixer_ch,
                source="live_input_trim_controller",
                current_trim_db=round(state.current_trim_db, 3),
                recommended_target_trim_db=round(state.target_trim_db, 3),
                delta_db=round(delta_db, 3),
                reason=state.waiting_reason or "target_peak_alignment",
                confidence=round(state.main_signal_confidence, 4),
                limited_by=state.blocked_reason or "none",
                blocked_reason=state.blocked_reason,
                role=state.role,
                analysis_state=state.state,
                live_apply_safe=live_apply_safe,
                dry_run_only=not live_apply_safe,
                level_source=state.level_source,
                metrics={
                    "rms_db": round(state.last_rms_db, 3),
                    "peak_dbfs": round(state.last_peak_dbfs, 3),
                    "true_peak_dbtp": round(state.last_true_peak_dbtp, 3),
                    "usb_rms_db": round(state.usb_rms_db, 3),
                    "usb_peak_dbfs": round(state.usb_peak_dbfs, 3),
                    "usb_true_peak_dbtp": round(state.usb_true_peak_dbtp, 3),
                    "meter_rms_db": round(state.meter_rms_db, 3),
                    "meter_peak_dbfs": round(state.meter_peak_dbfs, 3),
                    "meter_true_peak_dbtp": round(state.meter_true_peak_dbtp, 3),
                },
                metadata={
                    "channel_name": state.channel_name,
                    "role_confidence": round(state.role_confidence, 4),
                    "main_signal_sec": round(state.main_signal_sec, 3),
                    "main_signal_windows": int(state.main_signal_windows),
                    "bleed_ratio": round(state.bleed_ratio, 4),
                    "bleed_confidence": round(state.bleed_confidence, 4),
                    "bleed_source_channel": state.bleed_source_channel,
                    "meter_trusted": bool(state.meter_trusted),
                    "meter_status": state.meter_status,
                    "analysis_only_mode": bool(self.analysis_only_mode),
                    "trim_bounds_db": {"min": lo, "max": hi},
                },
            )
            recommendations[mixer_ch] = recommendation.to_dict()
        return recommendations

    def build_replay_snapshot(
        self,
        *,
        event_seq: Optional[int] = None,
        replay_metadata: Optional[Dict[str, Any]] = None,
    ) -> LiveInputTrimReplaySnapshot:
        base_payload = {
            "version": "replay_state_snapshot/v1",
            "channels": [int(channel) for channel in self.channels],
            "channel_mapping": {
                str(int(audio_channel)): int(mixer_channel)
                for audio_channel, mixer_channel in sorted(self.channel_mapping.items())
            },
            "states": {
                str(int(channel)): state.to_dict()
                for channel, state in sorted(self.states.items())
            },
            "transport_policy": {
                "live_apply_enabled": bool(self.live_apply_enabled),
                "analysis_only_mode": bool(self.analysis_only_mode),
                "confirm_live_apply": bool(self.confirm_live_apply),
                "dry_run_only": bool(not self.live_apply_enabled or self.analysis_only_mode),
                "shadow_mode_enabled": bool(self._shadow_mode_enabled()),
                "level_source": self.level_source,
                "require_meter_for_live_apply": bool(self.require_meter_for_live_apply),
                "require_meter_for_analysis": bool(self.require_meter_for_analysis),
            },
            "controller_state": {
                "enabled": bool(self.enabled),
                "started_at": self.started_at,
                "update_interval_sec": float(self.update_interval_sec),
                "window_sec": float(self.window_sec),
                "cooldown_sec": float(self.cooldown_sec),
                "deadband_db": float(self.deadband_db),
                "max_step_db_per_tick": float(self.max_step_db_per_tick),
                "max_boost_step_db": float(self.max_boost_step_db),
            },
            "replay_metadata": {
                "source": "live_input_trim_controller",
                "event_seq": int(event_seq) if event_seq is not None else None,
                **dict(replay_metadata or {}),
            },
            "last_state": dict(self.last_state),
        }
        snapshot_id = LiveInputTrimReplaySnapshot.build_snapshot_id(base_payload)
        return LiveInputTrimReplaySnapshot(
            version=str(base_payload["version"]),
            snapshot_id=snapshot_id,
            channels=list(self.channels),
            channel_mapping=dict(self.channel_mapping),
            states={channel: LiveInputTrimChannelState.from_dict(state.to_dict()) for channel, state in self.states.items()},
            transport_policy=dict(base_payload["transport_policy"]),
            controller_state=dict(base_payload["controller_state"]),
            replay_metadata=dict(base_payload["replay_metadata"]),
            last_state=dict(self.last_state),
        )

    def restore_replay_snapshot(
        self,
        snapshot: LiveInputTrimReplaySnapshot | Dict[str, Any],
    ) -> LiveInputTrimReplaySnapshot:
        restored = (
            snapshot
            if isinstance(snapshot, LiveInputTrimReplaySnapshot)
            else LiveInputTrimReplaySnapshot.from_dict(dict(snapshot))
        )
        self.channels = [int(channel) for channel in restored.channels]
        self.channel_mapping = {
            int(audio_channel): int(mixer_channel)
            for audio_channel, mixer_channel in restored.channel_mapping.items()
        }
        self.states = {
            int(channel): LiveInputTrimChannelState.from_dict(state.to_dict())
            for channel, state in restored.states.items()
        }
        self.last_state = dict(restored.last_state)
        self.enabled = bool(restored.controller_state.get("enabled", self.enabled))
        self.started_at = _maybe_float(restored.controller_state.get("started_at"))
        self.live_apply_enabled = bool(
            restored.transport_policy.get("live_apply_enabled", self.live_apply_enabled)
        )
        self.analysis_only_mode = bool(
            restored.transport_policy.get("analysis_only_mode", self.analysis_only_mode)
        )
        self.confirm_live_apply = bool(
            restored.transport_policy.get("confirm_live_apply", self.confirm_live_apply)
        )
        self.level_source = str(restored.transport_policy.get("level_source", self.level_source))
        self.require_meter_for_live_apply = bool(
            restored.transport_policy.get(
                "require_meter_for_live_apply",
                self.require_meter_for_live_apply,
            )
        )
        self.require_meter_for_analysis = bool(
            restored.transport_policy.get(
                "require_meter_for_analysis",
                self.require_meter_for_analysis,
            )
        )
        return restored

    def replay_decision(
        self,
        decision: LiveInputTrimDecision | Dict[str, Any],
        *,
        mixer_client=None,
        live_apply_service: Optional[LiveApplyService] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Replay a recorded trim decision through LiveApplyService."""
        planned = self._coerce_trim_decision(decision)
        result = planned.to_dict()
        apply_result = self._apply_trim_decision(
            planned,
            mixer_client=mixer_client,
            live_apply_service=live_apply_service,
            dry_run=dry_run,
        )
        if apply_result is None:
            return result

        result["audit_id"] = apply_result.audit_id
        result["replay_correlation_id"] = apply_result.replay_correlation_id or result.get(
            "replay_correlation_id",
            "",
        )
        result["send_status"] = apply_result.send_status
        result["readback_status"] = apply_result.readback_status
        result["confirmed_value"] = (
            round(float(apply_result.confirmed_value), 3)
            if apply_result.confirmed_value is not None
            else None
        )
        result["guard_reasons"] = list(apply_result.guard_reasons)
        result["sent"] = bool(apply_result.accepted and not apply_result.dry_run)
        if apply_result.desired_value is not None:
            result["target_trim_db"] = round(float(apply_result.desired_value), 3)
        if apply_result.accepted and not apply_result.dry_run:
            if apply_result.confirmed_value is not None:
                confirmed = float(apply_result.confirmed_value)
                result["current_trim_db"] = round(confirmed, 3)
                result["target_trim_db"] = round(confirmed, 3)
        else:
            result["blocked_reason"] = (
                result.get("blocked_reason")
                or apply_result.blocked_reason
                or apply_result.failure_reason
                or "live_apply_not_accepted"
            )
        return result

    def process_once(
        self,
        audio_by_channel: Optional[Dict[int, np.ndarray]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one realtime analysis/apply tick."""
        now = time.time() if timestamp is None else float(timestamp)
        audio = audio_by_channel or self._read_audio_buffers()
        metrics = {}
        band_energy = {}
        levels = {}
        centroids = {}
        metric_objects = {}
        level_metrics = {}
        meter_readings = self._read_meter_metrics(now)

        for mixer_ch, state in self.states.items():
            samples = _as_mono(audio.get(state.audio_channel, np.array([], dtype=np.float32)))
            metrics[mixer_ch] = self._channel_metrics(samples)
            level_metrics[mixer_ch] = self._select_level_metrics(
                metrics[mixer_ch],
                meter_readings.get(mixer_ch),
                state,
            )
            band_energy[mixer_ch] = _band_energy(samples, self._sample_rate)
            levels[mixer_ch] = metrics[mixer_ch]["rms_db"]
            centroids[mixer_ch] = _spectral_centroid(band_energy[mixer_ch])
            metric_objects[mixer_ch] = _metric_object(metrics[mixer_ch], band_energy[mixer_ch])

        bleed_by_channel = {}
        for mixer_ch, state in self.states.items():
            info = self.bleed_detector.detect_bleed(
                mixer_ch,
                levels.get(mixer_ch, -120.0),
                centroids.get(mixer_ch, 0.0),
                levels,
                centroids,
                metric_objects,
            )
            bleed_by_channel[mixer_ch] = info

        applied = []
        blocked = {}
        for mixer_ch, state in self.states.items():
            result = self._process_channel(
                state,
                metrics[mixer_ch],
                level_metrics[mixer_ch],
                band_energy[mixer_ch],
                bleed_by_channel[mixer_ch],
                metric_objects,
                now,
            )
            if result.get("sent"):
                applied.append(result)
            if result.get("blocked_reason"):
                blocked[mixer_ch] = result

        self.last_state = self._build_status(applied, blocked)
        if self.status_callback:
            self.status_callback(self.last_state)
        return self.last_state

    def _run_loop(self) -> None:
        logger.info("Live input trim loop started")
        while not self._stop_event.is_set():
            try:
                if self.channels:
                    self.process_once()
            except Exception as exc:
                logger.error("Live input trim loop error: %s", exc, exc_info=True)
            self._stop_event.wait(self.update_interval_sec)
        logger.info("Live input trim loop stopped")

    @staticmethod
    def _coerce_trim_decision(
        decision: LiveInputTrimDecision | Dict[str, Any],
    ) -> LiveInputTrimDecision:
        if isinstance(decision, LiveInputTrimDecision):
            return decision
        return LiveInputTrimDecision.from_dict(dict(decision))

    @staticmethod
    def _apply_trim_decision(
        decision: LiveInputTrimDecision | Dict[str, Any],
        *,
        mixer_client,
        live_apply_service: Optional[LiveApplyService],
        dry_run: bool,
    ):
        planned = LiveInputTrimController._coerce_trim_decision(decision)
        if not planned.write_plan:
            return None
        service = live_apply_service or LiveApplyService()
        request = planned.write_plan.to_live_apply_request(dry_run=dry_run)
        return service.apply(request, mixer_client)

    def _evaluate_trim_decision(
        self,
        state: LiveInputTrimChannelState,
        metrics: Dict[str, float],
        level_metrics: Dict[str, float],
        now: float,
        *,
        ready: bool,
        bleed_dominant: bool,
    ) -> LiveInputTrimDecision:
        """Extract the replayable trim decision before any live apply occurs."""

        target_info = self._target_for_role(state.role)
        target_peak_dbfs = float(target_info["peak_dbfs"])
        hot_peak = level_metrics["true_peak_dbtp"] > self.hot_peak_threshold_dbfs
        desired_delta = target_peak_dbfs - level_metrics["true_peak_dbtp"]
        reason = "target_peak_alignment"
        if hot_peak:
            desired_delta = self.hot_peak_target_dbfs - level_metrics["true_peak_dbtp"]
            reason = "hot_peak_guard"
        elif self.percussive_peak_mode and state.role in PERCUSSIVE_ROLES:
            if level_metrics["true_peak_dbtp"] < target_peak_dbfs - self.percussive_boost_margin_db:
                desired_delta = target_peak_dbfs - level_metrics["true_peak_dbtp"]
                reason = "percussive_low_peak_recovery"
            else:
                desired_delta = 0.0
                reason = "percussive_headroom_ok"

        blocked_reason = ""
        meter_required_for_decision = (
            self.require_meter_for_analysis
            and self.level_source == "wing_meter"
            and not state.meter_trusted
        )
        if meter_required_for_decision:
            blocked_reason = "meter_unavailable_for_level_decision"
            desired_delta = 0.0
            reason = "meter_unavailable"
        if desired_delta > 0.0:
            if not ready:
                blocked_reason = "waiting_for_main_signal"
            elif bleed_dominant:
                blocked_reason = "blocked_positive_trim_on_bleed_dominant_channel"
            desired_delta = min(desired_delta, self.max_boost_step_db)
        else:
            if (
                desired_delta < 0.0
                and not hot_peak
                and self.cut_requires_ready
                and not ready
            ):
                blocked_reason = "waiting_for_main_signal_before_cut"
            elif (
                desired_delta < 0.0
                and not hot_peak
                and self.block_bleed_regular_moves
                and bleed_dominant
            ):
                blocked_reason = "blocked_regular_cut_on_bleed_dominant_channel"
            desired_delta = max(desired_delta, -self.max_step_db_per_tick)

        desired_delta = max(-self.max_step_db_per_tick, min(self.max_step_db_per_tick, desired_delta))
        if desired_delta > 0.0 and self._should_hold_lower_trim_limit(
            state,
            level_metrics,
            target_peak_dbfs,
        ):
            blocked_reason = blocked_reason or "lower_trim_limit_hold"
            desired_delta = 0.0
        if blocked_reason:
            desired_delta = 0.0

        target_trim = self._clamp_trim_by_role(
            state.role,
            state.current_trim_db + desired_delta,
        )
        actual_delta = target_trim - state.current_trim_db
        if abs(actual_delta) < self.deadband_db:
            blocked_reason = blocked_reason or "deadband_no_send"
            actual_delta = 0.0
            target_trim = state.current_trim_db
        if state.last_apply_time is not None and now - state.last_apply_time < self.cooldown_sec:
            blocked_reason = blocked_reason or "cooldown"
            actual_delta = 0.0
            target_trim = state.current_trim_db

        return self._build_trim_decision(
            state,
            metrics,
            level_metrics,
            reason,
            blocked_reason,
            target_trim,
            actual_delta,
        )

    def _execute_trim_decision(
        self,
        state: LiveInputTrimChannelState,
        decision: LiveInputTrimDecision,
        now: float,
    ) -> LiveInputTrimDecision:
        """Execute a precomputed trim decision through the guarded apply path."""

        blocked_reason = decision.blocked_reason
        target_trim = decision.target_trim_db
        actual_delta = decision.delta_db
        if decision.write_plan and self._can_apply(state):
            try:
                previous_trim = state.current_trim_db
                apply_result = self._apply_trim_decision(
                    decision,
                    mixer_client=self.mixer_client,
                    live_apply_service=self.live_apply_service,
                    dry_run=False,
                )
                if apply_result is None:
                    state.target_trim_db = target_trim
                else:
                    decision.audit_id = apply_result.audit_id
                    decision.replay_correlation_id = (
                        apply_result.replay_correlation_id or decision.replay_correlation_id
                    )
                    decision.send_status = apply_result.send_status
                    decision.readback_status = apply_result.readback_status
                    decision.confirmed_value = apply_result.confirmed_value
                    decision.guard_reasons = tuple(apply_result.guard_reasons)
                if apply_result is not None and apply_result.accepted and not apply_result.dry_run:
                    verified = (
                        float(apply_result.confirmed_value)
                        if apply_result.confirmed_value is not None
                        else self._read_trim(state.mixer_channel, fallback=target_trim)
                    )
                    state.current_trim_db = verified
                    state.target_trim_db = verified
                    state.last_apply_time = now
                    state.last_applied_delta_db = verified - previous_trim
                    state.sent = True
                    state.state = "hold"
                    decision.current_trim_db = verified
                    decision.target_trim_db = verified
                    decision.delta_db = verified - previous_trim
                    decision.state = state.state
                    decision.sent = True
                    logger.info(
                        "Live input trim ch%s %s: %.2f dB -> %.2f dB (%s)",
                        state.mixer_channel,
                        state.channel_name,
                        previous_trim,
                        verified,
                        decision.reason,
                    )
                else:
                    blocked_reason = blocked_reason or (
                        apply_result.blocked_reason
                        if apply_result is not None
                        else None
                    ) or (
                        apply_result.failure_reason
                        if apply_result is not None
                        else None
                    ) or "live_apply_not_accepted"
                    desired_value = (
                        apply_result.desired_value
                        if apply_result is not None
                        and apply_result.desired_value is not None
                        else target_trim
                    )
                    state.target_trim_db = float(desired_value)
                    decision.target_trim_db = state.target_trim_db
            except Exception as exc:
                blocked_reason = f"osc_apply_failed:{exc}"
                logger.error("Live input trim OSC apply failed ch%s: %s", state.mixer_channel, exc)
        else:
            state.target_trim_db = target_trim
            if decision.write_plan and not self._can_apply(state):
                blocked_reason = blocked_reason or self._apply_block_reason(state)
            if not actual_delta and state.state == "hold":
                state.state = "monitoring"

        decision.blocked_reason = blocked_reason
        return decision

    def _build_trim_decision(
        self,
        state: LiveInputTrimChannelState,
        metrics: Dict[str, float],
        level_metrics: Dict[str, float],
        reason: str,
        blocked_reason: str,
        target_trim: float,
        actual_delta: float,
    ) -> LiveInputTrimDecision:
        replay_correlation_id = self._build_replay_correlation_id(
            state=state,
            reason=reason,
            target_trim=target_trim,
            actual_delta=actual_delta,
        )
        write_plan = None
        if actual_delta:
            lo, hi = self._role_trim_bounds(state.role)
            write_plan = LiveInputTrimWritePlan(
                source="auto_gain_live_input_trim",
                operation="set_gain",
                channel=state.mixer_channel,
                value=target_trim,
                confirm_live_apply=bool(self.confirm_live_apply),
                current_value=state.current_trim_db,
                min_value=lo,
                max_value=hi,
                max_step=self.max_step_db_per_tick,
                replay_correlation_id=replay_correlation_id,
                metadata={
                    "reason": reason,
                    "role": state.role,
                    "channel_name": state.channel_name,
                    "level_source": state.level_source,
                    "replay_correlation_id": replay_correlation_id,
                },
            )
        return LiveInputTrimDecision(
            channel=state.mixer_channel,
            name=state.channel_name,
            role=state.role,
            state=state.state,
            reason=reason,
            blocked_reason=blocked_reason,
            current_trim_db=state.current_trim_db,
            target_trim_db=target_trim,
            delta_db=actual_delta,
            main_signal_confidence=state.main_signal_confidence,
            main_signal_sec=state.main_signal_sec,
            main_signal_windows=state.main_signal_windows,
            bleed_ratio=state.bleed_ratio,
            bleed_confidence=state.bleed_confidence,
            bleed_source_channel=state.bleed_source_channel,
            level_source=state.level_source,
            rms_db=level_metrics["rms_db"],
            peak_dbfs=level_metrics["peak_dbfs"],
            true_peak_dbtp=level_metrics["true_peak_dbtp"],
            usb_rms_db=metrics["rms_db"],
            usb_peak_dbfs=metrics["peak_dbfs"],
            usb_true_peak_dbtp=metrics["true_peak_dbtp"],
            meter_status=state.meter_status,
            meter_trusted=state.meter_trusted,
            meter_peak_dbfs=state.meter_peak_dbfs,
            meter_rms_db=state.meter_rms_db,
            meter_true_peak_dbtp=state.meter_true_peak_dbtp,
            meter_age_sec=state.meter_age_sec,
            meter_activity_fallback=state.meter_activity_fallback,
            write_plan=write_plan,
            replay_correlation_id=replay_correlation_id,
        )

    @staticmethod
    def _build_replay_correlation_id(
        *,
        state: LiveInputTrimChannelState,
        reason: str,
        target_trim: float,
        actual_delta: float,
    ) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source": "auto_gain_live_input_trim",
                    "channel": int(state.mixer_channel),
                    "role": state.role,
                    "reason": reason,
                    "current_trim_db": round(float(state.current_trim_db), 3),
                    "target_trim_db": round(float(target_trim), 3),
                    "delta_db": round(float(actual_delta), 3),
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        return f"replay::live_input_trim::{digest[:16]}"

    def _process_channel(
        self,
        state: LiveInputTrimChannelState,
        metrics: Dict[str, float],
        level_metrics: Dict[str, float],
        bands: Dict[str, float],
        bleed_info: BleedInfo,
        metric_objects: Dict[int, Any],
        now: float,
    ) -> Dict[str, Any]:
        dt = self.window_sec
        if state.last_update_time is not None:
            dt = max(0.0, min(2.0, now - state.last_update_time))
        state.last_update_time = now
        state.sent = False

        state.usb_rms_db = metrics["rms_db"]
        state.usb_peak_dbfs = metrics["peak_dbfs"]
        state.usb_true_peak_dbtp = metrics["true_peak_dbtp"]
        state.last_rms_db = level_metrics["rms_db"]
        state.last_peak_dbfs = level_metrics["peak_dbfs"]
        state.last_true_peak_dbtp = level_metrics["true_peak_dbtp"]
        state.peak_hold_dbfs = max(state.peak_hold_dbfs, level_metrics["peak_dbfs"])
        state.bleed_ratio = float(bleed_info.bleed_ratio)
        state.bleed_confidence = float(bleed_info.confidence)
        state.bleed_source_channel = bleed_info.bleed_source_channel
        state.bleed_method = bleed_info.method_used

        if metrics["rms_db"] < state.noise_floor_db + 6.0:
            state.noise_floor_db = (
                (1.0 - self.noise_floor_alpha) * state.noise_floor_db
                + self.noise_floor_alpha * metrics["rms_db"]
            )

        activity = self.activity_detector.update(
            state.mixer_channel,
            rms_db=metrics["rms_db"],
            peak_db=metrics["peak_dbfs"],
            short_term_loudness_db=metrics["rms_db"],
            spectral_energy_db=max(bands.values()) if bands else metrics["rms_db"],
            noise_floor_db=state.noise_floor_db,
            timestamp=now,
        )
        source_metrics = (
            metric_objects.get(bleed_info.bleed_source_channel)
            if bleed_info.bleed_source_channel
            else None
        )
        state.compensated_level_db = get_compensated_level(
            metrics["rms_db"],
            bleed_info,
            channel_metrics=metric_objects.get(state.mixer_channel),
            source_metrics=source_metrics,
            compensation_factor_db=float(self.cfg.get("bleed_compensation_factor_db", 6.0)),
            compensation_mode=str(self.cfg.get("bleed_compensation_mode", "band_level")),
        )
        confidence_metrics = metrics
        confidence_activity = activity.activity_confidence
        state.meter_activity_fallback = self._should_use_meter_activity_fallback(
            metrics,
            level_metrics,
            state,
        )
        if state.meter_activity_fallback:
            confidence_metrics = level_metrics
            confidence_activity = max(
                confidence_activity,
                self._meter_activity_confidence(level_metrics, state),
            )

        state.main_signal_confidence = self._main_signal_confidence(
            state,
            confidence_activity,
            confidence_metrics,
            bands,
            bleed_info,
        )
        role_transient_evidence = self._has_role_transient_evidence(state, metrics, bands)
        bleed_dominant = self._is_bleed_dominant(bleed_info)
        if (
            bleed_dominant
            and role_transient_evidence
            and state.main_signal_confidence >= self.min_main_confidence
        ):
            bleed_dominant = False
        own_signal = state.main_signal_confidence >= self.min_main_confidence and not bleed_dominant
        if own_signal:
            state.main_signal_sec += dt
            state.main_signal_windows += 1
            if state.state == "waiting_for_signal":
                state.state = "learning_main_signal"
        else:
            state.main_signal_sec = max(0.0, state.main_signal_sec - dt * 0.5)
            state.main_signal_windows = max(0, state.main_signal_windows - 1)
            if state.main_signal_sec <= 0.0 and state.main_signal_windows == 0:
                state.state = "waiting_for_signal"

        ready = self._has_enough_main_signal(state)
        if ready and state.state in {"waiting_for_signal", "learning_main_signal"}:
            state.state = "ready_to_adjust"

        decision = self._evaluate_trim_decision(
            state,
            metrics,
            level_metrics,
            now,
            ready=ready,
            bleed_dominant=bleed_dominant,
        )
        decision = self._execute_trim_decision(state, decision, now)

        state.blocked_reason = decision.blocked_reason
        state.waiting_reason = decision.blocked_reason or decision.reason
        decision.state = state.state
        decision.blocked_reason = state.blocked_reason
        decision.current_trim_db = state.current_trim_db
        decision.target_trim_db = state.target_trim_db
        decision.main_signal_confidence = state.main_signal_confidence
        decision.main_signal_sec = state.main_signal_sec
        decision.main_signal_windows = state.main_signal_windows
        decision.bleed_ratio = state.bleed_ratio
        decision.bleed_confidence = state.bleed_confidence
        decision.bleed_source_channel = state.bleed_source_channel
        decision.level_source = state.level_source
        decision.meter_status = state.meter_status
        decision.meter_trusted = state.meter_trusted
        decision.meter_peak_dbfs = state.meter_peak_dbfs
        decision.meter_rms_db = state.meter_rms_db
        decision.meter_true_peak_dbtp = state.meter_true_peak_dbtp
        decision.meter_age_sec = state.meter_age_sec
        decision.meter_activity_fallback = state.meter_activity_fallback
        return decision.to_dict()

    def _main_signal_confidence(
        self,
        state: LiveInputTrimChannelState,
        activity_confidence: float,
        metrics: Dict[str, float],
        bands: Dict[str, float],
        bleed_info: BleedInfo,
    ) -> float:
        threshold = max(
            float(self.cfg.get("activity_threshold_db", -50.0)),
            state.noise_floor_db + float(self.cfg.get("noise_margin_db", 10.0)),
        )
        level_conf = _clamp01((max(metrics["rms_db"], metrics["peak_dbfs"] - 12.0) - threshold) / 18.0)
        role_conf = self._role_band_confidence(state.role, bands, metrics["rms_db"])
        transient_bonus = 0.0
        if state.role in {"kick", "snare", "tom"} and metrics["peak_dbfs"] >= threshold + 10.0:
            transient_bonus = 0.2
        bleed_penalty = float(bleed_info.bleed_ratio) * float(bleed_info.confidence)
        return _clamp01(
            0.42 * activity_confidence
            + 0.30 * level_conf
            + 0.22 * role_conf
            + transient_bonus
            - 0.38 * bleed_penalty
        )

    @staticmethod
    def _role_band_confidence(role: str, bands: Dict[str, float], rms_db: float) -> float:
        role_bands = {
            "kick": ("sub", "bass"),
            "bass": ("sub", "bass", "low_mid"),
            "snare": ("low_mid", "mid", "high_mid"),
            "tom": ("bass", "low_mid", "mid"),
            "overhead": ("high_mid", "high", "air"),
            "lead_vocal": ("mid", "high_mid"),
            "backing_vocal": ("mid", "high_mid"),
            "rhythm_guitar": ("low_mid", "mid", "high_mid"),
            "lead_guitar": ("mid", "high_mid"),
            "acoustic_guitar": ("mid", "high_mid", "high"),
            "keys": ("low_mid", "mid", "high_mid"),
            "pad": ("low_mid", "mid", "high_mid"),
        }.get(role, tuple(BAND_RANGES.keys()))
        if not bands:
            return 0.0
        band_level = max(float(bands.get(name, -120.0)) for name in role_bands)
        return _clamp01((band_level - rms_db + 10.0) / 18.0)

    def _has_enough_main_signal(self, state: LiveInputTrimChannelState) -> bool:
        target = self._target_for_role(state.role)
        return (
            state.main_signal_sec >= float(target["main_sec"])
            and state.main_signal_windows >= int(target["windows"])
        )

    def _is_bleed_dominant(self, bleed_info: BleedInfo) -> bool:
        return (
            float(bleed_info.bleed_ratio) >= self.high_bleed_ratio
            and float(bleed_info.confidence) >= self.high_bleed_confidence
            and bleed_info.bleed_source_channel is not None
        )

    def _has_role_transient_evidence(
        self,
        state: LiveInputTrimChannelState,
        metrics: Dict[str, float],
        bands: Dict[str, float],
    ) -> bool:
        if state.role not in PERCUSSIVE_ROLES:
            return False
        threshold = max(
            float(self.cfg.get("activity_threshold_db", -50.0)),
            state.noise_floor_db + float(self.cfg.get("noise_margin_db", 10.0)),
        )
        crest_db = metrics["peak_dbfs"] - metrics["rms_db"]
        return (
            metrics["peak_dbfs"] >= threshold + 10.0
            and crest_db >= float(self.cfg.get("percussive_min_crest_db", 10.0))
            and self._role_band_confidence(state.role, bands, metrics["rms_db"]) >= 0.25
        )

    def _should_hold_lower_trim_limit(
        self,
        state: LiveInputTrimChannelState,
        metrics: Dict[str, float],
        target_peak_dbfs: float,
    ) -> bool:
        lo, _ = self._role_trim_bounds(state.role)
        if state.current_trim_db > lo + self.deadband_db:
            return False
        return metrics["true_peak_dbtp"] > target_peak_dbfs - self.limit_release_margin_db

    def _target_for_role(self, role: str) -> Dict[str, float]:
        return dict(self.role_targets.get(role, self.role_targets["unknown"]))

    def _role_trim_bounds(self, role: str) -> tuple[float, float]:
        role_limits = self.role_trim_limits.get(role, self.role_trim_limits["unknown"])
        lo = max(self.trim_min_db, float(role_limits.get("min_db", self.trim_min_db)))
        hi = min(self.trim_max_db, float(role_limits.get("max_db", self.trim_max_db)))
        return lo, hi

    def _clamp_trim_by_role(self, role: str, trim_db: float) -> float:
        lo, hi = self._role_trim_bounds(role)
        return max(lo, min(hi, float(trim_db)))

    def _can_apply(self, state: Optional[LiveInputTrimChannelState] = None) -> bool:
        if (
            state is not None
            and self.require_meter_for_live_apply
            and self.level_source in {"wing_meter", "hybrid"}
            and not state.meter_trusted
        ):
            return False
        return (
            self.enabled
            and self.live_apply_enabled
            and not self.analysis_only_mode
            and self.mixer_client is not None
            and bool(getattr(self.mixer_client, "is_connected", False))
        )

    def _apply_block_reason(self, state: LiveInputTrimChannelState) -> str:
        if (
            self.require_meter_for_live_apply
            and self.level_source in {"wing_meter", "hybrid"}
            and not state.meter_trusted
        ):
            return "meter_unavailable_for_live_apply"
        return "analysis_only_or_live_apply_disabled"

    def _should_use_meter_activity_fallback(
        self,
        usb_metrics: Dict[str, float],
        level_metrics: Dict[str, float],
        state: LiveInputTrimChannelState,
    ) -> bool:
        return (
            self.meter_activity_fallback
            and state.meter_trusted
            and self.level_source in {"wing_meter", "hybrid"}
            and usb_metrics["peak_dbfs"] <= self.usb_silence_dbfs
            and level_metrics["peak_dbfs"] > self.usb_silence_dbfs
        )

    def _meter_activity_confidence(
        self,
        level_metrics: Dict[str, float],
        state: LiveInputTrimChannelState,
    ) -> float:
        threshold = max(
            float(self.cfg.get("activity_threshold_db", -50.0)),
            state.noise_floor_db + float(self.cfg.get("noise_margin_db", 10.0)),
        )
        level = max(level_metrics["rms_db"], level_metrics["peak_dbfs"] - 12.0)
        return _clamp01((level - threshold) / 18.0)

    def _read_meter_metrics(self, now: float) -> Dict[int, WingMeterReading]:
        if self.level_source not in {"wing_meter", "hybrid"} or not self.meter_reader:
            return {}
        try:
            return self.meter_reader.get_metrics(self.states.keys(), now=now)
        except Exception as exc:
            logger.debug("Could not read WING meter metrics: %s", exc)
            return {}

    def _select_level_metrics(
        self,
        usb_metrics: Dict[str, float],
        meter_reading: Optional[WingMeterReading],
        state: LiveInputTrimChannelState,
    ) -> Dict[str, float]:
        self._update_meter_state(state, meter_reading)
        if self.level_source == "wing_meter":
            if state.meter_trusted:
                state.level_source = "wing_meter"
                return {
                    "rms_db": state.meter_rms_db,
                    "peak_dbfs": state.meter_peak_dbfs,
                    "true_peak_dbtp": state.meter_true_peak_dbtp,
                }
            state.level_source = "wing_meter_unavailable"
            return dict(usb_metrics)
        if self.level_source == "hybrid" and state.meter_trusted:
            state.level_source = "wing_meter"
            return {
                "rms_db": state.meter_rms_db,
                "peak_dbfs": state.meter_peak_dbfs,
                "true_peak_dbtp": state.meter_true_peak_dbtp,
            }
        state.level_source = "usb_audio"
        return dict(usb_metrics)

    @staticmethod
    def _update_meter_state(
        state: LiveInputTrimChannelState,
        meter_reading: Optional[WingMeterReading],
    ) -> None:
        if not meter_reading:
            state.meter_status = "not_configured"
            state.meter_trusted = False
            state.meter_confidence = 0.0
            state.meter_age_sec = None
            return
        state.meter_status = meter_reading.status
        state.meter_trusted = bool(meter_reading.trusted)
        state.meter_confidence = float(meter_reading.confidence)
        state.meter_peak_dbfs = float(meter_reading.peak_dbfs)
        state.meter_rms_db = float(meter_reading.rms_db)
        state.meter_true_peak_dbtp = float(meter_reading.true_peak_dbtp)
        state.meter_age_sec = meter_reading.age_sec

    def _channel_metrics(self, samples: np.ndarray) -> Dict[str, float]:
        if samples.size == 0:
            return {"rms_db": -120.0, "peak_dbfs": -120.0, "true_peak_dbtp": -120.0}
        arr = np.asarray(samples, dtype=np.float64)
        rms = math.sqrt(float(np.mean(arr * arr)) + 1e-12)
        peak = float(np.max(np.abs(arr)))
        return {
            "rms_db": _amp_to_db(rms),
            "peak_dbfs": _amp_to_db(peak),
            "true_peak_dbtp": _true_peak_dbtp(arr),
        }

    def _read_audio_buffers(self) -> Dict[int, np.ndarray]:
        if not self.audio_capture or not getattr(self.audio_capture, "running", False):
            return {}
        samples = max(1, int(self._sample_rate * self.window_sec))
        return {
            audio_ch: self.audio_capture.get_buffer(audio_ch, samples)
            for audio_ch in self.channel_mapping
        }

    @property
    def _sample_rate(self) -> int:
        if self.audio_capture is not None:
            return int(getattr(self.audio_capture, "sample_rate", 48_000))
        return int(self.cfg.get("sample_rate", 48_000))

    def _read_channel_names(self, mixer_channels: Iterable[int]) -> Dict[int, str]:
        names = {}
        channel_list = list(mixer_channels)
        if self.mixer_client and hasattr(self.mixer_client, "get_all_channel_names"):
            try:
                names.update(
                    self.mixer_client.get_all_channel_names(max(channel_list or [40]))
                )
            except Exception as exc:
                logger.debug("Could not read channel names from mixer: %s", exc)
        for channel_id in channel_list:
            if (
                channel_id not in names
                and self.mixer_client
                and hasattr(self.mixer_client, "get_channel_name")
            ):
                try:
                    names[channel_id] = self.mixer_client.get_channel_name(channel_id)
                except Exception:
                    names[channel_id] = f"Ch {channel_id}"
            names.setdefault(channel_id, f"Ch {channel_id}")
        return names

    def _read_trim(self, channel_id: int, fallback: float = 0.0) -> float:
        address = f"/ch/{channel_id}/in/set/trim"
        timeout = float(self.cfg.get("trim_query_timeout_sec", 0.35))
        if self.mixer_client and hasattr(self.mixer_client, "get_channel_gain_live"):
            try:
                value = self.mixer_client.get_channel_gain_live(
                    channel_id,
                    timeout=timeout,
                )
                if value is not None and math.isfinite(float(value)):
                    return float(value)
            except Exception as exc:
                logger.debug("Could not live-query trim for channel %s: %s", channel_id, exc)
        if self.mixer_client and hasattr(self.mixer_client, "query_parameter"):
            try:
                value = self.mixer_client.query_parameter(address, timeout=timeout)
                if value is not None and math.isfinite(float(value)):
                    return float(value)
            except Exception as exc:
                logger.debug("Could not query trim parameter for channel %s: %s", channel_id, exc)
        if self.mixer_client and hasattr(self.mixer_client, "send"):
            try:
                self.mixer_client.send(address)
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if hasattr(self.mixer_client, "state"):
                        value = self.mixer_client.state.get(address)
                        if value is not None and math.isfinite(float(value)):
                            return float(value)
                    time.sleep(0.01)
            except Exception as exc:
                logger.debug("Could not query trim for channel %s: %s", channel_id, exc)
        if self.mixer_client and hasattr(self.mixer_client, "get_channel_gain"):
            try:
                value = self.mixer_client.get_channel_gain(channel_id)
                if value is not None and math.isfinite(float(value)):
                    return float(value)
            except Exception as exc:
                logger.debug("Could not read trim for channel %s: %s", channel_id, exc)
        return float(fallback)

    @staticmethod
    def _role_to_bleed_instrument(role: str, name: str) -> str:
        key = name.lower()
        if "playback" in key or "player" in key:
            return "playback"
        if role in {"kick", "snare", "overhead", "bass"}:
            return role
        if role == "tom":
            return "tom"
        return "unknown"

    def _build_status(
        self,
        applied: list[Dict[str, Any]],
        blocked: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "active": bool(self.channels),
            "enabled": self.enabled,
            "live_apply_enabled": self.live_apply_enabled,
            "analysis_only_mode": self.analysis_only_mode,
            "confirm_live_apply": self.confirm_live_apply,
            "shadow_mode_enabled": self._shadow_mode_enabled(),
            "level_source": self.level_source,
            "require_meter_for_live_apply": self.require_meter_for_live_apply,
            "require_meter_for_analysis": self.require_meter_for_analysis,
            "meter_status": self._meter_status_summary(),
            "started_at": self.started_at,
            "channels": {ch: state.to_dict() for ch, state in self.states.items()},
            "applied": applied,
            "blocked": blocked,
            "waiting_channels": [
                ch for ch, state in self.states.items() if state.state == "waiting_for_signal"
            ],
            "ready_channels": [
                ch for ch, state in self.states.items() if state.state == "ready_to_adjust"
            ],
            "message": "live input trim running" if self.channels else "idle",
        }

    @staticmethod
    def _empty_status() -> Dict[str, Any]:
        return {
            "active": False,
            "enabled": False,
            "live_apply_enabled": False,
            "analysis_only_mode": True,
            "confirm_live_apply": False,
            "shadow_mode_enabled": False,
            "level_source": "usb_audio",
            "require_meter_for_live_apply": False,
            "require_meter_for_analysis": False,
            "meter_status": {"enabled": False, "status": "idle", "trusted": False},
            "channels": {},
            "applied": [],
            "blocked": {},
            "waiting_channels": [],
            "ready_channels": [],
            "message": "idle",
        }

    def _shadow_mode_enabled(self) -> bool:
        if self.mixer_client is None:
            return False
        checker = getattr(self.mixer_client, "is_shadow_mode_enabled", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                logger.debug("Could not read shadow mode flag from mixer client", exc_info=True)
        return False

    def _meter_status_summary(self) -> Dict[str, Any]:
        if not self.meter_reader or self.level_source == "usb_audio":
            return {"enabled": False, "status": "not_configured", "trusted": False}
        try:
            return self.meter_reader.get_status()
        except Exception as exc:
            return {"enabled": True, "status": f"error:{exc}", "trusted": False}

    @staticmethod
    def _get_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
        automation = config.get("automation", {}) if config else {}
        return dict(
            config.get("live_input_trim")
            or automation.get("live_input_trim")
            or {}
        )


def _merge_role_map(defaults: Dict[str, Dict[str, float]], updates: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    merged = {role: dict(values) for role, values in defaults.items()}
    for role, values in (updates or {}).items():
        current = dict(merged.get(role, merged["unknown"]))
        current.update({key: float(value) for key, value in dict(values).items()})
        merged[role] = current
    return merged


def _metric_object(metrics: Dict[str, float], bands: Dict[str, float]):
    return type(
        "LiveInputTrimMetrics",
        (),
        {
            "rms_level": metrics["rms_db"],
            "band_energy_sub": float(bands.get("sub", -100.0)),
            "band_energy_bass": float(bands.get("bass", -100.0)),
            "band_energy_low_mid": float(bands.get("low_mid", -100.0)),
            "band_energy_mid": float(bands.get("mid", -100.0)),
            "band_energy_high_mid": float(bands.get("high_mid", -100.0)),
            "band_energy_high": float(bands.get("high", -100.0)),
            "band_energy_air": float(bands.get("air", -100.0)),
        },
    )()


def _as_mono(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1).astype(np.float32)


def _band_energy(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    x = _as_mono(samples)
    if x.size < 256:
        return {name: -120.0 for name in BAND_RANGES}
    n = min(x.size, int(sample_rate * 2.0))
    block = x[-n:].astype(np.float64, copy=False)
    windowed = block * np.hanning(block.size)
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(windowed.size, 1.0 / sample_rate)
    out = {}
    for name, (lo, hi) in BAND_RANGES.items():
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            out[name] = -120.0
            continue
        band_rms = float(np.sqrt(np.mean(np.square(spec[idx]))) / max(1, windowed.size))
        out[name] = _amp_to_db(band_rms)
    return out


def _spectral_centroid(bands: Dict[str, float]) -> float:
    centers = {
        "sub": 40.0,
        "bass": 120.0,
        "low_mid": 375.0,
        "mid": 1200.0,
        "high_mid": 3000.0,
        "high": 6000.0,
        "air": 11000.0,
    }
    weights = []
    weighted = []
    for name, center in centers.items():
        weight = 10.0 ** (float(bands.get(name, -120.0)) / 20.0)
        weights.append(weight)
        weighted.append(weight * center)
    return float(sum(weighted) / max(sum(weights), 1e-12))


def _true_peak_dbtp(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    try:
        from scipy import signal

        upsampled = signal.resample_poly(samples.astype(np.float64, copy=False), 4, 1)
        return _amp_to_db(float(np.max(np.abs(upsampled))))
    except Exception:
        return _amp_to_db(float(np.max(np.abs(samples))))


def _amp_to_db(value: float, floor_db: float = -120.0) -> float:
    value = abs(float(value))
    if value <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
