"""Safety checks specific to Mixing Station visualization/control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .config import load_yaml_file, resolve_repo_path
from .models import AutomixCorrection


DESTRUCTIVE_PREFIXES = (
    "scene.",
    "snapshot.",
    "cue_list.",
    "soft_key.",
    "action.",
    "destructive.",
)
FORBIDDEN_PARAMETERS = {
    "phantom_power",
    "phantom_power.enabled",
    "scene.recall",
    "scene.store",
    "snapshot.recall",
    "cue_list.recall",
    "soft_key.trigger",
    "action.trigger",
}


@dataclass(frozen=True)
class MixingStationSafetyConfig:
    """Configurable safety limits for Mixing Station commands."""

    enabled: bool = True
    dry_run_default: bool = True
    max_fader_step_db: float = 1.5
    max_send_step_db: float = 2.0
    max_eq_gain_step_db: float = 1.5
    max_eq_gain_absolute_db: float = 12.0
    max_eq_boost_db: float = 6.0
    min_hpf_hz: float = 20.0
    max_hpf_hz: float = 250.0
    min_delay_ms: float = 0.0
    max_delay_ms: float = 10.0
    min_compressor_threshold_db: float = -60.0
    max_compressor_threshold_db: float = 0.0
    min_compressor_ratio: float = 1.0
    max_compressor_ratio: float = 20.0
    min_compressor_mix: float = 0.0
    max_compressor_mix: float = 100.0
    min_compressor_makeup_gain_db: float = -12.0
    max_compressor_makeup_gain_db: float = 12.0
    min_compressor_time_ms: float = 0.1
    max_compressor_time_ms: float = 2000.0
    min_compressor_filter_hz: float = 20.0
    max_compressor_filter_hz: float = 20000.0
    min_compressor_filter_q: float = 0.1
    max_compressor_filter_q: float = 10.0
    allow_scene_recall: bool = False
    allow_phantom_power: bool = False
    allow_live_control: bool = False
    rate_limit_per_channel_hz: float = 5.0
    emergency_stop_file: str = "runtime/EMERGENCY_STOP"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        emergency_stop_file: Optional[str] = None,
    ) -> "MixingStationSafetyConfig":
        data = load_yaml_file(path)
        if emergency_stop_file is not None:
            data["emergency_stop_file"] = emergency_stop_file
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]] = None) -> "MixingStationSafetyConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            dry_run_default=bool(data.get("dry_run_default", True)),
            max_fader_step_db=float(data.get("max_fader_step_db", 1.5)),
            max_send_step_db=float(data.get("max_send_step_db", 2.0)),
            max_eq_gain_step_db=float(data.get("max_eq_gain_step_db", 1.5)),
            max_eq_gain_absolute_db=float(data.get("max_eq_gain_absolute_db", 12.0)),
            max_eq_boost_db=float(data.get("max_eq_boost_db", 6.0)),
            min_hpf_hz=float(data.get("min_hpf_hz", 20.0)),
            max_hpf_hz=float(data.get("max_hpf_hz", 250.0)),
            min_delay_ms=float(data.get("min_delay_ms", 0.0)),
            max_delay_ms=float(data.get("max_delay_ms", 10.0)),
            min_compressor_threshold_db=float(data.get("min_compressor_threshold_db", -60.0)),
            max_compressor_threshold_db=float(data.get("max_compressor_threshold_db", 0.0)),
            min_compressor_ratio=float(data.get("min_compressor_ratio", 1.0)),
            max_compressor_ratio=float(data.get("max_compressor_ratio", 20.0)),
            min_compressor_mix=float(data.get("min_compressor_mix", 0.0)),
            max_compressor_mix=float(data.get("max_compressor_mix", 100.0)),
            min_compressor_makeup_gain_db=float(data.get("min_compressor_makeup_gain_db", -12.0)),
            max_compressor_makeup_gain_db=float(data.get("max_compressor_makeup_gain_db", 12.0)),
            min_compressor_time_ms=float(data.get("min_compressor_time_ms", 0.1)),
            max_compressor_time_ms=float(data.get("max_compressor_time_ms", 2000.0)),
            min_compressor_filter_hz=float(data.get("min_compressor_filter_hz", 20.0)),
            max_compressor_filter_hz=float(data.get("max_compressor_filter_hz", 20000.0)),
            min_compressor_filter_q=float(data.get("min_compressor_filter_q", 0.1)),
            max_compressor_filter_q=float(data.get("max_compressor_filter_q", 10.0)),
            allow_scene_recall=bool(data.get("allow_scene_recall", False)),
            allow_phantom_power=bool(data.get("allow_phantom_power", False)),
            allow_live_control=bool(data.get("allow_live_control", False)),
            rate_limit_per_channel_hz=float(data.get("rate_limit_per_channel_hz", 5.0)),
            emergency_stop_file=str(data.get("emergency_stop_file", "runtime/EMERGENCY_STOP")),
        )


@dataclass(frozen=True)
class SafetyValidation:
    """Result of validating one correction."""

    correction: AutomixCorrection
    allowed: bool
    status: str
    message: str = ""
    clamped: bool = False
    rate_limited: bool = False
    emergency_stop: bool = False


class MixingStationSafetyLayer:
    """Dry-run-first safety layer for Mixing Station correction writes."""

    def __init__(
        self,
        config: Optional[MixingStationSafetyConfig] = None,
        *,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.config = config or MixingStationSafetyConfig()
        self.time_provider = time_provider or time.monotonic
        self._last_value: Dict[Tuple[Any, ...], Any] = {}
        self._last_sent_at: Dict[Tuple[Any, ...], float] = {}
        self._emergency_stop = False

    def set_emergency_stop(self, active: bool = True) -> None:
        self._emergency_stop = bool(active)

    def is_emergency_stop_active(self) -> bool:
        flag_path = resolve_repo_path(self.config.emergency_stop_file)
        return self._emergency_stop or flag_path.exists()

    def validate(
        self,
        correction: AutomixCorrection,
        *,
        live_control_enabled: bool = False,
    ) -> SafetyValidation:
        """Validate and possibly clamp a correction before mapping/sending."""
        correction = AutomixCorrection.from_dict(correction.to_dict())
        if not self.config.enabled:
            correction.safety_status = "allowed"
            return SafetyValidation(correction, allowed=True, status="allowed")

        correction.dry_run = bool(correction.dry_run)

        if self.is_emergency_stop_active():
            correction.safety_status = "blocked"
            return SafetyValidation(
                correction,
                allowed=False,
                status="blocked",
                message="Mixing Station emergency stop is active",
                emergency_stop=True,
            )

        destructive_message = self._destructive_block_message(correction.parameter)
        if destructive_message:
            correction.safety_status = "blocked"
            return SafetyValidation(
                correction,
                allowed=False,
                status="blocked",
                message=destructive_message,
            )

        if correction.mode == "live_control" or live_control_enabled:
            if not self.config.allow_live_control:
                correction.safety_status = "blocked"
                return SafetyValidation(
                    correction,
                    allowed=False,
                    status="blocked",
                    message="Mixing Station live_control is disabled by safety config",
                )

        if self._is_rate_limited(correction):
            correction.safety_status = "blocked"
            return SafetyValidation(
                correction,
                allowed=False,
                status="blocked",
                message="rate limited",
                rate_limited=True,
            )

        clamped, message = self._clamp_value(correction)
        correction.safety_status = "clamped" if clamped else "allowed"
        return SafetyValidation(
            correction,
            allowed=True,
            status=correction.safety_status,
            message=message,
            clamped=clamped,
        )

    def record_sent(self, correction: AutomixCorrection) -> None:
        """Record a successfully sent or dry-run accepted correction for rate/step limits."""
        key = self._key(correction)
        self._last_value[key] = correction.value
        self._last_sent_at[key] = self.time_provider()

    def _destructive_block_message(self, parameter: str) -> str:
        normalized = parameter.strip().lower()
        if normalized in FORBIDDEN_PARAMETERS:
            if normalized.startswith("scene.") and self.config.allow_scene_recall:
                return ""
            if normalized.startswith("phantom_power") and self.config.allow_phantom_power:
                return ""
            return f"parameter {parameter} is forbidden for Mixing Station live writes"
        if any(normalized.startswith(prefix) for prefix in DESTRUCTIVE_PREFIXES):
            if normalized.startswith("scene.") and self.config.allow_scene_recall:
                return ""
            return f"destructive parameter {parameter} is blocked"
        return ""

    def _is_rate_limited(self, correction: AutomixCorrection) -> bool:
        hz = max(0.0, float(self.config.rate_limit_per_channel_hz))
        if hz <= 0.0:
            return False
        last = self._last_sent_at.get(self._key(correction))
        if last is None:
            return False
        return (self.time_provider() - last) < (1.0 / hz)

    def _clamp_value(self, correction: AutomixCorrection) -> tuple[bool, str]:
        if not isinstance(correction.value, (int, float)) or isinstance(correction.value, bool):
            return False, ""

        value = float(correction.value)
        original = value
        parameter = correction.parameter.lower()
        previous = self._previous_numeric(correction)

        if parameter == "fader":
            if previous is not None:
                value = self._bounded_toward(previous, value, self.config.max_fader_step_db)
            value = min(0.0, value)
        elif parameter.startswith("send.") and parameter.endswith(".level"):
            if previous is not None:
                value = self._bounded_toward(previous, value, self.config.max_send_step_db)
        elif ".gain" in parameter and parameter.startswith("peq."):
            if previous is not None:
                value = self._bounded_toward(previous, value, self.config.max_eq_gain_step_db)
            value = max(-self.config.max_eq_gain_absolute_db, min(self.config.max_eq_boost_db, value))
        elif parameter == "hpf.frequency":
            value = max(self.config.min_hpf_hz, min(self.config.max_hpf_hz, value))
        elif parameter == "delay.time":
            value = max(self.config.min_delay_ms, min(self.config.max_delay_ms, value))
        elif parameter == "compressor.threshold":
            value = max(
                self.config.min_compressor_threshold_db,
                min(self.config.max_compressor_threshold_db, value),
            )
        elif parameter == "compressor.ratio":
            value = max(self.config.min_compressor_ratio, min(self.config.max_compressor_ratio, value))
        elif parameter == "compressor.mix":
            value = max(self.config.min_compressor_mix, min(self.config.max_compressor_mix, value))
        elif parameter == "compressor.makeup_gain":
            value = max(
                self.config.min_compressor_makeup_gain_db,
                min(self.config.max_compressor_makeup_gain_db, value),
            )
        elif parameter in {"compressor.attack", "compressor.hold", "compressor.release"}:
            value = max(
                self.config.min_compressor_time_ms,
                min(self.config.max_compressor_time_ms, value),
            )
        elif parameter == "compressor.filter.frequency" or (
            parameter.startswith("compressor.filter.band") and parameter.endswith(".frequency")
        ):
            value = max(
                self.config.min_compressor_filter_hz,
                min(self.config.max_compressor_filter_hz, value),
            )
        elif parameter.startswith("compressor.filter.band") and parameter.endswith(".q"):
            value = max(
                self.config.min_compressor_filter_q,
                min(self.config.max_compressor_filter_q, value),
            )

        if abs(value - original) <= 1e-9:
            return False, ""
        correction.value = value
        return True, f"{correction.parameter} clamped from {original} to {value}"

    def _previous_numeric(self, correction: AutomixCorrection) -> Optional[float]:
        previous = correction.previous_value
        if previous is None:
            previous = self._last_value.get(self._key(correction))
        if isinstance(previous, (int, float)) and not isinstance(previous, bool):
            return float(previous)
        return None

    @staticmethod
    def _bounded_toward(current: float, target: float, max_step: float) -> float:
        delta = target - current
        delta = max(-max_step, min(max_step, delta))
        return current + delta

    @staticmethod
    def _key(correction: AutomixCorrection) -> Tuple[Any, ...]:
        return (
            correction.console_profile,
            correction.strip_type,
            correction.channel_index,
            correction.parameter,
        )
