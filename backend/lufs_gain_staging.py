"""
LUFS-based Automatic Gain Staging Module

Реализует автоматическое управление усилением (AGC) на основе стандарта 
ITU-R BS.1770 / EBU R128 с использованием LUFS-метрик.

Ключевые компоненты:
- K-Weighting Filter для LUFS измерений
- Short-term LUFS Calculator (400ms окно)
- True Peak Detector с интерполяцией
- AGC Controller с Attack/Release/Hold envelope
"""

import numpy as np
import threading
import logging
import time
import re
from typing import Dict, List, Callable, Optional, Any
from collections import deque
from scipy import signal
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

INPUT_TRIM_AUTOGAIN_METHOD_ID = "input_trim_preamp_character"
INPUT_TRIM_ROLE = "input_headroom_not_mix_balance"


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_channel_tuple(value: Any, default: range) -> tuple[int, ...]:
    if isinstance(value, str):
        channels = []
        for part in value.split(","):
            try:
                channels.append(int(part.strip()))
            except ValueError:
                continue
        return tuple(channels) if channels else tuple(default)
    if isinstance(value, (list, tuple, set)):
        channels = []
        for item in value:
            try:
                channels.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(channels) if channels else tuple(default)
    return tuple(default)


_PREAMP_CHARACTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "clean": {
        "clean_gain_max_db": 6.0,
        "coloration_start_db": 9.0,
        "saturation_start_db": 12.0,
        "soundcheck_max_boost_db": 3.0,
    },
    "warm": {
        "clean_gain_max_db": 8.0,
        "coloration_start_db": 6.0,
        "saturation_start_db": 12.0,
        "soundcheck_max_boost_db": 2.0,
    },
    "drive_protected": {
        "clean_gain_max_db": 3.0,
        "coloration_start_db": 6.0,
        "saturation_start_db": 9.0,
        "soundcheck_allow_boost": False,
        "soundcheck_max_boost_db": 0.0,
    },
}


@dataclass(frozen=True)
class ChannelPersonalityProfile:
    """Preset-level gain-staging behavior, separate from artistic balance."""

    family: str
    boost_policy: str = "allow"
    min_presence: Optional[float] = None
    max_crest_db: Optional[float] = None
    min_lufs: Optional[float] = None
    max_lufs_std_db: Optional[float] = None
    min_noise_margin_db: Optional[float] = None
    isolated_peak_min_spread_db: Optional[float] = None
    confidence_bias: float = 0.0

    def as_report(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "boost_policy": self.boost_policy,
            "confidence_bias": float(self.confidence_bias),
        }


_DEFAULT_CHANNEL_PERSONALITY = ChannelPersonalityProfile(
    family="general",
    min_presence=0.25,
    max_crest_db=16.0,
    min_lufs=-48.0,
    max_lufs_std_db=7.0,
    min_noise_margin_db=10.0,
    isolated_peak_min_spread_db=6.0,
)

_PRESET_CHANNEL_PERSONALITIES: Dict[str, ChannelPersonalityProfile] = {
    "leadvocal": ChannelPersonalityProfile(
        family="lead_vocal",
        min_presence=0.35,
        max_crest_db=18.0,
        min_lufs=-48.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=12.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.03,
    ),
    "backingvocal": ChannelPersonalityProfile(
        family="backing_vocal",
        min_presence=0.30,
        max_crest_db=17.0,
        min_lufs=-48.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=12.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.01,
    ),
    "speech": ChannelPersonalityProfile(
        family="speech",
        min_presence=0.40,
        max_crest_db=16.0,
        min_lufs=-48.0,
        max_lufs_std_db=6.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.02,
    ),
    "kick": ChannelPersonalityProfile(
        family="transient_percussion",
        boost_policy="operator_review",
        min_presence=0.20,
        max_crest_db=14.0,
        min_lufs=-46.0,
        max_lufs_std_db=8.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "snare": ChannelPersonalityProfile(
        family="transient_percussion",
        boost_policy="operator_review",
        min_presence=0.18,
        max_crest_db=14.0,
        min_lufs=-46.0,
        max_lufs_std_db=8.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "tom": ChannelPersonalityProfile(
        family="transient_percussion",
        boost_policy="operator_review",
        min_presence=0.18,
        max_crest_db=14.0,
        min_lufs=-46.0,
        max_lufs_std_db=8.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "hihat": ChannelPersonalityProfile(
        family="cymbal",
        boost_policy="operator_review",
        min_presence=0.20,
        max_crest_db=18.0,
        min_lufs=-47.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "ride": ChannelPersonalityProfile(
        family="cymbal",
        boost_policy="operator_review",
        min_presence=0.20,
        max_crest_db=18.0,
        min_lufs=-47.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "cymbals": ChannelPersonalityProfile(
        family="cymbal",
        boost_policy="operator_review",
        min_presence=0.20,
        max_crest_db=18.0,
        min_lufs=-47.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.08,
    ),
    "overheads": ChannelPersonalityProfile(
        family="drum_image",
        boost_policy="operator_review",
        min_presence=0.30,
        max_crest_db=18.0,
        min_lufs=-47.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.06,
    ),
    "room": ChannelPersonalityProfile(
        family="drum_room",
        boost_policy="operator_review",
        min_presence=0.30,
        max_crest_db=18.0,
        min_lufs=-47.0,
        max_lufs_std_db=7.0,
        min_noise_margin_db=14.0,
        isolated_peak_min_spread_db=5.0,
        confidence_bias=-0.06,
    ),
    "bass": ChannelPersonalityProfile(
        family="sustained_instrument",
        min_presence=0.45,
        max_crest_db=14.0,
        min_lufs=-48.0,
        max_lufs_std_db=5.0,
        min_noise_margin_db=10.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.02,
    ),
    "electricguitar": ChannelPersonalityProfile(
        family="compressed_instrument",
        min_presence=0.40,
        max_crest_db=14.0,
        min_lufs=-48.0,
        max_lufs_std_db=5.5,
        min_noise_margin_db=10.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.01,
    ),
    "acousticguitar": ChannelPersonalityProfile(
        family="acoustic_instrument",
        min_presence=0.35,
        max_crest_db=16.0,
        min_lufs=-48.0,
        max_lufs_std_db=6.0,
        min_noise_margin_db=12.0,
        isolated_peak_min_spread_db=6.0,
    ),
    "keyboard": ChannelPersonalityProfile(
        family="keyboard",
        min_presence=0.55,
        max_crest_db=12.0,
        min_lufs=-50.0,
        max_lufs_std_db=4.0,
        min_noise_margin_db=8.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.04,
    ),
    "accordion": ChannelPersonalityProfile(
        family="sustained_instrument",
        min_presence=0.45,
        max_crest_db=14.0,
        min_lufs=-48.0,
        max_lufs_std_db=5.5,
        min_noise_margin_db=10.0,
        isolated_peak_min_spread_db=6.0,
        confidence_bias=0.01,
    ),
    "playback": ChannelPersonalityProfile(
        family="playback",
        min_presence=0.60,
        max_crest_db=10.0,
        min_lufs=-52.0,
        max_lufs_std_db=3.0,
        min_noise_margin_db=8.0,
        isolated_peak_min_spread_db=7.0,
        confidence_bias=0.06,
    ),
}


def _preset_key(preset: str) -> str:
    return "".join(ch for ch in str(preset or "custom").lower() if ch.isalnum())


_SPECTRAL_FREQUENCY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(khz|k|hz)?", re.IGNORECASE)


def _normalized_spectral_label(label: str) -> str:
    chars = [
        ch if ch.isalnum() else "_"
        for ch in str(label or "").strip().lower()
    ]
    normalized = "".join(chars)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _frequency_hz_from_spectral_label(label: str) -> Optional[float]:
    raw_label = str(label or "").strip().lower()
    for match in _SPECTRAL_FREQUENCY_PATTERN.finditer(raw_label):
        unit = (match.group(2) or "").lower()
        value = _coerce_float(match.group(1), 0.0)
        if unit in {"khz", "k"}:
            return float(value * 1000.0)
        if unit == "hz":
            return float(value)
        if value >= 20.0:
            return float(value)
    return None


def _canonical_spectral_band_name(label: str) -> str:
    normalized = _normalized_spectral_label(label)
    frequency_hz = _frequency_hz_from_spectral_label(label)
    if frequency_hz is not None:
        if frequency_hz < 80.0:
            return "sub"
        if frequency_hz < 180.0:
            return "low"
        if frequency_hz < 600.0:
            return "low_mid"
        if frequency_hz < 2000.0:
            return "mid"
        if frequency_hz < 7000.0:
            return "presence"
        return "air"
    if not normalized:
        return "unknown"
    if any(token in normalized for token in ("low_mid", "lowmid", "lowermid", "lower_mid")):
        return "low_mid"
    if any(token in normalized for token in ("upper_mid", "uppermid", "presence", "sibilance")):
        return "presence"
    if any(token in normalized for token in ("air", "brilliance", "cymbal")):
        return "air"
    if "sub" in normalized:
        return "sub"
    if "fundamental" in normalized:
        return "fundamental"
    if "body" in normalized:
        return "body"
    if "bass" in normalized or "low" in normalized:
        return "low"
    if "high" in normalized or "treble" in normalized:
        return "air"
    if "mid" in normalized:
        return "mid"
    return normalized


def _channel_personality_for_preset(preset: str) -> ChannelPersonalityProfile:
    return _PRESET_CHANNEL_PERSONALITIES.get(_preset_key(preset), _DEFAULT_CHANNEL_PERSONALITY)


_GENERIC_TRANSIENT_BODY_BAND_PROFILE: Dict[str, Any] = {
    "name": "generic_low_body",
    "applicable": True,
    "primary_tokens": ("low", "body", "fundamental"),
    "secondary_tokens": ("low_mid", "lowmid"),
    "match_policy": "primary_preferred_secondary_fallback",
}

_TRANSIENT_BODY_BAND_PROFILES: Dict[str, Dict[str, Any]] = {
    "kick": {
        "name": "kick_low_fundamental",
        "applicable": True,
        "primary_tokens": ("sub", "low", "fundamental", "kick", "thump"),
        "secondary_tokens": ("low_mid", "lowmid", "body"),
        "match_policy": "primary_preferred_secondary_fallback",
    },
    "snare": {
        "name": "snare_low_mid_body",
        "applicable": True,
        "primary_tokens": ("low_mid", "lowmid", "body", "snare_body"),
        "secondary_tokens": ("mid", "fundamental"),
        "match_policy": "primary_preferred_secondary_fallback",
    },
    "tom": {
        "name": "tom_body_fundamental",
        "applicable": True,
        "primary_tokens": ("low_mid", "lowmid", "body", "fundamental", "tom"),
        "secondary_tokens": ("low",),
        "match_policy": "primary_preferred_secondary_fallback",
    },
}

_TRANSIENT_NON_BODY_REFERENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cymbal": {
        "name": "cymbal_no_body_reference",
        "applicable": False,
        "primary_tokens": (),
        "secondary_tokens": (),
        "match_policy": "disabled_for_high_band_transient_source",
    },
    "drum_image": {
        "name": "drum_image_no_body_reference",
        "applicable": False,
        "primary_tokens": (),
        "secondary_tokens": (),
        "match_policy": "disabled_for_multi_source_image_channel",
    },
    "drum_room": {
        "name": "drum_room_no_body_reference",
        "applicable": False,
        "primary_tokens": (),
        "secondary_tokens": (),
        "match_policy": "disabled_for_ambient_image_channel",
    },
}


def _transient_body_band_profile_for_preset(preset: str, family: str) -> Dict[str, Any]:
    preset_profile = _TRANSIENT_BODY_BAND_PROFILES.get(_preset_key(preset))
    if preset_profile:
        return dict(preset_profile)
    family_profile = _TRANSIENT_NON_BODY_REFERENCE_PROFILES.get(str(family or "").strip().lower())
    if family_profile:
        return dict(family_profile)
    if family == "transient_percussion":
        return dict(_GENERIC_TRANSIENT_BODY_BAND_PROFILE)
    return {
        "name": "not_applicable",
        "applicable": False,
        "primary_tokens": (),
        "secondary_tokens": (),
        "match_policy": "not_transient_percussion",
    }


def _gain_direction(delta_db: Optional[float], threshold_db: float = 0.25) -> str:
    if delta_db is None:
        return "unknown"
    if delta_db > threshold_db:
        return "up"
    if delta_db < -threshold_db:
        return "down"
    return "hold"


def _rolling_average(previous: Optional[float], current: Optional[float], previous_windows: int) -> Optional[float]:
    if current is None:
        return previous
    if previous is None or previous_windows <= 0:
        return float(current)
    return float((float(previous) * previous_windows + float(current)) / (previous_windows + 1))


def _bucket_db(value: Optional[float], step_db: float) -> Optional[float]:
    if value is None:
        return None
    step = max(0.1, float(step_db))
    return float(round(float(value) / step) * step)


def _compact_confidence_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(factor.get("name") or ""),
        "delta": float(_coerce_float(factor.get("delta"), 0.0)),
        "reason": str(factor.get("reason") or ""),
    }


def _dominant_confidence_factors(
    factors: List[Dict[str, Any]],
    *,
    positive: bool,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    compact = [_compact_confidence_factor(factor) for factor in factors]
    if positive:
        selected = [factor for factor in compact if factor["delta"] > 0.0]
        selected.sort(key=lambda factor: factor["delta"], reverse=True)
    else:
        selected = [factor for factor in compact if factor["delta"] < 0.0]
        selected.sort(key=lambda factor: factor["delta"])
    return selected[:max(0, int(limit))]


@dataclass(frozen=True)
class LocalInputGainProfile:
    """Model the local mixer gain knob as a tone/drive stage, not just trim."""

    enabled: bool = True
    character: str = "clean"
    source: str = "wing_rack_local_midas_preamp"
    local_input_channels: tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 25)))
    min_gain_db: float = -18.0
    hard_max_gain_db: float = 18.0
    clean_gain_max_db: float = 6.0
    coloration_start_db: float = 9.0
    saturation_start_db: float = 12.0
    soundcheck_allow_boost: bool = True
    live_allow_boost: bool = False
    soundcheck_max_boost_db: float = 3.0
    soundcheck_convergence_enabled: bool = True
    soundcheck_convergence_large_gap_db: float = 12.0
    soundcheck_convergence_max_boost_db: float = 12.0
    soundcheck_convergence_vocal_max_boost_db: float = 9.0
    soundcheck_convergence_playback_max_boost_db: float = 6.0
    soundcheck_convergence_setup_check_db: float = 15.0
    soundcheck_convergence_gain_up_step_db: float = 3.0
    soundcheck_ensemble_context_enabled: bool = True
    soundcheck_ensemble_min_other_active_sources: int = 2
    soundcheck_ensemble_min_other_presence: float = 0.25
    soundcheck_ensemble_min_reference_peak_db: float = -18.0
    soundcheck_ensemble_min_reference_lufs: float = -30.0
    soundcheck_ensemble_vocal_under_peak_gap_db: float = 8.0
    soundcheck_ensemble_vocal_under_lufs_gap_db: float = 6.0
    soundcheck_ensemble_vocal_min_crest_db: float = 6.0
    soundcheck_ensemble_max_boost_db: float = 6.0
    soundcheck_ensemble_gain_up_step_db: float = 6.0
    soundcheck_ensemble_confidence_bonus: float = 0.08
    soundcheck_ensemble_time_coherence_enabled: bool = True
    soundcheck_ensemble_max_context_lag_samples: int = 2
    soundcheck_max_cut_db: float = 6.0
    live_max_cut_db: float = 1.0
    target_deadband_db: float = 1.0
    quiet_source_margin_db: float = 6.0
    soundcheck_boost_min_presence: float = 0.25
    soundcheck_boost_max_crest_db: float = 16.0
    soundcheck_boost_min_lufs: float = -48.0
    clip_safety_margin_db: float = 1.0
    isolated_peak_min_spread_db: float = 6.0
    soundcheck_boost_max_lufs_std_db: float = 7.0
    soundcheck_boost_max_peak_std_db: float = 8.0
    soundcheck_boost_max_peak_spread_db: float = 6.0
    soundcheck_boost_min_noise_margin_db: float = 10.0
    recommendation_hysteresis_db: float = 0.75
    opposite_direction_hold_db: float = 1.5
    gain_up_min_stable_windows: int = 2
    stable_window_tolerance_db: float = 1.0
    digital_clip_threshold_db: float = -0.1
    scream_peak_threshold_db: float = -4.0
    scream_lufs_threshold_db: float = -16.0
    predictive_clip_window_db: float = 2.0
    predictive_clip_density_threshold: float = 0.30
    predictive_clip_sustained_density_threshold: float = 0.55
    predictive_clip_min_rising_trend_db: float = 1.0
    predictive_clip_min_cut_db: float = 0.5
    predictive_clip_max_cut_db: float = 1.5
    vocal_gain_up_guard_enabled: bool = True
    vocal_gain_up_presence_min_bands: int = 1
    vocal_gain_up_presence_max_std_db: float = 4.0
    vocal_gain_up_presence_max_span_db: float = 10.0
    vocal_gain_up_scream_cooldown_windows: int = 2
    trim_smoothing_enabled: bool = True
    trim_smoothing_gain_up_step_db: float = 1.0
    trim_smoothing_gain_down_step_db: float = 1.5
    trend_memory_max_windows: int = 4
    behavior_flip_gain_up_hold: bool = True
    channel_profile_min_windows: int = 3
    channel_profile_max_windows: int = 8
    channel_profile_unstable_ratio_block: float = 0.5
    channel_profile_lufs_std_block_db: float = 6.5
    channel_profile_noise_margin_block_db: float = 12.0
    live_freeze_gain_up_profile: bool = True
    channel_profile_lifecycle_enabled: bool = True
    channel_profile_default_decay_factor: float = 0.5
    channel_profile_min_decay_factor: float = 0.0
    channel_profile_recovery_enabled: bool = True
    channel_profile_recovery_min_windows: int = 3
    channel_profile_recovery_max_lufs_std_db: float = 3.0
    channel_profile_recovery_max_peak_std_db: float = 3.0
    channel_profile_recovery_max_crest_db: float = 14.0
    channel_profile_recovery_min_noise_margin_db: float = 16.0
    source_signature_change_hold_enabled: bool = True
    source_signature_change_min_fields: int = 2
    decision_confidence_enabled: bool = True
    decision_confidence_gain_up_min_score: float = 0.72
    decision_confidence_hysteresis_enabled: bool = True
    decision_confidence_borderline_margin: float = 0.08
    decision_confidence_stable_windows: int = 2
    decision_confidence_score_tolerance: float = 0.08
    decision_confidence_time_memory_enabled: bool = True
    decision_confidence_default_sample_sec: float = 1.0
    decision_confidence_min_window_sec: float = 1.0
    decision_confidence_max_age_sec: float = 30.0
    decision_confidence_max_gap_ratio: float = 4.0
    decision_confidence_min_memory_strength: float = 0.5
    decision_confidence_decay_half_life_sec: float = 20.0
    decision_confidence_vocal_decay_half_life_sec: float = 12.0
    decision_confidence_playback_decay_half_life_sec: float = 45.0
    source_signature_drift_memory_penalty_enabled: bool = True
    source_signature_drift_penalty_per_field: float = 0.30
    source_signature_drift_max_penalty: float = 0.80
    source_signature_drift_vocal_noise_weight: float = 1.70
    source_signature_drift_vocal_crest_weight: float = 1.40
    source_signature_drift_playback_lufs_weight: float = 1.40
    source_signature_drift_playback_peak_gap_weight: float = 1.40
    source_signature_continuous_drift_enabled: bool = True
    source_signature_continuous_drift_penalty_per_unit: float = 0.35
    source_signature_continuous_drift_max_penalty: float = 0.60
    source_signature_spectral_drift_enabled: bool = True
    source_signature_spectral_drift_min_common_bands: int = 3
    source_signature_spectral_drift_penalty_per_db: float = 0.08
    source_signature_spectral_drift_max_penalty: float = 0.65
    source_signature_spectral_vocal_presence_weight: float = 1.35
    source_signature_spectral_playback_weight: float = 0.55
    source_signature_spectral_drum_high_band_weight: float = 0.60
    source_signature_spectral_quality_gate_enabled: bool = True
    source_signature_spectral_min_band_samples: int = 3
    source_signature_spectral_min_band_coverage: float = 0.60
    source_signature_spectral_max_band_std_db: float = 6.0
    source_signature_spectral_max_band_span_db: float = 14.0
    source_signature_spectral_stuck_band_std_db: float = 0.02
    source_signature_spectral_stuck_signal_std_db: float = 1.0
    source_signature_spectral_quality_penalty_per_rejected_band: float = 0.15
    source_signature_spectral_quality_max_penalty: float = 0.55
    source_signature_spectral_vocal_presence_max_std_db: float = 4.0
    source_signature_spectral_vocal_presence_max_span_db: float = 10.0
    source_signature_spectral_playback_max_band_std_db: float = 9.0
    source_signature_spectral_playback_max_band_span_db: float = 20.0
    source_signature_spectral_transient_high_band_max_std_db: float = 10.0
    source_signature_spectral_transient_high_band_max_span_db: float = 24.0
    source_signature_spectral_transient_high_band_identity_weight: float = 0.25
    source_signature_spectral_transient_attack_high_band_identity_weight: float = 0.10
    transient_window_profile_enabled: bool = True
    transient_window_min_samples: int = 6
    transient_window_attack_top_ratio: float = 0.20
    transient_window_attack_body_gap_db: float = 8.0
    transient_window_high_band_lift_db: float = 6.0
    transient_window_min_repeated_hits: int = 3
    transient_window_hit_body_std_max_db: float = 3.0
    transient_window_hit_gap_std_max_db: float = 4.0
    transient_window_body_reference_enabled: bool = True
    transient_window_source_specific_body_band_profiles: bool = True
    transient_window_body_band_reference_enabled: bool = True
    transient_window_body_band_std_max_db: float = 3.0

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "LocalInputGainProfile":
        config = dict(config or {})
        character = str(config.get("character") or "clean").strip().lower()
        preset = dict(_PREAMP_CHARACTER_PRESETS.get(character, {}))
        merged = {**preset, **config, "character": character}
        return cls(
            enabled=_coerce_bool(merged.get("enabled"), True),
            character=character,
            source=str(merged.get("source") or "wing_rack_local_midas_preamp"),
            local_input_channels=_coerce_channel_tuple(
                merged.get("local_input_channels"),
                range(1, 25),
            ),
            min_gain_db=_coerce_float(merged.get("min_gain_db"), -18.0),
            hard_max_gain_db=_coerce_float(merged.get("hard_max_gain_db"), 18.0),
            clean_gain_max_db=_coerce_float(merged.get("clean_gain_max_db"), 6.0),
            coloration_start_db=_coerce_float(merged.get("coloration_start_db"), 9.0),
            saturation_start_db=_coerce_float(merged.get("saturation_start_db"), 12.0),
            soundcheck_allow_boost=_coerce_bool(merged.get("soundcheck_allow_boost"), True),
            live_allow_boost=_coerce_bool(merged.get("live_allow_boost"), False),
            soundcheck_max_boost_db=_coerce_float(merged.get("soundcheck_max_boost_db"), 3.0),
            soundcheck_convergence_enabled=_coerce_bool(
                merged.get("soundcheck_convergence_enabled"),
                True,
            ),
            soundcheck_convergence_large_gap_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_convergence_large_gap_db"), 12.0),
            ),
            soundcheck_convergence_max_boost_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_convergence_max_boost_db"), 12.0),
            ),
            soundcheck_convergence_vocal_max_boost_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_convergence_vocal_max_boost_db"), 9.0),
            ),
            soundcheck_convergence_playback_max_boost_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_convergence_playback_max_boost_db"), 6.0),
            ),
            soundcheck_convergence_setup_check_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_convergence_setup_check_db"), 15.0),
            ),
            soundcheck_convergence_gain_up_step_db=max(
                0.1,
                _coerce_float(merged.get("soundcheck_convergence_gain_up_step_db"), 3.0),
            ),
            soundcheck_ensemble_context_enabled=_coerce_bool(
                merged.get("soundcheck_ensemble_context_enabled"),
                True,
            ),
            soundcheck_ensemble_min_other_active_sources=max(
                1,
                int(_coerce_float(merged.get("soundcheck_ensemble_min_other_active_sources"), 2)),
            ),
            soundcheck_ensemble_min_other_presence=float(np.clip(
                _coerce_float(merged.get("soundcheck_ensemble_min_other_presence"), 0.25),
                0.0,
                1.0,
            )),
            soundcheck_ensemble_min_reference_peak_db=_coerce_float(
                merged.get("soundcheck_ensemble_min_reference_peak_db"),
                -18.0,
            ),
            soundcheck_ensemble_min_reference_lufs=_coerce_float(
                merged.get("soundcheck_ensemble_min_reference_lufs"),
                -30.0,
            ),
            soundcheck_ensemble_vocal_under_peak_gap_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_ensemble_vocal_under_peak_gap_db"), 8.0),
            ),
            soundcheck_ensemble_vocal_under_lufs_gap_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_ensemble_vocal_under_lufs_gap_db"), 6.0),
            ),
            soundcheck_ensemble_vocal_min_crest_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_ensemble_vocal_min_crest_db"), 6.0),
            ),
            soundcheck_ensemble_max_boost_db=max(
                0.0,
                _coerce_float(merged.get("soundcheck_ensemble_max_boost_db"), 6.0),
            ),
            soundcheck_ensemble_gain_up_step_db=max(
                0.1,
                _coerce_float(merged.get("soundcheck_ensemble_gain_up_step_db"), 6.0),
            ),
            soundcheck_ensemble_confidence_bonus=float(np.clip(
                _coerce_float(merged.get("soundcheck_ensemble_confidence_bonus"), 0.08),
                0.0,
                0.20,
            )),
            soundcheck_ensemble_time_coherence_enabled=_coerce_bool(
                merged.get("soundcheck_ensemble_time_coherence_enabled"),
                True,
            ),
            soundcheck_ensemble_max_context_lag_samples=max(
                0,
                int(_coerce_float(merged.get("soundcheck_ensemble_max_context_lag_samples"), 2)),
            ),
            soundcheck_max_cut_db=_coerce_float(merged.get("soundcheck_max_cut_db"), 6.0),
            live_max_cut_db=_coerce_float(merged.get("live_max_cut_db"), 1.0),
            target_deadband_db=_coerce_float(merged.get("target_deadband_db"), 1.0),
            quiet_source_margin_db=_coerce_float(merged.get("quiet_source_margin_db"), 6.0),
            soundcheck_boost_min_presence=_coerce_float(merged.get("soundcheck_boost_min_presence"), 0.25),
            soundcheck_boost_max_crest_db=_coerce_float(merged.get("soundcheck_boost_max_crest_db"), 16.0),
            soundcheck_boost_min_lufs=_coerce_float(merged.get("soundcheck_boost_min_lufs"), -48.0),
            clip_safety_margin_db=_coerce_float(merged.get("clip_safety_margin_db"), 1.0),
            isolated_peak_min_spread_db=_coerce_float(merged.get("isolated_peak_min_spread_db"), 6.0),
            soundcheck_boost_max_lufs_std_db=_coerce_float(merged.get("soundcheck_boost_max_lufs_std_db"), 7.0),
            soundcheck_boost_max_peak_std_db=_coerce_float(merged.get("soundcheck_boost_max_peak_std_db"), 8.0),
            soundcheck_boost_max_peak_spread_db=_coerce_float(merged.get("soundcheck_boost_max_peak_spread_db"), 6.0),
            soundcheck_boost_min_noise_margin_db=_coerce_float(merged.get("soundcheck_boost_min_noise_margin_db"), 10.0),
            recommendation_hysteresis_db=_coerce_float(merged.get("recommendation_hysteresis_db"), 0.75),
            opposite_direction_hold_db=_coerce_float(merged.get("opposite_direction_hold_db"), 1.5),
            gain_up_min_stable_windows=max(1, int(_coerce_float(merged.get("gain_up_min_stable_windows"), 2))),
            stable_window_tolerance_db=_coerce_float(merged.get("stable_window_tolerance_db"), 1.0),
            digital_clip_threshold_db=_coerce_float(merged.get("digital_clip_threshold_db"), -0.1),
            scream_peak_threshold_db=_coerce_float(merged.get("scream_peak_threshold_db"), -4.0),
            scream_lufs_threshold_db=_coerce_float(merged.get("scream_lufs_threshold_db"), -16.0),
            predictive_clip_window_db=_coerce_float(merged.get("predictive_clip_window_db"), 2.0),
            predictive_clip_density_threshold=_coerce_float(merged.get("predictive_clip_density_threshold"), 0.30),
            predictive_clip_sustained_density_threshold=_coerce_float(
                merged.get("predictive_clip_sustained_density_threshold"),
                0.55,
            ),
            predictive_clip_min_rising_trend_db=_coerce_float(merged.get("predictive_clip_min_rising_trend_db"), 1.0),
            predictive_clip_min_cut_db=_coerce_float(merged.get("predictive_clip_min_cut_db"), 0.5),
            predictive_clip_max_cut_db=_coerce_float(merged.get("predictive_clip_max_cut_db"), 1.5),
            vocal_gain_up_guard_enabled=_coerce_bool(
                merged.get("vocal_gain_up_guard_enabled"),
                True,
            ),
            vocal_gain_up_presence_min_bands=max(
                1,
                int(_coerce_float(merged.get("vocal_gain_up_presence_min_bands"), 1)),
            ),
            vocal_gain_up_presence_max_std_db=max(
                0.0,
                _coerce_float(merged.get("vocal_gain_up_presence_max_std_db"), 4.0),
            ),
            vocal_gain_up_presence_max_span_db=max(
                0.0,
                _coerce_float(merged.get("vocal_gain_up_presence_max_span_db"), 10.0),
            ),
            vocal_gain_up_scream_cooldown_windows=max(
                0,
                int(_coerce_float(merged.get("vocal_gain_up_scream_cooldown_windows"), 2)),
            ),
            trim_smoothing_enabled=_coerce_bool(
                merged.get("trim_smoothing_enabled"),
                True,
            ),
            trim_smoothing_gain_up_step_db=max(
                0.1,
                _coerce_float(merged.get("trim_smoothing_gain_up_step_db"), 1.0),
            ),
            trim_smoothing_gain_down_step_db=max(
                0.1,
                _coerce_float(merged.get("trim_smoothing_gain_down_step_db"), 1.5),
            ),
            trend_memory_max_windows=max(1, int(_coerce_float(merged.get("trend_memory_max_windows"), 4))),
            behavior_flip_gain_up_hold=_coerce_bool(merged.get("behavior_flip_gain_up_hold"), True),
            channel_profile_min_windows=max(1, int(_coerce_float(merged.get("channel_profile_min_windows"), 3))),
            channel_profile_max_windows=max(1, int(_coerce_float(merged.get("channel_profile_max_windows"), 8))),
            channel_profile_unstable_ratio_block=_coerce_float(
                merged.get("channel_profile_unstable_ratio_block"),
                0.5,
            ),
            channel_profile_lufs_std_block_db=_coerce_float(
                merged.get("channel_profile_lufs_std_block_db"),
                6.5,
            ),
            channel_profile_noise_margin_block_db=_coerce_float(
                merged.get("channel_profile_noise_margin_block_db"),
                12.0,
            ),
            live_freeze_gain_up_profile=_coerce_bool(merged.get("live_freeze_gain_up_profile"), True),
            channel_profile_lifecycle_enabled=_coerce_bool(
                merged.get("channel_profile_lifecycle_enabled"),
                True,
            ),
            channel_profile_default_decay_factor=float(np.clip(
                _coerce_float(merged.get("channel_profile_default_decay_factor"), 0.5),
                0.0,
                1.0,
            )),
            channel_profile_min_decay_factor=float(np.clip(
                _coerce_float(merged.get("channel_profile_min_decay_factor"), 0.0),
                0.0,
                1.0,
            )),
            channel_profile_recovery_enabled=_coerce_bool(
                merged.get("channel_profile_recovery_enabled"),
                True,
            ),
            channel_profile_recovery_min_windows=max(
                1,
                int(_coerce_float(merged.get("channel_profile_recovery_min_windows"), 3)),
            ),
            channel_profile_recovery_max_lufs_std_db=_coerce_float(
                merged.get("channel_profile_recovery_max_lufs_std_db"),
                3.0,
            ),
            channel_profile_recovery_max_peak_std_db=_coerce_float(
                merged.get("channel_profile_recovery_max_peak_std_db"),
                3.0,
            ),
            channel_profile_recovery_max_crest_db=_coerce_float(
                merged.get("channel_profile_recovery_max_crest_db"),
                14.0,
            ),
            channel_profile_recovery_min_noise_margin_db=_coerce_float(
                merged.get("channel_profile_recovery_min_noise_margin_db"),
                16.0,
            ),
            source_signature_change_hold_enabled=_coerce_bool(
                merged.get("source_signature_change_hold_enabled"),
                True,
            ),
            source_signature_change_min_fields=max(
                1,
                int(_coerce_float(merged.get("source_signature_change_min_fields"), 2)),
            ),
            decision_confidence_enabled=_coerce_bool(
                merged.get("decision_confidence_enabled"),
                True,
            ),
            decision_confidence_gain_up_min_score=float(np.clip(
                _coerce_float(merged.get("decision_confidence_gain_up_min_score"), 0.72),
                0.0,
                0.95,
            )),
            decision_confidence_hysteresis_enabled=_coerce_bool(
                merged.get("decision_confidence_hysteresis_enabled"),
                True,
            ),
            decision_confidence_borderline_margin=float(np.clip(
                _coerce_float(merged.get("decision_confidence_borderline_margin"), 0.08),
                0.0,
                0.30,
            )),
            decision_confidence_stable_windows=max(
                1,
                int(_coerce_float(merged.get("decision_confidence_stable_windows"), 2)),
            ),
            decision_confidence_score_tolerance=float(np.clip(
                _coerce_float(merged.get("decision_confidence_score_tolerance"), 0.08),
                0.0,
                0.30,
            )),
            decision_confidence_time_memory_enabled=_coerce_bool(
                merged.get("decision_confidence_time_memory_enabled"),
                True,
            ),
            decision_confidence_default_sample_sec=max(
                0.01,
                _coerce_float(merged.get("decision_confidence_default_sample_sec"), 1.0),
            ),
            decision_confidence_min_window_sec=max(
                0.1,
                _coerce_float(merged.get("decision_confidence_min_window_sec"), 1.0),
            ),
            decision_confidence_max_age_sec=max(
                0.1,
                _coerce_float(merged.get("decision_confidence_max_age_sec"), 30.0),
            ),
            decision_confidence_max_gap_ratio=max(
                1.0,
                _coerce_float(merged.get("decision_confidence_max_gap_ratio"), 4.0),
            ),
            decision_confidence_min_memory_strength=float(np.clip(
                _coerce_float(merged.get("decision_confidence_min_memory_strength"), 0.5),
                0.0,
                1.0,
            )),
            decision_confidence_decay_half_life_sec=max(
                0.1,
                _coerce_float(merged.get("decision_confidence_decay_half_life_sec"), 20.0),
            ),
            decision_confidence_vocal_decay_half_life_sec=max(
                0.1,
                _coerce_float(merged.get("decision_confidence_vocal_decay_half_life_sec"), 12.0),
            ),
            decision_confidence_playback_decay_half_life_sec=max(
                0.1,
                _coerce_float(merged.get("decision_confidence_playback_decay_half_life_sec"), 45.0),
            ),
            source_signature_drift_memory_penalty_enabled=_coerce_bool(
                merged.get("source_signature_drift_memory_penalty_enabled"),
                True,
            ),
            source_signature_drift_penalty_per_field=float(np.clip(
                _coerce_float(merged.get("source_signature_drift_penalty_per_field"), 0.30),
                0.0,
                1.0,
            )),
            source_signature_drift_max_penalty=float(np.clip(
                _coerce_float(merged.get("source_signature_drift_max_penalty"), 0.80),
                0.0,
                1.0,
            )),
            source_signature_drift_vocal_noise_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_drift_vocal_noise_weight"), 1.70),
            ),
            source_signature_drift_vocal_crest_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_drift_vocal_crest_weight"), 1.40),
            ),
            source_signature_drift_playback_lufs_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_drift_playback_lufs_weight"), 1.40),
            ),
            source_signature_drift_playback_peak_gap_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_drift_playback_peak_gap_weight"), 1.40),
            ),
            source_signature_continuous_drift_enabled=_coerce_bool(
                merged.get("source_signature_continuous_drift_enabled"),
                True,
            ),
            source_signature_continuous_drift_penalty_per_unit=float(np.clip(
                _coerce_float(merged.get("source_signature_continuous_drift_penalty_per_unit"), 0.35),
                0.0,
                1.0,
            )),
            source_signature_continuous_drift_max_penalty=float(np.clip(
                _coerce_float(merged.get("source_signature_continuous_drift_max_penalty"), 0.60),
                0.0,
                1.0,
            )),
            source_signature_spectral_drift_enabled=_coerce_bool(
                merged.get("source_signature_spectral_drift_enabled"),
                True,
            ),
            source_signature_spectral_drift_min_common_bands=max(
                1,
                int(_coerce_float(merged.get("source_signature_spectral_drift_min_common_bands"), 3)),
            ),
            source_signature_spectral_drift_penalty_per_db=float(np.clip(
                _coerce_float(merged.get("source_signature_spectral_drift_penalty_per_db"), 0.08),
                0.0,
                1.0,
            )),
            source_signature_spectral_drift_max_penalty=float(np.clip(
                _coerce_float(merged.get("source_signature_spectral_drift_max_penalty"), 0.65),
                0.0,
                1.0,
            )),
            source_signature_spectral_vocal_presence_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_vocal_presence_weight"), 1.35),
            ),
            source_signature_spectral_playback_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_playback_weight"), 0.55),
            ),
            source_signature_spectral_drum_high_band_weight=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_drum_high_band_weight"), 0.60),
            ),
            source_signature_spectral_quality_gate_enabled=_coerce_bool(
                merged.get("source_signature_spectral_quality_gate_enabled"),
                True,
            ),
            source_signature_spectral_min_band_samples=max(
                1,
                int(_coerce_float(merged.get("source_signature_spectral_min_band_samples"), 3)),
            ),
            source_signature_spectral_min_band_coverage=float(np.clip(
                _coerce_float(merged.get("source_signature_spectral_min_band_coverage"), 0.60),
                0.0,
                1.0,
            )),
            source_signature_spectral_max_band_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_max_band_std_db"), 6.0),
            ),
            source_signature_spectral_max_band_span_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_max_band_span_db"), 14.0),
            ),
            source_signature_spectral_stuck_band_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_stuck_band_std_db"), 0.02),
            ),
            source_signature_spectral_stuck_signal_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_stuck_signal_std_db"), 1.0),
            ),
            source_signature_spectral_quality_penalty_per_rejected_band=float(np.clip(
                _coerce_float(
                    merged.get("source_signature_spectral_quality_penalty_per_rejected_band"),
                    0.15,
                ),
                0.0,
                1.0,
            )),
            source_signature_spectral_quality_max_penalty=float(np.clip(
                _coerce_float(merged.get("source_signature_spectral_quality_max_penalty"), 0.55),
                0.0,
                1.0,
            )),
            source_signature_spectral_vocal_presence_max_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_vocal_presence_max_std_db"), 4.0),
            ),
            source_signature_spectral_vocal_presence_max_span_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_vocal_presence_max_span_db"), 10.0),
            ),
            source_signature_spectral_playback_max_band_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_playback_max_band_std_db"), 9.0),
            ),
            source_signature_spectral_playback_max_band_span_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_playback_max_band_span_db"), 20.0),
            ),
            source_signature_spectral_transient_high_band_max_std_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_transient_high_band_max_std_db"), 10.0),
            ),
            source_signature_spectral_transient_high_band_max_span_db=max(
                0.0,
                _coerce_float(merged.get("source_signature_spectral_transient_high_band_max_span_db"), 24.0),
            ),
            source_signature_spectral_transient_high_band_identity_weight=max(
                0.0,
                _coerce_float(
                    merged.get("source_signature_spectral_transient_high_band_identity_weight"),
                    0.25,
                ),
            ),
            source_signature_spectral_transient_attack_high_band_identity_weight=max(
                0.0,
                _coerce_float(
                    merged.get("source_signature_spectral_transient_attack_high_band_identity_weight"),
                    0.10,
                ),
            ),
            transient_window_profile_enabled=_coerce_bool(
                merged.get("transient_window_profile_enabled"),
                True,
            ),
            transient_window_min_samples=max(
                1,
                int(_coerce_float(merged.get("transient_window_min_samples"), 6)),
            ),
            transient_window_attack_top_ratio=float(np.clip(
                _coerce_float(merged.get("transient_window_attack_top_ratio"), 0.20),
                0.05,
                0.50,
            )),
            transient_window_attack_body_gap_db=max(
                0.0,
                _coerce_float(merged.get("transient_window_attack_body_gap_db"), 8.0),
            ),
            transient_window_high_band_lift_db=max(
                0.0,
                _coerce_float(merged.get("transient_window_high_band_lift_db"), 6.0),
            ),
            transient_window_min_repeated_hits=max(
                1,
                int(_coerce_float(merged.get("transient_window_min_repeated_hits"), 3)),
            ),
            transient_window_hit_body_std_max_db=max(
                0.0,
                _coerce_float(merged.get("transient_window_hit_body_std_max_db"), 3.0),
            ),
            transient_window_hit_gap_std_max_db=max(
                0.0,
                _coerce_float(merged.get("transient_window_hit_gap_std_max_db"), 4.0),
            ),
            transient_window_body_reference_enabled=_coerce_bool(
                merged.get("transient_window_body_reference_enabled"),
                True,
            ),
            transient_window_source_specific_body_band_profiles=_coerce_bool(
                merged.get("transient_window_source_specific_body_band_profiles"),
                True,
            ),
            transient_window_body_band_reference_enabled=_coerce_bool(
                merged.get("transient_window_body_band_reference_enabled"),
                True,
            ),
            transient_window_body_band_std_max_db=max(
                0.0,
                _coerce_float(merged.get("transient_window_body_band_std_max_db"), 3.0),
            ),
        )

    def is_local_input(self, mixer_channel: Optional[int]) -> bool:
        if mixer_channel is None:
            return False
        try:
            return int(mixer_channel) in self.local_input_channels
        except (TypeError, ValueError):
            return False

    def zone_for_gain(self, gain_db: Optional[float]) -> str:
        if gain_db is None:
            return "unknown"
        if gain_db >= self.saturation_start_db:
            return "saturation_risk"
        if gain_db >= self.coloration_start_db:
            return "colored"
        if gain_db <= self.clean_gain_max_db:
            return "clean"
        return "transition"

    def distortion_risk(self, target_gain_db: Optional[float], *, local_input: bool) -> str:
        if not self.enabled or not local_input or target_gain_db is None:
            return "unknown"
        zone = self.zone_for_gain(target_gain_db)
        if zone == "saturation_risk":
            return "high"
        if zone in {"colored", "transition"}:
            return "medium"
        return "low"

    def max_cut_for_mode(self, mode: str) -> float:
        return self.live_max_cut_db if mode == "live" else self.soundcheck_max_cut_db

    def max_boost_for_mode(self, mode: str) -> float:
        if mode == "live":
            return self.soundcheck_max_boost_db if self.live_allow_boost else 0.0
        return self.soundcheck_max_boost_db if self.soundcheck_allow_boost else 0.0

    def soundcheck_convergence_cap_for(self, personality: ChannelPersonalityProfile) -> float:
        family = str(personality.family or "general")
        if family in {"lead_vocal", "backing_vocal", "speech"}:
            return float(self.soundcheck_convergence_vocal_max_boost_db)
        if family == "playback":
            return float(self.soundcheck_convergence_playback_max_boost_db)
        return float(self.soundcheck_convergence_max_boost_db)

    def soundcheck_boost_thresholds(self, preset: str) -> Dict[str, Any]:
        personality = _channel_personality_for_preset(preset)
        return {
            "personality": personality,
            "min_presence": max(
                float(self.soundcheck_boost_min_presence),
                float(personality.min_presence)
                if personality.min_presence is not None
                else float(self.soundcheck_boost_min_presence),
            ),
            "max_crest_db": (
                float(personality.max_crest_db)
                if personality.max_crest_db is not None
                else float(self.soundcheck_boost_max_crest_db)
            ),
            "min_lufs": max(
                float(self.soundcheck_boost_min_lufs),
                float(personality.min_lufs)
                if personality.min_lufs is not None
                else float(self.soundcheck_boost_min_lufs),
            ),
            "max_lufs_std_db": (
                float(personality.max_lufs_std_db)
                if personality.max_lufs_std_db is not None
                else float(self.soundcheck_boost_max_lufs_std_db)
            ),
            "max_peak_std_db": float(self.soundcheck_boost_max_peak_std_db),
            "max_peak_spread_db": float(self.soundcheck_boost_max_peak_spread_db),
            "min_noise_margin_db": max(
                float(self.soundcheck_boost_min_noise_margin_db),
                float(personality.min_noise_margin_db)
                if personality.min_noise_margin_db is not None
                else float(self.soundcheck_boost_min_noise_margin_db),
            ),
            "isolated_peak_min_spread_db": (
                float(personality.isolated_peak_min_spread_db)
                if personality.isolated_peak_min_spread_db is not None
                else float(self.isolated_peak_min_spread_db)
            ),
        }

    def soundcheck_boost_guard(
        self,
        *,
        signal_presence_ratio: float,
        crest_factor_db: float,
        integrated_lufs: float,
        preset: str,
        lufs_std_db: float,
        peak_std_db: float,
        isolated_peak_spread_db: float,
        noise_floor_margin_db: Optional[float],
    ) -> tuple[Optional[str], str, Dict[str, Any]]:
        thresholds = self.soundcheck_boost_thresholds(preset)
        personality = thresholds["personality"]
        evidence = {
            "personality": personality.as_report(),
            "min_presence": float(thresholds["min_presence"]),
            "max_crest_db": float(thresholds["max_crest_db"]),
            "min_lufs": float(thresholds["min_lufs"]),
            "max_lufs_std_db": float(thresholds["max_lufs_std_db"]),
            "max_peak_std_db": float(thresholds["max_peak_std_db"]),
            "max_peak_spread_db": float(thresholds["max_peak_spread_db"]),
            "min_noise_margin_db": float(thresholds["min_noise_margin_db"]),
            "noise_floor_margin_db": (
                float(noise_floor_margin_db)
                if noise_floor_margin_db is not None
                else None
            ),
        }
        if signal_presence_ratio < thresholds["min_presence"]:
            return (
                "unstable_presence_boost_block",
                "Soundcheck boost blocked: signal presence is too sparse for a stable input-gain decision.",
                evidence,
            )
        if crest_factor_db > thresholds["max_crest_db"]:
            return (
                "transient_dominant_boost_block",
                "Soundcheck boost blocked: high crest factor suggests transient-dominant material; preserve headroom.",
                evidence,
            )
        if integrated_lufs < thresholds["min_lufs"]:
            return (
                "noise_floor_boost_block",
                "Soundcheck boost blocked: integrated level is too close to the noise floor for automatic gain raise.",
                evidence,
            )
        if lufs_std_db > thresholds["max_lufs_std_db"]:
            return (
                "unstable_dynamics_boost_block",
                "Soundcheck boost blocked: loudness varies too much for a stable automatic gain raise.",
                evidence,
            )
        if peak_std_db > thresholds["max_peak_std_db"]:
            return (
                "unstable_dynamics_boost_block",
                "Soundcheck boost blocked: peak level varies too much for a stable automatic gain raise.",
                evidence,
            )
        if isolated_peak_spread_db > thresholds["max_peak_spread_db"]:
            return (
                "isolated_peak_boost_block",
                "Soundcheck boost blocked: an isolated peak already constrains safe input headroom.",
                evidence,
            )
        if (
            noise_floor_margin_db is not None
            and noise_floor_margin_db < thresholds["min_noise_margin_db"]
        ):
            return (
                "noise_floor_margin_boost_block",
                "Soundcheck boost blocked: source level is too close to measured noise floor.",
                evidence,
            )
        if personality.boost_policy == "operator_review":
            return (
                "source_personality_boost_review",
                "Soundcheck boost requires operator review for this source type.",
                evidence,
            )
        return None, "", evidence


class AnalysisState(Enum):
    """Analysis state for Safe Static Gain system."""
    IDLE = "idle"
    LEARNING = "learning"
    READY = "ready"
    APPLYING = "applying"


@dataclass
class SignalStats:
    """Signal statistics collected during analysis phase."""
    channel_id: int
    max_true_peak_db: float = -100.0
    integrated_lufs: float = -100.0
    signal_presence_ratio: float = 0.0
    total_samples: int = 0
    active_samples: int = 0
    rms_values: List[float] = field(default_factory=list)
    peak_values: List[float] = field(default_factory=list)
    inactive_rms_values: List[float] = field(default_factory=list)
    spectral_band_values: Dict[str, List[float]] = field(default_factory=dict)
    active_window_samples: List[Dict[str, Any]] = field(default_factory=list)
    active_sample_indices: List[int] = field(default_factory=list)

    noise_gate_threshold_db: float = -40.0

    crest_factor_db: float = 0.0

    suggested_gain_db: float = 0.0
    gain_limited_by: str = "none"
    target_peak_db: float = -10.0
    current_trim_db: Optional[float] = None
    target_trim_db: Optional[float] = None
    suggestion_confidence: float = 0.0
    operator_action: str = "none"
    operator_note: str = ""
    input_gain_character: Dict[str, Any] = field(default_factory=dict)
    safety_policy: Dict[str, Any] = field(default_factory=dict)

    def update_sample(
        self,
        true_peak_db: float,
        lufs: float,
        sample_rms_db: float,
        spectral_bands_db: Optional[Dict[str, float]] = None,
    ):
        """Update stats with new sample measurement."""
        self.total_samples += 1

        if true_peak_db > self.noise_gate_threshold_db:
            self.active_samples += 1
            self.active_sample_indices.append(int(self.total_samples))
            self.peak_values.append(true_peak_db)
            self.rms_values.append(lufs)
            clean_spectral_bands: Dict[str, float] = {}
            if isinstance(spectral_bands_db, dict):
                canonical_band_values: Dict[str, List[float]] = {}
                for band_name, band_db in spectral_bands_db.items():
                    try:
                        value = float(band_db)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue
                    key = _canonical_spectral_band_name(str(band_name))
                    if not key:
                        continue
                    canonical_band_values.setdefault(key, []).append(value)
                for key, values in canonical_band_values.items():
                    value = float(np.mean(values))
                    self.spectral_band_values.setdefault(key, []).append(value)
                    clean_spectral_bands[key] = value
            self.active_window_samples.append({
                "sample_index": int(self.total_samples),
                "peak_db": float(true_peak_db),
                "lufs": float(lufs),
                "spectral_bands_db": clean_spectral_bands,
            })

            if true_peak_db > self.max_true_peak_db:
                self.max_true_peak_db = true_peak_db
        else:
            self.inactive_rms_values.append(sample_rms_db)

        self.signal_presence_ratio = self.active_samples / max(self.total_samples, 1)

    def calculate_spectral_fingerprint(self) -> Dict[str, Any]:
        """Summarize active-window band energy as a level-normalized source fingerprint."""
        band_levels_db: Dict[str, float] = {}
        band_sample_counts: Dict[str, int] = {}
        band_std_db: Dict[str, float] = {}
        band_span_db: Dict[str, float] = {}
        active_sample_count = max(0, int(self.active_samples))
        signal_lufs_std_db = float(np.std(self.rms_values)) if self.rms_values else 0.0
        signal_peak_std_db = float(np.std(self.peak_values)) if self.peak_values else 0.0
        for band_name, values in sorted(self.spectral_band_values.items()):
            finite_values = [float(value) for value in values if np.isfinite(value)]
            if not finite_values:
                continue
            band_levels_db[band_name] = float(np.median(finite_values))
            band_sample_counts[band_name] = len(finite_values)
            band_std_db[band_name] = float(np.std(finite_values))
            band_span_db[band_name] = float(max(finite_values) - min(finite_values))

        if not band_levels_db:
            return {
                "available": False,
                "band_count": 0,
                "bands_db": {},
                "relative_bands_db": {},
                "band_sample_counts": {},
                "band_coverage_ratios": {},
                "band_std_db": {},
                "band_span_db": {},
                "active_sample_count": active_sample_count,
                "signal_lufs_std_db": signal_lufs_std_db,
                "signal_peak_std_db": signal_peak_std_db,
                "signal_variability_db": max(signal_lufs_std_db, signal_peak_std_db),
                "reference_db": None,
            }

        reference_db = float(np.mean(list(band_levels_db.values())))
        relative_bands_db = {
            band_name: float(level_db - reference_db)
            for band_name, level_db in band_levels_db.items()
        }
        band_coverage_ratios = {
            band_name: float(sample_count / max(active_sample_count, 1))
            for band_name, sample_count in band_sample_counts.items()
        }
        return {
            "available": True,
            "band_count": len(band_levels_db),
            "bands_db": band_levels_db,
            "relative_bands_db": relative_bands_db,
            "band_sample_counts": band_sample_counts,
            "band_coverage_ratios": band_coverage_ratios,
            "band_std_db": band_std_db,
            "band_span_db": band_span_db,
            "active_sample_count": active_sample_count,
            "signal_lufs_std_db": signal_lufs_std_db,
            "signal_peak_std_db": signal_peak_std_db,
            "signal_variability_db": max(signal_lufs_std_db, signal_peak_std_db),
            "reference_db": reference_db,
        }

    def calculate_transient_window_profile(
        self,
        *,
        enabled: bool,
        min_samples: int,
        attack_top_ratio: float,
        attack_body_gap_db: float,
        high_band_lift_db: float,
        min_repeated_hits: int,
        hit_body_std_max_db: float,
        hit_gap_std_max_db: float,
        body_band_reference_enabled: bool,
        body_band_std_max_db: float,
        body_band_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Split active samples into attack/body windows for transient-source gain staging."""
        samples = self.active_window_samples
        sample_count = len(samples)
        if not enabled:
            return {"enabled": False, "active": False, "reason": "disabled"}
        if sample_count < max(1, min_samples):
            return {
                "enabled": True,
                "active": False,
                "reason": "too_few_samples",
                "sample_count": int(sample_count),
                "min_samples": int(min_samples),
            }

        peaks = [float(sample["peak_db"]) for sample in samples]
        attack_percentile = 100.0 * (1.0 - float(np.clip(attack_top_ratio, 0.05, 0.50)))
        attack_threshold_db = float(np.percentile(peaks, attack_percentile))
        attack_samples = [
            sample
            for sample in samples
            if float(sample["peak_db"]) >= attack_threshold_db
        ]
        body_samples = [
            sample
            for sample in samples
            if float(sample["peak_db"]) < attack_threshold_db
        ]
        if not body_samples:
            body_samples = samples

        attack_peaks = [float(sample["peak_db"]) for sample in attack_samples]
        body_peaks = [float(sample["peak_db"]) for sample in body_samples]
        attack_peak_p95_db = float(np.percentile(attack_peaks, 95)) if attack_peaks else float(max(peaks))
        body_peak_median_db = float(np.median(body_peaks)) if body_peaks else float(np.median(peaks))
        attack_body_gap = float(attack_peak_p95_db - body_peak_median_db)

        high_band_tokens = ("presence", "air", "high", "upper", "sibilance", "cymbal")
        body_band_profile = dict(body_band_profile or _GENERIC_TRANSIENT_BODY_BAND_PROFILE)
        body_band_profile_name = str(body_band_profile.get("name") or "generic_low_body")
        body_band_profile_applicable = _coerce_bool(body_band_profile.get("applicable"), True)
        body_band_primary_tokens = tuple(
            str(token).strip().lower()
            for token in body_band_profile.get("primary_tokens", ())
            if str(token).strip()
        )
        body_band_secondary_tokens = tuple(
            str(token).strip().lower()
            for token in body_band_profile.get("secondary_tokens", ())
            if str(token).strip()
        )
        body_band_match_policy = str(
            body_band_profile.get("match_policy")
            or "primary_preferred_secondary_fallback"
        )
        effective_body_band_reference_enabled = bool(
            body_band_reference_enabled
            and body_band_profile_applicable
            and (body_band_primary_tokens or body_band_secondary_tokens)
        )

        def _band_values(window_samples: List[Dict[str, Any]], tokens: tuple[str, ...]) -> List[float]:
            values = []
            for sample in window_samples:
                spectral_bands = sample.get("spectral_bands_db")
                if not isinstance(spectral_bands, dict):
                    continue
                high_values = [
                    float(value)
                    for band_name, value in spectral_bands.items()
                    if any(token in str(band_name).lower() for token in tokens)
                ]
                if high_values:
                    values.append(float(np.mean(high_values)))
            return values

        def _high_band_values(window_samples: List[Dict[str, Any]]) -> List[float]:
            return _band_values(window_samples, high_band_tokens)

        def _normalized_band_name(band_name: str) -> str:
            chars = [
                ch if ch.isalnum() else "_"
                for ch in str(band_name).strip().lower()
            ]
            normalized = "".join(chars)
            while "__" in normalized:
                normalized = normalized.replace("__", "_")
            return normalized.strip("_")

        def _body_band_matches(band_name: str, tokens: tuple[str, ...]) -> bool:
            normalized_name = _normalized_band_name(band_name)
            compact_name = normalized_name.replace("_", "")
            if not normalized_name:
                return False
            for token in tokens:
                normalized_token = _normalized_band_name(token)
                compact_token = normalized_token.replace("_", "")
                if not normalized_token:
                    continue
                if normalized_name == normalized_token or compact_name == compact_token:
                    return True
                if normalized_name in {"low_mid", "lowmid"} and normalized_token == "low":
                    continue
                if normalized_token in normalized_name.split("_"):
                    return True
            return False

        def _body_band_summary(window_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            values = []
            primary_match_count = 0
            secondary_match_count = 0
            primary_used = False
            secondary_used = False
            if not effective_body_band_reference_enabled:
                return {
                    "values": values,
                    "primary_match_count": primary_match_count,
                    "secondary_match_count": secondary_match_count,
                    "source": "disabled",
                }
            for sample in window_samples:
                spectral_bands = sample.get("spectral_bands_db")
                if not isinstance(spectral_bands, dict):
                    continue
                primary_values = [
                    float(value)
                    for band_name, value in spectral_bands.items()
                    if _body_band_matches(band_name, body_band_primary_tokens)
                ]
                secondary_values = [
                    float(value)
                    for band_name, value in spectral_bands.items()
                    if _body_band_matches(band_name, body_band_secondary_tokens)
                ]
                primary_match_count += len(primary_values)
                secondary_match_count += len(secondary_values)
                if primary_values:
                    values.append(float(np.mean(primary_values)))
                    primary_used = True
                elif secondary_values:
                    values.append(float(np.mean(secondary_values)))
                    secondary_used = True
            source = "primary" if primary_used else "secondary_fallback" if secondary_used else "unavailable"
            return {
                "values": values,
                "primary_match_count": primary_match_count,
                "secondary_match_count": secondary_match_count,
                "source": source,
            }

        attack_high_values = _high_band_values(attack_samples)
        body_high_values = _high_band_values(body_samples)
        high_band_available = bool(attack_high_values and body_high_values)
        high_band_lift = (
            float(np.median(attack_high_values) - np.median(body_high_values))
            if high_band_available
            else None
        )
        attack_dominant = attack_body_gap >= attack_body_gap_db
        high_band_transient_dominant = bool(
            high_band_lift is not None
            and high_band_lift >= high_band_lift_db
        )
        attack_run_profiles: List[Dict[str, Any]] = []
        in_attack_run = False
        attack_run: List[Dict[str, Any]] = []
        preceding_body_run: List[Dict[str, Any]] = []
        body_run: List[Dict[str, Any]] = []
        for sample in samples:
            is_attack_sample = float(sample["peak_db"]) >= attack_threshold_db
            if is_attack_sample:
                if not in_attack_run:
                    attack_run = []
                    preceding_body_run = body_run
                    body_run = []
                    in_attack_run = True
                attack_run.append(sample)
            else:
                if in_attack_run and attack_run:
                    attack_peaks_for_run = [float(item["peak_db"]) for item in attack_run]
                    body_peaks_for_run = [
                        float(item["peak_db"])
                        for item in preceding_body_run
                    ]
                    if body_peaks_for_run:
                        hit_body_median_db = float(np.median(body_peaks_for_run))
                        hit_attack_p95_db = float(np.percentile(attack_peaks_for_run, 95))
                        body_band_summary_for_run = _body_band_summary(preceding_body_run)
                        body_band_values_for_run = body_band_summary_for_run["values"]
                        attack_run_profiles.append({
                            "attack_sample_count": int(len(attack_run)),
                            "body_sample_count": int(len(preceding_body_run)),
                            "attack_peak_p95_db": hit_attack_p95_db,
                            "body_peak_median_db": hit_body_median_db,
                            "attack_body_gap_db": float(hit_attack_p95_db - hit_body_median_db),
                            "body_band_median_db": (
                                float(np.median(body_band_values_for_run))
                                if body_band_values_for_run
                                else None
                            ),
                            "body_band_source": body_band_summary_for_run["source"],
                            "body_band_primary_match_count": int(
                                body_band_summary_for_run["primary_match_count"]
                            ),
                            "body_band_secondary_match_count": int(
                                body_band_summary_for_run["secondary_match_count"]
                            ),
                        })
                    in_attack_run = False
                    attack_run = []
                    preceding_body_run = []
                body_run.append(sample)
        if in_attack_run and attack_run:
            body_peaks_for_run = [float(item["peak_db"]) for item in preceding_body_run]
            if body_peaks_for_run:
                attack_peaks_for_run = [float(item["peak_db"]) for item in attack_run]
                hit_body_median_db = float(np.median(body_peaks_for_run))
                hit_attack_p95_db = float(np.percentile(attack_peaks_for_run, 95))
                body_band_summary_for_run = _body_band_summary(preceding_body_run)
                body_band_values_for_run = body_band_summary_for_run["values"]
                attack_run_profiles.append({
                    "attack_sample_count": int(len(attack_run)),
                    "body_sample_count": int(len(preceding_body_run)),
                    "attack_peak_p95_db": hit_attack_p95_db,
                    "body_peak_median_db": hit_body_median_db,
                    "attack_body_gap_db": float(hit_attack_p95_db - hit_body_median_db),
                    "body_band_median_db": (
                        float(np.median(body_band_values_for_run))
                        if body_band_values_for_run
                        else None
                    ),
                    "body_band_source": body_band_summary_for_run["source"],
                    "body_band_primary_match_count": int(
                        body_band_summary_for_run["primary_match_count"]
                    ),
                    "body_band_secondary_match_count": int(
                        body_band_summary_for_run["secondary_match_count"]
                    ),
                })

        hit_body_values = [profile["body_peak_median_db"] for profile in attack_run_profiles]
        hit_gap_values = [profile["attack_body_gap_db"] for profile in attack_run_profiles]
        hit_body_band_values = [
            profile["body_band_median_db"]
            for profile in attack_run_profiles
            if profile.get("body_band_median_db") is not None
        ]
        hit_count = len(attack_run_profiles)
        repeated_hit_evaluable = hit_count >= max(1, min_repeated_hits)
        hit_body_std_db = float(np.std(hit_body_values)) if hit_body_values else 0.0
        hit_gap_std_db = float(np.std(hit_gap_values)) if hit_gap_values else 0.0
        body_band_reference_available = bool(
            effective_body_band_reference_enabled
            and len(hit_body_band_values) >= max(1, min_repeated_hits)
        )
        hit_body_band_std_db = float(np.std(hit_body_band_values)) if hit_body_band_values else 0.0
        hit_body_band_median_db = (
            float(np.median(hit_body_band_values))
            if hit_body_band_values
            else None
        )
        repeated_hit_body_band_stable = bool(
            body_band_reference_available
            and hit_body_band_std_db <= body_band_std_max_db
        )
        repeated_hit_body_band_unstable = bool(
            body_band_reference_available
            and hit_body_band_std_db > body_band_std_max_db
        )
        repeated_hit_body_stable = bool(
            repeated_hit_evaluable
            and hit_body_std_db <= hit_body_std_max_db
        )
        repeated_hit_gap_stable = bool(
            repeated_hit_evaluable
            and hit_gap_std_db <= hit_gap_std_max_db
        )
        repeated_hit_unstable = bool(
            repeated_hit_evaluable
            and not (
                repeated_hit_body_stable
                and repeated_hit_gap_stable
                and not repeated_hit_body_band_unstable
            )
        )

        return {
            "enabled": True,
            "active": True,
            "reason": "ok",
            "sample_count": int(sample_count),
            "attack_sample_count": int(len(attack_samples)),
            "body_sample_count": int(len(body_samples)),
            "attack_top_ratio": float(attack_top_ratio),
            "attack_threshold_db": float(attack_threshold_db),
            "attack_peak_p95_db": float(attack_peak_p95_db),
            "body_peak_median_db": float(body_peak_median_db),
            "attack_body_gap_db": float(attack_body_gap),
            "attack_body_gap_threshold_db": float(attack_body_gap_db),
            "attack_dominant": bool(attack_dominant),
            "high_band_available": bool(high_band_available),
            "attack_high_band_median_db": (
                float(np.median(attack_high_values))
                if attack_high_values
                else None
            ),
            "body_high_band_median_db": (
                float(np.median(body_high_values))
                if body_high_values
                else None
            ),
            "high_band_lift_db": high_band_lift,
            "high_band_lift_threshold_db": float(high_band_lift_db),
            "high_band_transient_dominant": bool(high_band_transient_dominant),
            "hit_profiles": attack_run_profiles,
            "hit_count": int(hit_count),
            "min_repeated_hits": int(min_repeated_hits),
            "repeated_hit_evaluable": bool(repeated_hit_evaluable),
            "hit_body_std_db": float(hit_body_std_db),
            "hit_body_std_max_db": float(hit_body_std_max_db),
            "hit_gap_std_db": float(hit_gap_std_db),
            "hit_gap_std_max_db": float(hit_gap_std_max_db),
            "body_band_reference_enabled": bool(body_band_reference_enabled),
            "body_band_reference_applicable": bool(body_band_profile_applicable),
            "body_band_profile": body_band_profile_name,
            "body_band_primary_tokens": list(body_band_primary_tokens),
            "body_band_secondary_tokens": list(body_band_secondary_tokens),
            "body_band_match_policy": body_band_match_policy,
            "body_band_reference_available": bool(body_band_reference_available),
            "hit_body_band_median_db": hit_body_band_median_db,
            "hit_body_band_std_db": float(hit_body_band_std_db),
            "hit_body_band_std_max_db": float(body_band_std_max_db),
            "body_band_reference": {
                "enabled": bool(body_band_reference_enabled),
                "applicable": bool(body_band_profile_applicable),
                "available": bool(body_band_reference_available),
                "confident": bool(repeated_hit_body_band_stable),
                "profile": body_band_profile_name,
                "primary_tokens": list(body_band_primary_tokens),
                "secondary_tokens": list(body_band_secondary_tokens),
                "match_policy": body_band_match_policy,
                "hit_body_band_count": int(len(hit_body_band_values)),
                "body_band_median_db": hit_body_band_median_db,
                "hit_body_band_std_db": float(hit_body_band_std_db),
                "hit_body_band_std_max_db": float(body_band_std_max_db),
                "policy": "operator_review_evidence_only",
            },
            "repeated_hit_body_stable": bool(repeated_hit_body_stable),
            "repeated_hit_gap_stable": bool(repeated_hit_gap_stable),
            "repeated_hit_body_band_stable": bool(repeated_hit_body_band_stable),
            "repeated_hit_body_band_unstable": bool(repeated_hit_body_band_unstable),
            "repeated_hit_unstable": bool(repeated_hit_unstable),
        }

    def calculate_integrated_lufs(self):
        """Calculate integrated LUFS from collected RMS values.

        C-03 FIX: Implements two-pass relative gating per ITU-R BS.1770-4
        Section 2.8.  Without gating the integrated LUFS is underestimated
        by 1-3 dB because silent / near-silent blocks drag the average down.

        Pass 1: Absolute gate at -70 LUFS (unchanged).
        Pass 2: Relative gate — discard any block more than 10 LU below the
                Pass-1 ungated loudness.
        """
        if len(self.rms_values) == 0:
            self.integrated_lufs = -100.0
            return

        # Pass 1: absolute gate at -70 LUFS
        valid_lufs = [l for l in self.rms_values if l > -70.0]
        if len(valid_lufs) == 0:
            self.integrated_lufs = -100.0
            return

        lufs_linear = [10 ** (l / 10.0) for l in valid_lufs]
        ungated_mean_linear = float(np.mean(lufs_linear))
        ungated_loudness = 10 * np.log10(ungated_mean_linear + 1e-10)

        # Pass 2: relative gate — keep blocks >= (ungated_loudness - 10 LU)
        relative_gate_threshold = ungated_loudness - 10.0
        gated_lufs = [l for l in valid_lufs if l >= relative_gate_threshold]

        if len(gated_lufs) == 0:
            # Fallback: no blocks pass the relative gate; use ungated result
            self.integrated_lufs = ungated_loudness
            return

        gated_linear = [10 ** (l / 10.0) for l in gated_lufs]
        mean_linear = float(np.mean(gated_linear))
        self.integrated_lufs = 10 * np.log10(mean_linear + 1e-10)

    def calculate_crest_factor(self):
        """Calculate crest factor (difference between peak and RMS)."""
        if len(self.peak_values) == 0 or self.integrated_lufs < -70.0:
            self.crest_factor_db = 0.0
            return

        self.crest_factor_db = self.max_true_peak_db - self.integrated_lufs

    def calculate_safe_gain(self,
                           target_lufs: float = -18.0,
                           max_peak_limit: float = -3.0,
                           min_signal_presence: float = 0.05):
        """Calculate safe gain based on signal statistics and crest factor."""

        if self.signal_presence_ratio < min_signal_presence:
            self.suggested_gain_db = 0.0
            self.gain_limited_by = "silent_channel"
            logger.info(f"Ch{self.channel_id}: Silent/Bleed only (presence={self.signal_presence_ratio:.1%}), gain=0")
            return

        self.calculate_integrated_lufs()
        self.calculate_crest_factor()

        delta_lufs = target_lufs - self.integrated_lufs

        delta_peak = max_peak_limit - self.max_true_peak_db

        if self.crest_factor_db > 12.0:
            gain_correction = min(delta_lufs, delta_peak)
            if delta_peak < delta_lufs:
                self.gain_limited_by = "peak"
            else:
                self.gain_limited_by = "lufs"
        elif self.crest_factor_db < 6.0:
            gain_correction = delta_lufs * 0.7 + delta_peak * 0.3
            self.gain_limited_by = "lufs_priority"
        else:
            gain_correction = min(delta_lufs, delta_peak)
            if delta_peak < delta_lufs:
                self.gain_limited_by = "peak"
            else:
                self.gain_limited_by = "lufs"

        self.suggested_gain_db = np.clip(gain_correction, -24.0, 24.0)

        logger.info(f"Ch{self.channel_id}: Peak={self.max_true_peak_db:.1f}dB, LUFS={self.integrated_lufs:.1f}, "
                   f"Crest={self.crest_factor_db:.1f}dB, Gain={self.suggested_gain_db:+.1f}dB "
                   f"(limited_by={self.gain_limited_by})")

    def calculate_input_trim_recommendation(
        self,
        *,
        current_trim_db: Optional[float],
        target_peak_db: float = -10.0,
        max_peak_limit: float = -3.0,
        target_lufs: float = -18.0,
        min_signal_presence: float = 0.05,
        preamp_profile: Optional[LocalInputGainProfile] = None,
        mode: str = "soundcheck",
        mixer_channel: Optional[int] = None,
        preset: str = "custom",
        previous_recommendation: Optional[Dict[str, Any]] = None,
        analysis_timestamp_sec: Optional[float] = None,
        analysis_window_sec: Optional[float] = None,
        ensemble_context: Optional[Dict[str, Any]] = None,
    ):
        """Calculate the canonical autogain move for input trim.

        Input gain is treated as a preamp/drive control for local mixer inputs.
        The recommendation therefore targets headroom first and refuses to use
        live boosts or clean-range-breaking boosts as a loudness balancer.
        """

        profile = preamp_profile or LocalInputGainProfile()
        mode = "live" if str(mode).lower() == "live" else "soundcheck"
        current_trim = _coerce_float(current_trim_db, 0.0)
        current_analysis_timestamp_sec = _coerce_float(
            time.time() if analysis_timestamp_sec is None else analysis_timestamp_sec,
            time.time(),
        )
        current_analysis_window_sec = max(
            profile.decision_confidence_min_window_sec,
            _coerce_float(
                analysis_window_sec,
                self.total_samples * profile.decision_confidence_default_sample_sec,
            ),
        )
        self.current_trim_db = current_trim
        self.target_peak_db = float(target_peak_db)
        self.target_trim_db = current_trim
        self.suggested_gain_db = 0.0
        self.gain_limited_by = "none"
        self.operator_action = "none"
        self.operator_note = ""
        self.suggestion_confidence = 0.0

        local_input = profile.is_local_input(mixer_channel)
        current_zone = profile.zone_for_gain(current_trim) if profile.enabled else "disabled"

        self.safety_policy = {
            "method_id": INPUT_TRIM_AUTOGAIN_METHOD_ID,
            "role": INPUT_TRIM_ROLE,
            "mode": mode,
            "balance_surface": "fader",
            "live_policy": "trim_down_only" if mode == "live" else "operator_reviewed_soundcheck",
        }

        if self.signal_presence_ratio < min_signal_presence:
            self.gain_limited_by = "silent_or_bleed_only"
            self.operator_action = "check_source"
            self.operator_note = "Signal presence is too low; do not raise preamp gain from bleed."
            self.input_gain_character = {
                "source": profile.source,
                "local_input": local_input,
                "current_zone": current_zone,
                "target_zone": current_zone,
                "distortion_risk": "unknown",
            }
            logger.info(
                f"Ch{self.channel_id}: input trim skipped; signal presence={self.signal_presence_ratio:.1%}"
            )
            return

        self.calculate_integrated_lufs()
        self.calculate_crest_factor()

        boost_thresholds = profile.soundcheck_boost_thresholds(preset)
        channel_personality = boost_thresholds["personality"]
        peak_p95_db = (
            float(np.percentile(self.peak_values, 95))
            if self.peak_values
            else float(self.max_true_peak_db)
        )
        isolated_peak_spread_db = max(0.0, float(self.max_true_peak_db) - peak_p95_db)
        lufs_std_db = float(np.std(self.rms_values)) if self.rms_values else 0.0
        peak_std_db = float(np.std(self.peak_values)) if self.peak_values else 0.0
        inactive_noise_values = [v for v in self.inactive_rms_values if v > -95.0]
        noise_floor_db = (
            float(np.percentile(inactive_noise_values, 90))
            if inactive_noise_values
            else None
        )
        noise_floor_margin_db = (
            float(self.integrated_lufs - noise_floor_db)
            if noise_floor_db is not None
            else None
        )
        near_clip_threshold_db = float(max_peak_limit) - max(0.0, profile.predictive_clip_window_db)
        near_clip_peak_density = (
            sum(1 for peak in self.peak_values if peak >= near_clip_threshold_db) / len(self.peak_values)
            if self.peak_values
            else 0.0
        )
        trend_window_size = max(1, min(5, len(self.peak_values) // 3)) if self.peak_values else 0
        if trend_window_size > 0:
            first_peak_window = self.peak_values[:trend_window_size]
            recent_peak_window = self.peak_values[-trend_window_size:]
            peak_trend_db = float(np.percentile(recent_peak_window, 90) - np.percentile(first_peak_window, 90))
        else:
            peak_trend_db = 0.0
        predicted_peak_db = float(self.max_true_peak_db + max(0.0, peak_trend_db))
        predictive_clip_risk = (
            self.max_true_peak_db <= max_peak_limit
            and self.max_true_peak_db >= near_clip_threshold_db
            and near_clip_peak_density >= profile.predictive_clip_density_threshold
            and (
                peak_trend_db >= profile.predictive_clip_min_rising_trend_db
                or near_clip_peak_density >= profile.predictive_clip_sustained_density_threshold
            )
        )
        predictive_clip_state = {
            "active": bool(predictive_clip_risk),
            "near_clip_threshold_db": float(near_clip_threshold_db),
            "near_clip_peak_density": float(near_clip_peak_density),
            "density_threshold": float(profile.predictive_clip_density_threshold),
            "sustained_density_threshold": float(profile.predictive_clip_sustained_density_threshold),
            "peak_trend_db": float(peak_trend_db),
            "min_rising_trend_db": float(profile.predictive_clip_min_rising_trend_db),
            "predicted_peak_db": float(predicted_peak_db),
            "min_cut_db": float(profile.predictive_clip_min_cut_db),
            "max_cut_db": float(profile.predictive_clip_max_cut_db),
        }
        digital_clip_detected = self.max_true_peak_db >= profile.digital_clip_threshold_db
        preamp_saturation_detected = (
            bool(local_input)
            and profile.enabled
            and current_zone == "saturation_risk"
            and self.max_true_peak_db > max_peak_limit
        )
        vocal_scream_detected = (
            channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}
            and self.max_true_peak_db >= profile.scream_peak_threshold_db
            and self.integrated_lufs >= profile.scream_lufs_threshold_db
        )
        overload_classification = "normal"
        if digital_clip_detected:
            overload_classification = "digital_clip"
        elif preamp_saturation_detected:
            overload_classification = "preamp_saturation_risk"
        elif vocal_scream_detected:
            overload_classification = "vocal_scream_or_shout"
        elif predictive_clip_risk:
            overload_classification = "predictive_clip_risk"
        elif isolated_peak_spread_db > boost_thresholds["max_peak_spread_db"]:
            overload_classification = "isolated_spike"
        overload_state = {
            "classification": overload_classification,
            "digital_clip_detected": bool(digital_clip_detected),
            "preamp_saturation_detected": bool(preamp_saturation_detected),
            "vocal_scream_detected": bool(vocal_scream_detected),
            "predictive_clip_detected": bool(predictive_clip_risk),
            "digital_clip_threshold_db": float(profile.digital_clip_threshold_db),
            "scream_peak_threshold_db": float(profile.scream_peak_threshold_db),
            "scream_lufs_threshold_db": float(profile.scream_lufs_threshold_db),
        }
        transient_body_band_profile = (
            _transient_body_band_profile_for_preset(preset, channel_personality.family)
            if profile.transient_window_source_specific_body_band_profiles
            else dict(_GENERIC_TRANSIENT_BODY_BAND_PROFILE)
        )
        transient_window_profile = self.calculate_transient_window_profile(
            enabled=profile.transient_window_profile_enabled,
            min_samples=profile.transient_window_min_samples,
            attack_top_ratio=profile.transient_window_attack_top_ratio,
            attack_body_gap_db=profile.transient_window_attack_body_gap_db,
            high_band_lift_db=profile.transient_window_high_band_lift_db,
            min_repeated_hits=profile.transient_window_min_repeated_hits,
            hit_body_std_max_db=profile.transient_window_hit_body_std_max_db,
            hit_gap_std_max_db=profile.transient_window_hit_gap_std_max_db,
            body_band_reference_enabled=profile.transient_window_body_band_reference_enabled,
            body_band_std_max_db=profile.transient_window_body_band_std_max_db,
            body_band_profile=transient_body_band_profile,
        )
        body_reference_applicable = bool(channel_personality.family == "transient_percussion")
        if (
            profile.transient_window_body_reference_enabled
            and body_reference_applicable
        ):
            hit_profiles = transient_window_profile.get("hit_profiles")
            hit_body_values = [
                _coerce_float(hit_profile.get("body_peak_median_db"), 0.0)
                for hit_profile in hit_profiles
                if isinstance(hit_profile, dict)
                and hit_profile.get("body_peak_median_db") is not None
            ] if isinstance(hit_profiles, list) else []
            if hit_body_values:
                body_reference_db = float(np.median(hit_body_values))
                body_reference_raw_delta_db = float(self.target_peak_db - body_reference_db)
                attack_headroom_cap_db = float(
                    max_peak_limit - self.max_true_peak_db - max(0.0, profile.clip_safety_margin_db)
                )
                headroom_limited_delta_db = float(min(body_reference_raw_delta_db, attack_headroom_cap_db))
                review_delta_db = float(min(
                    max(0.0, headroom_limited_delta_db),
                    max(0.0, profile.soundcheck_max_boost_db),
                ))
                body_band_reference = transient_window_profile.get("body_band_reference")
                body_band_confident = (
                    not profile.transient_window_body_band_reference_enabled
                    or (
                        isinstance(body_band_reference, dict)
                        and body_band_reference.get("available")
                        and body_band_reference.get("confident")
                    )
                )
                body_reference_confident = bool(
                    transient_window_profile.get("repeated_hit_evaluable")
                    and transient_window_profile.get("repeated_hit_body_stable")
                    and transient_window_profile.get("repeated_hit_gap_stable")
                    and body_band_confident
                    and attack_headroom_cap_db > 0.0
                )
                transient_window_profile["body_reference"] = {
                    "enabled": True,
                    "applicable": True,
                    "available": True,
                    "confident": bool(body_reference_confident),
                    "body_peak_median_db": float(body_reference_db),
                    "raw_delta_to_target_db": float(body_reference_raw_delta_db),
                    "attack_headroom_cap_db": float(attack_headroom_cap_db),
                    "headroom_limited_delta_db": float(headroom_limited_delta_db),
                    "operator_review_delta_db": float(review_delta_db),
                    "soundcheck_max_boost_db": float(profile.soundcheck_max_boost_db),
                    "body_band_reference": (
                        body_band_reference
                        if isinstance(body_band_reference, dict)
                        else None
                    ),
                    "policy": "operator_review_only",
                }
            else:
                transient_window_profile["body_reference"] = {
                    "enabled": True,
                    "applicable": True,
                    "available": False,
                    "confident": False,
                    "policy": "operator_review_only",
                }
        else:
            if channel_personality.family in {"cymbal", "drum_image", "drum_room"}:
                body_reference_policy = "image_or_cymbal_review_only"
            else:
                body_reference_policy = "not_applicable"
            transient_window_profile["body_reference"] = {
                "enabled": bool(profile.transient_window_body_reference_enabled),
                "applicable": bool(body_reference_applicable),
                "available": False,
                "confident": False,
                "policy": body_reference_policy,
            }
        peak_error = self.target_peak_db - self.max_true_peak_db
        deadband = max(0.1, profile.target_deadband_db)
        desired_delta = 0.0
        reason = "within_input_headroom_window"
        boost_guard = {
            "active": False,
            "reason": None,
            "personality": channel_personality.as_report(),
            "min_presence": float(boost_thresholds["min_presence"]),
            "max_crest_db": float(boost_thresholds["max_crest_db"]),
            "min_lufs": float(boost_thresholds["min_lufs"]),
            "max_lufs_std_db": float(boost_thresholds["max_lufs_std_db"]),
            "max_peak_std_db": float(boost_thresholds["max_peak_std_db"]),
            "max_peak_spread_db": float(boost_thresholds["max_peak_spread_db"]),
            "min_noise_margin_db": float(boost_thresholds["min_noise_margin_db"]),
            "noise_floor_margin_db": (
                float(noise_floor_margin_db)
                if noise_floor_margin_db is not None
                else None
            ),
        }
        hysteresis_state = {
            "active": False,
            "reason": None,
            "recommendation_hysteresis_db": float(profile.recommendation_hysteresis_db),
            "opposite_direction_hold_db": float(profile.opposite_direction_hold_db),
            "previous_direction": "unknown",
            "current_direction": "hold",
        }

        if digital_clip_detected:
            clip_target = float(max_peak_limit) - max(0.0, profile.clip_safety_margin_db)
            desired_delta = clip_target - self.max_true_peak_db
            reason = "digital_clip_headroom"
        elif preamp_saturation_detected:
            clean_zone_delta = profile.clean_gain_max_db - current_trim
            desired_delta = min(peak_error, clean_zone_delta)
            reason = "preamp_saturation_trim_down"
        elif self.max_true_peak_db > max_peak_limit:
            if isolated_peak_spread_db >= boost_thresholds["isolated_peak_min_spread_db"]:
                clip_target = float(max_peak_limit) - max(0.0, profile.clip_safety_margin_db)
                desired_delta = clip_target - self.max_true_peak_db
                reason = "isolated_peak_headroom"
            else:
                desired_delta = peak_error
                reason = "peak_headroom"
        elif predictive_clip_risk:
            predictive_target_db = float(max_peak_limit) - max(0.0, profile.clip_safety_margin_db)
            predictive_delta = predictive_target_db - predicted_peak_db
            if predictive_delta >= 0.0:
                predictive_delta = -max(0.0, profile.predictive_clip_min_cut_db)
            desired_delta = min(predictive_delta, -max(0.0, profile.predictive_clip_min_cut_db))
            desired_delta = max(desired_delta, -max(0.0, profile.predictive_clip_max_cut_db))
            reason = "predictive_clip_risk"
        elif (
            overload_classification == "isolated_spike"
            and self.max_true_peak_db > self.target_peak_db + deadband
        ):
            desired_delta = 0.0
            reason = "isolated_spike_hold"
        elif self.max_true_peak_db > self.target_peak_db + deadband:
            desired_delta = peak_error
            reason = "target_peak"
        elif self.max_true_peak_db < self.target_peak_db - profile.quiet_source_margin_db:
            desired_delta = peak_error
            reason = "quiet_source"

        if desired_delta > 0.0 and mode == "live" and not profile.live_allow_boost:
            self.gain_limited_by = "live_down_only"
            self.operator_action = "operator_review"
            self.operator_note = "Live mode blocks input gain boosts; use fader or manual source check."
            desired_delta = 0.0

        raw_desired_delta = float(desired_delta)
        base_max_boost = max(0.0, profile.max_boost_for_mode(mode))
        clean_range_remaining_db = (
            max(0.0, profile.clean_gain_max_db - current_trim)
            if profile.enabled and local_input
            else None
        )
        ensemble_context = ensemble_context if isinstance(ensemble_context, dict) else {}
        ensemble_other_active_sources = int(_coerce_float(
            ensemble_context.get("other_active_sources"),
            0.0,
        ))
        ensemble_other_loud_sources = int(_coerce_float(
            ensemble_context.get("other_loud_sources"),
            ensemble_other_active_sources,
        ))
        ensemble_reference_peak_db = (
            _coerce_float(ensemble_context.get("reference_peak_db"), 0.0)
            if ensemble_context.get("reference_peak_db") is not None
            else None
        )
        ensemble_reference_lufs = (
            _coerce_float(ensemble_context.get("reference_lufs"), 0.0)
            if ensemble_context.get("reference_lufs") is not None
            else None
        )

        def _optional_ensemble_int(key: str) -> Optional[int]:
            value = ensemble_context.get(key)
            if value is None:
                return None
            return int(_coerce_float(value, 0.0))

        ensemble_time_coherence_enabled = bool(profile.soundcheck_ensemble_time_coherence_enabled)
        ensemble_time_coherence_evaluable = bool(
            _coerce_bool(ensemble_context.get("time_coherence_evaluable"), False)
            if ensemble_context
            else False
        )
        ensemble_time_coherent_loud_sources = int(_coerce_float(
            ensemble_context.get("time_coherent_loud_sources"),
            ensemble_other_loud_sources if not ensemble_time_coherence_enabled else 0.0,
        ))
        ensemble_required_time_coherent_sources = max(1, int(_coerce_float(
            ensemble_context.get("required_time_coherent_sources"),
            profile.soundcheck_ensemble_min_other_active_sources,
        )))
        explicit_time_coherent = ensemble_context.get("time_coherent")
        if not ensemble_time_coherence_enabled:
            ensemble_time_coherent = True
        elif explicit_time_coherent is not None:
            ensemble_time_coherent = bool(_coerce_bool(explicit_time_coherent, False))
        else:
            ensemble_time_coherent = bool(
                ensemble_time_coherence_evaluable
                and ensemble_time_coherent_loud_sources >= ensemble_required_time_coherent_sources
            )
        ensemble_context_lag_limit_samples = max(0, int(_coerce_float(
            ensemble_context.get("max_context_lag_samples"),
            profile.soundcheck_ensemble_max_context_lag_samples,
        )))
        ensemble_min_other_loud_sources = int(profile.soundcheck_ensemble_min_other_active_sources)
        ensemble_loud_source_margin = int(ensemble_other_loud_sources - ensemble_min_other_loud_sources)
        ensemble_time_coherent_source_margin = int(
            ensemble_time_coherent_loud_sources - ensemble_required_time_coherent_sources
        )
        ensemble_reference_max_lag_samples = _optional_ensemble_int("reference_max_lag_samples")
        ensemble_reference_lag_margin_samples = (
            int(ensemble_context_lag_limit_samples - ensemble_reference_max_lag_samples)
            if ensemble_reference_max_lag_samples is not None
            else None
        )
        ensemble_peak_gap_db = (
            float(ensemble_reference_peak_db - self.max_true_peak_db)
            if ensemble_reference_peak_db is not None
            else None
        )
        ensemble_lufs_gap_db = (
            float(ensemble_reference_lufs - self.integrated_lufs)
            if ensemble_reference_lufs is not None
            else None
        )
        ensemble_vocal_family = channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}
        ensemble_reference_loud_enough = bool(
            ensemble_reference_peak_db is not None
            and ensemble_reference_lufs is not None
            and ensemble_reference_peak_db >= profile.soundcheck_ensemble_min_reference_peak_db
            and ensemble_reference_lufs >= profile.soundcheck_ensemble_min_reference_lufs
        )
        ensemble_gap_supports_gain_up = bool(
            ensemble_peak_gap_db is not None
            and ensemble_peak_gap_db >= profile.soundcheck_ensemble_vocal_under_peak_gap_db
            and (
                (
                    ensemble_lufs_gap_db is not None
                    and ensemble_lufs_gap_db >= profile.soundcheck_ensemble_vocal_under_lufs_gap_db
                )
                or self.crest_factor_db >= profile.soundcheck_ensemble_vocal_min_crest_db
            )
        )
        soundcheck_ensemble_context_state = {
            "enabled": bool(profile.soundcheck_ensemble_context_enabled),
            "provided": bool(ensemble_context),
            "active": False,
            "reason": None,
            "source_family": channel_personality.family,
            "family_applicable": bool(ensemble_vocal_family),
            "other_active_sources": int(ensemble_other_active_sources),
            "other_loud_sources": int(ensemble_other_loud_sources),
            "min_other_active_sources": int(ensemble_min_other_loud_sources),
            "other_loud_source_margin": int(ensemble_loud_source_margin),
            "reference_peak_db": (
                float(ensemble_reference_peak_db)
                if ensemble_reference_peak_db is not None
                else None
            ),
            "reference_lufs": (
                float(ensemble_reference_lufs)
                if ensemble_reference_lufs is not None
                else None
            ),
            "min_reference_peak_db": float(profile.soundcheck_ensemble_min_reference_peak_db),
            "min_reference_lufs": float(profile.soundcheck_ensemble_min_reference_lufs),
            "vocal_peak_gap_db": (
                float(ensemble_peak_gap_db)
                if ensemble_peak_gap_db is not None
                else None
            ),
            "vocal_lufs_gap_db": (
                float(ensemble_lufs_gap_db)
                if ensemble_lufs_gap_db is not None
                else None
            ),
            "required_peak_gap_db": float(profile.soundcheck_ensemble_vocal_under_peak_gap_db),
            "required_lufs_gap_db": float(profile.soundcheck_ensemble_vocal_under_lufs_gap_db),
            "vocal_crest_factor_db": float(self.crest_factor_db),
            "required_vocal_min_crest_db": float(profile.soundcheck_ensemble_vocal_min_crest_db),
            "reference_loud_enough": bool(ensemble_reference_loud_enough),
            "gap_supports_gain_up": bool(ensemble_gap_supports_gain_up),
            "max_boost_db": float(profile.soundcheck_ensemble_max_boost_db),
            "gain_up_step_db": float(profile.soundcheck_ensemble_gain_up_step_db),
            "confidence_bonus": float(profile.soundcheck_ensemble_confidence_bonus),
            "time_coherence_enabled": bool(ensemble_time_coherence_enabled),
            "time_coherence_evaluable": bool(ensemble_time_coherence_evaluable),
            "time_coherent": bool(ensemble_time_coherent),
            "time_coherent_loud_sources": int(ensemble_time_coherent_loud_sources),
            "required_time_coherent_sources": int(ensemble_required_time_coherent_sources),
            "time_coherent_source_margin": int(ensemble_time_coherent_source_margin),
            "max_context_lag_samples": int(ensemble_context_lag_limit_samples),
            "target_first_active_sample_index": _optional_ensemble_int("target_first_active_sample_index"),
            "target_last_active_sample_index": _optional_ensemble_int("target_last_active_sample_index"),
            "reference_latest_active_sample_index": _optional_ensemble_int("reference_latest_active_sample_index"),
            "reference_max_lag_samples": ensemble_reference_max_lag_samples,
            "reference_lag_margin_samples": ensemble_reference_lag_margin_samples,
            "operator_review_only": True,
        }
        if not profile.soundcheck_ensemble_context_enabled:
            soundcheck_ensemble_context_state["reason"] = "disabled"
        elif not ensemble_context:
            soundcheck_ensemble_context_state["reason"] = "missing_ensemble_context"
        elif not ensemble_vocal_family:
            soundcheck_ensemble_context_state["reason"] = "non_vocal_source"
        elif ensemble_other_loud_sources < profile.soundcheck_ensemble_min_other_active_sources:
            soundcheck_ensemble_context_state["reason"] = "insufficient_loud_ensemble_sources"
        elif ensemble_time_coherence_enabled and not ensemble_time_coherence_evaluable:
            soundcheck_ensemble_context_state["reason"] = "ensemble_time_context_missing"
        elif ensemble_time_coherence_enabled and not ensemble_time_coherent:
            soundcheck_ensemble_context_state["reason"] = "ensemble_time_context_stale"
        elif not ensemble_reference_loud_enough:
            soundcheck_ensemble_context_state["reason"] = "ensemble_reference_not_loud_enough"
        elif not ensemble_gap_supports_gain_up:
            soundcheck_ensemble_context_state["reason"] = "vocal_not_understaged_vs_ensemble"
        elif raw_desired_delta <= 0.0:
            soundcheck_ensemble_context_state["reason"] = "no_gain_up_candidate"
        elif mode != "soundcheck":
            soundcheck_ensemble_context_state["reason"] = "not_soundcheck"
        else:
            soundcheck_ensemble_context_state.update({
                "active": True,
                "reason": "ensemble_context_vocal_understaged",
            })
        soundcheck_convergence_gap_db = max(0.0, raw_desired_delta)
        soundcheck_convergence_family_cap = (
            max(0.0, profile.soundcheck_convergence_cap_for(channel_personality))
            if profile.soundcheck_convergence_enabled
            else 0.0
        )
        soundcheck_ensemble_boost_cap = (
            max(0.0, min(
                profile.soundcheck_ensemble_max_boost_db,
                profile.soundcheck_convergence_cap_for(channel_personality),
            ))
            if soundcheck_ensemble_context_state.get("active")
            else 0.0
        )
        soundcheck_convergence_state = {
            "enabled": bool(profile.soundcheck_convergence_enabled),
            "active": False,
            "reason": None,
            "mode": mode,
            "source_family": channel_personality.family,
            "source_policy": channel_personality.boost_policy,
            "requested_delta_db": float(soundcheck_convergence_gap_db),
            "large_gap_threshold_db": float(profile.soundcheck_convergence_large_gap_db),
            "base_max_boost_db": float(base_max_boost),
            "family_cap_db": float(soundcheck_convergence_family_cap),
            "effective_max_boost_db": float(base_max_boost),
            "clean_range_remaining_db": (
                float(clean_range_remaining_db)
                if clean_range_remaining_db is not None
                else None
            ),
            "setup_check_threshold_db": float(profile.soundcheck_convergence_setup_check_db),
            "setup_check_required": False,
            "setup_check_reason": None,
            "ensemble_assisted": bool(soundcheck_ensemble_context_state.get("active")),
            "ensemble_boost_cap_db": float(soundcheck_ensemble_boost_cap),
            "operator_review_only": True,
        }
        if (
            mode == "soundcheck"
            and profile.soundcheck_convergence_enabled
            and soundcheck_convergence_gap_db > 0.0
        ):
            setup_check_reasons = []
            if soundcheck_convergence_gap_db >= profile.soundcheck_convergence_setup_check_db:
                setup_check_reasons.append("requested_gain_exceeds_setup_check_threshold")
            if (
                soundcheck_convergence_gap_db >= profile.soundcheck_convergence_large_gap_db
                and clean_range_remaining_db is not None
                and soundcheck_convergence_gap_db > clean_range_remaining_db
            ):
                setup_check_reasons.append("requested_gain_exceeds_clean_range")
            if setup_check_reasons:
                soundcheck_convergence_state.update({
                    "setup_check_required": True,
                    "setup_check_reason": setup_check_reasons[0],
                    "setup_check_reasons": setup_check_reasons,
                })
            if (
                soundcheck_convergence_gap_db >= profile.soundcheck_convergence_large_gap_db
                and channel_personality.boost_policy != "operator_review"
                and soundcheck_convergence_family_cap > base_max_boost
            ):
                base_max_boost = soundcheck_convergence_family_cap
                soundcheck_convergence_state.update({
                    "active": True,
                    "reason": "large_soundcheck_gain_gap",
                    "effective_max_boost_db": float(base_max_boost),
                })
            if (
                soundcheck_ensemble_context_state.get("active")
                and channel_personality.boost_policy != "operator_review"
                and soundcheck_ensemble_boost_cap > base_max_boost
            ):
                base_max_boost = soundcheck_ensemble_boost_cap
                soundcheck_convergence_state.update({
                    "active": True,
                    "reason": (
                        soundcheck_convergence_state.get("reason")
                        or "ensemble_context_vocal_understaged"
                    ),
                    "effective_max_boost_db": float(base_max_boost),
                })

        max_cut = max(0.0, profile.max_cut_for_mode(mode))
        max_boost = max(0.0, base_max_boost)
        if desired_delta < 0.0:
            desired_delta = max(desired_delta, -max_cut)
        elif desired_delta > 0.0:
            desired_delta = min(desired_delta, max_boost)
            if max_boost <= 0.0:
                self.gain_limited_by = "boost_disabled"
                self.operator_action = "operator_review"
                self.operator_note = "Configured preamp profile blocks automatic gain boosts."
        soundcheck_convergence_state["post_mode_cap_delta_db"] = float(desired_delta)

        if desired_delta > 0.0 and mode == "soundcheck":
            guard_reason, guard_note, guard_evidence = profile.soundcheck_boost_guard(
                signal_presence_ratio=self.signal_presence_ratio,
                crest_factor_db=self.crest_factor_db,
                integrated_lufs=self.integrated_lufs,
                preset=preset,
                lufs_std_db=lufs_std_db,
                peak_std_db=peak_std_db,
                isolated_peak_spread_db=isolated_peak_spread_db,
                noise_floor_margin_db=noise_floor_margin_db,
            )
            boost_guard.update(guard_evidence)
            if guard_reason:
                boost_guard["active"] = True
                boost_guard["reason"] = guard_reason
                self.gain_limited_by = guard_reason
                self.operator_action = "operator_review"
                self.operator_note = guard_note
                desired_delta = 0.0
        soundcheck_convergence_state["post_boost_guard_delta_db"] = float(desired_delta)

        target_trim = current_trim + desired_delta
        target_trim = max(profile.min_gain_db, min(profile.hard_max_gain_db, target_trim))
        desired_delta = target_trim - current_trim

        if (
            desired_delta > 0.0
            and profile.enabled
            and local_input
            and target_trim > profile.clean_gain_max_db
        ):
            capped_target = min(profile.clean_gain_max_db, profile.hard_max_gain_db)
            capped_delta = capped_target - current_trim
            if capped_delta > 0.0:
                target_trim = capped_target
                desired_delta = capped_delta
                self.gain_limited_by = "preamp_clean_range"
                self.operator_note = (
                    "Boost capped at the clean local preamp range; further gain would change tone."
                )
            else:
                target_trim = current_trim
                desired_delta = 0.0
                self.gain_limited_by = "preamp_character_limit"
                self.operator_action = "manual_gain_or_source_check"
                self.operator_note = (
                    "Local gain is already above the clean range; do not auto-boost into preamp drive."
                )
        soundcheck_convergence_state["post_clean_range_delta_db"] = float(desired_delta)

        previous_delta = None
        previous_character: Dict[str, Any] = {}
        if isinstance(previous_recommendation, dict):
            raw_previous_delta = previous_recommendation.get("suggested_gain_db")
            if raw_previous_delta is not None:
                previous_delta = _coerce_float(raw_previous_delta, 0.0)
            raw_previous_character = previous_recommendation.get("input_gain_character")
            if isinstance(raw_previous_character, dict):
                previous_character = raw_previous_character

        previous_direction = _gain_direction(previous_delta)
        current_direction = _gain_direction(desired_delta)
        hysteresis_state["previous_direction"] = previous_direction
        hysteresis_state["current_direction"] = current_direction
        urgent_headroom_cut = desired_delta < 0.0 and self.max_true_peak_db > max_peak_limit
        if (
            current_direction in {"up", "down"}
            and previous_direction in {"up", "down"}
            and current_direction != previous_direction
            and not urgent_headroom_cut
            and abs(desired_delta) <= max(0.0, profile.opposite_direction_hold_db)
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "anti_oscillation_hold"
            self.operator_action = "hold"
            self.operator_note = (
                "Recommendation held: small opposite-direction trim move would risk gain oscillation."
            )
            hysteresis_state.update({
                "active": True,
                "reason": "small_opposite_direction_change",
                "held_previous_delta_db": float(previous_delta or 0.0),
            })

        candidate_delta = desired_delta
        candidate_direction = _gain_direction(candidate_delta)
        previous_stability = {}
        if isinstance(previous_character, dict):
            previous_stability = previous_character.get("stability") or {}
        previous_candidate_direction = (
            str(previous_stability.get("candidate_direction") or previous_direction)
            if isinstance(previous_stability, dict)
            else previous_direction
        )
        previous_candidate_delta = None
        if isinstance(previous_stability, dict) and previous_stability.get("candidate_delta_db") is not None:
            previous_candidate_delta = _coerce_float(previous_stability.get("candidate_delta_db"), 0.0)
        previous_evidence_windows = 0
        if isinstance(previous_stability, dict):
            previous_evidence_windows = max(0, int(_coerce_float(previous_stability.get("evidence_windows"), 0.0)))
        same_stable_candidate = (
            candidate_direction in {"up", "down"}
            and previous_candidate_direction == candidate_direction
            and (
                previous_candidate_delta is None
                or abs(candidate_delta - previous_candidate_delta) <= max(0.0, profile.stable_window_tolerance_db)
            )
        )
        evidence_windows = (
            previous_evidence_windows + 1
            if same_stable_candidate
            else (1 if candidate_direction in {"up", "down"} else 0)
        )
        stability_state = {
            "active": False,
            "reason": None,
            "candidate_direction": candidate_direction,
            "candidate_delta_db": float(candidate_delta),
            "evidence_windows": int(evidence_windows),
            "required_gain_up_windows": int(profile.gain_up_min_stable_windows),
            "stable_window_tolerance_db": float(profile.stable_window_tolerance_db),
        }
        if (
            candidate_direction == "up"
            and mode == "soundcheck"
            and evidence_windows < profile.gain_up_min_stable_windows
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "gain_up_stability_wait"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held until the same gain-up evidence repeats across stable windows."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_repeated_gain_up_evidence",
            })

        target_zone = profile.zone_for_gain(target_trim) if profile.enabled else "disabled"
        distortion_risk = profile.distortion_risk(target_trim, local_input=local_input)
        direction = "down" if desired_delta < 0.0 else "up" if desired_delta > 0.0 else "hold"
        source_behavior = "stable_dynamic"
        if self.signal_presence_ratio < boost_thresholds["min_presence"]:
            source_behavior = "sparse_source"
        elif isolated_peak_spread_db > boost_thresholds["max_peak_spread_db"]:
            source_behavior = "isolated_peak_source"
        elif self.crest_factor_db > boost_thresholds["max_crest_db"]:
            source_behavior = "transient_dominant"
        elif (
            lufs_std_db > boost_thresholds["max_lufs_std_db"]
            or peak_std_db > boost_thresholds["max_peak_std_db"]
        ):
            source_behavior = "unstable_dynamic"
        elif (
            noise_floor_margin_db is not None
            and noise_floor_margin_db < boost_thresholds["min_noise_margin_db"]
        ):
            source_behavior = "low_noise_margin"
        elif channel_personality.boost_policy == "operator_review":
            source_behavior = "operator_review_source"
        elif self.crest_factor_db <= 8.0 and lufs_std_db <= 3.0 and peak_std_db <= 3.0:
            source_behavior = "compressed_stable"

        previous_trend_memory = previous_character.get("trend_memory") if isinstance(previous_character, dict) else {}
        if not isinstance(previous_trend_memory, dict):
            previous_trend_memory = {}
        previous_behavior = (
            previous_character.get("source_behavior")
            if isinstance(previous_character, dict)
            else None
        )
        previous_behavior_history = previous_trend_memory.get("behavior_history") or []
        if not isinstance(previous_behavior_history, list):
            previous_behavior_history = []
        memory_limit = max(1, int(profile.trend_memory_max_windows))
        behavior_history = [str(item) for item in previous_behavior_history[-(memory_limit - 1):]]
        behavior_history.append(source_behavior)
        previous_predictive_clip_windows = max(
            0,
            int(_coerce_float(previous_trend_memory.get("predictive_clip_windows"), 0.0)),
        )
        predictive_clip_windows = (
            previous_predictive_clip_windows + 1
            if predictive_clip_risk
            else 0
        )
        behavior_flip = bool(previous_behavior and previous_behavior != source_behavior)
        spectral_fingerprint = self.calculate_spectral_fingerprint()
        vocal_family = channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}
        previous_vocal_scream_cooldown_windows = max(
            0,
            int(_coerce_float(previous_trend_memory.get("vocal_scream_cooldown_windows"), 0.0)),
        )
        vocal_scream_cooldown_windows = (
            int(profile.vocal_gain_up_scream_cooldown_windows)
            if vocal_scream_detected
            else max(0, previous_vocal_scream_cooldown_windows - 1)
        )

        def _vocal_presence_band_state(band_name: str) -> Optional[Dict[str, Any]]:
            band_counts = spectral_fingerprint.get("band_sample_counts")
            band_std_values = spectral_fingerprint.get("band_std_db")
            band_span_values = spectral_fingerprint.get("band_span_db")
            if not isinstance(band_counts, dict) or band_name not in band_counts:
                return None
            band_count = int(_coerce_float(band_counts.get(band_name), 0.0))
            band_std_db = (
                _coerce_float(band_std_values.get(band_name), 0.0)
                if isinstance(band_std_values, dict)
                else 0.0
            )
            band_span_db = (
                _coerce_float(band_span_values.get(band_name), 0.0)
                if isinstance(band_span_values, dict)
                else 0.0
            )
            stable = bool(
                band_std_db <= profile.vocal_gain_up_presence_max_std_db
                and band_span_db <= profile.vocal_gain_up_presence_max_span_db
            )
            return {
                "samples": int(band_count),
                "std_db": float(band_std_db),
                "span_db": float(band_span_db),
                "max_std_db": float(profile.vocal_gain_up_presence_max_std_db),
                "max_span_db": float(profile.vocal_gain_up_presence_max_span_db),
                "stable": bool(stable),
            }

        vocal_presence_bands = {
            band_name: band_state
            for band_name in ("presence", "air")
            if (band_state := _vocal_presence_band_state(band_name)) is not None
        }
        vocal_presence_available = bool(
            len(vocal_presence_bands) >= profile.vocal_gain_up_presence_min_bands
        )
        vocal_presence_unstable = bool(
            vocal_presence_available
            and any(not band_state["stable"] for band_state in vocal_presence_bands.values())
        )
        vocal_scream_history_active = bool(
            vocal_family
            and vocal_scream_cooldown_windows > 0
            and not vocal_scream_detected
        )
        vocal_gain_up_guard_state = {
            "enabled": bool(profile.vocal_gain_up_guard_enabled),
            "family_applicable": bool(vocal_family),
            "presence_min_bands": int(profile.vocal_gain_up_presence_min_bands),
            "presence_available": bool(vocal_presence_available),
            "presence_bands": vocal_presence_bands,
            "presence_unstable": bool(vocal_presence_unstable),
            "scream_detected": bool(vocal_scream_detected),
            "previous_scream_cooldown_windows": int(previous_vocal_scream_cooldown_windows),
            "scream_cooldown_windows": int(vocal_scream_cooldown_windows),
            "scream_history_active": bool(vocal_scream_history_active),
            "active": False,
            "reason": None,
        }
        if profile.vocal_gain_up_guard_enabled and vocal_family:
            if vocal_scream_history_active:
                vocal_gain_up_guard_state.update({
                    "active": True,
                    "reason": "recent_vocal_scream_or_shout",
                })
            elif vocal_presence_unstable:
                vocal_gain_up_guard_state.update({
                    "active": True,
                    "reason": "unstable_vocal_presence_band",
                })
        source_signature = {
            "family": channel_personality.family,
            "behavior": source_behavior,
            "crest_db": float(self.crest_factor_db),
            "crest_bucket_db": _bucket_db(self.crest_factor_db, 3.0),
            "integrated_lufs_db": float(self.integrated_lufs),
            "integrated_lufs_bucket_db": _bucket_db(self.integrated_lufs, 6.0),
            "peak_lufs_gap_db": float(self.max_true_peak_db - self.integrated_lufs),
            "peak_lufs_gap_bucket_db": _bucket_db(self.max_true_peak_db - self.integrated_lufs, 3.0),
            "noise_margin_db": float(noise_floor_margin_db) if noise_floor_margin_db is not None else None,
            "noise_margin_bucket_db": _bucket_db(noise_floor_margin_db, 6.0),
            "spectral_fingerprint": spectral_fingerprint,
        }
        previous_source_signature = previous_trend_memory.get("source_signature")
        if not isinstance(previous_source_signature, dict):
            previous_source_signature = {}
        signature_compare_fields = [
            "family",
            "crest_bucket_db",
            "integrated_lufs_bucket_db",
            "peak_lufs_gap_bucket_db",
            "noise_margin_bucket_db",
        ]
        source_signature_changed_fields = [
            field_name
            for field_name in signature_compare_fields
            if (
                previous_source_signature.get(field_name) is not None
                and source_signature.get(field_name) is not None
                and previous_source_signature.get(field_name) != source_signature.get(field_name)
            )
        ]
        source_family_changed = bool(
            previous_source_signature.get("family") is not None
            and previous_source_signature.get("family") != source_signature["family"]
        )
        source_identity_changed = bool(
            previous_source_signature
            and (
                source_family_changed
                or len(source_signature_changed_fields) >= profile.source_signature_change_min_fields
            )
        )
        trend_memory_state = {
            "behavior_history": behavior_history,
            "previous_source_behavior": previous_behavior,
            "current_source_behavior": source_behavior,
            "behavior_flip": behavior_flip,
            "source_signature": source_signature,
            "previous_source_signature": previous_source_signature or None,
            "source_identity_changed": source_identity_changed,
            "source_family_changed": source_family_changed,
            "source_signature_changed_fields": source_signature_changed_fields,
            "source_signature_change_min_fields": int(profile.source_signature_change_min_fields),
            "predictive_clip_windows": int(predictive_clip_windows),
            "vocal_scream_cooldown_windows": int(vocal_scream_cooldown_windows),
            "vocal_gain_up_guard": vocal_gain_up_guard_state,
            "near_clip_peak_density": float(near_clip_peak_density),
            "peak_trend_db": float(peak_trend_db),
            "noise_floor_margin_db": (
                float(noise_floor_margin_db)
                if noise_floor_margin_db is not None
                else None
            ),
        }
        if (
            profile.behavior_flip_gain_up_hold
            and behavior_flip
            and candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "behavior_change_stability_wait"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: source behavior changed between analysis windows."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_consistent_source_behavior",
            })
        if (
            profile.source_signature_change_hold_enabled
            and source_identity_changed
            and candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "source_identity_change_wait"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: source signature changed between analysis windows."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_source_identity_confirmation",
            })

        previous_channel_profile = previous_character.get("channel_profile") if isinstance(previous_character, dict) else {}
        if not isinstance(previous_channel_profile, dict):
            previous_channel_profile = {}
        previous_profile_lifecycle = (
            previous_character.get("profile_lifecycle")
            if isinstance(previous_character, dict)
            else {}
        )
        if not isinstance(previous_profile_lifecycle, dict):
            previous_profile_lifecycle = {}
        requested_lifecycle_action = str(previous_profile_lifecycle.get("action") or "none").strip().lower()
        if requested_lifecycle_action not in {"none", "reset", "lock", "unlock", "decay"}:
            requested_lifecycle_action = "none"
        lifecycle_reason = previous_profile_lifecycle.get("reason")
        if lifecycle_reason is not None:
            lifecycle_reason = str(lifecycle_reason)
        profile_locked = _coerce_bool(previous_profile_lifecycle.get("profile_locked"), False)
        if "locked" in previous_profile_lifecycle:
            profile_locked = _coerce_bool(previous_profile_lifecycle.get("locked"), profile_locked)
        raw_previous_profile_windows = min(
            max(0, int(_coerce_float(previous_channel_profile.get("windows"), 0.0))),
            max(0, int(profile.channel_profile_max_windows) - 1),
        )
        original_previous_profile_windows = raw_previous_profile_windows
        applied_lifecycle_action = "none"
        lifecycle_decay_factor = float(np.clip(
            _coerce_float(
                previous_profile_lifecycle.get("decay_factor"),
                profile.channel_profile_default_decay_factor,
            ),
            profile.channel_profile_min_decay_factor,
            1.0,
        ))
        if profile.channel_profile_lifecycle_enabled:
            if requested_lifecycle_action == "reset":
                previous_channel_profile = {}
                raw_previous_profile_windows = 0
                profile_locked = False
                lifecycle_decay_factor = 0.0
                applied_lifecycle_action = "reset"
            elif requested_lifecycle_action == "lock":
                profile_locked = True
                applied_lifecycle_action = "lock"
            elif requested_lifecycle_action == "unlock":
                profile_locked = False
                applied_lifecycle_action = "unlock"
            elif requested_lifecycle_action == "decay":
                raw_previous_profile_windows = int(round(raw_previous_profile_windows * lifecycle_decay_factor))
                applied_lifecycle_action = "decay"
        else:
            requested_lifecycle_action = "none"
            profile_locked = False
        previous_profile_windows = min(
            raw_previous_profile_windows,
            max(0, int(profile.channel_profile_max_windows) - 1),
        )
        profile_lifecycle_state = {
            "enabled": bool(profile.channel_profile_lifecycle_enabled),
            "requested_action": requested_lifecycle_action,
            "applied_action": applied_lifecycle_action,
            "reason": lifecycle_reason,
            "profile_locked": bool(profile_locked),
            "decay_factor": float(lifecycle_decay_factor),
            "previous_windows": int(original_previous_profile_windows),
            "retained_windows": int(previous_profile_windows),
        }
        risky_behaviors = {
            "sparse_source",
            "isolated_peak_source",
            "transient_dominant",
            "unstable_dynamic",
            "low_noise_margin",
            "operator_review_source",
        }
        live_freezes_gain_up_profile = bool(mode == "live" and profile.live_freeze_gain_up_profile)
        gain_up_profile_frozen = bool(live_freezes_gain_up_profile or profile_locked)
        current_unstable_ratio = 1.0 if source_behavior in risky_behaviors else 0.0
        previous_noise_margin_min = (
            previous_channel_profile.get("noise_margin_min_db")
            if previous_profile_windows > 0
            else None
        )
        if gain_up_profile_frozen:
            crest_avg_db = previous_channel_profile.get("crest_avg_db")
            lufs_std_avg_db = previous_channel_profile.get("lufs_std_avg_db")
            near_clip_density_avg = previous_channel_profile.get("near_clip_density_avg")
            unstable_behavior_ratio = _coerce_float(
                previous_channel_profile.get("unstable_behavior_ratio"),
                0.0,
            )
            noise_margin_min_db = (
                float(previous_noise_margin_min)
                if previous_noise_margin_min is not None
                else None
            )
            channel_profile_windows = previous_profile_windows
        else:
            crest_avg_db = _rolling_average(
                previous_channel_profile.get("crest_avg_db"),
                self.crest_factor_db,
                previous_profile_windows,
            )
            lufs_std_avg_db = _rolling_average(
                previous_channel_profile.get("lufs_std_avg_db"),
                lufs_std_db,
                previous_profile_windows,
            )
            near_clip_density_avg = _rolling_average(
                previous_channel_profile.get("near_clip_density_avg"),
                near_clip_peak_density,
                previous_profile_windows,
            )
            unstable_behavior_ratio = _rolling_average(
                previous_channel_profile.get("unstable_behavior_ratio"),
                current_unstable_ratio,
                previous_profile_windows,
            ) or 0.0
            if noise_floor_margin_db is None:
                noise_margin_min_db = (
                    float(previous_noise_margin_min)
                    if previous_noise_margin_min is not None
                    else None
                )
            elif previous_noise_margin_min is None:
                noise_margin_min_db = float(noise_floor_margin_db)
            else:
                noise_margin_min_db = min(float(previous_noise_margin_min), float(noise_floor_margin_db))
            channel_profile_windows = min(previous_profile_windows + 1, int(profile.channel_profile_max_windows))
        if gain_up_profile_frozen and noise_floor_margin_db is None:
            noise_margin_min_db = (
                float(previous_noise_margin_min)
                if previous_noise_margin_min is not None
                else None
            )
        current_live_risk = bool(
            mode == "live"
            and (
                predictive_clip_risk
                or candidate_direction == "down"
                or overload_classification != "normal"
                or self.max_true_peak_db > self.target_peak_db + deadband
            )
        )
        previous_live_risk_windows = max(
            0,
            int(_coerce_float(previous_channel_profile.get("live_risk_windows"), 0.0)),
        )
        previous_live_overload_events = max(
            0,
            int(_coerce_float(previous_channel_profile.get("live_overload_events"), 0.0)),
        )
        previous_live_peak_density_max = _coerce_float(
            previous_channel_profile.get("live_peak_density_max"),
            0.0,
        )
        live_risk_windows = previous_live_risk_windows + (1 if current_live_risk else 0)
        live_overload_events = previous_live_overload_events + (
            1 if mode == "live" and overload_classification != "normal" else 0
        )
        live_peak_density_max = max(
            previous_live_peak_density_max,
            near_clip_peak_density if mode == "live" else 0.0,
        )
        previous_recovery_windows = (
            max(0, int(_coerce_float(previous_channel_profile.get("recovery_windows"), 0.0)))
            if previous_profile_windows > 0
            else 0
        )
        recovery_policy_family = channel_personality.family
        recovery_allowed = bool(profile.channel_profile_recovery_enabled)
        recovery_required_windows = int(profile.channel_profile_recovery_min_windows)
        recovery_max_lufs_std_db = float(profile.channel_profile_recovery_max_lufs_std_db)
        recovery_max_peak_std_db = float(profile.channel_profile_recovery_max_peak_std_db)
        recovery_max_crest_db = float(profile.channel_profile_recovery_max_crest_db)
        recovery_min_noise_margin_db = float(profile.channel_profile_recovery_min_noise_margin_db)
        if recovery_policy_family in {"lead_vocal", "backing_vocal", "speech"}:
            recovery_required_windows = max(recovery_required_windows, 4)
            recovery_max_lufs_std_db = min(recovery_max_lufs_std_db, 2.5)
            recovery_max_peak_std_db = min(recovery_max_peak_std_db, 2.5)
            recovery_min_noise_margin_db = max(recovery_min_noise_margin_db, 18.0)
        elif recovery_policy_family == "playback":
            recovery_required_windows = min(recovery_required_windows, 2)
            recovery_max_lufs_std_db = min(recovery_max_lufs_std_db, 2.0)
            recovery_max_peak_std_db = min(recovery_max_peak_std_db, 2.0)
            recovery_max_crest_db = min(recovery_max_crest_db, 11.0)
        if channel_personality.boost_policy == "operator_review":
            recovery_allowed = False
        recovery_candidate = bool(
            recovery_allowed
            and mode == "soundcheck"
            and not gain_up_profile_frozen
            and source_behavior in {"stable_dynamic", "compressed_stable"}
            and lufs_std_db <= recovery_max_lufs_std_db
            and peak_std_db <= recovery_max_peak_std_db
            and self.crest_factor_db <= recovery_max_crest_db
            and (
                noise_floor_margin_db is None
                or noise_floor_margin_db >= recovery_min_noise_margin_db
            )
        )
        if gain_up_profile_frozen:
            recovery_windows = previous_recovery_windows
        else:
            recovery_windows = previous_recovery_windows + 1 if recovery_candidate else 0
        profile_blocks_gain_up_by_history_raw = (
            channel_profile_windows >= profile.channel_profile_min_windows
            and (
                unstable_behavior_ratio >= profile.channel_profile_unstable_ratio_block
                or (lufs_std_avg_db is not None and lufs_std_avg_db > profile.channel_profile_lufs_std_block_db)
                or (
                    noise_margin_min_db is not None
                    and noise_margin_min_db < profile.channel_profile_noise_margin_block_db
                )
            )
        )
        profile_recovered_from_history = bool(
            profile_blocks_gain_up_by_history_raw
            and recovery_windows >= recovery_required_windows
            and recovery_candidate
        )
        profile_blocks_gain_up_by_history = bool(
            profile_blocks_gain_up_by_history_raw
            and not profile_recovered_from_history
        )
        profile_blocks_gain_up = bool(profile_locked or profile_blocks_gain_up_by_history)
        if profile_locked:
            learning_mode = "profile_locked"
        elif live_freezes_gain_up_profile:
            learning_mode = "live_risk_only"
        else:
            learning_mode = "soundcheck_learning"
        channel_profile_state = {
            "phase": mode,
            "learning_mode": learning_mode,
            "gain_up_profile_frozen": gain_up_profile_frozen,
            "profile_locked": bool(profile_locked),
            "windows": int(channel_profile_windows),
            "profile_ready": bool(channel_profile_windows >= profile.channel_profile_min_windows),
            "profile_blocks_gain_up": bool(profile_blocks_gain_up),
            "profile_blocks_gain_up_by_history": bool(profile_blocks_gain_up_by_history),
            "profile_blocks_gain_up_by_history_raw": bool(profile_blocks_gain_up_by_history_raw),
            "profile_recovered_from_history": bool(profile_recovered_from_history),
            "recovery_allowed": bool(recovery_allowed),
            "recovery_policy_family": recovery_policy_family,
            "recovery_candidate": bool(recovery_candidate),
            "recovery_windows": int(recovery_windows),
            "recovery_required_windows": int(recovery_required_windows),
            "recovery_max_lufs_std_db": float(recovery_max_lufs_std_db),
            "recovery_max_peak_std_db": float(recovery_max_peak_std_db),
            "recovery_max_crest_db": float(recovery_max_crest_db),
            "recovery_min_noise_margin_db": float(recovery_min_noise_margin_db),
            "crest_avg_db": float(crest_avg_db) if crest_avg_db is not None else None,
            "lufs_std_avg_db": float(lufs_std_avg_db) if lufs_std_avg_db is not None else None,
            "near_clip_density_avg": (
                float(near_clip_density_avg)
                if near_clip_density_avg is not None
                else None
            ),
            "noise_margin_min_db": (
                float(noise_margin_min_db)
                if noise_margin_min_db is not None
                else None
            ),
            "unstable_behavior_ratio": float(unstable_behavior_ratio),
            "unstable_ratio_block": float(profile.channel_profile_unstable_ratio_block),
            "lufs_std_block_db": float(profile.channel_profile_lufs_std_block_db),
            "noise_margin_block_db": float(profile.channel_profile_noise_margin_block_db),
            "live_risk_windows": int(live_risk_windows),
            "live_overload_events": int(live_overload_events),
            "live_peak_density_max": float(live_peak_density_max),
            "live_current_risk": current_live_risk,
        }
        if (
            profile_locked
            and candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "channel_profile_locked_review"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: this channel profile is locked for engineer review."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_profile_unlock_or_engineer_review",
            })
        elif (
            profile_blocks_gain_up_by_history
            and candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "channel_profile_stability_wait"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: this channel's recent profile is not stable enough for input gain-up."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_channel_profile_stability",
            })

        if (
            vocal_gain_up_guard_state.get("active")
            and candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
        ):
            desired_delta = 0.0
            target_trim = current_trim
            guard_reason = str(vocal_gain_up_guard_state.get("reason") or "")
            if guard_reason == "recent_vocal_scream_or_shout":
                self.gain_limited_by = "vocal_scream_history_wait"
                self.operator_note = (
                    "Soundcheck boost held: recent vocal shout history needs another stable window."
                )
                stability_reason = "waiting_after_recent_vocal_scream_or_shout"
            else:
                self.gain_limited_by = "vocal_presence_stability_wait"
                self.operator_note = (
                    "Soundcheck boost held: vocal presence band is not stable enough for input gain-up."
                )
                stability_reason = "waiting_for_stable_vocal_presence_band"
            self.operator_action = "operator_review"
            stability_state.update({
                "active": True,
                "reason": stability_reason,
            })

        previous_trim_smoothing = (
            previous_character.get("trim_smoothing")
            if isinstance(previous_character, dict)
            else {}
        )
        if not isinstance(previous_trim_smoothing, dict):
            previous_trim_smoothing = {}
        previous_smoothed_delta = (
            _coerce_float(previous_trim_smoothing.get("smoothed_delta_db"), 0.0)
            if previous_trim_smoothing.get("smoothed_delta_db") is not None
            else previous_delta
        )
        previous_apply_status = ""
        previous_applied_value = None
        if isinstance(previous_recommendation, dict):
            previous_apply_status = str(
                previous_recommendation.get("status")
                or previous_recommendation.get("apply_status")
                or previous_recommendation.get("operator_status")
                or ""
            ).strip().lower()
            if previous_recommendation.get("applied") is not None:
                previous_applied_value = _coerce_bool(previous_recommendation.get("applied"), False)
        if previous_trim_smoothing.get("applied") is not None:
            previous_applied_value = _coerce_bool(previous_trim_smoothing.get("applied"), False)
        previous_explicitly_unapplied = bool(
            previous_applied_value is False
            or previous_apply_status in {
                "dismissed",
                "rejected",
                "ignored",
                "expired",
                "apply_blocked",
                "apply_error",
            }
        )
        smoothing_reference_delta = (
            0.0
            if previous_explicitly_unapplied
            else previous_smoothed_delta
        )
        convergence_smoothing_start = bool(
            soundcheck_convergence_state.get("active")
            and desired_delta > 0.0
            and smoothing_reference_delta is None
        )
        if convergence_smoothing_start:
            smoothing_reference_delta = 0.0
        effective_gain_up_step_db = (
            max(
                0.1,
                profile.soundcheck_ensemble_gain_up_step_db
                if soundcheck_ensemble_context_state.get("active")
                else profile.soundcheck_convergence_gain_up_step_db,
            )
            if soundcheck_convergence_state.get("active")
            else profile.trim_smoothing_gain_up_step_db
        )
        trim_smoothing_state = {
            "enabled": bool(profile.trim_smoothing_enabled),
            "candidate_delta_db": float(desired_delta),
            "previous_smoothed_delta_db": (
                float(previous_smoothed_delta)
                if previous_smoothed_delta is not None
                else None
            ),
            "previous_apply_status": previous_apply_status or None,
            "previous_applied": previous_applied_value,
            "previous_explicitly_unapplied": bool(previous_explicitly_unapplied),
            "reference_delta_db": (
                float(smoothing_reference_delta)
                if smoothing_reference_delta is not None
                else None
            ),
            "reference_source": (
                "current_trim_due_to_unapplied_previous"
                if previous_explicitly_unapplied
                else (
                    "soundcheck_convergence_start"
                    if convergence_smoothing_start
                    else "previous_smoothed_delta"
                )
            ),
            "gain_up_step_db": float(effective_gain_up_step_db),
            "base_gain_up_step_db": float(profile.trim_smoothing_gain_up_step_db),
            "soundcheck_convergence_gain_up_step_db": float(
                profile.soundcheck_convergence_gain_up_step_db
            ),
            "soundcheck_ensemble_gain_up_step_db": float(
                profile.soundcheck_ensemble_gain_up_step_db
            ),
            "soundcheck_convergence_active": bool(soundcheck_convergence_state.get("active")),
            "soundcheck_ensemble_context_active": bool(soundcheck_ensemble_context_state.get("active")),
            "gain_down_step_db": float(profile.trim_smoothing_gain_down_step_db),
            "urgent_headroom_bypass": bool(urgent_headroom_cut),
            "active": False,
            "reason": None,
            "smoothed_delta_db": float(desired_delta),
        }
        if (
            profile.trim_smoothing_enabled
            and smoothing_reference_delta is not None
            and desired_delta != 0.0
            and not urgent_headroom_cut
        ):
            smoothed_delta = desired_delta
            smoothing_reason = None
            previous_delta_for_smoothing = float(smoothing_reference_delta)
            if desired_delta > 0.0:
                max_gain_up_delta = max(0.0, previous_delta_for_smoothing) + effective_gain_up_step_db
                if desired_delta > max_gain_up_delta:
                    smoothed_delta = max_gain_up_delta
                    smoothing_reason = "gain_up_slew_limited"
            elif desired_delta < 0.0:
                max_gain_down_delta = min(0.0, previous_delta_for_smoothing) - profile.trim_smoothing_gain_down_step_db
                if desired_delta < max_gain_down_delta:
                    smoothed_delta = max_gain_down_delta
                    smoothing_reason = "gain_down_slew_limited"
            if smoothing_reason is not None and abs(smoothed_delta - desired_delta) > 1e-9:
                desired_delta = float(smoothed_delta)
                target_trim = current_trim + desired_delta
                self.gain_limited_by = f"trim_smoothing_{smoothing_reason}"
                self.operator_note = (
                    "Trim proposal slew-limited to avoid abrupt input gain changes."
                )
                trim_smoothing_state.update({
                    "active": True,
                    "reason": smoothing_reason,
                    "smoothed_delta_db": float(desired_delta),
                })
        else:
            trim_smoothing_state["smoothed_delta_db"] = float(desired_delta)
        soundcheck_convergence_state["post_smoothing_delta_db"] = float(desired_delta)

        confidence_candidate_delta = desired_delta
        confidence_candidate_direction = _gain_direction(confidence_candidate_delta)
        decision_confidence_factors = []
        decision_score = 0.82

        def _confidence_factor(name: str, delta: float, reason_text: str, active: bool = True) -> None:
            nonlocal decision_score
            if not active:
                return
            decision_score += float(delta)
            decision_confidence_factors.append({
                "name": name,
                "delta": float(delta),
                "reason": reason_text,
            })

        _confidence_factor(
            "channel_personality_bias",
            channel_personality.confidence_bias,
            f"{channel_personality.family} baseline",
            abs(channel_personality.confidence_bias) > 0.0,
        )
        _confidence_factor(
            "gain_up_caution",
            -0.08,
            "positive input-gain changes can alter preamp structure",
            confidence_candidate_direction == "up",
        )
        _confidence_factor(
            "large_soundcheck_gain_gap",
            -0.03,
            "large soundcheck gain gap needs engineer attention while converging",
            (
                confidence_candidate_direction == "up"
                and bool(soundcheck_convergence_state.get("active"))
                and not bool(soundcheck_ensemble_context_state.get("active"))
            ),
        )
        _confidence_factor(
            "ensemble_context_vocal_understaged",
            profile.soundcheck_ensemble_confidence_bonus,
            "other soundcheck sources are already loud enough and the vocal input is under-staged",
            (
                confidence_candidate_direction == "up"
                and bool(soundcheck_ensemble_context_state.get("active"))
            ),
        )
        _confidence_factor(
            "headroom_cut_preferred",
            0.05,
            "trim-down protects input headroom",
            confidence_candidate_direction == "down",
        )
        _confidence_factor(
            "operator_review_personality",
            -0.25,
            "source personality requires engineer review for gain-up",
            channel_personality.boost_policy == "operator_review",
        )
        _confidence_factor(
            "distortion_risk_medium",
            -0.10,
            "target trim approaches coloration range",
            distortion_risk == "medium",
        )
        _confidence_factor(
            "distortion_risk_high",
            -0.25,
            "target trim approaches saturation range",
            distortion_risk == "high",
        )
        _confidence_factor(
            "non_local_input",
            -0.05,
            "channel is outside the configured local-input preamp set",
            local_input is False and profile.enabled,
        )
        _confidence_factor(
            "unstable_source_behavior",
            -0.08,
            f"source behavior is {source_behavior}",
            source_behavior in risky_behaviors,
        )
        _confidence_factor(
            "behavior_flip",
            -0.18,
            "source behavior changed between windows",
            behavior_flip,
        )
        _confidence_factor(
            "source_identity_change",
            -0.28,
            "source signature changed between windows",
            source_identity_changed,
        )
        _confidence_factor(
            "profile_locked",
            -0.30,
            "channel profile is locked for engineer review",
            profile_locked,
        )
        _confidence_factor(
            "profile_history_risk",
            -0.18,
            "long-term channel profile still blocks gain-up",
            profile_blocks_gain_up_by_history,
        )
        _confidence_factor(
            "profile_recovery",
            0.05,
            "short-term recovery lane is stable",
            recovery_candidate,
        )
        _confidence_factor(
            "transient_hit_instability",
            -0.12,
            "percussion hit-to-hit body level or attack gap is unstable",
            bool(transient_window_profile.get("repeated_hit_unstable")),
        )
        _confidence_factor(
            "transient_body_band_instability",
            -0.10,
            "percussion low/low-mid body-band reference is unstable",
            bool(transient_window_profile.get("repeated_hit_body_band_unstable")),
        )
        _confidence_factor(
            "recovered_history_caution",
            -0.04,
            "long-term risk was overridden by repeated recovery evidence",
            profile_recovered_from_history,
        )
        _confidence_factor(
            "predictive_clip_cut",
            0.04,
            "preemptive trim-down is protecting near-clip headroom",
            predictive_clip_risk and confidence_candidate_direction == "down",
        )
        _confidence_factor(
            "clip_or_overload_risk",
            -0.35,
            "clipping or overload evidence makes gain-up unsafe",
            (
                digital_clip_detected
                or preamp_saturation_detected
                or vocal_scream_detected
                or predictive_clip_risk
            )
            and confidence_candidate_direction == "up",
        )
        _confidence_factor(
            "low_noise_margin",
            -0.20,
            "source is too close to the noise floor",
            (
                noise_floor_margin_db is not None
                and noise_floor_margin_db < boost_thresholds["min_noise_margin_db"]
            ),
        )
        _confidence_factor(
            "healthy_noise_margin",
            0.04,
            "noise floor margin is healthy",
            (
                noise_floor_margin_db is not None
                and noise_floor_margin_db >= boost_thresholds["min_noise_margin_db"] + 6.0
            ),
        )
        _confidence_factor(
            "stability_wait",
            -0.16,
            "candidate does not yet have stable repeated evidence",
            bool(stability_state.get("active")),
        )
        _confidence_factor(
            "stable_gain_up_evidence",
            0.04,
            "gain-up evidence repeated across stable windows",
            (
                confidence_candidate_direction == "up"
                and not stability_state.get("active")
                and evidence_windows >= profile.gain_up_min_stable_windows
            ),
        )
        _confidence_factor(
            "boost_guard_active",
            -0.20,
            "boost guard rejected the gain-up candidate",
            bool(boost_guard.get("active")),
        )
        _confidence_factor(
            "anti_oscillation_active",
            -0.12,
            "hysteresis is holding a small direction change",
            bool(hysteresis_state.get("active")),
        )
        decision_score = float(np.clip(decision_score, 0.0, 0.95))
        confidence_required_score = (
            float(profile.decision_confidence_gain_up_min_score)
            if confidence_candidate_direction == "up"
            else None
        )
        confidence_threshold_met = (
            True
            if confidence_required_score is None
            else decision_score >= confidence_required_score
        )
        previous_decision_confidence = (
            previous_character.get("decision_confidence")
            if isinstance(previous_character, dict)
            else {}
        )
        if not isinstance(previous_decision_confidence, dict):
            previous_decision_confidence = {}
        previous_confidence_score = (
            _coerce_float(previous_decision_confidence.get("score"), -1.0)
            if previous_decision_confidence.get("score") is not None
            else None
        )
        previous_confidence_windows = max(
            0,
            int(_coerce_float(previous_decision_confidence.get("confidence_windows"), 0.0)),
        )
        previous_confidence_timestamp_sec = (
            _coerce_float(previous_decision_confidence.get("analysis_timestamp_sec"), 0.0)
            if previous_decision_confidence.get("analysis_timestamp_sec") is not None
            else None
        )
        previous_confidence_window_sec = (
            max(
                profile.decision_confidence_min_window_sec,
                _coerce_float(previous_decision_confidence.get("analysis_window_sec"), 0.0),
            )
            if previous_decision_confidence.get("analysis_window_sec") is not None
            else None
        )
        confidence_elapsed_sec = (
            max(0.0, current_analysis_timestamp_sec - previous_confidence_timestamp_sec)
            if previous_confidence_timestamp_sec is not None
            else None
        )
        confidence_max_gap_sec = min(
            profile.decision_confidence_max_age_sec,
            profile.decision_confidence_max_gap_ratio
            * max(
                current_analysis_window_sec,
                previous_confidence_window_sec or profile.decision_confidence_min_window_sec,
            ),
        )
        previous_confidence_timing_known = bool(
            previous_confidence_timestamp_sec is not None
            and previous_confidence_window_sec is not None
        )
        previous_confidence_expired = bool(
            profile.decision_confidence_time_memory_enabled
            and previous_confidence_windows > 0
            and (
                not previous_confidence_timing_known
                or (
                    confidence_elapsed_sec is not None
                    and confidence_elapsed_sec > confidence_max_gap_sec
                )
            )
        )
        if channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}:
            confidence_decay_family = "vocal"
            confidence_decay_half_life_sec = profile.decision_confidence_vocal_decay_half_life_sec
        elif channel_personality.family == "playback":
            confidence_decay_family = "playback"
            confidence_decay_half_life_sec = profile.decision_confidence_playback_decay_half_life_sec
        else:
            confidence_decay_family = "general"
            confidence_decay_half_life_sec = profile.decision_confidence_decay_half_life_sec
        if (
            profile.decision_confidence_time_memory_enabled
            and previous_confidence_timing_known
            and confidence_elapsed_sec is not None
            and not previous_confidence_expired
        ):
            confidence_memory_strength = float(
                0.5 ** (confidence_elapsed_sec / max(0.1, confidence_decay_half_life_sec))
            )
        elif not profile.decision_confidence_time_memory_enabled and previous_confidence_windows > 0:
            confidence_memory_strength = 1.0
        else:
            confidence_memory_strength = 0.0
        confidence_time_memory_strength = confidence_memory_strength
        source_signature_drift_field_count = len(source_signature_changed_fields)
        source_signature_drift_weights = {field_name: 1.0 for field_name in source_signature_changed_fields}
        def _signature_drift_weight(bucket_field: str) -> float:
            if channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}:
                if bucket_field == "noise_margin_bucket_db":
                    return float(profile.source_signature_drift_vocal_noise_weight)
                if bucket_field == "crest_bucket_db":
                    return float(profile.source_signature_drift_vocal_crest_weight)
            elif channel_personality.family == "playback":
                if bucket_field == "integrated_lufs_bucket_db":
                    return float(profile.source_signature_drift_playback_lufs_weight)
                if bucket_field == "peak_lufs_gap_bucket_db":
                    return float(profile.source_signature_drift_playback_peak_gap_weight)
            return 1.0
        continuous_drift_specs = {
            "crest_db": ("crest_bucket_db", 3.0),
            "integrated_lufs_db": ("integrated_lufs_bucket_db", 6.0),
            "peak_lufs_gap_db": ("peak_lufs_gap_bucket_db", 3.0),
            "noise_margin_db": ("noise_margin_bucket_db", 6.0),
        }
        if channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}:
            if "noise_margin_bucket_db" in source_signature_drift_weights:
                source_signature_drift_weights["noise_margin_bucket_db"] = _signature_drift_weight(
                    "noise_margin_bucket_db"
                )
            if "crest_bucket_db" in source_signature_drift_weights:
                source_signature_drift_weights["crest_bucket_db"] = _signature_drift_weight("crest_bucket_db")
        elif channel_personality.family == "playback":
            if "integrated_lufs_bucket_db" in source_signature_drift_weights:
                source_signature_drift_weights["integrated_lufs_bucket_db"] = _signature_drift_weight(
                    "integrated_lufs_bucket_db"
                )
            if "peak_lufs_gap_bucket_db" in source_signature_drift_weights:
                source_signature_drift_weights["peak_lufs_gap_bucket_db"] = _signature_drift_weight(
                    "peak_lufs_gap_bucket_db"
                )
        source_signature_continuous_distances = {}
        source_signature_continuous_weighted_units = {}
        for raw_field, (bucket_field, normalizer_db) in continuous_drift_specs.items():
            previous_raw_value = previous_source_signature.get(raw_field)
            current_raw_value = source_signature.get(raw_field)
            if previous_raw_value is None or current_raw_value is None:
                continue
            raw_distance_db = abs(
                _coerce_float(current_raw_value, 0.0)
                - _coerce_float(previous_raw_value, 0.0)
            )
            normalized_distance = raw_distance_db / max(0.1, normalizer_db)
            field_weight = source_signature_drift_weights.get(bucket_field, _signature_drift_weight(bucket_field))
            source_signature_continuous_distances[raw_field] = {
                "previous_db": float(_coerce_float(previous_raw_value, 0.0)),
                "current_db": float(_coerce_float(current_raw_value, 0.0)),
                "distance_db": float(raw_distance_db),
                "normalizer_db": float(normalizer_db),
                "normalized": float(normalized_distance),
                "weight": float(field_weight),
            }
            source_signature_continuous_weighted_units[raw_field] = float(normalized_distance * field_weight)
        source_signature_drift_weight_sum = float(sum(source_signature_drift_weights.values()))
        source_signature_continuous_weight_sum = float(sum(source_signature_continuous_weighted_units.values()))
        def _signature_relative_spectral_bands(signature: Dict[str, Any]) -> Dict[str, float]:
            fingerprint = signature.get("spectral_fingerprint")
            if not isinstance(fingerprint, dict):
                return {}
            relative_bands = fingerprint.get("relative_bands_db")
            if not isinstance(relative_bands, dict):
                return {}
            result = {}
            for band_name, band_db in relative_bands.items():
                try:
                    value = float(band_db)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    result[str(band_name)] = value
            return result

        def _signature_spectral_band_quality(signature: Dict[str, Any], band_name: str) -> Dict[str, Any]:
            if not profile.source_signature_spectral_quality_gate_enabled:
                return {"usable": True, "reason": "quality_gate_disabled"}
            fingerprint = signature.get("spectral_fingerprint")
            if not isinstance(fingerprint, dict):
                return {"usable": False, "reason": "missing_fingerprint"}
            band_counts = fingerprint.get("band_sample_counts")
            band_coverages = fingerprint.get("band_coverage_ratios")
            band_std_values = fingerprint.get("band_std_db")
            band_span_values = fingerprint.get("band_span_db")
            if not isinstance(band_counts, dict) or not isinstance(band_coverages, dict):
                return {"usable": True, "reason": "legacy_fingerprint_without_quality"}
            band_count = int(_coerce_float(band_counts.get(band_name), 0.0))
            band_coverage = _coerce_float(band_coverages.get(band_name), 0.0)
            band_std_db = (
                _coerce_float(band_std_values.get(band_name), 0.0)
                if isinstance(band_std_values, dict)
                else 0.0
            )
            band_span_db = (
                _coerce_float(band_span_values.get(band_name), 0.0)
                if isinstance(band_span_values, dict)
                else 0.0
            )
            signal_variability_db = _coerce_float(fingerprint.get("signal_variability_db"), 0.0)
            normalized_name = str(band_name).lower()
            is_high_or_presence_band = any(
                token in normalized_name
                for token in ("presence", "air", "high", "upper", "sibilance", "cymbal")
            )
            quality_profile = "general"
            max_band_std_db = float(profile.source_signature_spectral_max_band_std_db)
            max_band_span_db = float(profile.source_signature_spectral_max_band_span_db)
            if channel_personality.family == "playback":
                quality_profile = "playback"
                max_band_std_db = float(profile.source_signature_spectral_playback_max_band_std_db)
                max_band_span_db = float(profile.source_signature_spectral_playback_max_band_span_db)
            elif (
                channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}
                and is_high_or_presence_band
            ):
                quality_profile = "vocal_presence"
                max_band_std_db = float(profile.source_signature_spectral_vocal_presence_max_std_db)
                max_band_span_db = float(profile.source_signature_spectral_vocal_presence_max_span_db)
            elif (
                channel_personality.family
                in {"transient_percussion", "cymbal", "drum_image", "drum_room"}
                and is_high_or_presence_band
            ):
                quality_profile = "transient_high_band"
                max_band_std_db = float(profile.source_signature_spectral_transient_high_band_max_std_db)
                max_band_span_db = float(profile.source_signature_spectral_transient_high_band_max_span_db)
            if band_count < profile.source_signature_spectral_min_band_samples:
                return {
                    "usable": False,
                    "reason": "too_few_band_samples",
                    "quality_profile": quality_profile,
                    "samples": int(band_count),
                    "min_samples": int(profile.source_signature_spectral_min_band_samples),
                    "coverage": float(band_coverage),
                }
            if band_coverage < profile.source_signature_spectral_min_band_coverage:
                return {
                    "usable": False,
                    "reason": "low_band_coverage",
                    "quality_profile": quality_profile,
                    "samples": int(band_count),
                    "coverage": float(band_coverage),
                    "min_coverage": float(profile.source_signature_spectral_min_band_coverage),
                }
            if band_std_db > max_band_std_db:
                return {
                    "usable": False,
                    "reason": "high_band_variance",
                    "quality_profile": quality_profile,
                    "samples": int(band_count),
                    "coverage": float(band_coverage),
                    "std_db": float(band_std_db),
                    "max_std_db": float(max_band_std_db),
                    "span_db": float(band_span_db),
                }
            if band_span_db > max_band_span_db:
                return {
                    "usable": False,
                    "reason": "high_band_span",
                    "quality_profile": quality_profile,
                    "samples": int(band_count),
                    "coverage": float(band_coverage),
                    "std_db": float(band_std_db),
                    "span_db": float(band_span_db),
                    "max_span_db": float(max_band_span_db),
                }
            if (
                signal_variability_db >= profile.source_signature_spectral_stuck_signal_std_db
                and band_std_db <= profile.source_signature_spectral_stuck_band_std_db
            ):
                return {
                    "usable": False,
                    "reason": "stuck_band_output",
                    "quality_profile": quality_profile,
                    "samples": int(band_count),
                    "coverage": float(band_coverage),
                    "std_db": float(band_std_db),
                    "max_stuck_std_db": float(profile.source_signature_spectral_stuck_band_std_db),
                    "signal_variability_db": float(signal_variability_db),
                    "min_signal_variability_db": float(
                        profile.source_signature_spectral_stuck_signal_std_db
                    ),
                }
            return {
                "usable": True,
                "reason": "usable",
                "quality_profile": quality_profile,
                "samples": int(band_count),
                "coverage": float(band_coverage),
                "std_db": float(band_std_db),
                "span_db": float(band_span_db),
                "max_std_db": float(max_band_std_db),
                "max_span_db": float(max_band_span_db),
            }

        previous_spectral_bands = _signature_relative_spectral_bands(previous_source_signature)
        current_spectral_bands = _signature_relative_spectral_bands(source_signature)
        source_signature_spectral_all_common_bands = sorted(
            set(previous_spectral_bands).intersection(current_spectral_bands)
        )
        source_signature_spectral_common_bands = []
        source_signature_spectral_rejected_bands = {}
        for band_name in source_signature_spectral_all_common_bands:
            previous_quality = _signature_spectral_band_quality(previous_source_signature, band_name)
            current_quality = _signature_spectral_band_quality(source_signature, band_name)
            if previous_quality.get("usable") and current_quality.get("usable"):
                source_signature_spectral_common_bands.append(band_name)
            else:
                source_signature_spectral_rejected_bands[band_name] = {
                    "previous": previous_quality,
                    "current": current_quality,
                }
        source_signature_spectral_distances = {}
        source_signature_spectral_band_weights = {}
        source_signature_spectral_distance_db = 0.0
        source_signature_spectral_unweighted_distance_db = 0.0
        source_signature_spectral_quality_rejected_count = int(len(source_signature_spectral_rejected_bands))
        source_signature_spectral_quality_penalty = 0.0
        if (
            profile.source_signature_spectral_quality_gate_enabled
            and source_signature_spectral_quality_rejected_count > 0
            and not source_identity_changed
        ):
            source_signature_spectral_quality_penalty = min(
                profile.source_signature_spectral_quality_max_penalty,
                source_signature_spectral_quality_rejected_count
                * profile.source_signature_spectral_quality_penalty_per_rejected_band,
            )
            confidence_memory_strength *= max(0.0, 1.0 - source_signature_spectral_quality_penalty)
        def _spectral_drift_weight(band_name: str) -> float:
            normalized_name = str(band_name).lower()
            is_high_or_presence_band = any(
                token in normalized_name
                for token in ("presence", "air", "high", "upper", "sibilance", "cymbal")
            )
            if channel_personality.family == "playback":
                return float(profile.source_signature_spectral_playback_weight)
            if (
                channel_personality.family in {"lead_vocal", "backing_vocal", "speech"}
                and is_high_or_presence_band
            ):
                return float(profile.source_signature_spectral_vocal_presence_weight)
            if (
                channel_personality.family
                in {"transient_percussion", "cymbal", "drum_image", "drum_room"}
                and is_high_or_presence_band
            ):
                if transient_window_profile.get("high_band_transient_dominant"):
                    return float(
                        profile.source_signature_spectral_transient_attack_high_band_identity_weight
                    )
                return float(profile.source_signature_spectral_transient_high_band_identity_weight)
            return 1.0

        if (
            len(source_signature_spectral_common_bands)
            >= profile.source_signature_spectral_drift_min_common_bands
        ):
            weighted_distance_sum = 0.0
            for band_name in source_signature_spectral_common_bands:
                previous_band_db = previous_spectral_bands[band_name]
                current_band_db = current_spectral_bands[band_name]
                band_distance_db = abs(current_band_db - previous_band_db)
                band_weight = _spectral_drift_weight(band_name)
                source_signature_spectral_distances[band_name] = {
                    "previous_relative_db": float(previous_band_db),
                    "current_relative_db": float(current_band_db),
                    "distance_db": float(band_distance_db),
                    "weight": float(band_weight),
                }
                source_signature_spectral_band_weights[band_name] = float(band_weight)
                weighted_distance_sum += float(band_distance_db * band_weight)
            source_signature_spectral_unweighted_distance_db = float(
                np.mean([
                    band_state["distance_db"]
                    for band_state in source_signature_spectral_distances.values()
                ])
            )
            source_signature_spectral_distance_db = float(
                weighted_distance_sum / max(len(source_signature_spectral_common_bands), 1)
            )
        source_signature_drift_penalty = 0.0
        if (
            profile.source_signature_drift_memory_penalty_enabled
            and source_signature_drift_field_count > 0
            and not source_identity_changed
        ):
            source_signature_drift_penalty = min(
                profile.source_signature_drift_max_penalty,
                source_signature_drift_weight_sum * profile.source_signature_drift_penalty_per_field,
            )
            confidence_memory_strength *= max(0.0, 1.0 - source_signature_drift_penalty)
        source_signature_continuous_drift_penalty = 0.0
        if (
            profile.source_signature_continuous_drift_enabled
            and source_signature_continuous_weight_sum > 0.0
            and not source_identity_changed
        ):
            source_signature_continuous_drift_penalty = min(
                profile.source_signature_continuous_drift_max_penalty,
                source_signature_continuous_weight_sum
                * profile.source_signature_continuous_drift_penalty_per_unit,
            )
            confidence_memory_strength *= max(0.0, 1.0 - source_signature_continuous_drift_penalty)
        source_signature_spectral_drift_penalty = 0.0
        if (
            profile.source_signature_spectral_drift_enabled
            and source_signature_spectral_distance_db > 0.0
            and not source_identity_changed
        ):
            source_signature_spectral_drift_penalty = min(
                profile.source_signature_spectral_drift_max_penalty,
                source_signature_spectral_distance_db
                * profile.source_signature_spectral_drift_penalty_per_db,
            )
            confidence_memory_strength *= max(0.0, 1.0 - source_signature_spectral_drift_penalty)
        confidence_borderline = bool(
            confidence_candidate_direction == "up"
            and confidence_required_score is not None
            and confidence_threshold_met
            and decision_score <= confidence_required_score + profile.decision_confidence_borderline_margin
        )
        same_confidence_candidate = bool(
            confidence_candidate_direction == "up"
            and previous_decision_confidence.get("candidate_direction") == "up"
            and previous_confidence_score is not None
            and previous_confidence_score >= float(confidence_required_score or 0.0)
            and not previous_confidence_expired
            and confidence_memory_strength >= profile.decision_confidence_min_memory_strength
            and abs(decision_score - previous_confidence_score) <= profile.decision_confidence_score_tolerance
        )
        confidence_windows = (
            previous_confidence_windows + 1
            if confidence_threshold_met and same_confidence_candidate
            else (1 if confidence_threshold_met and confidence_candidate_direction == "up" else 0)
        )
        confidence_hysteresis_stable = bool(
            not confidence_borderline
            or confidence_windows >= profile.decision_confidence_stable_windows
        )
        decision_confidence_state = {
            "enabled": bool(profile.decision_confidence_enabled),
            "candidate_direction": confidence_candidate_direction,
            "candidate_delta_db": float(confidence_candidate_delta),
            "score": float(decision_score),
            "gain_up_min_score": float(profile.decision_confidence_gain_up_min_score),
            "required_score": confidence_required_score,
            "threshold_met": bool(confidence_threshold_met),
            "hysteresis_enabled": bool(profile.decision_confidence_hysteresis_enabled),
            "borderline": bool(confidence_borderline),
            "borderline_margin": float(profile.decision_confidence_borderline_margin),
            "score_tolerance": float(profile.decision_confidence_score_tolerance),
            "time_memory_enabled": bool(profile.decision_confidence_time_memory_enabled),
            "analysis_timestamp_sec": float(current_analysis_timestamp_sec),
            "analysis_window_sec": float(current_analysis_window_sec),
            "previous_timestamp_sec": previous_confidence_timestamp_sec,
            "previous_window_sec": previous_confidence_window_sec,
            "elapsed_since_previous_sec": confidence_elapsed_sec,
            "max_gap_sec": float(confidence_max_gap_sec),
            "previous_confidence_expired": bool(previous_confidence_expired),
            "memory_decay_family": confidence_decay_family,
            "memory_decay_half_life_sec": float(confidence_decay_half_life_sec),
            "time_memory_strength": float(confidence_time_memory_strength),
            "source_signature_drift_penalty_enabled": bool(
                profile.source_signature_drift_memory_penalty_enabled
            ),
            "source_signature_drift_fields": source_signature_changed_fields,
            "source_signature_drift_field_weights": source_signature_drift_weights,
            "source_signature_drift_field_count": int(source_signature_drift_field_count),
            "source_signature_drift_weight_sum": float(source_signature_drift_weight_sum),
            "source_signature_drift_penalty": float(source_signature_drift_penalty),
            "source_signature_continuous_drift_enabled": bool(
                profile.source_signature_continuous_drift_enabled
            ),
            "source_signature_continuous_distances": source_signature_continuous_distances,
            "source_signature_continuous_weighted_units": source_signature_continuous_weighted_units,
            "source_signature_continuous_weight_sum": float(source_signature_continuous_weight_sum),
            "source_signature_continuous_drift_penalty": float(source_signature_continuous_drift_penalty),
            "source_signature_spectral_drift_enabled": bool(
                profile.source_signature_spectral_drift_enabled
            ),
            "source_signature_spectral_min_common_bands": int(
                profile.source_signature_spectral_drift_min_common_bands
            ),
            "source_signature_spectral_quality_gate_enabled": bool(
                profile.source_signature_spectral_quality_gate_enabled
            ),
            "source_signature_spectral_min_band_samples": int(
                profile.source_signature_spectral_min_band_samples
            ),
            "source_signature_spectral_min_band_coverage": float(
                profile.source_signature_spectral_min_band_coverage
            ),
            "source_signature_spectral_all_common_bands": source_signature_spectral_all_common_bands,
            "source_signature_spectral_rejected_bands": source_signature_spectral_rejected_bands,
            "source_signature_spectral_quality_rejected_count": int(
                source_signature_spectral_quality_rejected_count
            ),
            "source_signature_spectral_quality_penalty_per_rejected_band": float(
                profile.source_signature_spectral_quality_penalty_per_rejected_band
            ),
            "source_signature_spectral_quality_max_penalty": float(
                profile.source_signature_spectral_quality_max_penalty
            ),
            "source_signature_spectral_quality_penalty": float(
                source_signature_spectral_quality_penalty
            ),
            "source_signature_spectral_common_bands": source_signature_spectral_common_bands,
            "source_signature_spectral_common_band_count": int(len(source_signature_spectral_common_bands)),
            "source_signature_spectral_distances": source_signature_spectral_distances,
            "source_signature_spectral_band_weights": source_signature_spectral_band_weights,
            "source_signature_spectral_unweighted_distance_db": float(
                source_signature_spectral_unweighted_distance_db
            ),
            "source_signature_spectral_distance_db": float(source_signature_spectral_distance_db),
            "source_signature_spectral_drift_penalty": float(source_signature_spectral_drift_penalty),
            "memory_strength": float(confidence_memory_strength),
            "min_memory_strength": float(profile.decision_confidence_min_memory_strength),
            "confidence_windows": int(confidence_windows),
            "required_confidence_windows": int(profile.decision_confidence_stable_windows),
            "hysteresis_stable": bool(confidence_hysteresis_stable),
            "previous_score": previous_confidence_score,
            "gated": False,
            "gate_reason": None,
            "factors": decision_confidence_factors,
        }
        if (
            profile.decision_confidence_enabled
            and confidence_candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
            and not confidence_threshold_met
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "decision_confidence_gain_up_hold"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: combined confidence is too low for automatic input gain-up."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_higher_gain_up_confidence",
            })
            decision_confidence_state.update({
                "gated": True,
                "gate_reason": "gain_up_confidence_below_threshold",
            })
        elif (
            profile.decision_confidence_enabled
            and profile.decision_confidence_hysteresis_enabled
            and confidence_candidate_direction == "up"
            and desired_delta > 0.0
            and mode == "soundcheck"
            and confidence_borderline
            and not confidence_hysteresis_stable
        ):
            desired_delta = 0.0
            target_trim = current_trim
            self.gain_limited_by = "decision_confidence_stability_wait"
            self.operator_action = "operator_review"
            self.operator_note = (
                "Soundcheck boost held: confidence is borderline and needs a repeated stable window."
            )
            stability_state.update({
                "active": True,
                "reason": "waiting_for_stable_gain_up_confidence",
            })
            decision_confidence_state.update({
                "gated": True,
                "gate_reason": "gain_up_confidence_borderline_wait",
            })

        if abs(desired_delta) < 0.25:
            desired_delta = 0.0
            target_trim = current_trim
            if self.gain_limited_by == "none":
                self.gain_limited_by = reason
            if self.operator_action == "none":
                self.operator_action = "hold"
            self.operator_note = self.operator_note or "Input trim is already inside the safe headroom window."
        else:
            self.gain_limited_by = self.gain_limited_by if self.gain_limited_by != "none" else reason
            self.operator_action = "apply_trim_proposal"
            self.operator_note = self.operator_note or (
                "Trim down protects input headroom."
                if desired_delta < 0.0
                else "Soundcheck-only clean-range boost; do not use as mix balance."
            )

        target_zone = profile.zone_for_gain(target_trim) if profile.enabled else "disabled"
        distortion_risk = profile.distortion_risk(target_trim, local_input=local_input)
        direction = "down" if desired_delta < 0.0 else "up" if desired_delta > 0.0 else "hold"
        confidence = decision_score
        if desired_delta == 0.0:
            confidence = min(confidence, 0.55)

        self.suggested_gain_db = float(np.clip(desired_delta, -24.0, 24.0))
        self.target_trim_db = float(target_trim)
        self.suggestion_confidence = max(0.0, min(0.95, confidence))
        decision_confidence_state["final_direction"] = direction
        decision_confidence_state["final_score"] = float(self.suggestion_confidence)
        if direction == "down":
            foh_headline = f"Trim down {abs(self.suggested_gain_db):.1f} dB: {self.gain_limited_by}"
        elif direction == "up":
            foh_headline = f"Trim up {self.suggested_gain_db:.1f} dB: {self.gain_limited_by}"
        elif self.operator_action == "operator_review":
            foh_headline = f"Hold for operator review: {self.gain_limited_by}"
        else:
            foh_headline = f"Hold: {self.gain_limited_by}"
        body_reference = transient_window_profile.get("body_reference")
        body_band_reference = None
        if isinstance(body_reference, dict):
            body_band_reference = body_reference.get("body_band_reference")
        risk_flags = []
        if mode == "live" and confidence_candidate_direction == "up":
            risk_flags.append("live_gain_up_blocked")
        if self.max_true_peak_db > max_peak_limit:
            risk_flags.append("peak_over_limit")
        if overload_classification != "normal":
            risk_flags.append(overload_classification)
        if bool(boost_guard.get("active")):
            risk_flags.append(str(boost_guard.get("reason") or "boost_guard_active"))
        if bool(vocal_gain_up_guard_state.get("active")):
            risk_flags.append(str(vocal_gain_up_guard_state.get("reason") or "vocal_gain_up_guard"))
        if bool(trim_smoothing_state.get("active")):
            risk_flags.append(str(trim_smoothing_state.get("reason") or "trim_smoothing_active"))
        if bool(soundcheck_convergence_state.get("active")):
            risk_flags.append(str(soundcheck_convergence_state.get("reason") or "soundcheck_convergence_active"))
        ensemble_context_reason = str(soundcheck_ensemble_context_state.get("reason") or "")
        if bool(soundcheck_ensemble_context_state.get("active")):
            risk_flags.append("ensemble_context_vocal_understaged")
        elif (
            confidence_candidate_direction == "up"
            and bool(soundcheck_ensemble_context_state.get("provided"))
            and ensemble_context_reason in {
                "ensemble_time_context_missing",
                "ensemble_time_context_stale",
            }
        ):
            risk_flags.append(ensemble_context_reason)
        if bool(soundcheck_convergence_state.get("setup_check_required")):
            risk_flags.append("source_setup_check_required")
        if bool(transient_window_profile.get("repeated_hit_unstable")):
            risk_flags.append("transient_hit_unstable")
        if channel_personality.boost_policy == "operator_review" and confidence_candidate_direction == "up":
            risk_flags.append("operator_review_source")

        safe_to_auto_apply = bool(self.operator_action == "apply_trim_proposal" and direction == "down")
        if safe_to_auto_apply:
            foh_next_action = "apply_trim_down"
            foh_readiness = "auto_apply_trim_down"
        elif (
            bool(soundcheck_convergence_state.get("setup_check_required"))
            and (
                direction == "up"
                or confidence_candidate_direction == "up"
                or soundcheck_convergence_state.get("requested_delta_db", 0.0) > 0.0
            )
        ):
            foh_next_action = "check_source_setup_and_review_gain_up"
            foh_readiness = "operator_review_gain_up"
        elif direction == "up":
            foh_next_action = "review_soundcheck_gain_up"
            foh_readiness = "operator_review_gain_up"
        elif self.operator_action == "operator_review":
            foh_next_action = "operator_review"
            foh_readiness = "hold_for_review"
        elif self.operator_action == "check_source":
            foh_next_action = "check_source"
            foh_readiness = "insufficient_signal"
        else:
            foh_next_action = "hold"
            foh_readiness = "hold_safe"

        spectral_available = bool(spectral_fingerprint.get("available"))
        spectral_band_count = int(_coerce_float(spectral_fingerprint.get("band_count"), 0.0))
        measurement_warnings = []
        if self.total_samples < profile.transient_window_min_samples:
            measurement_warnings.append("short_analysis_window")
        if self.signal_presence_ratio < min_signal_presence:
            measurement_warnings.append("insufficient_signal_presence")
        if not spectral_available:
            measurement_warnings.append("missing_spectral_fingerprint")
        elif spectral_band_count < profile.source_signature_spectral_drift_min_common_bands:
            measurement_warnings.append("limited_spectral_band_count")
        if noise_floor_db is None:
            measurement_warnings.append("missing_noise_floor_reference")
        if self.active_samples < profile.source_signature_spectral_min_band_samples:
            measurement_warnings.append("few_active_samples")
        if "insufficient_signal_presence" in measurement_warnings:
            measurement_readiness = "insufficient_signal"
        elif "short_analysis_window" in measurement_warnings:
            measurement_readiness = "short_window"
        elif "missing_spectral_fingerprint" in measurement_warnings:
            measurement_readiness = "level_only"
        elif "missing_noise_floor_reference" in measurement_warnings:
            measurement_readiness = "usable_without_noise_floor"
        elif "limited_spectral_band_count" in measurement_warnings:
            measurement_readiness = "limited_spectral_context"
        else:
            measurement_readiness = "full_context"

        foh_summary = {
            "policy": "display_only_no_apply_authority",
            "headline": foh_headline,
            "readiness": foh_readiness,
            "next_action": foh_next_action,
            "safe_to_auto_apply": safe_to_auto_apply,
            "candidate_direction": confidence_candidate_direction,
            "candidate_delta_db": float(confidence_candidate_delta),
            "final_direction": direction,
            "suggested_delta_db": float(self.suggested_gain_db),
            "operator_action": self.operator_action,
            "limited_by": self.gain_limited_by,
            "gate_reason": decision_confidence_state.get("gate_reason"),
            "risk_flags": risk_flags,
            "score": float(decision_score),
            "final_score": float(self.suggestion_confidence),
            "required_score": confidence_required_score,
            "dominant_blockers": _dominant_confidence_factors(
                decision_confidence_factors,
                positive=False,
            ),
            "supporting_evidence": _dominant_confidence_factors(
                decision_confidence_factors,
                positive=True,
            ),
            "evidence_window": {
                "analysis_timestamp_sec": float(current_analysis_timestamp_sec),
                "analysis_window_sec": float(current_analysis_window_sec),
                "samples_analyzed": int(self.total_samples),
                "active_samples": int(self.active_samples),
                "source_behavior": source_behavior,
                "stability_active": bool(stability_state.get("active")),
                "stability_reason": stability_state.get("reason"),
                "stability_evidence_windows": int(stability_state.get("evidence_windows") or 0),
                "required_gain_up_windows": int(stability_state.get("required_gain_up_windows") or 0),
                "confidence_windows": int(confidence_windows),
                "required_confidence_windows": int(profile.decision_confidence_stable_windows),
                "threshold_met": bool(confidence_threshold_met),
                "borderline": bool(confidence_borderline),
                "hysteresis_stable": bool(confidence_hysteresis_stable),
                "memory_strength": float(confidence_memory_strength),
                "min_memory_strength": float(profile.decision_confidence_min_memory_strength),
                "trim_smoothing_active": bool(trim_smoothing_state.get("active")),
                "trim_smoothing_reason": trim_smoothing_state.get("reason"),
                "transient_repeated_hit_evaluable": bool(
                    transient_window_profile.get("repeated_hit_evaluable")
                ),
                "transient_repeated_hit_unstable": bool(
                    transient_window_profile.get("repeated_hit_unstable")
                ),
                "transient_body_band_confident": (
                    bool(body_band_reference.get("confident"))
                    if isinstance(body_band_reference, dict)
                    else None
                ),
            },
            "gain_structure": {
                "current_trim_db": float(current_trim),
                "target_trim_db": float(target_trim),
                "current_zone": current_zone,
                "target_zone": target_zone,
                "distortion_risk": distortion_risk,
                "gain_changes_tone": bool(profile.enabled and local_input),
            },
            "headroom": {
                "max_true_peak_db": float(self.max_true_peak_db),
                "peak_p95_db": float(peak_p95_db),
                "target_peak_db": float(self.target_peak_db),
                "max_peak_limit_db": float(max_peak_limit),
                "predicted_peak_db": float(predicted_peak_db),
                "near_clip_peak_density": float(near_clip_peak_density),
                "overload_classification": overload_classification,
            },
            "soundcheck_ensemble_context": soundcheck_ensemble_context_state,
            "soundcheck_convergence": soundcheck_convergence_state,
            "measurement_quality": {
                "readiness": measurement_readiness,
                "warnings": measurement_warnings,
                "samples_analyzed": int(self.total_samples),
                "active_samples": int(self.active_samples),
                "signal_presence_ratio": float(self.signal_presence_ratio),
                "spectral_available": spectral_available,
                "spectral_band_count": int(spectral_band_count),
                "noise_floor_available": noise_floor_db is not None,
                "noise_floor_db": (
                    float(noise_floor_db)
                    if noise_floor_db is not None
                    else None
                ),
            },
        }
        decision_confidence_state["foh_summary"] = foh_summary
        self.input_gain_character = {
            "source": profile.source,
            "character": profile.character,
            "local_input": bool(local_input),
            "current_zone": current_zone,
            "target_zone": target_zone,
            "distortion_risk": distortion_risk,
            "clean_gain_max_db": float(profile.clean_gain_max_db),
            "coloration_start_db": float(profile.coloration_start_db),
            "saturation_start_db": float(profile.saturation_start_db),
            "direction": direction,
            "gain_changes_tone": bool(profile.enabled and local_input),
            "source_behavior": source_behavior,
            "boost_guard": boost_guard,
            "hysteresis": hysteresis_state,
            "stability": stability_state,
            "trim_smoothing": trim_smoothing_state,
            "soundcheck_ensemble_context": soundcheck_ensemble_context_state,
            "soundcheck_convergence": soundcheck_convergence_state,
            "overload": overload_state,
            "predictive_clip": predictive_clip_state,
            "vocal_gain_up_guard": vocal_gain_up_guard_state,
            "transient_window_profile": transient_window_profile,
            "trend_memory": trend_memory_state,
            "channel_profile": channel_profile_state,
            "profile_lifecycle": profile_lifecycle_state,
            "decision_confidence": decision_confidence_state,
            "foh_summary": foh_summary,
            "channel_personality": channel_personality.as_report(),
            "integrated_lufs": float(self.integrated_lufs),
            "crest_factor_db": float(self.crest_factor_db),
            "signal_presence_ratio": float(self.signal_presence_ratio),
            "peak_p95_db": float(peak_p95_db),
            "isolated_peak_spread_db": float(isolated_peak_spread_db),
            "isolated_peak_min_spread_db": float(boost_thresholds["isolated_peak_min_spread_db"]),
            "lufs_std_db": float(lufs_std_db),
            "peak_std_db": float(peak_std_db),
            "noise_floor_db": float(noise_floor_db) if noise_floor_db is not None else None,
            "noise_floor_margin_db": (
                float(noise_floor_margin_db)
                if noise_floor_margin_db is not None
                else None
            ),
        }

        logger.info(
            f"Ch{self.channel_id}: input trim {current_trim:+.1f}->{target_trim:+.1f} dB "
            f"(delta={self.suggested_gain_db:+.1f}, peak={self.max_true_peak_db:.1f}, "
            f"target_peak={self.target_peak_db:.1f}, mode={mode}, zone={current_zone}->{target_zone}, "
            f"limited_by={self.gain_limited_by})"
        )

    def get_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        return {
            'channel': self.channel_id,
            'peak_db': round(self.max_true_peak_db, 1),
            'lufs': round(self.integrated_lufs, 1),
            'crest_factor_db': round(self.crest_factor_db, 1),
            'signal_presence': round(self.signal_presence_ratio * 100, 1),
            'suggested_gain_db': round(self.suggested_gain_db, 1),
            'limited_by': self.gain_limited_by,
            'method_id': INPUT_TRIM_AUTOGAIN_METHOD_ID,
            'input_trim_role': INPUT_TRIM_ROLE,
            'target_peak_db': round(float(self.target_peak_db), 1),
            'current_trim_db': round(float(self.current_trim_db), 1) if self.current_trim_db is not None else None,
            'target_trim_db': round(float(self.target_trim_db), 1) if self.target_trim_db is not None else None,
            'confidence': round(float(self.suggestion_confidence), 2),
            'operator_action': self.operator_action,
            'operator_note': self.operator_note,
            'auto_apply_allowed': self.operator_action == "apply_trim_proposal" and self.suggested_gain_db < 0.0,
            'input_gain_character': dict(self.input_gain_character),
            'safety_policy': dict(self.safety_policy),
            'samples_analyzed': self.total_samples,
            'active_samples': self.active_samples
        }


class KWeightingFilter:
    """
    K-Weighting фильтр по стандарту ITU-R BS.1770.
    
    Состоит из двух фильтров:
    1. High-shelf filter (подъем ВЧ для компенсации головы)
    2. High-pass filter (срез НЧ ниже ~60Hz)
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._design_filters()
        self._reset_state()
    
    def _design_filters(self):
        """Проектирование K-weighting фильтров."""
        fs = self.sample_rate
        
        # Stage 1: High-shelf filter (+4dB выше 1500Hz)
        # Коэффициенты для 48kHz по стандарту ITU-R BS.1770-4
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196
        
        K = np.tan(np.pi * f0 / fs)
        Vh = 10 ** (G / 20)
        Vb = Vh ** 0.4996667741545416
        
        a0 = 1 + K / Q + K * K
        self.shelf_b = np.array([
            (Vh + Vb * K / Q + K * K) / a0,
            2 * (K * K - Vh) / a0,
            (Vh - Vb * K / Q + K * K) / a0
        ])
        self.shelf_a = np.array([
            1,
            2 * (K * K - 1) / a0,
            (1 - K / Q + K * K) / a0
        ])
        
        # Stage 2: High-pass filter (срез ниже ~60Hz)
        f0 = 38.13547087602444
        Q = 0.5003270373238773
        
        K = np.tan(np.pi * f0 / fs)
        a0 = 1 + K / Q + K * K
        self.hp_b = np.array([1 / a0, -2 / a0, 1 / a0])
        self.hp_a = np.array([
            1,
            2 * (K * K - 1) / a0,
            (1 - K / Q + K * K) / a0
        ])
    
    def _reset_state(self):
        """Сброс состояния фильтров."""
        self.shelf_zi = np.zeros(2)
        self.hp_zi = np.zeros(2)
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Применение K-weighting фильтра к сэмплам.
        
        Args:
            samples: Входные аудио сэмплы (float32, -1 to 1)
            
        Returns:
            Отфильтрованные сэмплы
        """
        # Stage 1: High-shelf
        filtered, self.shelf_zi = signal.lfilter(
            self.shelf_b, self.shelf_a, samples, zi=self.shelf_zi
        )
        
        # Stage 2: High-pass
        filtered, self.hp_zi = signal.lfilter(
            self.hp_b, self.hp_a, filtered, zi=self.hp_zi
        )
        
        return filtered.astype(np.float32)
    
    def reset(self):
        """Сброс состояния фильтра."""
        self._reset_state()


class LUFSMeter:
    """
    Измеритель Short-term LUFS по стандарту ITU-R BS.1770.
    
    Использует скользящее окно 400ms для вычисления текущего уровня громкости.
    """
    
    def __init__(self, sample_rate: int = 48000, window_ms: float = 400.0):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_ms / 1000)
        
        # K-weighting фильтр
        self.k_filter = KWeightingFilter(sample_rate)
        
        # Буфер для скользящего окна
        self.buffer = deque(maxlen=self.window_samples)
        
        # Заполняем буфер тишиной
        for _ in range(self.window_samples):
            self.buffer.append(0.0)
        
        # Кэш для быстрого расчета
        self._sum_squares = 0.0
    
    def process(self, samples: np.ndarray) -> float:
        """
        Обработка новых сэмплов и возврат текущего LUFS.
        
        Args:
            samples: Входные аудио сэмплы
            
        Returns:
            Текущий Short-term LUFS (dB)
        """
        # Применяем K-weighting
        filtered = self.k_filter.process(samples)
        
        # Обновляем буфер и сумму квадратов
        for sample in filtered:
            # Удаляем старый сэмпл из суммы
            old_sample = self.buffer[0]
            self._sum_squares -= old_sample * old_sample

            # Добавляем новый сэмпл
            self.buffer.append(sample)
            self._sum_squares += sample * sample

        # C-12 FIX: Floating-point accumulation can make _sum_squares drift
        # slightly negative after long runs (catastrophic cancellation).
        # Clamp to zero to prevent NaN / negative-log from max() bypass.
        if self._sum_squares < 0.0:
            self._sum_squares = 0.0

        # Вычисляем LUFS
        mean_square = max(self._sum_squares / self.window_samples, 1e-10)
        lufs = -0.691 + 10 * np.log10(mean_square)
        
        return float(lufs)
    
    def get_current_lufs(self) -> float:
        """Возврат текущего LUFS без добавления новых сэмплов."""
        mean_square = max(self._sum_squares / self.window_samples, 1e-10)
        lufs = -0.691 + 10 * np.log10(mean_square)
        return float(lufs)
    
    def reset(self):
        """Сброс измерителя."""
        self.k_filter.reset()
        self.buffer.clear()
        for _ in range(self.window_samples):
            self.buffer.append(0.0)
        self._sum_squares = 0.0


class TruePeakMeter:
    """
    Измеритель True Peak с 4x интерполяцией.
    
    True Peak учитывает межсэмпловые пики, которые могут возникнуть
    при цифро-аналоговом преобразовании.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.oversample_factor = 4
        
        # Проектируем интерполяционный фильтр
        self._design_interpolation_filter()
        
        # Состояние
        self._max_peak = 0.0
        self._current_peak = 0.0
    
    def _design_interpolation_filter(self):
        """Проектирование low-pass фильтра для интерполяции."""
        # FIR фильтр для 4x oversampling
        num_taps = 49
        cutoff = 0.5 / self.oversample_factor
        self.interp_filter = signal.firwin(num_taps, cutoff)
        self.filter_delay = (num_taps - 1) // 2
    
    def process(self, samples: np.ndarray) -> float:
        """
        Измерение True Peak для блока сэмплов.
        
        Args:
            samples: Входные аудио сэмплы
            
        Returns:
            True Peak в dBTP
        """
        # Upsample сигнал
        upsampled = np.zeros(len(samples) * self.oversample_factor)
        upsampled[::self.oversample_factor] = samples
        
        # Применяем интерполяционный фильтр
        interpolated = signal.lfilter(self.interp_filter, 1, upsampled)
        interpolated *= self.oversample_factor  # Компенсация амплитуды
        
        # Находим максимальный пик
        peak_linear = np.max(np.abs(interpolated))
        self._current_peak = peak_linear
        self._max_peak = max(self._max_peak, peak_linear)
        
        # Конвертируем в dBTP
        if peak_linear > 0:
            peak_dbtp = 20 * np.log10(peak_linear)
        else:
            peak_dbtp = -100.0
        
        return float(peak_dbtp)
    
    def get_current_peak_dbtp(self) -> float:
        """Возврат текущего True Peak в dBTP."""
        if self._current_peak > 0:
            return float(20 * np.log10(self._current_peak))
        return -100.0
    
    def get_max_peak_dbtp(self) -> float:
        """Возврат максимального True Peak в dBTP."""
        if self._max_peak > 0:
            return float(20 * np.log10(self._max_peak))
        return -100.0
    
    def reset_max(self):
        """Сброс максимального пика."""
        self._max_peak = 0.0
    
    def reset(self):
        """Полный сброс."""
        self._max_peak = 0.0
        self._current_peak = 0.0


class AGCEnvelope:
    """
    Envelope generator для плавного управления gain.
    
    Реализует Attack/Release/Hold логику для предотвращения
    резких скачков усиления.
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 attack_ms: float = 50.0,
                 release_ms: float = 500.0,
                 hold_ms: float = 200.0,
                 update_interval_ms: float = 100.0):
        self.sample_rate = sample_rate
        self.update_interval_ms = update_interval_ms
        
        # Вычисляем коэффициенты
        self.set_times(attack_ms, release_ms, hold_ms)
        
        # Состояние
        self._current_gain = 0.0
        self._hold_counter = 0
        self._is_holding = False
    
    def set_times(self, attack_ms: float, release_ms: float, hold_ms: float):
        """Установка временных констант."""
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.hold_ms = hold_ms
        
        # Коэффициенты для экспоненциального сглаживания
        # alpha = 1 - exp(-update_interval / time_constant)
        # Это правильная формула для дискретного обновления каждые update_interval_ms
        self.attack_coef = 1 - np.exp(-self.update_interval_ms / attack_ms) if attack_ms > 0 else 1.0
        self.release_coef = 1 - np.exp(-self.update_interval_ms / release_ms) if release_ms > 0 else 1.0
        
        # Hold в итерациях (не сэмплах)
        self.hold_iterations = int(hold_ms / self.update_interval_ms)
    
    def process(self, target_gain: float) -> float:
        """
        Обработка целевого gain с применением envelope.
        
        Args:
            target_gain: Целевое усиление в dB
            
        Returns:
            Сглаженное усиление в dB
        """
        # Определяем направление изменения
        if target_gain < self._current_gain:
            # Снижение gain (Attack) - быстрая реакция
            self._current_gain += self.attack_coef * (target_gain - self._current_gain)
            self._is_holding = True
            self._hold_counter = self.hold_iterations
        elif self._is_holding:
            # Hold period - удерживаем текущее значение
            self._hold_counter -= 1
            if self._hold_counter <= 0:
                self._is_holding = False
        else:
            # Повышение gain (Release) - медленное восстановление
            self._current_gain += self.release_coef * (target_gain - self._current_gain)
        
        return self._current_gain
    
    def get_current_gain(self) -> float:
        """Возврат текущего gain."""
        return self._current_gain
    
    def reset(self, initial_gain: float = 0.0):
        """Сброс envelope."""
        self._current_gain = initial_gain
        self._hold_counter = 0
        self._is_holding = False


class ChannelAGC:
    """AGC для одного канала."""
    
    def __init__(self, 
                 channel_id: int,
                 sample_rate: int = 48000,
                 target_lufs: float = -23.0,
                 true_peak_limit: float = -1.0,
                 max_gain_db: float = 12.0,
                 min_gain_db: float = -12.0,
                 ratio: float = 4.0,
                 attack_ms: float = 50.0,
                 release_ms: float = 500.0,
                 hold_ms: float = 200.0,
                 gate_threshold_lufs: float = -50.0,
                 update_interval_ms: float = 100.0):
        
        self.channel_id = channel_id
        self.sample_rate = sample_rate
        self.update_interval_ms = update_interval_ms
        
        # Параметры AGC
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.ratio = ratio
        self.gate_threshold_lufs = gate_threshold_lufs
        
        # Компоненты
        self.lufs_meter = LUFSMeter(sample_rate)
        self.true_peak_meter = TruePeakMeter(sample_rate)
        self.envelope = AGCEnvelope(sample_rate, attack_ms, release_ms, hold_ms, update_interval_ms)
        
        # Состояние
        self.current_lufs = -60.0
        self.current_true_peak = -60.0
        self.current_gain = 0.0
        self.applied_gain = 0.0
        self.is_gated = True
        self.status = "idle"  # idle, measuring, adjusting, limiting
        
        # Base TRIM (начальное значение с микшера)
        self.base_trim = 0.0
    
    def process(self, samples: np.ndarray) -> Dict[str, Any]:
        """
        Обработка аудио сэмплов и расчет коррекции gain.
        
        Args:
            samples: Входные аудио сэмплы
            
        Returns:
            Словарь с метриками и рекомендуемым gain
        """
        # Измеряем LUFS
        self.current_lufs = self.lufs_meter.process(samples)
        
        # Измеряем True Peak
        self.current_true_peak = self.true_peak_meter.process(samples)
        
        # Проверяем gate
        if self.current_lufs < self.gate_threshold_lufs:
            self.is_gated = True
            self.status = "idle"
            # При gated не меняем gain
            target_gain = self.current_gain
        else:
            self.is_gated = False
            self.status = "measuring"
            
            # Вычисляем ошибку
            error = self.target_lufs - self.current_lufs
            
            # Применяем ratio (компрессия)
            target_gain = error / self.ratio
            
            # Лимитируем gain
            target_gain = max(self.min_gain_db, min(self.max_gain_db, target_gain))
            
            # Проверяем True Peak limit
            predicted_peak = self.current_true_peak + target_gain
            if predicted_peak > self.true_peak_limit:
                # Ограничиваем gain чтобы не превысить True Peak limit
                target_gain = self.true_peak_limit - self.current_true_peak
                target_gain = max(self.min_gain_db, target_gain)
                self.status = "limiting"
            else:
                self.status = "adjusting"
        
        # Bleed detection and gain blocking now handled externally via bleed_service
        
        # Применяем envelope для плавности
        self.current_gain = self.envelope.process(target_gain)
        
        # Итоговый applied gain
        self.applied_gain = self.base_trim + self.current_gain
        
        return {
            "channel": self.channel_id,
            "lufs": self.current_lufs,
            "true_peak": self.current_true_peak,
            "gain": self.current_gain,
            "applied_gain": self.applied_gain,
            "is_gated": self.is_gated,
            "status": self.status,
        }
    
    def set_base_trim(self, trim: float):
        """Установка базового TRIM."""
        self.base_trim = trim
    
    def get_recommended_trim(self) -> float:
        """Возврат рекомендуемого TRIM для микшера."""
        return self.applied_gain
    
    def reset(self):
        """Сброс состояния канала."""
        self.lufs_meter.reset()
        self.true_peak_meter.reset()
        self.envelope.reset()
        self.current_lufs = -60.0
        self.current_true_peak = -60.0
        self.current_gain = 0.0
        self.applied_gain = self.base_trim
        self.is_gated = True
        self.status = "idle"


class LUFSGainStagingController:
    """
    Главный контроллер LUFS-based Gain Staging.
    
    Управляет захватом аудио, анализом LUFS и применением
    TRIM коррекций к микшеру через OSC.
    """
    
    def __init__(self,
                 mixer_client=None,
                 sample_rate: int = 48000,
                 chunk_size: int = 2048,
                 config: Optional[Dict[str, Any]] = None,
                 bleed_service=None,
                 audio_capture=None):

        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._audio_capture = audio_capture
        
        # Загрузка конфигурации
        self.config = config or {}
        self.bleed_service = bleed_service
        agc_config = self.config.get('automation', {}).get('lufs_gain_staging', {})
        peak_config = self.config.get('automation', {}).get('peak_gain_staging', {})
        
        # Параметры AGC
        self.target_lufs = agc_config.get('target_lufs', -23.0)
        self.true_peak_limit = agc_config.get('true_peak_limit', -1.0)
        self.max_gain_db = agc_config.get('max_gain_db', 12.0)
        self.min_gain_db = agc_config.get('min_gain_db', -12.0)
        self.ratio = agc_config.get('ratio', 4.0)
        self.attack_ms = agc_config.get('attack_ms', 50.0)
        self.release_ms = agc_config.get('release_ms', 500.0)
        self.hold_ms = agc_config.get('hold_ms', 200.0)
        self.gate_threshold_lufs = agc_config.get('gate_threshold_lufs', -50.0)
        self.update_interval = agc_config.get('update_interval_ms', 100) / 1000.0
        
        # Centralized bleed detection service
        self.bleed_service = bleed_service
        
        # Параметры Peak Gain Staging
        self.peak_staging_mode = False  # Режим работы: False = LUFS, True = Peak
        self.default_peak_threshold = peak_config.get('default_peak_threshold', -6.0)
        self.trim_reduction_step_db = peak_config.get('trim_reduction_step_db', 1.0)
        self.min_trim_db = peak_config.get('min_trim_db', -18.0)
        # Live mode: ограничение снижения trim (для концерта не глубже -8 dB, чтобы канал оставался слышимым)
        self.live_mode = agc_config.get('live_mode', False)
        self.min_trim_db_live = agc_config.get('min_trim_db_live', -8.0)
        # Пресеты для soundcheck / live (target_lufs, ratio, attack/release, gate)
        presets = agc_config.get('presets', {})
        self._presets = {
            'soundcheck': presets.get('soundcheck', {
                'target_lufs': -18.0, 'ratio': 2.0, 'attack_ms': 50.0,
                'release_ms': 500.0, 'gate_threshold_lufs': -50.0,
            }),
            'live': presets.get('live', {
                'target_lufs': -16.0, 'ratio': 1.5, 'attack_ms': 200.0,
                'release_ms': 2000.0, 'gate_threshold_lufs': -40.0,
            }),
        }
        
        # Каналы
        self.channels: Dict[int, ChannelAGC] = {}
        self.channel_mapping: Dict[int, int] = {}  # audio_ch -> mixer_ch
        self.channel_settings: Dict[int, Dict] = {}
        
        # Измеренные уровни (для совместимости с server.py)
        self.measured_levels: Dict[int, Dict] = {}
        
        # Отслеживание состояния коррекции для Peak Staging режима
        self.trim_correction_active: Dict[int, bool] = {}  # mixer_ch -> активна ли коррекция
        self.last_trim_value: Dict[int, float] = {}  # mixer_ch -> последнее значение Trim
        self.trim_reduction_count: Dict[int, int] = {}  # mixer_ch -> количество снижений
        
        # PyAudio
        self.pa = None
        self.stream = None
        self.device_index = None
        self._num_channels = 0
        
        # Threading
        self.is_active = False
        self.realtime_correction_enabled = False
        self._stop_event = threading.Event()
        self._correction_thread = None
        
        # Callbacks
        self.on_status_update: Optional[Callable[[Dict], None]] = None
        self.on_levels_updated: Optional[Callable[[Dict], None]] = None
        
        # Буферы аудио
        self._audio_buffers: Dict[int, deque] = {}
        
        # Freeze: when True, do not apply TRIM corrections (for emergency stop / live)
        self.automation_frozen = False
        
        logger.info(f"LUFSGainStagingController initialized: target={self.target_lufs} LUFS, "
                   f"ratio={self.ratio}:1, attack={self.attack_ms}ms, release={self.release_ms}ms")
    
    def start(self, 
              device_id: int,
              channels: List[int],
              channel_settings: Dict[int, Dict],
              channel_mapping: Dict[int, int],
              on_status_callback: Callable = None) -> bool:
        """
        Запуск захвата аудио и анализа.
        
        Args:
            device_id: Индекс аудио устройства
            channels: Список номеров каналов для анализа
            channel_settings: Настройки каналов (preset и т.д.)
            channel_mapping: Маппинг audio_ch -> mixer_ch
            on_status_callback: Callback для обновлений статуса
            
        Returns:
            True если успешно запущен
        """
        if self.is_active:
            logger.warning("Controller already active")
            return False

        self.device_index = device_id
        self.channel_settings = channel_settings
        self.channel_mapping = channel_mapping
        self.on_status_update = on_status_callback

        def _init_channels(channels, channel_mapping):
            self.channels.clear()
            self._audio_buffers.clear()
            self.measured_levels.clear()
            logger.info(f"Cleared old channels, initializing {len(channels)} new channels")
            update_interval_ms = self.update_interval * 1000
            for audio_ch in channels:
                mixer_ch = channel_mapping.get(audio_ch, audio_ch)
                self.channels[audio_ch] = ChannelAGC(
                    channel_id=mixer_ch,
                    sample_rate=self.sample_rate,
                    target_lufs=self.target_lufs,
                    true_peak_limit=self.true_peak_limit,
                    max_gain_db=self.max_gain_db,
                    min_gain_db=self.min_gain_db,
                    ratio=self.ratio,
                    attack_ms=self.attack_ms,
                    release_ms=self.release_ms,
                    hold_ms=self.hold_ms,
                    gate_threshold_lufs=self.gate_threshold_lufs,
                    update_interval_ms=update_interval_ms,
                )
                self._audio_buffers[audio_ch] = deque(maxlen=10)

        # Use unified AudioCapture if available
        if self._audio_capture is not None:
            try:
                self.sample_rate = self._audio_capture.sample_rate
                self._num_channels = max(channels) if channels else 2
                _init_channels(channels, channel_mapping)
                self._audio_capture.subscribe('lufs_gain_staging', self._audio_capture_poll)
                self.is_active = True
                self._stop_event.clear()
                logger.info(f"Audio capture started via AudioCapture: {len(channels)} channels")
                return True
            except Exception as e:
                logger.warning(f"AudioCapture integration failed, falling back to PyAudio: {e}")
                self._audio_capture = None

        # Fallback: direct PyAudio stream
        try:
            import pyaudio

            self.pa = pyaudio.PyAudio()

            # Получаем информацию об устройстве
            device_info = self.pa.get_device_info_by_index(int(device_id))
            max_channels = int(device_info.get('maxInputChannels', 2))
            device_sample_rate = int(device_info.get('defaultSampleRate', 48000))

            logger.info(f"Audio device: {device_info.get('name')}, "
                       f"max channels: {max_channels}, sample rate: {device_sample_rate}")

            self.sample_rate = device_sample_rate

            # Определяем количество каналов
            required_channels = max(channels) if channels else 2
            self._num_channels = min(required_channels, max_channels)

            _init_channels(channels, channel_mapping)

            # Открываем аудио поток
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self._num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(device_id),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            self.stream.start_stream()
            self.is_active = True
            self._stop_event.clear()

            logger.info(f"Audio capture started: {len(channels)} channels")

            return True

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}", exc_info=True)
            self.stop()
            return False
    
    def _audio_capture_poll(self):
        """Poll AudioCapture buffers and fill local _audio_buffers."""
        if not self._audio_capture:
            return
        for audio_ch in list(self.channels.keys()):
            data = self._audio_capture.get_buffer(audio_ch, self.chunk_size)
            if data is not None and len(data) > 0:
                self._audio_buffers[audio_ch].append(data.copy())

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback для обработки аудио."""
        import pyaudio

        if not self.is_active:
            return (None, pyaudio.paComplete)
        
        try:
            # Конвертируем в numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Разделяем по каналам
            if self._num_channels > 1:
                audio_data = audio_data.reshape(-1, self._num_channels)
            else:
                audio_data = audio_data.reshape(-1, 1)
            
            # Обновляем буферы для каждого канала
            for audio_ch, agc in list(self.channels.items()):
                if audio_ch <= self._num_channels:
                    channel_data = audio_data[:, audio_ch - 1]
                    self._audio_buffers[audio_ch].append(channel_data.copy())
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def start_realtime_correction(self) -> bool:
        """
        Запуск real-time TRIM коррекции.
        
        Returns:
            True если успешно запущен
        """
        if not self.is_active:
            logger.error("Cannot start correction: controller not active")
            return False
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.error("Cannot start correction: mixer not connected")
            return False
        
        if self.realtime_correction_enabled:
            logger.warning("Real-time correction already enabled")
            return True
        
        # Получаем текущие TRIM значения с микшера
        for audio_ch, agc in list(self.channels.items()):
            mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)
            try:
                current_trim = self.mixer_client.get_channel_gain(mixer_ch) or 0.0
                agc.set_base_trim(current_trim)
                logger.info(f"Channel {mixer_ch}: base TRIM = {current_trim:.1f} dB")
                
                # Инициализация состояния для Peak Staging режима
                if self.peak_staging_mode:
                    self.trim_correction_active[mixer_ch] = True
                    self.last_trim_value[mixer_ch] = current_trim
                    self.trim_reduction_count[mixer_ch] = 0
                    logger.info(f"Channel {mixer_ch}: Peak staging initialized, active=True, trim={current_trim:.1f} dB")
            except Exception as e:
                logger.warning(f"Could not get TRIM for channel {mixer_ch}: {e}")
                agc.set_base_trim(0.0)
                if self.peak_staging_mode:
                    self.trim_correction_active[mixer_ch] = True
                    self.last_trim_value[mixer_ch] = 0.0
                    self.trim_reduction_count[mixer_ch] = 0
        
        self.realtime_correction_enabled = True
        self._stop_event.clear()
        
        # Запускаем поток коррекции
        self._correction_thread = threading.Thread(
            target=self._correction_loop, 
            daemon=True
        )
        self._correction_thread.start()
        
        mode_name = "Peak" if self.peak_staging_mode else "LUFS"
        logger.info(f"Real-time {mode_name} correction started")
        
        if self.on_status_update:
            self.on_status_update({
                'type': 'realtime_correction_started',
                'active': True
            })
        
        return True

    def handle_external_trim_change(self, mixer_channel: int, new_trim_db: float) -> None:
        """Обработка внешнего изменения TRIM для канала микшера."""
        if not self.realtime_correction_enabled:
            return

        try:
            new_trim_db = float(new_trim_db)
        except (TypeError, ValueError):
            return

        if not np.isfinite(new_trim_db):
            return

        safe_trim_db = float(np.clip(new_trim_db, -18.0, 18.0))
        try:
            mixer_channel_id = int(mixer_channel)
        except (TypeError, ValueError):
            return

        handled = False
        for agc in self.channels.values():
            if agc.channel_id == mixer_channel_id:
                agc.set_base_trim(safe_trim_db)
                if self.peak_staging_mode and mixer_channel_id in self.trim_correction_active:
                    self.last_trim_value[agc.channel_id] = safe_trim_db
                    self.trim_reduction_count[agc.channel_id] = 0
                handled = True
                break

        if not handled:
            logger.debug("No active TRIM channel state for mixer channel %s", mixer_channel)
    
    def _correction_loop(self):
        """Основной цикл коррекции."""
        logger.info("Correction loop started")
        
        iteration = 0
        while self.realtime_correction_enabled and not self._stop_event.is_set():
            try:
                levels_update = {}
                trim_updates = {}
                iteration += 1
                
                # Логируем каждые 50 итераций статистику буферов
                if iteration % 50 == 0:
                    buffer_sizes = {ch: len(buf) for ch, buf in self._audio_buffers.items()}
                    logger.info(f"Iteration {iteration}: buffer sizes = {buffer_sizes}")
                
                for audio_ch, agc in list(self.channels.items()):
                    # Получаем накопленные аудио данные
                    if audio_ch in self._audio_buffers and len(self._audio_buffers[audio_ch]) > 0:
                        # Объединяем буферы
                        chunks = list(self._audio_buffers[audio_ch])
                        self._audio_buffers[audio_ch].clear()
                        
                        if chunks:
                            try:
                                audio_data = np.concatenate(chunks)
                                
                                # Обрабатываем (включая тихие данные - AGC сам отметит gated/idle)
                                result = agc.process(audio_data)
                            except Exception as e:
                                logger.error(f"Error processing audio for channel {audio_ch}: {e}", exc_info=True)
                                continue
                            
                            mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)
                            
                            # Логируем первые несколько каналов каждые 50 итераций
                            if iteration % 50 == 0 and audio_ch <= 3:
                                logger.info(f"Ch{audio_ch}: LUFS={result['lufs']:.1f}, TruePeak={result['true_peak']:.1f}, "
                                           f"gain={result['gain']:.2f}, gated={result['is_gated']}, status={result['status']}")
                            
                            # Обновляем measured_levels для совместимости
                            self.measured_levels[audio_ch] = {
                                'peak': result['true_peak'],
                                'signal_present': not result['is_gated'],
                                'lufs': result['lufs'],
                                'true_peak': result['true_peak'],
                                'gain': result['gain'],
                                'applied_gain': result['applied_gain'],
                                'status': result['status'],
                            }
                            
                            # Bleed detection: блокируем повышение gain при высоком bleed_ratio
                            bleed_info = None
                            bleed_ratio = 0.0
                            if self.bleed_service and self.bleed_service.enabled:
                                bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                                if bleed_info:
                                    bleed_ratio = float(bleed_info.bleed_ratio)
                                    # Блокируем повышение gain при высоком bleed (>0.5)
                                    if bleed_ratio > 0.5 and result['gain'] > agc.current_gain:
                                        # Не повышаем gain - оставляем текущий
                                        logger.debug(f"Ch{audio_ch}: High bleed (ratio={bleed_ratio:.2f}), blocking gain increase")
                                        # Можно также компенсировать LUFS перед расчетом коррекции
                                    # Компенсируем LUFS для более точного расчета
                                    compensated_lufs = self.bleed_service.get_compensated_level(audio_ch, result['lufs'])
                                    if compensated_lufs != result['lufs']:
                                        # Пересчитываем gain с компенсированным LUFS
                                        error = agc.target_lufs - compensated_lufs
                                        compensated_gain = error / agc.ratio
                                        compensated_gain = max(agc.min_gain_db, min(agc.max_gain_db, compensated_gain))
                                        # Используем компенсированный gain только если он меньше (меньше усиление при блиде)
                                        if compensated_gain < result['gain']:
                                            agc.current_gain = compensated_gain
                                            result['gain'] = compensated_gain
                                            result['applied_gain'] = agc.base_trim + compensated_gain
                                            logger.debug(f"Ch{audio_ch}: Compensated LUFS {result['lufs']:.1f} -> {compensated_lufs:.1f}, gain {result['gain']:.2f} -> {compensated_gain:.2f}")
                            
                            self.measured_levels[audio_ch]['bleed_ratio'] = bleed_ratio
                            self.measured_levels[audio_ch]['bleed_source'] = int(bleed_info.bleed_source_channel) if bleed_info and bleed_info.bleed_source_channel else None
                            
                            levels_update[audio_ch] = self.measured_levels[audio_ch]
                            
                            # Peak Staging режим (пропуск при freeze)
                            if self.peak_staging_mode and not self.automation_frozen:
                                effective_min_trim = self.min_trim_db_live if self.live_mode else self.min_trim_db
                                # Проверяем, активна ли коррекция для этого канала
                                if self.trim_correction_active.get(mixer_ch, False):
                                    # Получаем порог пика из channel_settings
                                    channel_setting = self.channel_settings.get(audio_ch, {})
                                    peak_threshold = channel_setting.get('peak_threshold', self.default_peak_threshold)
                                    
                                    # Проверяем превышение порога
                                    if result['true_peak'] > peak_threshold:
                                        logger.info(f"Peak threshold exceeded for channel {mixer_ch}: "
                                                   f"{result['true_peak']:.1f} > {peak_threshold:.1f} dBTP")
                                        
                                        # Получаем текущий Trim
                                        try:
                                            current_trim = self.mixer_client.get_channel_gain(mixer_ch) or 0.0
                                            
                                            # Вычисляем новый Trim (снижаем на 1дБ)
                                            new_trim = current_trim - self.trim_reduction_step_db
                                            
                                            # Ограничиваем диапазон (в live_mode не глубже min_trim_db_live)
                                            new_trim = max(effective_min_trim, new_trim)
                                            
                                            # Применяем новый Trim
                                            logger.info(f"Reducing Trim for channel {mixer_ch}: "
                                                       f"{current_trim:.1f} -> {new_trim:.1f} dB")
                                            self.mixer_client.set_channel_gain(mixer_ch, new_trim)
                                            
                                            # Проверяем фактический Trim после применения
                                            actual_trim = self.mixer_client.get_channel_gain(mixer_ch) or new_trim
                                            
                                            # Определяем остановку коррекции
                                            last_trim = self.last_trim_value.get(mixer_ch, current_trim)
                                            
                                            # Остановка если:
                                            # 1. Trim не изменился после попытки снижения (actual_trim == current_trim)
                                            #    Это означает, что микшер не принял изменение или достигнут минимум
                                            # 2. Trim достиг минимального значения
                                            # 3. Новый Trim равен предыдущему значению (Trim перестал уменьшаться)
                                            trim_unchanged = abs(actual_trim - current_trim) < 0.01
                                            trim_at_minimum = actual_trim <= effective_min_trim + 0.01
                                            trim_no_progress = abs(actual_trim - last_trim) < 0.01 and self.trim_reduction_count.get(mixer_ch, 0) > 0
                                            
                                            if trim_unchanged or trim_at_minimum or trim_no_progress:
                                                self.trim_correction_active[mixer_ch] = False
                                                if trim_at_minimum:
                                                    reason = "reached minimum limit"
                                                elif trim_unchanged:
                                                    reason = "trim did not change after reduction attempt"
                                                else:
                                                    reason = "no progress detected (trim stopped decreasing)"
                                                logger.info(f"Trim correction stopped for channel {mixer_ch}: "
                                                           f"{reason} (trim={actual_trim:.1f} dB, was={current_trim:.1f} dB, "
                                                           f"last={last_trim:.1f} dB)")
                                            else:
                                                # Обновляем состояние
                                                self.last_trim_value[mixer_ch] = actual_trim
                                                self.trim_reduction_count[mixer_ch] = self.trim_reduction_count.get(mixer_ch, 0) + 1
                                                logger.debug(f"Channel {mixer_ch}: Trim reduced, count={self.trim_reduction_count[mixer_ch]}")
                                                
                                        except Exception as e:
                                            logger.error(f"Error processing peak staging for channel {mixer_ch}: {e}")
                            else:
                                # LUFS режим (оригинальная логика)
                                # Если не gated и коррекция достаточно значительная
                                if not result['is_gated'] and abs(result['gain']) > 0.1:
                                    trim_updates[mixer_ch] = result['applied_gain']
                
                # Применяем TRIM коррекции (только для LUFS режима; пропуск при freeze)
                if not self.peak_staging_mode and trim_updates and self.mixer_client and not self.automation_frozen:
                    for mixer_ch, new_trim in trim_updates.items():
                        try:
                            # Ограничиваем диапазон
                            new_trim = max(-18.0, min(18.0, new_trim))
                            self.mixer_client.set_channel_gain(mixer_ch, new_trim)
                            logger.info(f"Channel {mixer_ch}: TRIM -> {new_trim:.1f} dB")
                        except Exception as e:
                            logger.error(f"Error setting TRIM for channel {mixer_ch}: {e}")
                
                # Отправляем обновление уровней
                if levels_update and self.on_levels_updated:
                    self.on_levels_updated(levels_update)
                
            except Exception as e:
                logger.error(f"Correction loop error: {e}", exc_info=True)
            
            # Ждем следующего цикла
            self._stop_event.wait(self.update_interval)
        
        logger.info("Correction loop stopped")
    
    def stop_realtime_correction(self):
        """Остановка real-time коррекции."""
        if not self.realtime_correction_enabled:
            return
        
        self.realtime_correction_enabled = False
        self._stop_event.set()
        
        if self._correction_thread and self._correction_thread.is_alive():
            self._correction_thread.join(timeout=1.0)
        
        logger.info("Real-time correction stopped")
        
        if self.on_status_update:
            self.on_status_update({
                'type': 'realtime_correction_stopped',
                'active': False
            })
    
    def stop(self):
        """Полная остановка контроллера."""
        self.stop_realtime_correction()

        self.is_active = False
        self._stop_event.set()

        if self._audio_capture is not None:
            try:
                self._audio_capture.unsubscribe('lufs_gain_staging')
            except Exception:
                pass

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None

        self.channels.clear()
        self._audio_buffers.clear()
        self.measured_levels.clear()

        logger.info("Controller stopped")
    
    def get_status(self) -> Dict:
        """Возврат текущего статуса."""
        # Преобразуем NumPy типы в нативные Python типы
        levels = {}
        for ch, data in self.measured_levels.items():
            levels[str(ch)] = {
                key: bool(value) if isinstance(value, np.bool_) else 
                     float(value) if isinstance(value, (np.floating, np.float_, np.float16, np.float32, np.float64)) else
                     int(value) if isinstance(value, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                                      np.uint8, np.uint16, np.uint32, np.uint64)) else
                     value
                for key, value in data.items()
            }
        
        return {
            'active': bool(self.is_active),
            'realtime_enabled': bool(self.realtime_correction_enabled),
            'target_lufs': float(self.target_lufs),
            'true_peak_limit': float(self.true_peak_limit),
            'ratio': float(self.ratio),
            'channels': int(len(self.channels)),
            'levels': levels
        }
    
    def update_parameters(self,
                          target_lufs: Optional[float] = None,
                          true_peak_limit: Optional[float] = None,
                          max_gain_db: Optional[float] = None,
                          min_gain_db: Optional[float] = None,
                          ratio: Optional[float] = None,
                          attack_ms: Optional[float] = None,
                          release_ms: Optional[float] = None,
                          hold_ms: Optional[float] = None,
                          gate_threshold_lufs: Optional[float] = None):
        """Обновление параметров AGC в реальном времени."""
        if target_lufs is not None:
            self.target_lufs = target_lufs
        if true_peak_limit is not None:
            self.true_peak_limit = true_peak_limit
        if max_gain_db is not None:
            self.max_gain_db = max_gain_db
        if min_gain_db is not None:
            self.min_gain_db = min_gain_db
        if ratio is not None:
            self.ratio = ratio
        if attack_ms is not None:
            self.attack_ms = attack_ms
        if release_ms is not None:
            self.release_ms = release_ms
        if hold_ms is not None:
            self.hold_ms = hold_ms
        if gate_threshold_lufs is not None:
            self.gate_threshold_lufs = gate_threshold_lufs
        
        # Обновляем параметры для всех каналов
        for agc in list(self.channels.values()):
            agc.target_lufs = self.target_lufs
            agc.true_peak_limit = self.true_peak_limit
            agc.max_gain_db = self.max_gain_db
            agc.min_gain_db = self.min_gain_db
            agc.ratio = self.ratio
            agc.gate_threshold_lufs = self.gate_threshold_lufs
            agc.envelope.set_times(self.attack_ms, self.release_ms, self.hold_ms)
        
        logger.info(f"AGC parameters updated: target={self.target_lufs} LUFS, ratio={self.ratio}:1, gate={self.gate_threshold_lufs}")
    
    def apply_preset(self, name: str) -> bool:
        """Применить пресет soundcheck или live (target_lufs, ratio, attack/release, gate)."""
        preset = self._presets.get(name)
        if not preset:
            logger.warning(f"Unknown preset: {name}")
            return False
        self.update_parameters(
            target_lufs=preset.get('target_lufs'),
            ratio=preset.get('ratio'),
            attack_ms=preset.get('attack_ms'),
            release_ms=preset.get('release_ms'),
            gate_threshold_lufs=preset.get('gate_threshold_lufs'),
        )
        self.live_mode = name == 'live'
        logger.info(f"Applied preset '{name}': target={self.target_lufs} LUFS, ratio={self.ratio}:1, gate={self.gate_threshold_lufs}")
        return True
    
    # Свойство analyzer для совместимости с server.py
    @property
    def analyzer(self):
        """Совместимость с server.py - возвращает self."""
        return self
    
    @property
    def is_audio_stream_active(self) -> bool:
        """Check if audio stream is actually running."""
        if not self.stream:
            return False
        try:
            return self.stream.is_active()
        except Exception:
            return False


class SafeGainCalibrator:
    """
    Safe Gain Calibrator для анализа сигнала и рекомендации безопасных уровней gain.

    Выполняет фазу обучения (LEARNING), анализирует пики и LUFS,
    затем предоставляет рекомендации по коррекции gain через микшер.
    """

    def __init__(self,
                 mixer_client=None,
                 sample_rate: int = 48000,
                 config: Optional[Dict[str, Any]] = None):

        self.mixer_client = mixer_client
        self.sample_rate = sample_rate

        self.config = config or {}
        cal_config = self.config.get('automation', {}).get('safe_gain_calibration', {})

        self.target_lufs = cal_config.get('target_lufs', -18.0)
        self.max_peak_limit = cal_config.get('max_peak_limit', -3.0)
        self.target_peak_db = cal_config.get('target_peak_db', -10.0)
        self.noise_gate_threshold = cal_config.get('noise_gate_threshold', -40.0)
        self.min_signal_presence = cal_config.get('min_signal_presence', 0.05)
        self.learning_duration = cal_config.get('learning_duration_sec', 30.0)  # По умолчанию 30 секунд
        self.method_id = str(cal_config.get('method_id') or INPUT_TRIM_AUTOGAIN_METHOD_ID)
        self.default_mode = str(cal_config.get('default_mode') or cal_config.get('mode') or 'soundcheck')
        self.local_input_preamp = LocalInputGainProfile.from_config(cal_config.get('local_input_preamp', {}))
        logger.info(f"SafeGainCalibrator initialized with learning_duration={self.learning_duration} seconds")

        # Channel settings for preset-based metrics
        self.channel_settings: Dict[int, Dict] = {}
        self.preset_targets = cal_config.get('preset_targets', {})
        self.state = AnalysisState.IDLE

        self.channels: Dict[int, SignalStats] = {}

        self.channel_mapping: Dict[int, int] = {}

        self.lufs_meters: Dict[int, LUFSMeter] = {}
        self.true_peak_meters: Dict[int, TruePeakMeter] = {}

        self.learning_start_time: Optional[float] = None
        self.learning_progress: float = 0.0

        self.suggestions: Dict[int, Dict[str, Any]] = {}

        self.on_progress_update: Optional[Callable] = None
        self.on_suggestions_ready: Optional[Callable] = None

        logger.info(f"SafeGainCalibrator initialized: target={self.target_lufs} LUFS, "
                   f"peak_limit={self.max_peak_limit} dBTP, gate={self.noise_gate_threshold} dB")

    def add_channel(self, audio_channel: int, mixer_channel: Optional[int] = None):
        """
        Добавление канала для анализа.

        Args:
            audio_channel: Номер аудио канала
            mixer_channel: Номер канала в микшере (по умолчанию = audio_channel)
        """
        mixer_ch = mixer_channel if mixer_channel is not None else audio_channel
        self.channel_mapping[audio_channel] = mixer_ch

        self.channels[audio_channel] = SignalStats(
            channel_id=audio_channel,
            noise_gate_threshold_db=self.noise_gate_threshold
        )

        self.lufs_meters[audio_channel] = LUFSMeter(
            sample_rate=self.sample_rate,
            window_ms=400.0
        )

        self.true_peak_meters[audio_channel] = TruePeakMeter(
            sample_rate=self.sample_rate
        )

        logger.info(f"Channel {audio_channel} -> mixer {mixer_ch} added to calibrator")

    def remove_channel(self, audio_channel: int):
        """Удаление канала из анализа."""
        if audio_channel in self.channels:
            del self.channels[audio_channel]
            del self.lufs_meters[audio_channel]
            del self.true_peak_meters[audio_channel]
            del self.channel_mapping[audio_channel]
            logger.info(f"Channel {audio_channel} removed from calibrator")

    def start_analysis(self) -> bool:
        """
        Запуск фазы анализа (LEARNING).

        Returns:
            True если анализ успешно запущен
        """
        if self.state != AnalysisState.IDLE:
            logger.warning(f"Cannot start analysis: current state is {self.state}")
            return False

        if len(self.channels) == 0:
            logger.error("Cannot start analysis: no channels configured")
            return False

        for ch_id, stats in self.channels.items():
            self.channels[ch_id] = SignalStats(
                channel_id=ch_id,
                noise_gate_threshold_db=self.noise_gate_threshold
            )
            self.lufs_meters[ch_id].reset()
            self.true_peak_meters[ch_id].reset()

        self.suggestions.clear()
        self.learning_start_time = time.time()
        self.learning_progress = 0.0
        self.state = AnalysisState.LEARNING

        logger.info(f"Analysis started: learning for {self.learning_duration} seconds")

        if self.on_progress_update:
            self.on_progress_update({
                'state': self.state.value,
                'progress': 0.0,
                'message': f'Analyzing signal... {self.learning_duration}s'
            })

        return True

    def process_audio(self, channel: int, samples: np.ndarray):
        """
        Обработка аудио сэмплов для анализа.

        Args:
            channel: Номер канала
            samples: Аудио сэмплы (numpy array)
        """
        if self.state != AnalysisState.LEARNING:
            return

        if channel not in self.channels:
            return

        stats = self.channels[channel]
        lufs_meter = self.lufs_meters[channel]
        peak_meter = self.true_peak_meters[channel]

        lufs = lufs_meter.process(samples)
        true_peak = peak_meter.process(samples)

        rms = np.sqrt(np.mean(samples ** 2) + 1e-10)
        rms_db = 20 * np.log10(rms) if rms > 0 else -100.0

        stats.update_sample(true_peak, lufs, rms_db)

        elapsed = time.time() - self.learning_start_time
        self.learning_progress = min(elapsed / self.learning_duration, 1.0)

        if self.learning_progress >= 1.0 and self.state == AnalysisState.LEARNING:
            self._finalize_analysis()

    def _soundcheck_ensemble_context_for_channel(self, audio_channel: int) -> Dict[str, Any]:
        """Summarize other active sources so gain staging can separate solo-mic caution from ensemble context."""
        if not self.local_input_preamp.soundcheck_ensemble_context_enabled:
            return {"enabled": False, "other_active_sources": 0, "other_loud_sources": 0}

        target_stats = self.channels.get(audio_channel)
        target_active_indices = list(getattr(target_stats, "active_sample_indices", []) or [])
        target_first_active = min(target_active_indices) if target_active_indices else None
        target_last_active = max(target_active_indices) if target_active_indices else None
        max_context_lag_samples = max(
            0,
            int(self.local_input_preamp.soundcheck_ensemble_max_context_lag_samples),
        )

        def _source_time_context(other_stats: SignalStats) -> Dict[str, Any]:
            other_active_indices = list(getattr(other_stats, "active_sample_indices", []) or [])
            first_active = min(other_active_indices) if other_active_indices else None
            last_active = max(other_active_indices) if other_active_indices else None
            overlaps_target = bool(
                target_first_active is not None
                and target_last_active is not None
                and first_active is not None
                and last_active is not None
                and first_active <= target_last_active
                and last_active >= target_first_active
            )
            lag_samples = (
                max(0, int(target_last_active - last_active))
                if target_last_active is not None and last_active is not None
                else None
            )
            recent_to_target = bool(
                lag_samples is not None
                and lag_samples <= max_context_lag_samples
            )
            return {
                "first_active_sample_index": first_active,
                "last_active_sample_index": last_active,
                "overlaps_target_window": overlaps_target,
                "lag_samples": lag_samples,
                "time_coherent": bool(overlaps_target and recent_to_target),
            }

        active_sources = []
        loud_sources = []
        for other_channel, other_stats in self.channels.items():
            if other_channel == audio_channel or other_stats.total_samples <= 0:
                continue
            if other_stats.signal_presence_ratio < self.local_input_preamp.soundcheck_ensemble_min_other_presence:
                continue
            time_context = _source_time_context(other_stats)
            source = {
                "channel": int(other_channel),
                "presence": float(other_stats.signal_presence_ratio),
                "peak_db": float(other_stats.max_true_peak_db),
                "lufs": float(other_stats.integrated_lufs),
                "crest_factor_db": float(other_stats.crest_factor_db),
                "preset": str(self.channel_settings.get(other_channel, {}).get("preset", "custom")),
                **time_context,
            }
            active_sources.append(source)
            if (
                other_stats.max_true_peak_db >= self.local_input_preamp.soundcheck_ensemble_min_reference_peak_db
                or other_stats.integrated_lufs >= self.local_input_preamp.soundcheck_ensemble_min_reference_lufs
            ):
                loud_sources.append(source)

        coherent_loud_sources = [
            source
            for source in loud_sources
            if bool(source.get("time_coherent"))
        ]
        if self.local_input_preamp.soundcheck_ensemble_time_coherence_enabled:
            reference_sources = coherent_loud_sources
        else:
            reference_sources = loud_sources or active_sources
        if reference_sources:
            reference_peak_db = float(np.percentile(
                [source["peak_db"] for source in reference_sources],
                75,
            ))
            reference_lufs = float(np.percentile(
                [source["lufs"] for source in reference_sources],
                75,
            ))
            reference_crest_db = float(reference_peak_db - reference_lufs)
        else:
            reference_peak_db = None
            reference_lufs = None
            reference_crest_db = None
        timestamp_sources = reference_sources or loud_sources
        latest_reference_active = (
            max(
                int(source["last_active_sample_index"])
                for source in timestamp_sources
                if source.get("last_active_sample_index") is not None
            )
            if any(source.get("last_active_sample_index") is not None for source in timestamp_sources)
            else None
        )
        reference_max_lag_samples = (
            max(0, int(target_last_active - latest_reference_active))
            if target_last_active is not None and latest_reference_active is not None
            else None
        )
        time_coherence_evaluable = bool(
            target_last_active is not None
            and any(source.get("last_active_sample_index") is not None for source in loud_sources)
        )
        required_time_coherent_sources = int(self.local_input_preamp.soundcheck_ensemble_min_other_active_sources)
        other_loud_source_margin = int(len(loud_sources) - required_time_coherent_sources)
        time_coherent_source_margin = int(len(coherent_loud_sources) - required_time_coherent_sources)
        reference_lag_margin_samples = (
            int(max_context_lag_samples - reference_max_lag_samples)
            if reference_max_lag_samples is not None
            else None
        )

        return {
            "enabled": True,
            "other_active_sources": int(len(active_sources)),
            "other_loud_sources": int(len(loud_sources)),
            "other_loud_source_margin": int(other_loud_source_margin),
            "time_coherence_enabled": bool(self.local_input_preamp.soundcheck_ensemble_time_coherence_enabled),
            "time_coherence_evaluable": bool(time_coherence_evaluable),
            "time_coherent": bool(len(coherent_loud_sources) >= required_time_coherent_sources),
            "time_coherent_loud_sources": int(len(coherent_loud_sources)),
            "required_time_coherent_sources": int(required_time_coherent_sources),
            "time_coherent_source_margin": int(time_coherent_source_margin),
            "max_context_lag_samples": int(max_context_lag_samples),
            "target_first_active_sample_index": target_first_active,
            "target_last_active_sample_index": target_last_active,
            "reference_latest_active_sample_index": latest_reference_active,
            "reference_max_lag_samples": reference_max_lag_samples,
            "reference_lag_margin_samples": reference_lag_margin_samples,
            "reference_peak_db": reference_peak_db,
            "reference_lufs": reference_lufs,
            "reference_crest_db": reference_crest_db,
            "sources": loud_sources[:8],
        }

    def _finalize_analysis(self):
        """Завершение фазы анализа и расчет рекомендаций."""
        logger.info("Learning phase completed. Calculating suggestions...")

        for stats in self.channels.values():
            if stats.total_samples > 0:
                stats.calculate_integrated_lufs()
                stats.calculate_crest_factor()

        for ch_id, stats in self.channels.items():
            # Получаем пресет для канала
            channel_setting = self.channel_settings.get(ch_id, {})
            preset = channel_setting.get('preset', 'custom')
            
            # Используем метрики для пресета или значения по умолчанию
            preset_metrics = self.preset_targets.get(preset, {})
            target_lufs = preset_metrics.get('target_lufs', self.target_lufs)
            max_peak_limit = preset_metrics.get('max_peak_limit', self.max_peak_limit)
            
            logger.info(f"Ch{ch_id} ({preset}): using target_lufs={target_lufs:.1f}, max_peak_limit={max_peak_limit:.1f}")
            
            target_peak_db = preset_metrics.get(
                'target_peak_db',
                self._default_target_peak_for_preset(preset, max_peak_limit),
            )
            mixer_ch = self.channel_mapping.get(ch_id, ch_id)
            current_trim = self._current_trim_for_mixer_channel(mixer_ch)
            previous_recommendation = self.suggestions.get(ch_id)

            if self.method_id == "legacy_lufs_safe_gain":
                stats.calculate_safe_gain(
                    target_lufs=target_lufs,
                    max_peak_limit=max_peak_limit,
                    min_signal_presence=self.min_signal_presence
                )
            else:
                stats.calculate_input_trim_recommendation(
                    current_trim_db=current_trim,
                    target_peak_db=target_peak_db,
                    max_peak_limit=max_peak_limit,
                    target_lufs=target_lufs,
                    min_signal_presence=self.min_signal_presence,
                    preamp_profile=self.local_input_preamp,
                    mode=self.default_mode,
                    mixer_channel=mixer_ch,
                    preset=preset,
                    previous_recommendation=previous_recommendation,
                    ensemble_context=self._soundcheck_ensemble_context_for_channel(ch_id),
                )

            self.suggestions[ch_id] = stats.get_report()

        self.state = AnalysisState.READY

        logger.info(f"Analysis complete: {len(self.suggestions)} channels ready")

        if self.on_suggestions_ready:
            self.on_suggestions_ready(self.suggestions)

        if self.on_progress_update:
            self.on_progress_update({
                'state': self.state.value,
                'progress': 1.0,
                'message': 'Analysis complete. Ready to apply.'
            })

    def get_suggestions(self) -> Dict[int, Dict[str, Any]]:
        """
        Получить рекомендации по корректировке gain.

        Returns:
            Словарь с рекомендациями для каждого канала
        """
        if self.state != AnalysisState.READY:
            logger.warning(f"Suggestions not ready yet. Current state: {self.state}")
            return {}

        return self.suggestions.copy()

    def _current_trim_for_mixer_channel(self, mixer_channel: int) -> Optional[float]:
        if self.mixer_client is None:
            return None
        try:
            if hasattr(self.mixer_client, "get_channel_gain"):
                value = self.mixer_client.get_channel_gain(mixer_channel)
                if value is not None:
                    return float(value)
        except Exception:
            logger.debug(f"Could not read current trim for channel {mixer_channel}", exc_info=True)
        state = getattr(self.mixer_client, "state", None)
        if isinstance(state, dict):
            for address in (
                f"/ch/{mixer_channel}/in/set/trim",
                f"/ch/{mixer_channel:02d}/in/set/trim",
                f"/ch/{mixer_channel}/gain",
            ):
                try:
                    value = state.get(address)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _default_target_peak_for_preset(self, preset: str, max_peak_limit: float) -> float:
        preset_defaults = {
            "kick": -9.0,
            "snare": -9.0,
            "tom": -9.0,
            "hihat": -14.0,
            "ride": -14.0,
            "cymbals": -14.0,
            "overheads": -14.0,
            "room": -16.0,
            "bass": -10.0,
            "leadVocal": -8.0,
            "backVocal": -10.0,
            "backingVocal": -10.0,
            "electricGuitar": -10.0,
            "acousticGuitar": -10.0,
            "accordion": -10.0,
            "synth": -14.0,
            "keyboard": -14.0,
            "playback": -14.0,
            "custom": float(self.target_peak_db),
        }
        target = preset_defaults.get(str(preset), float(self.target_peak_db))
        return min(float(max_peak_limit) - 1.0, float(target))

    def apply_corrections(self, channel_ids: Optional[List[int]] = None) -> bool:
        """
        Применение рекомендованных коррекций к микшеру.

        Args:
            channel_ids: Список каналов для применения (по умолчанию все готовые)

        Returns:
            True если хотя бы одна коррекция была применена
        """
        if self.state != AnalysisState.READY:
            logger.error(f"Cannot apply corrections: state is {self.state}, must be READY")
            return False

        if not self.mixer_client:
            logger.error("Cannot apply corrections: no mixer client")
            return False

        channels_to_apply = channel_ids if channel_ids else list(self.suggestions.keys())

        self.state = AnalysisState.APPLYING

        applied_count = 0
        for ch_id in channels_to_apply:
            if ch_id not in self.suggestions:
                logger.warning(f"No suggestion for channel {ch_id}, skipping")
                continue

            suggestion = self.suggestions[ch_id]
            if suggestion.get('auto_apply_allowed') is False:
                logger.info(f"Channel {ch_id}: SafeGain suggestion queued only; auto apply blocked by policy")
                continue
            gain = suggestion['suggested_gain_db']

            mixer_ch = self.channel_mapping.get(ch_id, ch_id)

            try:
                current_trim = self.mixer_client.get_channel_gain(mixer_ch) or 0.0
                new_trim = suggestion.get('target_trim_db')
                if new_trim is None:
                    new_trim = current_trim + gain

                new_trim = np.clip(new_trim, self.local_input_preamp.min_gain_db, self.local_input_preamp.hard_max_gain_db)

                self.mixer_client.set_channel_gain(mixer_ch, new_trim)
                logger.info(f"Channel {mixer_ch}: TRIM {current_trim:.1f} -> {new_trim:.1f} dB "
                           f"(gain={gain:+.1f}dB)")
                applied_count += 1

            except Exception as e:
                logger.error(f"Failed to apply correction to channel {mixer_ch}: {e}")

        self.state = AnalysisState.IDLE

        logger.info(f"Corrections applied: {applied_count}/{len(channels_to_apply)} channels")

        if self.on_progress_update:
            self.on_progress_update({
                'state': self.state.value,
                'progress': 1.0,
                'message': f'Applied {applied_count} corrections'
            })

        return applied_count > 0

    def reset(self):
        """Сброс калибратора в IDLE состояние."""
        self.state = AnalysisState.IDLE
        for stats in self.channels.values():
            stats.total_samples = 0
            stats.active_samples = 0
            stats.rms_values.clear()
            stats.peak_values.clear()
            stats.inactive_rms_values.clear()
        self.suggestions.clear()
        self.learning_start_time = None
        self.learning_progress = 0.0
        logger.info("Calibrator reset to IDLE state")

    def get_status(self) -> Dict[str, Any]:
        """Получить текущий статус калибратора."""
        return {
            'state': self.state.value,
            'learning_progress': self.learning_progress,
            'channels_count': len(self.channels),
            'suggestions_ready': len(self.suggestions) > 0,
            'target_lufs': self.target_lufs,
            'max_peak_limit': self.max_peak_limit,
            'target_peak_db': self.target_peak_db,
            'learning_duration': self.learning_duration,
            'method_id': self.method_id,
            'default_mode': self.default_mode,
            'local_input_preamp': {
                'enabled': self.local_input_preamp.enabled,
                'character': self.local_input_preamp.character,
                'source': self.local_input_preamp.source,
                'clean_gain_max_db': self.local_input_preamp.clean_gain_max_db,
                'coloration_start_db': self.local_input_preamp.coloration_start_db,
                'saturation_start_db': self.local_input_preamp.saturation_start_db,
                'soundcheck_allow_boost': self.local_input_preamp.soundcheck_allow_boost,
                'live_allow_boost': self.local_input_preamp.live_allow_boost,
                'soundcheck_max_boost_db': self.local_input_preamp.soundcheck_max_boost_db,
                'soundcheck_convergence_enabled': self.local_input_preamp.soundcheck_convergence_enabled,
                'soundcheck_convergence_large_gap_db': self.local_input_preamp.soundcheck_convergence_large_gap_db,
                'soundcheck_convergence_max_boost_db': self.local_input_preamp.soundcheck_convergence_max_boost_db,
                'soundcheck_convergence_vocal_max_boost_db': self.local_input_preamp.soundcheck_convergence_vocal_max_boost_db,
                'soundcheck_convergence_playback_max_boost_db': self.local_input_preamp.soundcheck_convergence_playback_max_boost_db,
                'soundcheck_convergence_setup_check_db': self.local_input_preamp.soundcheck_convergence_setup_check_db,
                'soundcheck_convergence_gain_up_step_db': self.local_input_preamp.soundcheck_convergence_gain_up_step_db,
                'soundcheck_ensemble_context_enabled': self.local_input_preamp.soundcheck_ensemble_context_enabled,
                'soundcheck_ensemble_min_other_active_sources': self.local_input_preamp.soundcheck_ensemble_min_other_active_sources,
                'soundcheck_ensemble_min_other_presence': self.local_input_preamp.soundcheck_ensemble_min_other_presence,
                'soundcheck_ensemble_min_reference_peak_db': self.local_input_preamp.soundcheck_ensemble_min_reference_peak_db,
                'soundcheck_ensemble_min_reference_lufs': self.local_input_preamp.soundcheck_ensemble_min_reference_lufs,
                'soundcheck_ensemble_vocal_under_peak_gap_db': self.local_input_preamp.soundcheck_ensemble_vocal_under_peak_gap_db,
                'soundcheck_ensemble_vocal_under_lufs_gap_db': self.local_input_preamp.soundcheck_ensemble_vocal_under_lufs_gap_db,
                'soundcheck_ensemble_vocal_min_crest_db': self.local_input_preamp.soundcheck_ensemble_vocal_min_crest_db,
                'soundcheck_ensemble_max_boost_db': self.local_input_preamp.soundcheck_ensemble_max_boost_db,
                'soundcheck_ensemble_gain_up_step_db': self.local_input_preamp.soundcheck_ensemble_gain_up_step_db,
                'soundcheck_ensemble_confidence_bonus': self.local_input_preamp.soundcheck_ensemble_confidence_bonus,
                'soundcheck_ensemble_time_coherence_enabled': self.local_input_preamp.soundcheck_ensemble_time_coherence_enabled,
                'soundcheck_ensemble_max_context_lag_samples': self.local_input_preamp.soundcheck_ensemble_max_context_lag_samples,
                'soundcheck_boost_min_presence': self.local_input_preamp.soundcheck_boost_min_presence,
                'soundcheck_boost_max_crest_db': self.local_input_preamp.soundcheck_boost_max_crest_db,
                'soundcheck_boost_min_lufs': self.local_input_preamp.soundcheck_boost_min_lufs,
                'clip_safety_margin_db': self.local_input_preamp.clip_safety_margin_db,
                'isolated_peak_min_spread_db': self.local_input_preamp.isolated_peak_min_spread_db,
                'soundcheck_boost_max_lufs_std_db': self.local_input_preamp.soundcheck_boost_max_lufs_std_db,
                'soundcheck_boost_max_peak_std_db': self.local_input_preamp.soundcheck_boost_max_peak_std_db,
                'soundcheck_boost_max_peak_spread_db': self.local_input_preamp.soundcheck_boost_max_peak_spread_db,
                'soundcheck_boost_min_noise_margin_db': self.local_input_preamp.soundcheck_boost_min_noise_margin_db,
                'recommendation_hysteresis_db': self.local_input_preamp.recommendation_hysteresis_db,
                'opposite_direction_hold_db': self.local_input_preamp.opposite_direction_hold_db,
                'gain_up_min_stable_windows': self.local_input_preamp.gain_up_min_stable_windows,
                'stable_window_tolerance_db': self.local_input_preamp.stable_window_tolerance_db,
                'digital_clip_threshold_db': self.local_input_preamp.digital_clip_threshold_db,
                'scream_peak_threshold_db': self.local_input_preamp.scream_peak_threshold_db,
                'scream_lufs_threshold_db': self.local_input_preamp.scream_lufs_threshold_db,
                'predictive_clip_window_db': self.local_input_preamp.predictive_clip_window_db,
                'predictive_clip_density_threshold': self.local_input_preamp.predictive_clip_density_threshold,
                'predictive_clip_sustained_density_threshold': self.local_input_preamp.predictive_clip_sustained_density_threshold,
                'predictive_clip_min_rising_trend_db': self.local_input_preamp.predictive_clip_min_rising_trend_db,
                'predictive_clip_min_cut_db': self.local_input_preamp.predictive_clip_min_cut_db,
                'predictive_clip_max_cut_db': self.local_input_preamp.predictive_clip_max_cut_db,
                'vocal_gain_up_guard_enabled': self.local_input_preamp.vocal_gain_up_guard_enabled,
                'vocal_gain_up_presence_min_bands': self.local_input_preamp.vocal_gain_up_presence_min_bands,
                'vocal_gain_up_presence_max_std_db': self.local_input_preamp.vocal_gain_up_presence_max_std_db,
                'vocal_gain_up_presence_max_span_db': self.local_input_preamp.vocal_gain_up_presence_max_span_db,
                'vocal_gain_up_scream_cooldown_windows': self.local_input_preamp.vocal_gain_up_scream_cooldown_windows,
                'trim_smoothing_enabled': self.local_input_preamp.trim_smoothing_enabled,
                'trim_smoothing_gain_up_step_db': self.local_input_preamp.trim_smoothing_gain_up_step_db,
                'trim_smoothing_gain_down_step_db': self.local_input_preamp.trim_smoothing_gain_down_step_db,
                'trend_memory_max_windows': self.local_input_preamp.trend_memory_max_windows,
                'behavior_flip_gain_up_hold': self.local_input_preamp.behavior_flip_gain_up_hold,
                'channel_profile_min_windows': self.local_input_preamp.channel_profile_min_windows,
                'channel_profile_max_windows': self.local_input_preamp.channel_profile_max_windows,
                'channel_profile_unstable_ratio_block': self.local_input_preamp.channel_profile_unstable_ratio_block,
                'channel_profile_lufs_std_block_db': self.local_input_preamp.channel_profile_lufs_std_block_db,
                'channel_profile_noise_margin_block_db': self.local_input_preamp.channel_profile_noise_margin_block_db,
                'live_freeze_gain_up_profile': self.local_input_preamp.live_freeze_gain_up_profile,
                'channel_profile_lifecycle_enabled': self.local_input_preamp.channel_profile_lifecycle_enabled,
                'channel_profile_default_decay_factor': self.local_input_preamp.channel_profile_default_decay_factor,
                'channel_profile_min_decay_factor': self.local_input_preamp.channel_profile_min_decay_factor,
                'channel_profile_recovery_enabled': self.local_input_preamp.channel_profile_recovery_enabled,
                'channel_profile_recovery_min_windows': self.local_input_preamp.channel_profile_recovery_min_windows,
                'channel_profile_recovery_max_lufs_std_db': self.local_input_preamp.channel_profile_recovery_max_lufs_std_db,
                'channel_profile_recovery_max_peak_std_db': self.local_input_preamp.channel_profile_recovery_max_peak_std_db,
                'channel_profile_recovery_max_crest_db': self.local_input_preamp.channel_profile_recovery_max_crest_db,
                'channel_profile_recovery_min_noise_margin_db': self.local_input_preamp.channel_profile_recovery_min_noise_margin_db,
                'source_signature_change_hold_enabled': self.local_input_preamp.source_signature_change_hold_enabled,
                'source_signature_change_min_fields': self.local_input_preamp.source_signature_change_min_fields,
                'decision_confidence_enabled': self.local_input_preamp.decision_confidence_enabled,
                'decision_confidence_gain_up_min_score': self.local_input_preamp.decision_confidence_gain_up_min_score,
                'decision_confidence_hysteresis_enabled': self.local_input_preamp.decision_confidence_hysteresis_enabled,
                'decision_confidence_borderline_margin': self.local_input_preamp.decision_confidence_borderline_margin,
                'decision_confidence_stable_windows': self.local_input_preamp.decision_confidence_stable_windows,
                'decision_confidence_score_tolerance': self.local_input_preamp.decision_confidence_score_tolerance,
                'decision_confidence_time_memory_enabled': self.local_input_preamp.decision_confidence_time_memory_enabled,
                'decision_confidence_default_sample_sec': self.local_input_preamp.decision_confidence_default_sample_sec,
                'decision_confidence_min_window_sec': self.local_input_preamp.decision_confidence_min_window_sec,
                'decision_confidence_max_age_sec': self.local_input_preamp.decision_confidence_max_age_sec,
                'decision_confidence_max_gap_ratio': self.local_input_preamp.decision_confidence_max_gap_ratio,
                'decision_confidence_min_memory_strength': self.local_input_preamp.decision_confidence_min_memory_strength,
                'decision_confidence_decay_half_life_sec': self.local_input_preamp.decision_confidence_decay_half_life_sec,
                'decision_confidence_vocal_decay_half_life_sec': self.local_input_preamp.decision_confidence_vocal_decay_half_life_sec,
                'decision_confidence_playback_decay_half_life_sec': self.local_input_preamp.decision_confidence_playback_decay_half_life_sec,
                'source_signature_drift_memory_penalty_enabled': self.local_input_preamp.source_signature_drift_memory_penalty_enabled,
                'source_signature_drift_penalty_per_field': self.local_input_preamp.source_signature_drift_penalty_per_field,
                'source_signature_drift_max_penalty': self.local_input_preamp.source_signature_drift_max_penalty,
                'source_signature_drift_vocal_noise_weight': self.local_input_preamp.source_signature_drift_vocal_noise_weight,
                'source_signature_drift_vocal_crest_weight': self.local_input_preamp.source_signature_drift_vocal_crest_weight,
                'source_signature_drift_playback_lufs_weight': self.local_input_preamp.source_signature_drift_playback_lufs_weight,
                'source_signature_drift_playback_peak_gap_weight': self.local_input_preamp.source_signature_drift_playback_peak_gap_weight,
                'source_signature_continuous_drift_enabled': self.local_input_preamp.source_signature_continuous_drift_enabled,
                'source_signature_continuous_drift_penalty_per_unit': self.local_input_preamp.source_signature_continuous_drift_penalty_per_unit,
                'source_signature_continuous_drift_max_penalty': self.local_input_preamp.source_signature_continuous_drift_max_penalty,
                'source_signature_spectral_drift_enabled': self.local_input_preamp.source_signature_spectral_drift_enabled,
                'source_signature_spectral_drift_min_common_bands': self.local_input_preamp.source_signature_spectral_drift_min_common_bands,
                'source_signature_spectral_drift_penalty_per_db': self.local_input_preamp.source_signature_spectral_drift_penalty_per_db,
                'source_signature_spectral_drift_max_penalty': self.local_input_preamp.source_signature_spectral_drift_max_penalty,
                'source_signature_spectral_vocal_presence_weight': self.local_input_preamp.source_signature_spectral_vocal_presence_weight,
                'source_signature_spectral_playback_weight': self.local_input_preamp.source_signature_spectral_playback_weight,
                'source_signature_spectral_drum_high_band_weight': self.local_input_preamp.source_signature_spectral_drum_high_band_weight,
                'source_signature_spectral_quality_gate_enabled': self.local_input_preamp.source_signature_spectral_quality_gate_enabled,
                'source_signature_spectral_min_band_samples': self.local_input_preamp.source_signature_spectral_min_band_samples,
                'source_signature_spectral_min_band_coverage': self.local_input_preamp.source_signature_spectral_min_band_coverage,
                'source_signature_spectral_max_band_std_db': self.local_input_preamp.source_signature_spectral_max_band_std_db,
                'source_signature_spectral_max_band_span_db': self.local_input_preamp.source_signature_spectral_max_band_span_db,
                'source_signature_spectral_stuck_band_std_db': self.local_input_preamp.source_signature_spectral_stuck_band_std_db,
                'source_signature_spectral_stuck_signal_std_db': self.local_input_preamp.source_signature_spectral_stuck_signal_std_db,
                'source_signature_spectral_quality_penalty_per_rejected_band': self.local_input_preamp.source_signature_spectral_quality_penalty_per_rejected_band,
                'source_signature_spectral_quality_max_penalty': self.local_input_preamp.source_signature_spectral_quality_max_penalty,
                'source_signature_spectral_vocal_presence_max_std_db': self.local_input_preamp.source_signature_spectral_vocal_presence_max_std_db,
                'source_signature_spectral_vocal_presence_max_span_db': self.local_input_preamp.source_signature_spectral_vocal_presence_max_span_db,
                'source_signature_spectral_playback_max_band_std_db': self.local_input_preamp.source_signature_spectral_playback_max_band_std_db,
                'source_signature_spectral_playback_max_band_span_db': self.local_input_preamp.source_signature_spectral_playback_max_band_span_db,
                'source_signature_spectral_transient_high_band_max_std_db': self.local_input_preamp.source_signature_spectral_transient_high_band_max_std_db,
                'source_signature_spectral_transient_high_band_max_span_db': self.local_input_preamp.source_signature_spectral_transient_high_band_max_span_db,
                'source_signature_spectral_transient_high_band_identity_weight': self.local_input_preamp.source_signature_spectral_transient_high_band_identity_weight,
                'source_signature_spectral_transient_attack_high_band_identity_weight': self.local_input_preamp.source_signature_spectral_transient_attack_high_band_identity_weight,
                'transient_window_profile_enabled': self.local_input_preamp.transient_window_profile_enabled,
                'transient_window_min_samples': self.local_input_preamp.transient_window_min_samples,
                'transient_window_attack_top_ratio': self.local_input_preamp.transient_window_attack_top_ratio,
                'transient_window_attack_body_gap_db': self.local_input_preamp.transient_window_attack_body_gap_db,
                'transient_window_high_band_lift_db': self.local_input_preamp.transient_window_high_band_lift_db,
                'transient_window_min_repeated_hits': self.local_input_preamp.transient_window_min_repeated_hits,
                'transient_window_hit_body_std_max_db': self.local_input_preamp.transient_window_hit_body_std_max_db,
                'transient_window_hit_gap_std_max_db': self.local_input_preamp.transient_window_hit_gap_std_max_db,
                'transient_window_body_reference_enabled': self.local_input_preamp.transient_window_body_reference_enabled,
                'transient_window_source_specific_body_band_profiles': self.local_input_preamp.transient_window_source_specific_body_band_profiles,
                'transient_window_body_band_reference_enabled': self.local_input_preamp.transient_window_body_band_reference_enabled,
                'transient_window_body_band_std_max_db': self.local_input_preamp.transient_window_body_band_std_max_db,
            }
        }
    
    def update_settings(self, settings: Dict[str, Any]):
        """Обновить настройки калибратора."""
        if 'learning_duration_sec' in settings:
            new_duration = float(settings['learning_duration_sec'])
            if new_duration > 0:
                self.learning_duration = new_duration
                logger.info(f"Updated learning_duration to {self.learning_duration} seconds")
                # Если анализ уже идет, пересчитываем прогресс
                if self.state == AnalysisState.LEARNING and self.learning_start_time:
                    elapsed = time.time() - self.learning_start_time
                    self.learning_progress = min(100.0, (elapsed / self.learning_duration) * 100.0)
        
        if 'target_lufs' in settings:
            self.target_lufs = float(settings['target_lufs'])
            logger.info(f"Updated target_lufs to {self.target_lufs}")
        
        if 'max_peak_limit' in settings:
            self.max_peak_limit = float(settings['max_peak_limit'])
            logger.info(f"Updated max_peak_limit to {self.max_peak_limit}")

        if 'target_peak_db' in settings:
            self.target_peak_db = float(settings['target_peak_db'])
            logger.info(f"Updated target_peak_db to {self.target_peak_db}")

        if 'default_mode' in settings or 'mode' in settings:
            self.default_mode = str(settings.get('default_mode') or settings.get('mode') or self.default_mode)
            logger.info(f"Updated SafeGain mode to {self.default_mode}")

        if 'method_id' in settings:
            self.method_id = str(settings['method_id'] or INPUT_TRIM_AUTOGAIN_METHOD_ID)
            logger.info(f"Updated SafeGain method_id to {self.method_id}")

        if isinstance(settings.get('local_input_preamp'), dict):
            self.local_input_preamp = LocalInputGainProfile.from_config(settings['local_input_preamp'])
            logger.info(
                f"Updated local input preamp profile: character={self.local_input_preamp.character}, "
                f"clean_max={self.local_input_preamp.clean_gain_max_db}, "
                f"saturation={self.local_input_preamp.saturation_start_db}"
            )
