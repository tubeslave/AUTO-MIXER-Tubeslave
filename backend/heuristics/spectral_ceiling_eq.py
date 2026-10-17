"""Role-aware spectral ceiling EQ proposals.

This module implements a conservative, explainable EQ suggestion layer. It is
inspired by public spectral-ceiling/noise-slope guide ideas, but it deliberately
avoids match EQ and proprietary behavior. It returns structured proposals that
the existing live safety layer or offline renderer can accept, merge, or skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "spectral_ceiling_profiles.yaml"
)

GUIDE_SLOPES_DB_PER_OCT = {
    "white": 0.0,
    "pink": -3.0,
    "brown": -6.0,
    "custom": None,
}

POSITION_RULES: Dict[str, Dict[str, Any]] = {
    "foreground": {
        "tilt_offset_db_per_oct": 1.0,
        "presence_priority": "high",
    },
    "midground": {
        "tilt_offset_db_per_oct": 0.0,
        "presence_priority": "medium",
    },
    "background": {
        "tilt_offset_db_per_oct": -1.5,
        "presence_priority": "low",
    },
}

ROLE_ALIASES = {
    "lead": "lead_vocal",
    "leadvox": "lead_vocal",
    "leadvocal": "lead_vocal",
    "lead_vocal": "lead_vocal",
    "main_vocal": "lead_vocal",
    "vocal_main": "lead_vocal",
    "vocal": "lead_vocal",
    "vocals": "lead_vocal",
    "backvocal": "backing_vocal",
    "back_vocal": "backing_vocal",
    "backing": "backing_vocal",
    "backing_vocal": "backing_vocal",
    "bgv": "backing_vocal",
    "bvox": "backing_vocal",
    "harmony": "backing_vocal",
    "electricguitar": "electric_guitar",
    "guitar": "electric_guitar",
    "guitars": "electric_guitar",
    "lead_guitar": "electric_guitar",
    "rhythm_guitar": "electric_guitar",
    "acousticguitar": "acoustic_guitar",
    "acoustic_guitar": "acoustic_guitar",
    "tom": "tom",
    "rack_tom": "tom",
    "ftom": "floor_tom",
    "f_tom": "floor_tom",
    "floor_tom": "floor_tom",
    "hihat": "hihat",
    "hi_hat": "hihat",
    "hi-hat": "hihat",
    "hh": "hihat",
    "hat": "hihat",
    "ride": "ride",
    "synth": "synth",
    "lead_synth": "synth",
    "percussion": "percussion",
    "perc": "percussion",
    "pad": "playback",
    "pads": "playback",
    "synth_percussion_pad": "playback",
    "synth_perc_pad": "playback",
    "synth+percussion+pad": "playback",
    "piano": "keys",
    "organ": "keys",
    "overhead": "overheads",
    "oh": "overheads",
    "oh_l": "overheads",
    "oh_r": "overheads",
    "cymbals": "overheads",
    "drums": "drums_bus",
    "drum_bus": "drums_bus",
    "drums_bus": "drums_bus",
    "drum_group": "drums_bus",
    "fx": "fx_return",
    "fx_return": "fx_return",
    "reverb_return": "fx_return",
    "delay_return": "fx_return",
    "playback": "playback",
    "track": "playback",
    "tracks": "playback",
    "trax": "playback",
    "master": "mix_bus",
    "master_bus": "mix_bus",
    "mix": "mix_bus",
    "mix_bus": "mix_bus",
}

VOCAL_COMPETING_ROLES = {
    "backing_vocal",
    "electric_guitar",
    "acoustic_guitar",
    "keys",
    "fx_return",
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _to_float_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low = float(value[0])
        high = float(value[1])
        return (min(low, high), max(low, high))
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, number)
    return default


def _as_zone_map(value: Any) -> Dict[str, Tuple[float, float]]:
    if not isinstance(value, Mapping):
        return {}
    zones: Dict[str, Tuple[float, float]] = {}
    for name, bounds in value.items():
        zones[str(name)] = _to_float_pair(bounds, (20.0, 20000.0))
    return zones


def _center_freq(range_hz: Tuple[float, float]) -> float:
    low, high = range_hz
    return float(math.sqrt(max(low, 1.0) * max(high, 1.0)))


def _q_for_range(range_hz: Tuple[float, float], max_q: float) -> float:
    low, high = range_hz
    center = _center_freq(range_hz)
    width = max(1.0, high - low)
    return _clamp(center / width, 0.5, max_q)


def normalize_role(role: str | None) -> str:
    """Normalize channel/stem labels into profile role names."""
    normalized = str(role or "unknown").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return ROLE_ALIASES.get(normalized, normalized)


def guide_slope_db_per_oct(guide: str, custom_slope_db_per_oct: float) -> float:
    """Return the dB/octave slope for a named noise guide."""
    key = str(guide or "custom").lower()
    if key not in GUIDE_SLOPES_DB_PER_OCT:
        raise ValueError(f"unknown noise-slope guide: {guide}")
    slope = GUIDE_SLOPES_DB_PER_OCT[key]
    return float(custom_slope_db_per_oct if slope is None else slope)


def generate_noise_slope_curve(
    frequencies_hz: np.ndarray,
    *,
    slope_db_per_oct: float,
    reference_freq_hz: float = 1000.0,
    reference_db: float = 0.0,
) -> np.ndarray:
    """Build a white/pink/brown/custom target curve around a reference point."""
    freqs = np.maximum(np.asarray(frequencies_hz, dtype=np.float64), EPS)
    reference = max(float(reference_freq_hz), EPS)
    return (
        float(reference_db)
        + float(slope_db_per_oct) * np.log2(freqs / reference)
    ).astype(np.float64)


@dataclass
class SpectralCeilingEQConfig:
    enabled: bool = True
    dry_run: bool = False
    correction_strength: float = 0.4
    smoothing_octaves: float = 0.333
    reference_freq_hz: float = 1000.0
    max_bands_per_track: int = 5
    max_abs_gain_db: float = 3.0
    min_confidence_to_apply: float = 0.65
    allow_master_bus_eq: bool = True
    master_bus_max_abs_gain_db: float = 1.0
    max_q: float = 2.0
    min_band_gain_db: float = 0.25
    log_verbose: bool = True
    mix_amount: float = 1.0
    profiles_path: str = str(DEFAULT_PROFILE_PATH)

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> "SpectralCeilingEQConfig":
        payload = dict(payload or {})
        max_bands = payload.get("max_bands_per_track", payload.get("max_total_bands", 5))
        return cls(
            enabled=bool(payload.get("enabled", True)),
            dry_run=bool(payload.get("dry_run", False)),
            correction_strength=_clamp(float(payload.get("correction_strength", 0.4)), 0.0, 1.0),
            smoothing_octaves=max(0.05, float(payload.get("smoothing_octaves", 0.333))),
            reference_freq_hz=max(20.0, float(payload.get("reference_freq_hz", 1000.0))),
            max_bands_per_track=max(0, int(max_bands)),
            max_abs_gain_db=max(0.0, float(payload.get("max_abs_gain_db", 3.0))),
            min_confidence_to_apply=_clamp(
                float(payload.get("min_confidence_to_apply", 0.65)),
                0.0,
                1.0,
            ),
            allow_master_bus_eq=bool(payload.get("allow_master_bus_eq", True)),
            master_bus_max_abs_gain_db=max(
                0.0,
                float(payload.get("master_bus_max_abs_gain_db", 1.0)),
            ),
            max_q=max(0.5, float(payload.get("max_q", 2.0))),
            min_band_gain_db=max(0.0, float(payload.get("min_band_gain_db", 0.25))),
            log_verbose=bool(payload.get("log_verbose", True)),
            mix_amount=_clamp(
                float(payload.get("mix_amount", payload.get("dry_wet", 1.0))),
                0.0,
                1.0,
            ),
            profiles_path=str(payload.get("profiles_path", DEFAULT_PROFILE_PATH)),
        )


@dataclass(frozen=True)
class SpectralCeilingProfile:
    role: str
    description: str = ""
    target_profile: str = ""
    guide: str = "custom"
    slope_db_per_oct: float = -3.0
    low_cut_hz: Tuple[float, float] = (20.0, 30.0)
    high_cut_hz: Tuple[float, float] = (16000.0, 20000.0)
    important_zones: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    avoid: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_eq_gain_db: float = 3.0
    max_eq_cut_db: float = -4.0
    front_back_position: str = "midground"

    @property
    def effective_target_name(self) -> str:
        return self.target_profile or self.role

    @property
    def position_rule(self) -> Dict[str, Any]:
        return POSITION_RULES.get(self.front_back_position, POSITION_RULES["midground"])

    @property
    def effective_slope_db_per_oct(self) -> float:
        return float(self.slope_db_per_oct) + float(
            self.position_rule.get("tilt_offset_db_per_oct", 0.0)
        )

    @classmethod
    def from_mapping(cls, role: str, payload: Mapping[str, Any]) -> "SpectralCeilingProfile":
        normalized_role = normalize_role(role)
        guide = str(payload.get("guide", "custom")).lower()
        raw_slope = float(payload.get("slope_db_per_oct", -3.0))
        slope = guide_slope_db_per_oct(guide, raw_slope)
        return cls(
            role=normalized_role,
            description=str(payload.get("description", "")),
            target_profile=str(payload.get("target_profile", normalized_role)),
            guide=guide,
            slope_db_per_oct=slope,
            low_cut_hz=_to_float_pair(payload.get("low_cut_hz"), (20.0, 30.0)),
            high_cut_hz=_to_float_pair(payload.get("high_cut_hz"), (16000.0, 20000.0)),
            important_zones=_as_zone_map(payload.get("important_zones", {})),
            avoid=_as_zone_map(payload.get("avoid", {})),
            max_eq_gain_db=max(0.0, float(payload.get("max_eq_gain_db", 3.0))),
            max_eq_cut_db=min(0.0, float(payload.get("max_eq_cut_db", -4.0))),
            front_back_position=str(payload.get("front_back_position", "midground")),
        )


@dataclass
class SpectralEQBandSuggestion:
    name: str
    range_hz: Tuple[float, float]
    action: str
    gain_db: float
    reason: str
    freq_hz: float = 0.0
    q: float = 1.0
    rule: str = "spectral_ceiling"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "range_hz": [float(self.range_hz[0]), float(self.range_hz[1])],
            "action": self.action,
            "gain_db": round(float(self.gain_db), 3),
            "freq_hz": round(float(self.freq_hz or _center_freq(self.range_hz)), 3),
            "q": round(float(self.q), 3),
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass
class SpectralCeilingEQProposal:
    enabled: bool
    track_id: str
    role: str
    target_profile: str
    guide: str
    slope_db_per_oct: float
    measured_tilt_db_per_oct: float
    target_tilt_db_per_oct: float
    low_cut_hz: Optional[float]
    high_cut_hz: Optional[float]
    bands: List[SpectralEQBandSuggestion] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    safety: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    applied: bool = False
    dry_run: bool = False
    warnings: List[str] = field(default_factory=list)
    spectral_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_apply(self) -> bool:
        return bool(self.enabled and self.applied and not self.dry_run)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "track_id": self.track_id,
            "role": self.role,
            "target_profile": self.target_profile,
            "guide": self.guide,
            "slope_db_per_oct": round(float(self.slope_db_per_oct), 3),
            "measured_tilt": round(float(self.measured_tilt_db_per_oct), 3),
            "target_tilt": round(float(self.target_tilt_db_per_oct), 3),
            "low_cut_hz": self.low_cut_hz,
            "high_cut_hz": self.high_cut_hz,
            "bands": [band.to_dict() for band in self.bands],
            "skipped": list(self.skipped),
            "safety": dict(self.safety),
            "confidence": round(float(self.confidence), 3),
            "applied": bool(self.should_apply),
            "dry_run": bool(self.dry_run),
            "warnings": list(self.warnings),
            "spectral_summary": dict(self.spectral_summary),
        }


def load_spectral_ceiling_profiles(
    path: str | Path | None = None,
) -> Dict[str, SpectralCeilingProfile]:
    """Load role profiles from YAML."""
    profile_path = Path(path or DEFAULT_PROFILE_PATH).expanduser()
    if not profile_path.is_absolute() and not profile_path.exists():
        profile_path = Path(__file__).resolve().parents[2] / profile_path
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is in project deps.
        raise RuntimeError("PyYAML is required to load spectral ceiling profiles") from exc

    with profile_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    profile_payload = payload.get("profiles", payload)
    if not isinstance(profile_payload, Mapping):
        raise ValueError(f"Invalid spectral ceiling profile file: {profile_path}")

    profiles: Dict[str, SpectralCeilingProfile] = {}
    for role, data in profile_payload.items():
        if not isinstance(data, Mapping):
            continue
        profile = SpectralCeilingProfile.from_mapping(str(role), data)
        profiles[profile.role] = profile
    return profiles


def select_spectral_ceiling_profile(
    role: str | None,
    profiles: Mapping[str, SpectralCeilingProfile],
) -> SpectralCeilingProfile:
    """Select the best profile for a role, falling back to mix bus/unknown."""
    normalized = normalize_role(role)
    if normalized in profiles:
        return profiles[normalized]
    if normalized in {"unknown", ""} and "unknown" in profiles:
        return profiles["unknown"]
    if normalized.endswith("_vocal") and "lead_vocal" in profiles:
        return profiles["lead_vocal"]
    if "mix_bus" in profiles:
        return profiles["mix_bus"]
    if profiles:
        return next(iter(profiles.values()))
    return SpectralCeilingProfile(role="unknown", target_profile="unknown")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 2:
        axis = 1 if data.shape[0] > data.shape[1] else 0
        return np.nan_to_num(np.mean(data, axis=axis).astype(np.float32))
    return np.nan_to_num(data.reshape(-1).astype(np.float32))


def smoothed_log_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    *,
    smoothing_octaves: float = 0.333,
    min_hz: float = 20.0,
    max_hz: float = 20000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a perceptually smoothed LTAS on logarithmic frequency bins."""
    data = _to_mono(audio)
    if data.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    data = data - float(np.mean(data))
    n_fft = min(16384, max(1024, 2 ** int(math.ceil(math.log2(max(2, min(data.size, 16384)))))))
    if data.size < n_fft:
        data = np.pad(data, (0, n_fft - data.size))
    hop = max(512, n_fft // 2)
    window = np.hanning(n_fft).astype(np.float32)
    powers = []
    for start in range(0, max(1, data.size - n_fft + 1), hop):
        frame = data[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spectrum = np.fft.rfft(frame * window)
        powers.append(np.square(np.abs(spectrum)).astype(np.float64) + EPS)
    if not powers:
        spectrum = np.fft.rfft(data[:n_fft] * window)
        powers.append(np.square(np.abs(spectrum)).astype(np.float64) + EPS)

    mean_power = np.mean(np.vstack(powers), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / int(sample_rate))
    valid = (freqs >= min_hz) & (freqs <= min(max_hz, sample_rate * 0.49))
    freqs = freqs[valid]
    mean_power = mean_power[valid]
    if freqs.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    bins_per_octave = max(6, int(round(1.0 / max(smoothing_octaves, 0.05) * 4)))
    count = max(24, int(math.ceil(math.log2(freqs[-1] / freqs[0]) * bins_per_octave)))
    centers = np.geomspace(freqs[0], freqs[-1], count)
    half_width = max(0.05, smoothing_octaves) / 2.0
    smoothed = []
    for center in centers:
        low = center / (2.0 ** half_width)
        high = center * (2.0 ** half_width)
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(freqs - center)))
            value = mean_power[idx]
        else:
            value = float(np.mean(mean_power[mask]))
        smoothed.append(10.0 * math.log10(max(value, EPS)))
    return centers.astype(np.float64), np.asarray(smoothed, dtype=np.float64)


def estimate_spectral_tilt_db_per_oct(
    frequencies_hz: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    low_hz: float = 80.0,
    high_hz: float = 14000.0,
) -> float:
    """Estimate broad dB/oct spectral tilt from a smoothed spectrum."""
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    mags = np.asarray(magnitude_db, dtype=np.float64)
    valid = (
        (freqs >= low_hz)
        & (freqs <= high_hz)
        & np.isfinite(freqs)
        & np.isfinite(mags)
    )
    if int(np.sum(valid)) < 4:
        return 0.0
    return float(np.polyfit(np.log2(freqs[valid]), mags[valid], 1)[0])


def _reference_db(
    frequencies_hz: np.ndarray,
    magnitude_db: np.ndarray,
    reference_freq_hz: float,
) -> float:
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    mags = np.asarray(magnitude_db, dtype=np.float64)
    low = reference_freq_hz / math.sqrt(2.0)
    high = reference_freq_hz * math.sqrt(2.0)
    mask = (freqs >= low) & (freqs <= high)
    if np.any(mask):
        return float(np.median(mags[mask]))
    if freqs.size:
        return float(np.interp(reference_freq_hz, freqs, mags))
    return -90.0


def _zone_mean(
    frequencies_hz: np.ndarray,
    values_db: np.ndarray,
    range_hz: Tuple[float, float],
) -> float:
    low, high = range_hz
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    values = np.asarray(values_db, dtype=np.float64)
    mask = (freqs >= low) & (freqs <= high)
    if np.any(mask):
        return float(np.mean(values[mask]))
    center = _center_freq(range_hz)
    if freqs.size:
        return float(np.interp(center, freqs, values))
    return -90.0


def _confidence_for_audio(
    audio: np.ndarray,
    sample_rate: int,
    role_confidence: float,
    role: str,
) -> Tuple[float, List[str]]:
    data = _to_mono(audio)
    warnings: List[str] = []
    if data.size == 0:
        return 0.0, ["empty audio buffer"]
    rms = float(np.sqrt(np.mean(np.square(data)) + EPS))
    rms_db = 20.0 * math.log10(max(rms, EPS))
    signal_conf = _clamp((rms_db + 70.0) / 35.0, 0.0, 1.0)
    duration_conf = _clamp((data.size / max(1, sample_rate)) / 0.25, 0.0, 1.0)
    role_conf = _clamp(float(role_confidence), 0.0, 1.0)
    if normalize_role(role) == "unknown":
        role_conf *= 0.35
        warnings.append("role detection uncertain")
    if rms_db < -65.0:
        warnings.append("signal level too low for confident EQ")
    confidence = min(role_conf, 0.65 * signal_conf + 0.35 * duration_conf)
    return _clamp(confidence, 0.0, 1.0), warnings


class SpectralCeilingEQAnalyzer:
    """Analyze one track/stem and return a bounded spectral EQ proposal."""

    def __init__(
        self,
        config: SpectralCeilingEQConfig | Mapping[str, Any] | None = None,
        profiles: Mapping[str, SpectralCeilingProfile] | None = None,
    ):
        self.config = (
            config
            if isinstance(config, SpectralCeilingEQConfig)
            else SpectralCeilingEQConfig.from_mapping(config)
        )
        self.profiles = dict(profiles or load_spectral_ceiling_profiles(self.config.profiles_path))

    def analyze(
        self,
        audio: np.ndarray,
        *,
        instrument_role: str,
        sample_rate: int,
        track_id: str = "",
        role_confidence: float = 1.0,
        lead_vocal_active: bool = False,
        lead_vocal_confidence: float = 0.0,
    ) -> SpectralCeilingEQProposal:
        cfg = self.config
        role = normalize_role(instrument_role)
        profile = select_spectral_ceiling_profile(role, self.profiles)
        safety = self._safety_for_profile(profile)
        if not cfg.enabled:
            return SpectralCeilingEQProposal(
                enabled=False,
                track_id=str(track_id),
                role=role,
                target_profile=profile.effective_target_name,
                guide=profile.guide,
                slope_db_per_oct=profile.slope_db_per_oct,
                measured_tilt_db_per_oct=0.0,
                target_tilt_db_per_oct=profile.effective_slope_db_per_oct,
                low_cut_hz=None,
                high_cut_hz=None,
                safety=safety,
                confidence=0.0,
                applied=False,
                dry_run=cfg.dry_run,
                skipped=[{"reason": "spectral_ceiling_eq disabled by config"}],
            )

        frequencies, measured_db = smoothed_log_spectrum(
            audio,
            sample_rate,
            smoothing_octaves=cfg.smoothing_octaves,
        )
        confidence, warnings = _confidence_for_audio(
            audio,
            sample_rate,
            role_confidence,
            role,
        )
        if frequencies.size == 0:
            warnings.append("no usable spectral bins")
            measured_tilt = 0.0
            target_db = np.asarray([], dtype=np.float64)
        else:
            measured_tilt = estimate_spectral_tilt_db_per_oct(frequencies, measured_db)
            reference_db = _reference_db(frequencies, measured_db, cfg.reference_freq_hz)
            target_db = generate_noise_slope_curve(
                frequencies,
                slope_db_per_oct=profile.effective_slope_db_per_oct,
                reference_freq_hz=cfg.reference_freq_hz,
                reference_db=reference_db,
            )

        proposal = SpectralCeilingEQProposal(
            enabled=True,
            track_id=str(track_id),
            role=role,
            target_profile=profile.effective_target_name,
            guide=profile.guide,
            slope_db_per_oct=profile.slope_db_per_oct,
            measured_tilt_db_per_oct=measured_tilt,
            target_tilt_db_per_oct=profile.effective_slope_db_per_oct,
            low_cut_hz=self._low_cut_decision(profile, frequencies, measured_db, target_db),
            high_cut_hz=self._high_cut_decision(profile, frequencies, measured_db, target_db),
            safety=safety,
            confidence=confidence,
            dry_run=cfg.dry_run,
            warnings=warnings,
            spectral_summary=self._spectral_summary(frequencies, measured_db, target_db),
        )

        if confidence < cfg.min_confidence_to_apply:
            proposal.skipped.append(
                {
                    "reason": "confidence below apply threshold",
                    "confidence": round(confidence, 3),
                    "threshold": cfg.min_confidence_to_apply,
                }
            )

        if role == "mix_bus" and not cfg.allow_master_bus_eq:
            proposal.skipped.append({"reason": "master/mix bus EQ disabled by config"})

        if frequencies.size:
            self._add_zone_suggestions(proposal, profile, frequencies, measured_db, target_db)
            self._add_vocal_priority_demasking(
                proposal,
                profile,
                frequencies,
                measured_db,
                target_db,
                lead_vocal_active=lead_vocal_active,
                lead_vocal_confidence=lead_vocal_confidence,
            )

        proposal.bands.sort(key=lambda band: abs(band.gain_db), reverse=True)
        proposal.bands = proposal.bands[: cfg.max_bands_per_track]
        proposal.applied = bool(
            proposal.bands
            and confidence >= cfg.min_confidence_to_apply
            and not cfg.dry_run
            and (role != "mix_bus" or cfg.allow_master_bus_eq)
        )
        if cfg.dry_run:
            proposal.skipped.append({"reason": "dry_run true; proposal logged only"})
        return proposal

    def _safety_for_profile(self, profile: SpectralCeilingProfile) -> Dict[str, Any]:
        max_abs = min(
            self.config.max_abs_gain_db,
            max(abs(profile.max_eq_cut_db), abs(profile.max_eq_gain_db)),
        )
        if profile.role == "mix_bus":
            max_abs = min(max_abs, self.config.master_bus_max_abs_gain_db)
        return {
            "max_abs_gain_db": float(max_abs),
            "max_boost_db": float(min(profile.max_eq_gain_db, max_abs)),
            "max_cut_db": float(max(profile.max_eq_cut_db, -max_abs)),
            "max_q": float(min(self.config.max_q, 2.0)),
            "phase_safe": True,
            "no_narrow_aggressive_eq": True,
        }

    def _bounded_correction(
        self,
        delta_db: float,
        profile: SpectralCeilingProfile,
        action: str,
    ) -> float:
        safety = self._safety_for_profile(profile)
        if action == "lift":
            low, high = 0.0, float(safety["max_boost_db"])
        else:
            low, high = float(safety["max_cut_db"]), 0.0
        correction = _clamp(delta_db, low, high) * self.config.correction_strength
        correction *= self.config.mix_amount
        return round(_clamp(correction, low, high), 3)

    def _append_band(
        self,
        proposal: SpectralCeilingEQProposal,
        profile: SpectralCeilingProfile,
        *,
        name: str,
        range_hz: Tuple[float, float],
        action: str,
        raw_delta_db: float,
        reason: str,
        rule: str,
    ) -> None:
        gain = self._bounded_correction(raw_delta_db, profile, action)
        if abs(gain) < self.config.min_band_gain_db:
            proposal.skipped.append(
                {
                    "zone": name,
                    "reason": "correction below minimum useful move",
                    "suggested_gain_db": gain,
                }
            )
            return
        proposal.bands.append(
            SpectralEQBandSuggestion(
                name=name,
                range_hz=range_hz,
                action=action,
                gain_db=gain,
                freq_hz=_center_freq(range_hz),
                q=_q_for_range(range_hz, float(proposal.safety.get("max_q", 2.0))),
                rule=rule,
                reason=reason,
            )
        )

    def _add_zone_suggestions(
        self,
        proposal: SpectralCeilingEQProposal,
        profile: SpectralCeilingProfile,
        frequencies: np.ndarray,
        measured_db: np.ndarray,
        target_db: np.ndarray,
    ) -> None:
        handled_ranges: List[Tuple[float, float]] = []
        for zone, range_hz in profile.avoid.items():
            measured = _zone_mean(frequencies, measured_db, range_hz)
            target = _zone_mean(frequencies, target_db, range_hz)
            excess = measured - target
            if excess < 1.0:
                proposal.skipped.append(
                    {
                        "zone": zone,
                        "reason": "avoid zone within spectral ceiling",
                        "excess_db": round(excess, 3),
                    }
                )
                continue
            self._append_band(
                proposal,
                profile,
                name=zone,
                range_hz=range_hz,
                action="attenuate",
                raw_delta_db=target - measured,
                reason="energy exceeds target ceiling",
                rule="avoid_zone_ceiling",
            )
            handled_ranges.append(range_hz)

        presence_priority = str(profile.position_rule.get("presence_priority", "medium"))
        for zone, range_hz in profile.important_zones.items():
            measured = _zone_mean(frequencies, measured_db, range_hz)
            target = _zone_mean(frequencies, target_db, range_hz)
            deficit = target - measured
            if deficit <= 1.0:
                continue
            if presence_priority == "low" and _range_intersects(range_hz, (1000.0, 6000.0)):
                proposal.skipped.append(
                    {
                        "zone": zone,
                        "reason": "background role avoids presence boost",
                        "deficit_db": round(deficit, 3),
                    }
                )
                continue
            if any(_range_intersects(range_hz, existing) for existing in handled_ranges):
                continue
            self._append_band(
                proposal,
                profile,
                name=zone,
                range_hz=range_hz,
                action="lift",
                raw_delta_db=deficit,
                reason="below role spectral target",
                rule="important_zone_support",
            )

    def _add_vocal_priority_demasking(
        self,
        proposal: SpectralCeilingEQProposal,
        profile: SpectralCeilingProfile,
        frequencies: np.ndarray,
        measured_db: np.ndarray,
        target_db: np.ndarray,
        *,
        lead_vocal_active: bool,
        lead_vocal_confidence: float,
    ) -> None:
        if proposal.role not in VOCAL_COMPETING_ROLES:
            return
        if not lead_vocal_active or lead_vocal_confidence < 0.65:
            proposal.skipped.append(
                {
                    "rule": "vocal_priority_demasking",
                    "reason": "lead vocal not active or confidence too low",
                }
            )
            return
        range_hz = (
            profile.avoid.get("vocal_conflict")
            or profile.avoid.get("lead_conflict")
            or (1500.0, 4500.0)
        )
        measured = _zone_mean(frequencies, measured_db, range_hz)
        target = _zone_mean(frequencies, target_db, range_hz)
        excess = max(1.0, measured - target + 0.5)
        self._append_band(
            proposal,
            profile,
            name="vocal_priority_demasking",
            range_hz=range_hz,
            action="attenuate",
            raw_delta_db=-min(3.0, excess),
            reason="instrument presence overlaps active vocal intelligibility band",
            rule="vocal_priority_demasking",
        )

    def _low_cut_decision(
        self,
        profile: SpectralCeilingProfile,
        frequencies: np.ndarray,
        measured_db: np.ndarray,
        target_db: np.ndarray,
    ) -> Optional[float]:
        if frequencies.size == 0:
            return None
        low, high = profile.low_cut_hz
        below = (20.0, max(low, 20.0))
        if below[1] <= below[0]:
            return float(low)
        excess = (
            _zone_mean(frequencies, measured_db, below)
            - _zone_mean(frequencies, target_db, below)
        )
        if excess > 1.5:
            return round(float(high), 1)
        return round(float((low + high) / 2.0), 1)

    def _high_cut_decision(
        self,
        profile: SpectralCeilingProfile,
        frequencies: np.ndarray,
        measured_db: np.ndarray,
        target_db: np.ndarray,
    ) -> Optional[float]:
        if frequencies.size == 0:
            return None
        low, high = profile.high_cut_hz
        upper_band = (low, min(high, float(frequencies[-1])))
        if upper_band[1] <= upper_band[0]:
            return float(low)
        excess = (
            _zone_mean(frequencies, measured_db, upper_band)
            - _zone_mean(frequencies, target_db, upper_band)
        )
        if excess > 2.5:
            return round(float(low), 1)
        return round(float((low + high) / 2.0), 1)

    @staticmethod
    def _spectral_summary(
        frequencies: np.ndarray,
        measured_db: np.ndarray,
        target_db: np.ndarray,
    ) -> Dict[str, Any]:
        zones = {
            "low": (60.0, 150.0),
            "low_mid": (180.0, 500.0),
            "mid": (500.0, 2000.0),
            "presence": (2500.0, 5000.0),
            "air": (8000.0, 14000.0),
        }
        summary: Dict[str, Any] = {}
        for name, bounds in zones.items():
            if frequencies.size == 0:
                continue
            measured = _zone_mean(frequencies, measured_db, bounds)
            target = _zone_mean(frequencies, target_db, bounds)
            summary[name] = {
                "measured_db": round(measured, 3),
                "target_db": round(target, 3),
                "delta_db": round(measured - target, 3),
            }
        return summary


def _range_intersects(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _console_band_for_freq(freq_hz: float) -> int:
    if freq_hz < 180.0:
        return 1
    if freq_hz < 1200.0:
        return 2
    if freq_hz < 5500.0:
        return 3
    return 4


def merge_spectral_proposal_into_eq_bands(
    eq_bands: Sequence[Tuple[float, float, float]],
    proposal: SpectralCeilingEQProposal,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """Merge proposal moves into a four-band console EQ target list."""
    merged = [tuple(map(float, band)) for band in eq_bands]
    report: Dict[str, Any] = {
        "accepted": [],
        "skipped": [],
        "dry_run": bool(proposal.dry_run),
        "confidence": float(proposal.confidence),
    }
    if not proposal.should_apply:
        report["skipped"].append(
            {
                "reason": "proposal not eligible for application",
                "dry_run": proposal.dry_run,
                "confidence": proposal.confidence,
            }
        )
        return merged, report

    max_abs = float(proposal.safety.get("max_abs_gain_db", 3.0))
    for suggestion in proposal.bands:
        if suggestion.action not in {"attenuate", "lift"}:
            report["skipped"].append(
                {"zone": suggestion.name, "reason": "non-parametric suggestion logged only"}
            )
            continue
        band_idx = _console_band_for_freq(suggestion.freq_hz) - 1
        if band_idx >= len(merged):
            report["skipped"].append(
                {"zone": suggestion.name, "reason": "no console EQ band slot available"}
            )
            continue

        current_freq, current_gain, current_q = merged[band_idx]
        direction = 1.0 if suggestion.gain_db > 0.0 else -1.0
        if abs(current_gain) > 0.1 and current_gain * suggestion.gain_db > 0.0:
            if abs(current_gain) >= abs(suggestion.gain_db):
                report["skipped"].append(
                    {
                        "zone": suggestion.name,
                        "reason": "existing EQ decision already addresses zone",
                        "existing_gain_db": round(current_gain, 3),
                    }
                )
                continue
        if current_gain * suggestion.gain_db < 0.0:
            if suggestion.gain_db > 0.0:
                report["skipped"].append(
                    {
                        "zone": suggestion.name,
                        "reason": "existing safer cut conflicts with proposed boost",
                        "existing_gain_db": round(current_gain, 3),
                    }
                )
                continue
            new_gain = min(0.0, current_gain + suggestion.gain_db * 0.5)
            merge_reason = "reduced conflicting boost toward safer cut"
        else:
            new_gain = current_gain + suggestion.gain_db
            merge_reason = "combined with existing EQ target"

        new_gain = _clamp(new_gain, -max_abs, max_abs)
        new_freq = current_freq
        new_q = current_q
        if abs(current_gain) <= 0.1:
            new_freq = suggestion.freq_hz
            new_q = suggestion.q
        else:
            new_q = min(current_q, suggestion.q, float(proposal.safety.get("max_q", 2.0)))

        merged[band_idx] = (float(new_freq), float(round(new_gain, 3)), float(new_q))
        report["accepted"].append(
            {
                "zone": suggestion.name,
                "band": band_idx + 1,
                "freq_hz": round(new_freq, 3),
                "gain_db": round(new_gain, 3),
                "direction": "boost" if direction > 0 else "cut",
                "reason": merge_reason,
            }
        )
    return merged, report


def format_spectral_ceiling_log(
    proposal: SpectralCeilingEQProposal,
    merge_report: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render a human-readable log block for operators."""
    lines = [
        "[SPECTRAL_CEILING_EQ]",
        f"track: {proposal.track_id}",
        f"role: {proposal.role}",
        f"profile: {proposal.target_profile}",
        f"measured_tilt: {proposal.measured_tilt_db_per_oct:.1f} dB/oct",
        f"target_tilt: {proposal.target_tilt_db_per_oct:.1f} dB/oct",
        "decision:",
    ]
    if proposal.low_cut_hz is not None:
        lines.append(f"  - high-pass guide around {proposal.low_cut_hz:.0f} Hz")
    if proposal.high_cut_hz is not None:
        lines.append(f"  - high roll-off guide around {proposal.high_cut_hz:.0f} Hz")
    for band in proposal.bands:
        verb = "cut" if band.gain_db < 0.0 else "boost"
        lines.append(
            "  - "
            f"{verb} {band.gain_db:+.1f} dB at {band.freq_hz:.0f} Hz "
            f"({band.name}) because {band.reason}"
        )
    skipped = list(proposal.skipped)
    if merge_report:
        skipped.extend(list(merge_report.get("skipped", [])))
        for item in merge_report.get("accepted", []):
            lines.append(
                "  - accepted "
                f"band {item.get('band')} {item.get('gain_db'):+.1f} dB "
                f"at {item.get('freq_hz'):.0f} Hz"
            )
    for item in skipped[:6]:
        reason = item.get("reason", "skipped")
        zone = item.get("zone") or item.get("rule")
        prefix = f"{zone}: " if zone else ""
        lines.append(f"  - no change: {prefix}{reason}")
    lines.append(f"confidence: {proposal.confidence:.2f}")
    lines.append(f"applied: {str(proposal.should_apply).lower()}")
    return "\n".join(lines)
