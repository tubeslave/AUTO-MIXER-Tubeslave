"""Explainable, safety-limited AutoEQ engine for offline/live automixing.

The engine is intentionally rule/metric based. It produces auditable EQ
decisions with metrics, limits, validation and rollback instead of opaque
"black box" curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np

from .models import amp_to_db, db_to_amp, ensure_stereo, jsonable


EPS = 1e-12

LOG_BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 50.0),
    "bass": (50.0, 100.0),
    "punch": (100.0, 180.0),
    "mud": (180.0, 350.0),
    "boxiness": (350.0, 700.0),
    "intelligibility": (700.0, 1500.0),
    "presence": (1500.0, 3000.0),
    "upper_presence": (3000.0, 5000.0),
    "harshness": (5000.0, 9000.0),
    "air": (9000.0, 14000.0),
    "polish": (14000.0, 18000.0),
}


DEFAULT_AUTOEQ_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "offline",
    "safety_profile": "conservative",
    "apply_static_eq": True,
    "apply_dynamic_eq": True,
    "apply_group_eq": True,
    "apply_master_eq": False,
    "report_only": False,
    "osc_dry_run": True,
    "osc_apply": False,
    "live_limits": {
        "max_static_gain_db": 1.5,
        "max_dynamic_cut_db": 3.0,
        "max_dynamic_boost_db": 1.0,
        "max_master_eq_db": 0.75,
        "max_filters_per_channel": 3,
        "max_dynamic_filters_per_channel": 2,
        "min_confidence_to_apply": 0.6,
        "half_strength_confidence_until": 0.8,
        "smoothing_ms": 1000,
        "hold_time_ms": 2000,
        "max_q": 4.0,
        "notch_max_q": 10.0,
    },
    "offline_limits": {
        "max_static_gain_db": 3.0,
        "max_dynamic_cut_db": 5.0,
        "max_dynamic_boost_db": 1.5,
        "max_master_eq_db": 1.5,
        "max_filters_per_channel": 5,
        "max_dynamic_filters_per_channel": 3,
        "min_confidence_to_apply": 0.55,
        "half_strength_confidence_until": 0.8,
        "smoothing_ms": 0,
        "hold_time_ms": 0,
        "max_q": 4.0,
        "notch_max_q": 12.0,
    },
    "priorities": {
        "lead_vocal": 100,
        "kick": 90,
        "snare": 85,
        "snare_top": 85,
        "snare_bottom": 80,
        "bass": 85,
        "back_vocal": 65,
        "backing_vocal": 65,
        "guitar": 55,
        "guitar_l": 55,
        "guitar_r": 55,
        "accordion": 55,
        "keys": 55,
        "playback": 50,
        "playback_l": 50,
        "playback_r": 50,
        "overhead_l": 45,
        "overhead_r": 45,
        "overhead_pair": 45,
        "cymbals": 40,
        "master": 100,
    },
}


@dataclass(frozen=True)
class FrequencyRange:
    low_hz: float
    high_hz: float
    label: str = ""
    target: str = ""

    @property
    def center_hz(self) -> float:
        return math.sqrt(max(self.low_hz, 1.0) * max(self.high_hz, self.low_hz + 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_hz": round(float(self.low_hz), 2),
            "high_hz": round(float(self.high_hz), 2),
            "label": self.label,
            "target": self.target,
        }


@dataclass(frozen=True)
class InstrumentProfile:
    instrument: str
    useful_bands: tuple[FrequencyRange, ...] = ()
    problematic_bands: tuple[FrequencyRange, ...] = ()
    protected_bands: tuple[FrequencyRange, ...] = ()
    hpf_range: tuple[float, float] = (20.0, 80.0)
    lpf_range: tuple[float, float] = (9000.0, 20000.0)
    max_static_gain_db: float = 3.0
    max_dynamic_cut_db: float = 5.0
    max_dynamic_boost_db: float = 1.0
    preferred_q_range: tuple[float, float] = (0.7, 2.5)
    role_priority: int = 50
    masking_priority: int = 50
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["useful_bands"] = [b.to_dict() for b in self.useful_bands]
        payload["problematic_bands"] = [b.to_dict() for b in self.problematic_bands]
        payload["protected_bands"] = [b.to_dict() for b in self.protected_bands]
        return jsonable(payload)


@dataclass(frozen=True)
class AutoEQSettings:
    enabled: bool = False
    mode: str = "offline"
    safety_profile: str = "conservative"
    apply_static_eq: bool = True
    apply_dynamic_eq: bool = True
    apply_group_eq: bool = True
    apply_master_eq: bool = False
    report_only: bool = False
    osc_dry_run: bool = True
    osc_apply: bool = False
    live_limits: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_AUTOEQ_CONFIG["live_limits"]))
    offline_limits: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_AUTOEQ_CONFIG["offline_limits"]))
    priorities: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_AUTOEQ_CONFIG["priorities"]))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None = None, **overrides: Any) -> "AutoEQSettings":
        merged = _deep_merge(DEFAULT_AUTOEQ_CONFIG, dict(payload or {}))
        merged = _deep_merge(merged, {k: v for k, v in overrides.items() if v is not None})
        return cls(
            enabled=bool(merged.get("enabled", False)),
            mode=str(merged.get("mode", "offline")),
            safety_profile=str(merged.get("safety_profile", "conservative")),
            apply_static_eq=bool(merged.get("apply_static_eq", True)),
            apply_dynamic_eq=bool(merged.get("apply_dynamic_eq", True)),
            apply_group_eq=bool(merged.get("apply_group_eq", True)),
            apply_master_eq=bool(merged.get("apply_master_eq", False)),
            report_only=bool(merged.get("report_only", False)),
            osc_dry_run=bool(merged.get("osc_dry_run", True)),
            osc_apply=bool(merged.get("osc_apply", False)),
            live_limits=dict(merged.get("live_limits", {})),
            offline_limits=dict(merged.get("offline_limits", {})),
            priorities={str(k): int(v) for k, v in dict(merged.get("priorities", {})).items()},
        )

    @property
    def limits(self) -> dict[str, Any]:
        return self.live_limits if self.mode == "live" else self.offline_limits

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass
class AutoEQTrack:
    channel: int
    file: str
    instrument: str
    audio: np.ndarray
    sample_rate: int
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": int(self.channel),
            "file": self.file,
            "instrument": self.instrument,
            "group": self.group,
            "sample_rate": int(self.sample_rate),
            "samples": int(ensure_stereo(self.audio).shape[0]),
        }


@dataclass(frozen=True)
class MaskingRelation:
    masked_channel: int | None
    masked_source: str
    masked_instrument: str
    masker_channel: int | None
    masker_source: str
    masker_instrument: str
    suggested_band: tuple[float, float]
    suggested_action: str
    overlapping_band_energy: float
    masking_score: float
    confidence: float
    reason: str
    temporary_conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class EQDecision:
    channel: int | None
    file: str
    instrument: str
    eq_type: str
    frequency_hz: float
    q: float
    gain_db: float
    target_type: str = "channel"
    dynamic: bool = False
    sidechain_source: str | None = None
    trigger_band_hz: tuple[float, float] | None = None
    attack_ms: float | None = None
    release_ms: float | None = None
    threshold_metric: str = ""
    reason: str = ""
    confidence: float = 0.0
    safety_limited: bool = False
    source_metrics_before: dict[str, Any] = field(default_factory=dict)
    linked_group: str | None = None
    applied: bool = False
    applied_to_audio: bool = False
    rollback: bool = False
    rejected: bool = False
    rejection_reason: str = ""
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.eq_type
        return jsonable(payload)


@dataclass(frozen=True)
class AutoEQRunResult:
    processed_audio_by_channel: dict[int, np.ndarray]
    report: dict[str, Any]


class InstrumentProfileLibrary:
    """Instrument-aware EQ limits and role priorities."""

    def __init__(self, priorities: Mapping[str, int] | None = None):
        self.priorities = {**DEFAULT_AUTOEQ_CONFIG["priorities"], **dict(priorities or {})}
        self._profiles = self._build_profiles()

    def get(self, instrument: str) -> InstrumentProfile:
        key = normalize_instrument(instrument)
        if key in self._profiles:
            return self._profiles[key]
        return self._profiles["playback"]

    def all_profiles(self) -> dict[str, InstrumentProfile]:
        return dict(self._profiles)

    def _profile(
        self,
        instrument: str,
        useful: list[tuple[float, float, str]],
        problematic: list[tuple[float, float, str]],
        protected: list[tuple[float, float, str]],
        *,
        hpf: tuple[float, float],
        lpf: tuple[float, float],
        static: float = 3.0,
        dynamic_cut: float = 5.0,
        dynamic_boost: float = 1.0,
        q_range: tuple[float, float] = (0.7, 2.5),
        notes: tuple[str, ...] = (),
    ) -> InstrumentProfile:
        priority = int(self.priorities.get(instrument, 50))
        return InstrumentProfile(
            instrument=instrument,
            useful_bands=tuple(FrequencyRange(lo, hi, label, "useful") for lo, hi, label in useful),
            problematic_bands=tuple(FrequencyRange(lo, hi, label, "problematic") for lo, hi, label in problematic),
            protected_bands=tuple(FrequencyRange(lo, hi, label, "protected") for lo, hi, label in protected),
            hpf_range=hpf,
            lpf_range=lpf,
            max_static_gain_db=float(static),
            max_dynamic_cut_db=float(dynamic_cut),
            max_dynamic_boost_db=float(dynamic_boost),
            preferred_q_range=q_range,
            role_priority=priority,
            masking_priority=priority,
            notes=notes,
        )

    def _build_profiles(self) -> dict[str, InstrumentProfile]:
        p: dict[str, InstrumentProfile] = {}
        p["kick"] = self._profile(
            "kick",
            [(45, 80, "fundamental"), (90, 130, "punch"), (2500, 4500, "beater")],
            [(180, 350, "mud"), (5000, 9000, "bleed click harshness")],
            [(45, 130, "low-end anchor")],
            hpf=(25, 45),
            lpf=(5500, 9000),
            notes=("Kick owns punch/fundamental before bass sustain in many rock/live mixes.",),
        )
        p["snare_top"] = self._profile(
            "snare_top",
            [(180, 250, "body"), (3500, 6000, "crack")],
            [(500, 1000, "boxiness"), (6500, 9500, "cymbal spill")],
            [(180, 250, "body")],
            hpf=(80, 130),
            lpf=(6500, 10000),
        )
        p["snare_bottom"] = self._profile(
            "snare_bottom",
            [(180, 260, "wire body"), (4500, 7500, "wire brightness")],
            [(700, 1200, "honk"), (6500, 11000, "spill")],
            [(180, 260, "body after polarity check")],
            hpf=(100, 160),
            lpf=(6500, 9500),
            static=2.0,
            notes=("Snare bottom is polarity/phase sensitive and should be corrected cautiously.",),
        )
        p["tom"] = self._profile("tom", [(90, 170, "body"), (3500, 5500, "attack")], [(250, 500, "mud"), (6000, 10000, "spill")], [(80, 180, "body")], hpf=(50, 90), lpf=(5000, 8000))
        p["floor_tom"] = self._profile("floor_tom", [(65, 120, "body"), (3000, 5000, "attack")], [(180, 420, "mud"), (6000, 10000, "spill")], [(60, 130, "body")], hpf=(40, 70), lpf=(5000, 8000))
        overhead = self._profile(
            "overhead_pair",
            [(3000, 7000, "kit presence"), (9000, 14000, "air")],
            [(250, 600, "low-mid wash"), (5000, 9000, "harsh cymbal wash")],
            [(500, 6000, "stereo image")],
            hpf=(100, 180),
            lpf=(9000, 14000),
            static=1.5,
            dynamic_cut=3.0,
            notes=("Prefer linked stereo EQ for overheads; avoid independent L/R moves that narrow or tilt the image.",),
        )
        p["overhead_pair"] = overhead
        p["overhead_l"] = replace(overhead, instrument="overhead_l")
        p["overhead_r"] = replace(overhead, instrument="overhead_r")
        p["bass"] = self._profile("bass", [(55, 100, "fundamental"), (100, 180, "sustain"), (700, 1000, "note definition")], [(180, 350, "mud"), (2500, 5000, "string noise")], [(55, 180, "low-end sustain")], hpf=(30, 50), lpf=(5000, 9000))
        guitar = self._profile("guitar", [(700, 1500, "body readability"), (1800, 3500, "presence")], [(180, 350, "mud"), (2500, 4500, "vocal masking"), (5000, 8000, "fizz")], [(700, 2500, "musical identity")], hpf=(70, 120), lpf=(5500, 9000), notes=("Guitar/accordion/playback should yield to lead vocal in intelligibility bands.",))
        p["guitar"] = guitar
        p["guitar_l"] = replace(guitar, instrument="guitar_l")
        p["guitar_r"] = replace(guitar, instrument="guitar_r")
        p["accordion"] = self._profile("accordion", [(350, 800, "body"), (1800, 3500, "reed presence")], [(250, 450, "mud"), (2500, 4500, "vocal masking"), (6500, 9000, "edge")], [(350, 3000, "musical identity")], hpf=(80, 140), lpf=(6500, 10000))
        p["keys"] = self._profile("keys", [(250, 700, "body"), (1500, 4000, "definition")], [(180, 350, "mud"), (2500, 4500, "vocal masking")], [(250, 4000, "harmony")], hpf=(70, 140), lpf=(8000, 13000))
        playback = self._profile("playback", [(150, 500, "body"), (1500, 5000, "arrangement detail")], [(180, 350, "mud"), (2500, 4500, "vocal masking"), (6000, 10000, "hiss")], [(150, 5000, "stem integrity")], hpf=(25, 80), lpf=(8000, 14000))
        p["playback"] = playback
        p["playback_l"] = replace(playback, instrument="playback_l")
        p["playback_r"] = replace(playback, instrument="playback_r")
        p["lead_vocal"] = self._profile("lead_vocal", [(1500, 5000, "intelligibility"), (9000, 14000, "air")], [(180, 350, "mud"), (5000, 9000, "sibilance")], [(1500, 5000, "lead clarity")], hpf=(70, 120), lpf=(10000, 16000), dynamic_boost=0.8, notes=("Do not boost vocal first when masking can be solved by carving lower-priority sources.",))
        p["back_vocal"] = self._profile("back_vocal", [(1200, 3500, "blend definition")], [(180, 350, "mud"), (1500, 5000, "lead vocal masking"), (5000, 9000, "sibilance stack")], [(700, 3000, "blend")], hpf=(90, 140), lpf=(9000, 14000), static=2.0)
        p["vocal_group"] = self._profile("vocal_group", [(1500, 5000, "group clarity")], [(180, 350, "mud"), (5000, 9000, "sibilance stack")], [(1500, 5000, "clarity")], hpf=(70, 130), lpf=(10000, 16000), static=1.5)
        p["drums_group"] = self._profile("drums_group", [(70, 160, "impact"), (3000, 7000, "attack")], [(180, 400, "mud"), (5000, 9000, "cymbal harshness")], [(70, 160, "impact")], hpf=(20, 50), lpf=(9000, 14000), static=1.5)
        p["harmonic_group"] = self._profile("harmonic_group", [(300, 4000, "harmony")], [(180, 350, "mud"), (2500, 4500, "vocal masking")], [(300, 2500, "body")], hpf=(50, 120), lpf=(7000, 14000), static=1.5)
        p["low_end_group"] = self._profile("low_end_group", [(45, 120, "foundation")], [(120, 350, "mud")], [(45, 120, "foundation")], hpf=(20, 45), lpf=(5000, 9000), static=1.5)
        p["cymbals_group"] = self._profile("cymbals_group", [(6000, 12000, "cymbal detail")], [(350, 700, "wash"), (5000, 9000, "harshness")], [(6000, 12000, "stereo air")], hpf=(150, 300), lpf=(10000, 16000), static=1.0, dynamic_cut=2.5)
        p["master"] = self._profile("master", [(80, 12000, "translation")], [(180, 350, "mud"), (5000, 9000, "harshness")], [(60, 120, "foundation"), (1500, 5000, "vocal clarity")], hpf=(20, 35), lpf=(14000, 18000), static=0.75, dynamic_cut=1.5, notes=("Master EQ should be minimal and should not correct problems that belong to channels.",))
        return p


class SpectralAnalyzer:
    """Dependency-light spectral and loudness analysis."""

    def __init__(self, bands: Mapping[str, tuple[float, float]] | None = None):
        self.bands = dict(bands or LOG_BANDS)

    def analyze_track(self, track: AutoEQTrack) -> dict[str, Any]:
        audio = ensure_stereo(track.audio)
        mono = np.mean(audio, axis=1) if audio.size else np.zeros(0, dtype=np.float32)
        spectrum = _spectrum(mono, track.sample_rate)
        freqs = spectrum["freqs"]
        power = spectrum["power"]
        mag = spectrum["mag"]
        band_power = {
            name: _sum_band_power(freqs, power, low, high)
            for name, (low, high) in self.bands.items()
        }
        total = max(sum(band_power.values()), EPS)
        band_energy_db = {name: _power_to_db(value) for name, value in band_power.items()}
        band_ratios = {name: float(value / total) for name, value in band_power.items()}
        centroid = _centroid(freqs, mag)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)) + EPS)) if audio.size else 0.0
        integrated_lufs = _integrated_lufs(audio, track.sample_rate)
        short_lufs = _short_term_lufs(audio, track.sample_rate)
        mono_compat = _phase_correlation(audio)
        resonance = _resonance_candidates(freqs, mag)
        scores = _spectral_scores(band_ratios, track.instrument)
        activity = _activity_metrics(mono, track.sample_rate)
        return {
            "channel": int(track.channel),
            "file": track.file,
            "instrument": track.instrument,
            "band_energy_db": {k: round(v, 3) for k, v in band_energy_db.items()},
            "band_ratios": {k: round(v, 6) for k, v in band_ratios.items()},
            "spectral_centroid_hz": round(centroid, 3),
            "spectral_tilt_db_per_oct": round(_spectral_tilt(freqs, mag), 3),
            "low_mid_buildup_score": scores["low_mid_buildup_score"],
            "harshness_score": scores["harshness_score"],
            "mud_score": scores["mud_score"],
            "presence_score": scores["presence_score"],
            "air_score": scores["air_score"],
            "resonance_candidates": resonance,
            "sibilance_score": scores["sibilance_score"],
            "low_end_excess_score": scores["low_end_excess_score"],
            "mono_compatibility": round(mono_compat, 4),
            "short_term_lufs": round(short_lufs, 3),
            "integrated_lufs": round(integrated_lufs, 3),
            "peak_dbfs": round(amp_to_db(peak), 3),
            "rms_dbfs": round(amp_to_db(rms), 3),
            "active_ratio": round(activity["active_ratio"], 4),
            "activity_windows": activity["windows"],
        }

    def analyze_tracks(self, tracks: list[AutoEQTrack]) -> dict[int, dict[str, Any]]:
        return {track.channel: self.analyze_track(track) for track in tracks}

    def analyze_audio(self, audio: np.ndarray, sample_rate: int, *, channel: int = 0, file: str = "", instrument: str = "master", group: str = "") -> dict[str, Any]:
        return self.analyze_track(AutoEQTrack(channel, file, instrument, ensure_stereo(audio), sample_rate, group))


class MaskingAnalyzer:
    """Cross-channel masking analysis for explainable EQ moves."""

    def __init__(self, profile_library: InstrumentProfileLibrary):
        self.profile_library = profile_library

    def analyze(self, tracks: list[AutoEQTrack], metrics: Mapping[int, Mapping[str, Any]]) -> list[MaskingRelation]:
        relations: list[MaskingRelation] = []
        lead_vocals = _tracks_matching(tracks, {"lead_vocal"})
        back_vocals = _tracks_matching(tracks, {"back_vocal", "backing_vocal"})
        guitars = _tracks_matching(tracks, {"guitar", "guitar_l", "guitar_r", "accordion", "keys", "playback", "playback_l", "playback_r"})
        cymbals = _tracks_matching(tracks, {"overhead_l", "overhead_r", "overhead_pair", "cymbals"})
        kicks = _tracks_matching(tracks, {"kick"})
        basses = _tracks_matching(tracks, {"bass"})
        snares = _tracks_matching(tracks, {"snare", "snare_top", "snare_bottom"})
        close_drums = _tracks_matching(tracks, {"kick", "snare", "snare_top", "snare_bottom", "tom", "floor_tom"})

        for vocal in lead_vocals:
            for source in guitars:
                relations.extend(self._relation(vocal, source, metrics, (1500.0, 4000.0), "dynamic_bell_cut", "Guitar/accordion/playback masks lead vocal presence; carve the masker before boosting vocal."))
            for source in cymbals:
                relations.extend(self._relation(vocal, source, metrics, (5000.0, 9000.0), "dynamic_bell_cut", "Cymbal/overhead harshness competes with vocal presence and sibilance."))
            for source in back_vocals:
                relations.extend(self._relation(vocal, source, metrics, (1500.0, 5000.0), "dynamic_bell_cut", "Backing vocals overlap lead vocal clarity band; keep lead intelligibility protected."))

        for kick in kicks:
            for bass in basses:
                relations.extend(self._low_end_relation(kick, bass, metrics))

        for snare in snares:
            for source in [*guitars, *cymbals]:
                relations.extend(self._relation(snare, source, metrics, (2000.0, 5000.0), "dynamic_bell_cut", "Snare attack is masked by accompaniment or cymbal energy."))

        for overhead in cymbals:
            for close in close_drums:
                relations.extend(self._relation(overhead, close, metrics, (5000.0, 9000.0), "static_bell_cut", "Close drum mic cymbal spill stacks against the overhead picture."))

        return relations

    def _relation(
        self,
        masked: AutoEQTrack,
        masker: AutoEQTrack,
        metrics: Mapping[int, Mapping[str, Any]],
        band: tuple[float, float],
        action: str,
        reason: str,
    ) -> list[MaskingRelation]:
        masked_metrics = dict(metrics.get(masked.channel, {}) or {})
        masker_metrics = dict(metrics.get(masker.channel, {}) or {})
        if not masked_metrics or not masker_metrics:
            return []
        masked_energy = _band_energy_from_metrics(masked_metrics, band)
        masker_energy = _band_energy_from_metrics(masker_metrics, band)
        if masked_energy <= -118.0 or masker_energy <= -118.0:
            return []
        overlap = min(masked_energy, masker_energy)
        diff = masker_energy - masked_energy
        priority_delta = self.profile_library.get(masked.instrument).masking_priority - self.profile_library.get(masker.instrument).masking_priority
        score = _clamp01(0.45 + diff / 18.0 + priority_delta / 180.0)
        confidence = _clamp01(0.45 + score * 0.45 + max(0.0, diff) / 30.0)
        if score < 0.42 or confidence < 0.45:
            return []
        masked_active = float(masked_metrics.get("active_ratio", 1.0))
        masker_active = float(masker_metrics.get("active_ratio", 1.0))
        temporary = 0.12 <= min(masked_active, masker_active) <= 0.55
        return [
            MaskingRelation(
                masked_channel=masked.channel,
                masked_source=masked.file,
                masked_instrument=masked.instrument,
                masker_channel=masker.channel,
                masker_source=masker.file,
                masker_instrument=masker.instrument,
                suggested_band=band,
                suggested_action=action,
                overlapping_band_energy=round(overlap, 3),
                masking_score=round(score, 4),
                confidence=round(confidence, 4),
                reason=reason,
                temporary_conflict=temporary,
            )
        ]

    def _low_end_relation(self, kick: AutoEQTrack, bass: AutoEQTrack, metrics: Mapping[int, Mapping[str, Any]]) -> list[MaskingRelation]:
        kick_metrics = dict(metrics.get(kick.channel, {}) or {})
        bass_metrics = dict(metrics.get(bass.channel, {}) or {})
        if not kick_metrics or not bass_metrics:
            return []
        band = (50.0, 100.0)
        kick_energy = _band_energy_from_metrics(kick_metrics, band)
        bass_energy = _band_energy_from_metrics(bass_metrics, band)
        if bass_energy < kick_energy - 6.0:
            return []
        score = _clamp01(0.5 + (bass_energy - kick_energy) / 18.0)
        return [
            MaskingRelation(
                masked_channel=kick.channel,
                masked_source=kick.file,
                masked_instrument=kick.instrument,
                masker_channel=bass.channel,
                masker_source=bass.file,
                masker_instrument=bass.instrument,
                suggested_band=band,
                suggested_action="dynamic_bell_cut",
                overlapping_band_energy=round(min(kick_energy, bass_energy), 3),
                masking_score=round(score, 4),
                confidence=round(_clamp01(0.58 + score * 0.35), 4),
                reason="Kick and bass overlap in 50-100 Hz; preserve kick punch by cutting the lower-priority low-end source slightly.",
                temporary_conflict=True,
            )
        ]


class EQDecisionEngine:
    """Generate static channel, group and master EQ candidates."""

    def __init__(self, profile_library: InstrumentProfileLibrary):
        self.profile_library = profile_library

    def generate_channel_decisions(
        self,
        tracks: list[AutoEQTrack],
        metrics: Mapping[int, Mapping[str, Any]],
        masking: list[MaskingRelation],
        settings: AutoEQSettings,
    ) -> list[EQDecision]:
        decisions: list[EQDecision] = []
        if settings.apply_static_eq:
            for track in tracks:
                decisions.extend(self._static_profile_decisions(track, dict(metrics.get(track.channel, {}) or {})))
        for relation in masking:
            if relation.suggested_action.startswith("static") and relation.masker_channel is not None:
                masker = next((track for track in tracks if track.channel == relation.masker_channel), None)
                if masker is None:
                    continue
                decisions.append(
                    _decision_from_relation(
                        relation,
                        masker,
                        dynamic=False,
                        gain_db=-_scale_cut(relation.masking_score, 0.6, 2.2),
                        q=1.4,
                        reason=relation.reason,
                        metrics=dict(metrics.get(masker.channel, {}) or {}),
                    )
                )
        return decisions

    def generate_group_decisions(
        self,
        tracks: list[AutoEQTrack],
        metrics: Mapping[int, Mapping[str, Any]],
        settings: AutoEQSettings,
    ) -> list[EQDecision]:
        if not settings.apply_group_eq:
            return []
        decisions: list[EQDecision] = []
        groups: dict[str, list[AutoEQTrack]] = {}
        for track in tracks:
            groups.setdefault(track.group or _group_for_instrument(track.instrument), []).append(track)
        for group, members in groups.items():
            if len(members) < 2:
                continue
            mud = float(np.mean([float(metrics[m.channel].get("mud_score", 0.0)) for m in members if m.channel in metrics]))
            harsh = float(np.mean([float(metrics[m.channel].get("harshness_score", 0.0)) for m in members if m.channel in metrics]))
            if mud > 0.72:
                decisions.append(_bus_decision(group, 260.0, -0.8, 1.1, "group EQ", f"{group} low-mid buildup is high; small bus cleanup is safer than wider master EQ.", 0.72))
            if group in {"drums", "cymbals", "drums_group"} and harsh > 0.68:
                decisions.append(_bus_decision(group, 6500.0, -0.7, 1.2, "group EQ", f"{group} harshness is elevated; use a small linked group cut.", 0.7))
        return decisions

    def generate_master_decisions(self, master_metrics: Mapping[str, Any] | None, settings: AutoEQSettings) -> list[EQDecision]:
        if not settings.apply_master_eq or not master_metrics:
            return []
        decisions: list[EQDecision] = []
        mud = float(master_metrics.get("mud_score", 0.0))
        harsh = float(master_metrics.get("harshness_score", 0.0))
        if mud > 0.78:
            decisions.append(EQDecision(None, "MASTER", "master", "master_tilt_correction", 260.0, 0.8, -0.5, target_type="master", reason="Master mud is high after channel analysis; apply only a small final correction.", confidence=0.65, source_metrics_before=dict(master_metrics)))
        if harsh > 0.78:
            decisions.append(EQDecision(None, "MASTER", "master", "master_tilt_correction", 6500.0, 0.8, -0.5, target_type="master", reason="Master harshness is high after channel analysis; use minimal final correction.", confidence=0.65, source_metrics_before=dict(master_metrics)))
        return decisions

    def _static_profile_decisions(self, track: AutoEQTrack, metrics: Mapping[str, Any]) -> list[EQDecision]:
        if not metrics:
            return []
        profile = self.profile_library.get(track.instrument)
        decisions: list[EQDecision] = []
        peak = float(metrics.get("peak_dbfs", -120.0))
        mud = float(metrics.get("mud_score", 0.0))
        harsh = float(metrics.get("harshness_score", 0.0))
        low = float(metrics.get("low_end_excess_score", 0.0))
        sibilance = float(metrics.get("sibilance_score", 0.0))
        resonance = list(metrics.get("resonance_candidates", []) or [])

        if mud > 0.66:
            band = _nearest_profile_band(profile.problematic_bands, (180.0, 450.0)) or FrequencyRange(220, 350, "mud")
            decisions.append(_channel_decision(track, metrics, "static_bell_cut", band.center_hz, -_scale_cut(mud, 0.7, 2.2), 1.15, f"{track.file} has elevated low-mid/mud score; apply a small corrective cut.", mud))
        if harsh > 0.66:
            band = _nearest_profile_band(profile.problematic_bands, (3000.0, 9000.0)) or FrequencyRange(4500, 8000, "harshness")
            decisions.append(_channel_decision(track, metrics, "static_bell_cut", band.center_hz, -_scale_cut(harsh, 0.6, 2.0), 1.25, f"{track.file} harshness score is elevated; use bounded cut instead of global darkening.", harsh))
        if low > 0.78 and track.instrument not in {"kick", "bass", "low_end_group"}:
            decisions.append(_channel_decision(track, metrics, "static_bell_cut", 90.0, -_scale_cut(low, 0.5, 1.5), 0.9, f"{track.file} has low-end excess outside protected low-end roles.", low))
        if normalize_instrument(track.instrument) in {"lead_vocal", "back_vocal"} and sibilance > 0.58:
            decisions.append(_channel_decision(track, metrics, "static_bell_cut", 7200.0, -_scale_cut(sibilance, 0.4, 1.4), 2.0, f"{track.file} sibilance score is high; start with a modest de-ess cut.", sibilance))
        if resonance:
            strongest = max(resonance, key=lambda item: float(item.get("prominence_db", 0.0)))
            if float(strongest.get("prominence_db", 0.0)) > 8.0 and 80.0 <= float(strongest.get("frequency_hz", 0.0)) <= 9000.0:
                decisions.append(_channel_decision(track, metrics, "notch", float(strongest["frequency_hz"]), -1.5, 5.5, f"{track.file} has a narrow resonance candidate at {float(strongest['frequency_hz']):.0f} Hz.", min(0.95, 0.62 + float(strongest.get("prominence_db", 0.0)) / 40.0)))
        if peak < -3.0 and float(metrics.get("presence_score", 0.0)) < 0.22 and normalize_instrument(track.instrument) in {"lead_vocal", "snare_top", "kick"}:
            useful = profile.useful_bands[0] if profile.useful_bands else FrequencyRange(1500, 3000, "presence")
            decisions.append(_channel_decision(track, metrics, "static_bell_boost", useful.center_hz, 0.6, 1.0, f"{track.file} has low useful-band presence and enough peak headroom; use a small bounded boost.", 0.62))
        return decisions


class DynamicEQDecisionEngine:
    """Generate dynamic EQ decisions for temporary conflicts."""

    def generate(
        self,
        tracks: list[AutoEQTrack],
        metrics: Mapping[int, Mapping[str, Any]],
        masking: list[MaskingRelation],
        settings: AutoEQSettings,
    ) -> list[EQDecision]:
        if not settings.apply_dynamic_eq:
            return []
        decisions: list[EQDecision] = []
        for relation in masking:
            if relation.masker_channel is None:
                continue
            if not relation.temporary_conflict and relation.suggested_action != "dynamic_bell_cut":
                continue
            masker = next((track for track in tracks if track.channel == relation.masker_channel), None)
            if masker is None:
                continue
            cut = -_scale_cut(relation.masking_score, 0.6, 2.4)
            decisions.append(
                _decision_from_relation(
                    relation,
                    masker,
                    dynamic=True,
                    gain_db=cut,
                    q=1.25,
                    reason=f"{relation.reason} Use dynamic EQ because the conflict is time-dependent.",
                    metrics=dict(metrics.get(masker.channel, {}) or {}),
                )
            )

        for track in tracks:
            m = dict(metrics.get(track.channel, {}) or {})
            inst = normalize_instrument(track.instrument)
            if inst in {"overhead_l", "overhead_r", "overhead_pair"} and float(m.get("harshness_score", 0.0)) > 0.6:
                decisions.append(_channel_decision(track, m, "dynamic_bell_cut", 7000.0, -1.2, 1.2, "Overhead/cymbal harshness spikes; linked dynamic cut preserves stereo image better than broad static darkening.", 0.72, dynamic=True, trigger_band=(5000.0, 9000.0), linked_group="overhead_pair"))
            if inst in {"lead_vocal", "back_vocal"} and float(m.get("sibilance_score", 0.0)) > 0.55:
                decisions.append(_channel_decision(track, m, "dynamic_bell_cut", 7200.0, -1.4, 2.2, "Vocal sibilance is intermittent; use de-ess style dynamic EQ rather than static dulling.", 0.74, dynamic=True, trigger_band=(5000.0, 9000.0)))
            if inst in {"snare", "snare_top"} and float(m.get("harshness_score", 0.0)) > 0.58:
                decisions.append(_channel_decision(track, m, "dynamic_bell_cut", 3800.0, -1.2, 1.6, "Snare attack harshness is transient; dynamic cut tracks hits without thinning the drum.", 0.68, dynamic=True, trigger_band=(2000.0, 5000.0)))
        return decisions


class EQSafetyLimiter:
    """Clamp or reject decisions before they can affect audio or OSC."""

    def __init__(self, profile_library: InstrumentProfileLibrary):
        self.profile_library = profile_library

    def limit(self, decisions: list[EQDecision], settings: AutoEQSettings) -> tuple[list[EQDecision], list[EQDecision]]:
        limited: list[EQDecision] = []
        rejected: list[EQDecision] = []
        per_channel: dict[int | None, int] = {}
        per_channel_dynamic: dict[int | None, int] = {}
        limits = settings.limits
        min_conf = float(limits.get("min_confidence_to_apply", 0.6))
        half_until = float(limits.get("half_strength_confidence_until", 0.8))
        max_filters = int(limits.get("max_filters_per_channel", 3))
        max_dyn = int(limits.get("max_dynamic_filters_per_channel", 2))
        max_q = float(limits.get("max_q", 4.0))
        notch_max_q = float(limits.get("notch_max_q", 10.0))

        for decision in sorted(decisions, key=lambda item: (-item.confidence, item.channel or 10_000, item.frequency_hz)):
            if decision.confidence < min_conf:
                rejected.append(replace(decision, rejected=True, rejection_reason="confidence too low"))
                continue
            if decision.channel is not None:
                if per_channel.get(decision.channel, 0) >= max_filters:
                    rejected.append(replace(decision, rejected=True, rejection_reason="max filters per channel exceeded"))
                    continue
                if decision.dynamic and per_channel_dynamic.get(decision.channel, 0) >= max_dyn:
                    rejected.append(replace(decision, rejected=True, rejection_reason="max dynamic filters per channel exceeded"))
                    continue

            gain = float(decision.gain_db)
            q = float(decision.q)
            safety_limited = bool(decision.safety_limited)
            if decision.confidence < half_until:
                gain *= 0.5
                safety_limited = True
            if decision.dynamic:
                if gain < 0:
                    max_cut = float(limits.get("max_dynamic_cut_db", 3.0))
                    if gain < -max_cut:
                        gain = -max_cut
                        safety_limited = True
                else:
                    max_boost = float(limits.get("max_dynamic_boost_db", 1.0))
                    if gain > max_boost:
                        gain = max_boost
                        safety_limited = True
            elif decision.target_type == "master":
                max_master = float(limits.get("max_master_eq_db", 0.75))
                if abs(gain) > max_master:
                    gain = math.copysign(max_master, gain)
                    safety_limited = True
            else:
                max_static = float(limits.get("max_static_gain_db", 1.5))
                profile_limit = self.profile_library.get(decision.instrument).max_static_gain_db
                max_static = min(max_static, profile_limit)
                if abs(gain) > max_static:
                    gain = math.copysign(max_static, gain)
                    safety_limited = True
            if decision.eq_type == "notch":
                if q > notch_max_q:
                    q = notch_max_q
                    safety_limited = True
            elif q > max_q:
                q = max_q
                safety_limited = True
            if gain > 0.0 and float(decision.source_metrics_before.get("peak_dbfs", -120.0)) > -3.0:
                rejected.append(replace(decision, rejected=True, rejection_reason="never boost when channel is near peak limit"))
                continue
            if gain > 0.0 and "mask" in decision.reason.lower():
                rejected.append(replace(decision, rejected=True, rejection_reason="do not boost problematic band while masking is high"))
                continue

            out = replace(decision, gain_db=round(gain, 3), q=round(q, 3), safety_limited=safety_limited)
            limited.append(out)
            if decision.channel is not None:
                per_channel[decision.channel] = per_channel.get(decision.channel, 0) + 1
                if decision.dynamic:
                    per_channel_dynamic[decision.channel] = per_channel_dynamic.get(decision.channel, 0) + 1
        return limited, rejected


class EQResultValidator:
    """Apply channel decisions with rollback/retry validation."""

    def __init__(self, analyzer: SpectralAnalyzer):
        self.analyzer = analyzer

    def apply_and_validate(
        self,
        tracks: list[AutoEQTrack],
        decisions: list[EQDecision],
        settings: AutoEQSettings,
    ) -> tuple[dict[int, np.ndarray], list[EQDecision], list[EQDecision], dict[str, Any]]:
        audio_by_channel = {track.channel: ensure_stereo(track.audio).copy() for track in tracks}
        track_by_channel = {track.channel: track for track in tracks}
        accepted: list[EQDecision] = []
        rejected: list[EQDecision] = []

        for decision in decisions:
            if settings.report_only:
                accepted.append(replace(decision, applied=False, validation={"status": "report_only"}))
                continue
            if decision.target_type != "channel" or decision.channel is None:
                accepted.append(replace(decision, applied=True, applied_to_audio=False, validation={"status": "accepted_for_command_export"}))
                continue
            track = track_by_channel.get(decision.channel)
            if track is None:
                rejected.append(replace(decision, rejected=True, rejection_reason="channel audio not found"))
                continue

            current_audio = audio_by_channel[decision.channel]
            before_metrics = self.analyzer.analyze_audio(current_audio, track.sample_rate, channel=track.channel, file=track.file, instrument=track.instrument, group=track.group)
            proposed_audio = _apply_eq_decision(current_audio, track.sample_rate, decision, audio_by_channel=audio_by_channel, tracks=track_by_channel)
            after_metrics = self.analyzer.analyze_audio(proposed_audio, track.sample_rate, channel=track.channel, file=track.file, instrument=track.instrument, group=track.group)
            verdict = _validation_verdict(decision, before_metrics, after_metrics)
            if verdict["accepted"]:
                audio_by_channel[decision.channel] = proposed_audio
                accepted.append(replace(decision, applied=True, applied_to_audio=True, validation=verdict))
                continue

            retry = replace(decision, gain_db=round(decision.gain_db * 0.5, 3), safety_limited=True, rollback=True)
            retry_audio = _apply_eq_decision(current_audio, track.sample_rate, retry, audio_by_channel=audio_by_channel, tracks=track_by_channel)
            retry_metrics = self.analyzer.analyze_audio(retry_audio, track.sample_rate, channel=track.channel, file=track.file, instrument=track.instrument, group=track.group)
            retry_verdict = _validation_verdict(retry, before_metrics, retry_metrics)
            if retry_verdict["accepted"]:
                audio_by_channel[decision.channel] = retry_audio
                accepted.append(replace(retry, applied=True, applied_to_audio=True, validation={**retry_verdict, "retry_after_rollback": True}))
            else:
                rejected.append(replace(decision, rejected=True, rollback=True, rejection_reason=retry_verdict["reason"], validation={**retry_verdict, "retry_failed": True}))

        summary = {
            "proposed_count": len(decisions),
            "applied_count": len([d for d in accepted if d.applied]),
            "applied_to_audio_count": len([d for d in accepted if d.applied_to_audio]),
            "rejected_count": len(rejected),
            "rollback_retry_count": len([d for d in accepted if d.rollback]),
            "report_only": bool(settings.report_only),
        }
        return audio_by_channel, accepted, rejected, summary


class OSCEQCommandAdapter:
    """Create abstract dry-run/apply-ready EQ commands without sending them."""

    def commands_for(self, decisions: list[EQDecision], settings: AutoEQSettings) -> list[dict[str, Any]]:
        commands = []
        mode = "apply" if settings.osc_apply and not settings.osc_dry_run else "dry_run"
        smoothing_ms = int(settings.limits.get("smoothing_ms", 1000 if settings.mode == "live" else 0))
        band_index_by_channel: dict[tuple[str, int | None], int] = {}
        for decision in decisions:
            if not decision.applied:
                continue
            key = (decision.target_type, decision.channel)
            band = band_index_by_channel.get(key, 0) + 1
            band_index_by_channel[key] = band
            target = decision.target_type
            commands.extend(
                [
                    _osc_command(decision, f"eq.band.{band}.frequency", decision.frequency_hz, target, mode, smoothing_ms),
                    _osc_command(decision, f"eq.band.{band}.gain", decision.gain_db, target, mode, smoothing_ms),
                    _osc_command(decision, f"eq.band.{band}.q", decision.q, target, mode, smoothing_ms),
                ]
            )
        return commands


class EQReportWriter:
    """Format engine results using stable report keys."""

    @staticmethod
    def build(
        *,
        settings: AutoEQSettings,
        channel_analysis: list[dict[str, Any]],
        masking_relations: list[MaskingRelation],
        proposed: list[EQDecision],
        applied: list[EQDecision],
        rejected: list[EQDecision],
        group_decisions: list[EQDecision],
        master_decisions: list[EQDecision],
        validation_summary: dict[str, Any],
        osc_commands: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "autoeq_settings": {
                "enabled": settings.enabled,
                "mode": settings.mode,
                "safety_profile": settings.safety_profile,
                "max_static_gain_db": settings.limits.get("max_static_gain_db"),
                "max_dynamic_cut_db": settings.limits.get("max_dynamic_cut_db"),
                "report_only": settings.report_only,
                "osc_dry_run": settings.osc_dry_run,
            },
            "autoeq_channel_analysis": channel_analysis,
            "autoeq_decisions": [d.to_dict() for d in applied],
            "autoeq_proposed_decisions": [d.to_dict() for d in proposed],
            "autoeq_rejected_decisions": [d.to_dict() for d in rejected],
            "autoeq_group_decisions": [d.to_dict() for d in group_decisions],
            "autoeq_master_decisions": [d.to_dict() for d in master_decisions],
            "autoeq_masking_relations": [m.to_dict() for m in masking_relations],
            "autoeq_validation_summary": validation_summary,
            "autoeq_osc_commands": osc_commands,
        }


class AutoEQEngine:
    """High-level AutoEQ orchestrator."""

    def __init__(self, settings: AutoEQSettings | Mapping[str, Any] | None = None):
        self.settings = settings if isinstance(settings, AutoEQSettings) else AutoEQSettings.from_mapping(settings)
        self.profile_library = InstrumentProfileLibrary(self.settings.priorities)
        self.spectral_analyzer = SpectralAnalyzer()
        self.masking_analyzer = MaskingAnalyzer(self.profile_library)
        self.decision_engine = EQDecisionEngine(self.profile_library)
        self.dynamic_engine = DynamicEQDecisionEngine()
        self.safety_limiter = EQSafetyLimiter(self.profile_library)
        self.validator = EQResultValidator(self.spectral_analyzer)
        self.osc_adapter = OSCEQCommandAdapter()

    def run(self, tracks: list[AutoEQTrack], *, master_audio: np.ndarray | None = None) -> AutoEQRunResult:
        processed = {track.channel: ensure_stereo(track.audio).copy() for track in tracks}
        if not self.settings.enabled:
            report = EQReportWriter.build(
                settings=self.settings,
                channel_analysis=[],
                masking_relations=[],
                proposed=[],
                applied=[],
                rejected=[],
                group_decisions=[],
                master_decisions=[],
                validation_summary={"enabled": False, "reason": "autoeq_disabled"},
                osc_commands=[],
            )
            return AutoEQRunResult(processed, report)

        metrics = self.spectral_analyzer.analyze_tracks(tracks)
        masking = self.masking_analyzer.analyze(tracks, metrics)
        static_decisions = self.decision_engine.generate_channel_decisions(tracks, metrics, masking, self.settings)
        dynamic_decisions = self.dynamic_engine.generate(tracks, metrics, masking, self.settings)
        group_decisions = self.decision_engine.generate_group_decisions(tracks, metrics, self.settings)
        master_metrics = self.spectral_analyzer.analyze_audio(master_audio, tracks[0].sample_rate, file="MASTER", instrument="master") if master_audio is not None and tracks else None
        master_decisions = self.decision_engine.generate_master_decisions(master_metrics, self.settings)
        proposed = [*static_decisions, *dynamic_decisions, *group_decisions, *master_decisions]
        limited, safety_rejected = self.safety_limiter.limit(proposed, self.settings)
        processed, applied, validation_rejected, validation_summary = self.validator.apply_and_validate(tracks, limited, self.settings)
        rejected = [*safety_rejected, *validation_rejected]
        osc_commands = self.osc_adapter.commands_for(applied, self.settings)
        report = EQReportWriter.build(
            settings=self.settings,
            channel_analysis=[
                {
                    "channel": item["channel"],
                    "file": item["file"],
                    "instrument": item["instrument"],
                    "spectral_metrics": item,
                    "masking_relations": [
                        rel.to_dict()
                        for rel in masking
                        if rel.masked_channel == item["channel"] or rel.masker_channel == item["channel"]
                    ],
                    "detected_problems": _detected_problems(item),
                }
                for item in metrics.values()
            ],
            masking_relations=masking,
            proposed=proposed,
            applied=applied,
            rejected=rejected,
            group_decisions=[d for d in applied if d.target_type == "bus"],
            master_decisions=[d for d in applied if d.target_type == "master"],
            validation_summary=validation_summary,
            osc_commands=osc_commands,
        )
        return AutoEQRunResult(processed, report)


def normalize_instrument(instrument: str) -> str:
    key = " ".join(str(instrument).lower().replace("-", "_").replace(" ", "_").split())
    aliases = {
        "bass_guitar": "bass",
        "backing_vocal": "back_vocal",
        "backs": "back_vocal",
        "bvox": "back_vocal",
        "vocal": "lead_vocal",
        "vocals": "lead_vocal",
        "snare": "snare_top",
        "rack_tom": "tom",
        "overhead": "overhead_pair",
        "oh_l": "overhead_l",
        "oh_r": "overhead_r",
        "electric_guitar": "guitar",
        "music_stem": "playback",
    }
    return aliases.get(key, key)


def _detected_problems(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    problems = []
    thresholds = {
        "mud_score": 0.66,
        "harshness_score": 0.66,
        "sibilance_score": 0.58,
        "low_end_excess_score": 0.78,
        "low_mid_buildup_score": 0.66,
    }
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 0.0))
        if value >= threshold:
            problems.append({"metric": key, "score": round(value, 3), "threshold": threshold})
    if float(metrics.get("peak_dbfs", -120.0)) > -1.0:
        problems.append({"metric": "peak_dbfs", "score": metrics.get("peak_dbfs"), "threshold": -1.0})
    return problems


def _channel_decision(
    track: AutoEQTrack,
    metrics: Mapping[str, Any],
    eq_type: str,
    frequency_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    confidence: float,
    *,
    dynamic: bool = False,
    trigger_band: tuple[float, float] | None = None,
    linked_group: str | None = None,
) -> EQDecision:
    return EQDecision(
        channel=track.channel,
        file=track.file,
        instrument=track.instrument,
        eq_type=eq_type if not dynamic else "dynamic_bell_cut" if gain_db < 0 else "dynamic_bell_boost",
        frequency_hz=float(frequency_hz),
        q=float(q),
        gain_db=float(gain_db),
        dynamic=dynamic,
        sidechain_source=None,
        trigger_band_hz=trigger_band,
        attack_ms=25.0 if dynamic else None,
        release_ms=180.0 if dynamic else None,
        threshold_metric="band_activity" if dynamic else "integrated_spectral_score",
        reason=reason,
        confidence=float(_clamp01(confidence)),
        source_metrics_before=dict(metrics),
        linked_group=linked_group,
    )


def _decision_from_relation(
    relation: MaskingRelation,
    masker: AutoEQTrack,
    *,
    dynamic: bool,
    gain_db: float,
    q: float,
    reason: str,
    metrics: Mapping[str, Any],
) -> EQDecision:
    low, high = relation.suggested_band
    return EQDecision(
        channel=masker.channel,
        file=masker.file,
        instrument=masker.instrument,
        eq_type="dynamic_bell_cut" if dynamic else "static_bell_cut",
        frequency_hz=math.sqrt(low * high),
        q=q,
        gain_db=gain_db,
        dynamic=dynamic,
        sidechain_source=relation.masked_source if dynamic else None,
        trigger_band_hz=relation.suggested_band,
        attack_ms=35.0 if dynamic else None,
        release_ms=220.0 if dynamic else None,
        threshold_metric="masking_score",
        reason=reason,
        confidence=relation.confidence,
        source_metrics_before=dict(metrics),
        linked_group="overhead_pair" if masker.instrument in {"overhead_l", "overhead_r", "overhead_pair"} else None,
    )


def _bus_decision(group: str, freq: float, gain: float, q: float, eq_type: str, reason: str, confidence: float) -> EQDecision:
    return EQDecision(None, group, group, "static_bell_cut", freq, q, gain, target_type="bus", reason=reason, confidence=confidence)


def _osc_command(decision: EQDecision, parameter: str, value: float, target: str, mode: str, smoothing_ms: int) -> dict[str, Any]:
    return {
        "target": target,
        "channel": decision.channel,
        "parameter": parameter,
        "value": round(float(value), 4),
        "frequency_hz": round(float(decision.frequency_hz), 2),
        "q": round(float(decision.q), 3),
        "mode": mode,
        "smoothing_ms": smoothing_ms,
        "reason": decision.reason,
    }


def _nearest_profile_band(bands: tuple[FrequencyRange, ...], target: tuple[float, float]) -> FrequencyRange | None:
    low, high = target
    candidates = [band for band in bands if band.high_hz >= low and band.low_hz <= high]
    if not candidates:
        return None
    center = math.sqrt(low * high)
    return min(candidates, key=lambda band: abs(math.log2(band.center_hz / center)))


def _scale_cut(score: float, minimum: float, maximum: float) -> float:
    return float(np.clip(minimum + max(0.0, score - 0.55) * 2.1, minimum, maximum))


def _tracks_matching(tracks: list[AutoEQTrack], instruments: set[str]) -> list[AutoEQTrack]:
    return [track for track in tracks if normalize_instrument(track.instrument) in instruments or track.instrument in instruments]


def _group_for_instrument(instrument: str) -> str:
    key = normalize_instrument(instrument)
    if key in {"kick", "snare_top", "snare_bottom", "tom", "floor_tom", "overhead_l", "overhead_r", "overhead_pair"}:
        return "drums"
    if key == "bass":
        return "low_end_group"
    if key in {"lead_vocal", "back_vocal"}:
        return "vocal_group"
    if key in {"guitar", "guitar_l", "guitar_r", "accordion", "keys", "playback", "playback_l", "playback_r"}:
        return "harmonic_group"
    return "playback"


def _validation_verdict(decision: EQDecision, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_score = _target_problem_score(decision, before)
    after_score = _target_problem_score(decision, after)
    reason_text = decision.reason.lower()
    problematic_boost = decision.gain_db > 0.0 and any(
        token in reason_text
        for token in ("mud", "harsh", "sibilance", "mask", "low-end", "resonance")
    )
    peak_ok = float(after.get("peak_dbfs", -120.0)) <= 0.0
    mono_before = float(before.get("mono_compatibility", 1.0))
    mono_after = float(after.get("mono_compatibility", 1.0))
    mono_ok = mono_after >= mono_before - 0.08
    tilt_before = abs(float(before.get("spectral_tilt_db_per_oct", 0.0)))
    tilt_after = abs(float(after.get("spectral_tilt_db_per_oct", 0.0)))
    tilt_ok = tilt_after <= tilt_before + 1.5
    improvement = before_score - after_score if decision.gain_db < 0 else after_score - before_score
    accepted = (not problematic_boost) and peak_ok and mono_ok and tilt_ok and improvement >= -0.03
    reason = "accepted" if accepted else "worsened target metrics"
    if problematic_boost:
        reason = "problematic band boost rejected"
    elif not peak_ok:
        reason = "LUFS/peak not safe after EQ"
    elif not mono_ok:
        reason = "worsened mono compatibility"
    elif not tilt_ok:
        reason = "spectral tilt worsened"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "before_metric": round(before_score, 4),
        "after_metric": round(after_score, 4),
        "improvement_score": round(improvement, 4),
        "peak_ok": bool(peak_ok),
        "mono_ok": bool(mono_ok),
        "spectral_tilt_ok": bool(tilt_ok),
    }


def _target_problem_score(decision: EQDecision, metrics: Mapping[str, Any]) -> float:
    reason = decision.reason.lower()
    if "mud" in reason or 160.0 <= decision.frequency_hz <= 450.0:
        return float(metrics.get("mud_score", 0.0))
    if "harsh" in reason or 3000.0 <= decision.frequency_hz <= 9000.0:
        return float(metrics.get("harshness_score", 0.0))
    if "sibilance" in reason or "de-ess" in reason:
        return float(metrics.get("sibilance_score", 0.0))
    if "low-end" in reason or decision.frequency_hz < 140.0:
        return float(metrics.get("low_end_excess_score", 0.0))
    if "mask" in reason or "vocal" in reason:
        band = dict(metrics.get("band_ratios", {}) or {})
        return float(band.get("presence", 0.0) + band.get("upper_presence", 0.0))
    return float(metrics.get("low_mid_buildup_score", 0.0))


def _apply_eq_decision(
    audio: np.ndarray,
    sample_rate: int,
    decision: EQDecision,
    *,
    audio_by_channel: Mapping[int, np.ndarray],
    tracks: Mapping[int, AutoEQTrack],
) -> np.ndarray:
    stereo = ensure_stereo(audio)
    if decision.eq_type == "hpf":
        return _filter_stereo(stereo, sample_rate, "highpass", decision.frequency_hz)
    if decision.eq_type == "lpf":
        return _filter_stereo(stereo, sample_rate, "lowpass", decision.frequency_hz)
    processed = _peaking_stereo(stereo, sample_rate, decision.frequency_hz, decision.gain_db, decision.q)
    if not decision.dynamic:
        return processed
    env = _dynamic_envelope(stereo, sample_rate, decision.trigger_band_hz or (decision.frequency_hz / 1.4, decision.frequency_hz * 1.4))
    if decision.sidechain_source:
        sidechain_track = next((track for track in tracks.values() if track.file == decision.sidechain_source), None)
        if sidechain_track is not None and sidechain_track.channel in audio_by_channel:
            env = _dynamic_envelope(audio_by_channel[sidechain_track.channel], sample_rate, decision.trigger_band_hz or (decision.frequency_hz / 1.4, decision.frequency_hz * 1.4))
    return (stereo * (1.0 - env[:, None]) + processed * env[:, None]).astype(np.float32)


def _filter_stereo(audio: np.ndarray, sample_rate: int, btype: str, freq: float) -> np.ndarray:
    try:
        from scipy.signal import butter, lfilter

        freq = float(np.clip(freq, 20.0, sample_rate * 0.49))
        b, a = butter(2, freq / (sample_rate * 0.5), btype=btype)
        return np.column_stack([lfilter(b, a, audio[:, 0]), lfilter(b, a, audio[:, 1])]).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def _peaking_stereo(audio: np.ndarray, sample_rate: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    return np.column_stack(
        [
            _peaking_eq(audio[:, 0], sample_rate, freq, gain_db, q),
            _peaking_eq(audio[:, 1], sample_rate, freq, gain_db, q),
        ]
    ).astype(np.float32)


def _peaking_eq(x: np.ndarray, sample_rate: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    if abs(gain_db) < 1e-5 or freq <= 0.0 or freq >= sample_rate * 0.49:
        return np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import lfilter

        a = db_to_amp(gain_db)
        w0 = 2.0 * math.pi * float(freq) / sample_rate
        alpha = math.sin(w0) / (2.0 * max(float(q), 0.05))
        cos_w0 = math.cos(w0)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        aa = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return lfilter(b, aa, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _dynamic_envelope(audio: np.ndarray, sample_rate: int, band: tuple[float, float]) -> np.ndarray:
    mono = np.mean(ensure_stereo(audio), axis=1)
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    banded = _filter_mono(_filter_mono(mono, sample_rate, "highpass", band[0]), sample_rate, "lowpass", band[1])
    frame = max(128, int(0.030 * sample_rate))
    hop = max(64, frame // 2)
    env = np.zeros_like(mono, dtype=np.float32)
    values = []
    for start in range(0, max(1, len(mono) - frame + 1), hop):
        value = float(np.sqrt(np.mean(np.square(banded[start:start + frame])) + EPS))
        values.append(value)
    if not values:
        return env
    threshold = float(np.percentile(values, 70))
    scale = max(float(np.percentile(values, 95)) - threshold, EPS)
    for idx, start in enumerate(range(0, max(1, len(mono) - frame + 1), hop)):
        amount = float(np.clip((values[idx] - threshold) / scale, 0.0, 1.0))
        env[start:min(len(env), start + frame)] = np.maximum(env[start:min(len(env), start + frame)], amount)
    try:
        from scipy.ndimage import uniform_filter1d

        env = uniform_filter1d(env, size=max(1, int(0.060 * sample_rate))).astype(np.float32)
    except Exception:
        pass
    return np.clip(env, 0.0, 1.0).astype(np.float32)


def _filter_mono(x: np.ndarray, sample_rate: int, btype: str, freq: float) -> np.ndarray:
    try:
        from scipy.signal import butter, lfilter

        freq = float(np.clip(freq, 20.0, sample_rate * 0.49))
        b, a = butter(2, freq / (sample_rate * 0.5), btype=btype)
        return lfilter(b, a, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _spectrum(mono: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        return {"freqs": np.array([], dtype=np.float64), "mag": np.array([], dtype=np.float64), "power": np.array([], dtype=np.float64)}
    fft_size = int(2 ** np.ceil(np.log2(max(2048, min(65536, mono.size)))))
    block = _active_block(mono, fft_size)
    windowed = block * np.hanning(block.size)
    mag = np.abs(np.fft.rfft(windowed)) + EPS
    freqs = np.fft.rfftfreq(block.size, 1.0 / float(sample_rate))
    return {"freqs": freqs, "mag": mag, "power": np.square(mag)}


def _active_block(mono: np.ndarray, fft_size: int) -> np.ndarray:
    if mono.size < fft_size:
        return np.pad(mono, (0, fft_size - mono.size))
    step = max(1, fft_size // 2)
    best_start = 0
    best_power = -1.0
    for start in range(0, mono.size - fft_size + 1, step):
        block = mono[start:start + fft_size]
        power = float(np.mean(block * block))
        if power > best_power:
            best_power = power
            best_start = start
    return mono[best_start:best_start + fft_size]


def _sum_band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    if freqs.size == 0:
        return EPS
    idx = (freqs >= low) & (freqs < high)
    if not np.any(idx):
        return EPS
    return float(np.sum(power[idx]) + EPS)


def _band_energy_from_metrics(metrics: Mapping[str, Any], band: tuple[float, float]) -> float:
    ratios = dict(metrics.get("band_energy_db", {}) or {})
    powers = []
    for name, (low, high) in LOG_BANDS.items():
        overlap = max(0.0, min(high, band[1]) - max(low, band[0]))
        if overlap <= 0:
            continue
        db = float(ratios.get(name, -120.0))
        powers.append((10.0 ** (db / 10.0)) * (overlap / max(1.0, high - low)))
    if not powers:
        return -120.0
    return _power_to_db(sum(powers))


def _power_to_db(value: float) -> float:
    return float(10.0 * math.log10(max(float(value), EPS)))


def _centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    if freqs.size == 0 or mag.size == 0:
        return 0.0
    total = float(np.sum(mag) + EPS)
    return float(np.sum(freqs * mag) / total)


def _spectral_tilt(freqs: np.ndarray, mag: np.ndarray) -> float:
    if freqs.size < 8:
        return 0.0
    idx = (freqs >= 80.0) & (freqs <= 12000.0)
    if np.count_nonzero(idx) < 8:
        return 0.0
    x = np.log2(freqs[idx])
    y = 20.0 * np.log10(mag[idx] + EPS)
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return 0.0


def _resonance_candidates(freqs: np.ndarray, mag: np.ndarray) -> list[dict[str, Any]]:
    if freqs.size < 16:
        return []
    db = 20.0 * np.log10(mag + EPS)
    idx = (freqs >= 80.0) & (freqs <= 9000.0)
    if np.count_nonzero(idx) < 8:
        return []
    sub_freqs = freqs[idx]
    sub_db = db[idx]
    try:
        from scipy.signal import find_peaks

        peaks, props = find_peaks(sub_db, prominence=6.0, distance=4)
        candidates = [
            {
                "frequency_hz": round(float(sub_freqs[p]), 2),
                "level_db": round(float(sub_db[p]), 2),
                "prominence_db": round(float(props["prominences"][i]), 2),
            }
            for i, p in enumerate(peaks)
        ]
    except Exception:
        local_median = float(np.median(sub_db))
        candidates = [
            {"frequency_hz": round(float(f), 2), "level_db": round(float(v), 2), "prominence_db": round(float(v - local_median), 2)}
            for f, v in zip(sub_freqs, sub_db)
            if v - local_median > 8.0
        ]
    candidates.sort(key=lambda item: float(item["prominence_db"]), reverse=True)
    return candidates[:6]


def _spectral_scores(ratios: Mapping[str, float], instrument: str) -> dict[str, float]:
    low_mid = float(ratios.get("mud", 0.0) + ratios.get("boxiness", 0.0))
    harsh = float(ratios.get("upper_presence", 0.0) + ratios.get("harshness", 0.0))
    mud = float(ratios.get("mud", 0.0))
    presence = float(ratios.get("intelligibility", 0.0) + ratios.get("presence", 0.0) + ratios.get("upper_presence", 0.0))
    air = float(ratios.get("air", 0.0) + ratios.get("polish", 0.0))
    low = float(ratios.get("sub", 0.0) + ratios.get("bass", 0.0) + ratios.get("punch", 0.0))
    vocalish = normalize_instrument(instrument) in {"lead_vocal", "back_vocal"}
    return {
        "low_mid_buildup_score": round(_clamp01(low_mid / 0.35), 4),
        "harshness_score": round(_clamp01(harsh / 0.32), 4),
        "mud_score": round(_clamp01(mud / 0.22), 4),
        "presence_score": round(_clamp01(presence / 0.34), 4),
        "air_score": round(_clamp01(air / 0.22), 4),
        "sibilance_score": round(_clamp01((float(ratios.get("harshness", 0.0)) + float(ratios.get("air", 0.0))) / (0.20 if vocalish else 0.30)), 4),
        "low_end_excess_score": round(_clamp01(low / 0.45), 4),
    }


def _activity_metrics(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if mono.size == 0:
        return {"active_ratio": 0.0, "windows": []}
    frame = max(256, int(0.100 * sample_rate))
    hop = max(128, frame // 2)
    starts = list(range(0, max(1, len(mono) - frame + 1), hop))
    rms = np.asarray([float(np.sqrt(np.mean(np.square(mono[start:start + frame])) + EPS)) for start in starts], dtype=np.float64)
    if rms.size == 0:
        return {"active_ratio": 0.0, "windows": []}
    threshold = max(float(np.percentile(rms, 65)), float(np.max(rms)) * 0.18)
    active = rms >= threshold
    active_ratio = float(np.mean(active))
    windows = [
        {"start": round(starts[i] / sample_rate, 3), "end": round((starts[i] + frame) / sample_rate, 3)}
        for i, value in enumerate(active)
        if bool(value)
    ][:8]
    return {"active_ratio": active_ratio, "windows": windows}


def _short_term_lufs(audio: np.ndarray, sample_rate: int) -> float:
    arr = ensure_stereo(audio)
    if arr.shape[0] <= 0:
        return -120.0
    window = min(arr.shape[0], max(1, int(3.0 * sample_rate)))
    return _integrated_lufs(arr[:window], sample_rate)


def _integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    try:
        import pyloudnorm as pyln

        if arr.shape[0] > int(0.4 * sample_rate):
            value = float(pyln.Meter(int(sample_rate)).integrated_loudness(arr.astype(np.float64, copy=False)))
            if np.isfinite(value):
                return value
    except Exception:
        pass
    return amp_to_db(float(np.sqrt(np.mean(np.square(arr)) + EPS))) - 0.691


def _phase_correlation(stereo: np.ndarray) -> float:
    audio = ensure_stereo(stereo)
    if audio.shape[0] < 2:
        return 1.0
    left = audio[:, 0]
    right = audio[:, 1]
    if float(np.std(left)) <= EPS or float(np.std(right)) <= EPS:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in dict(base).items()}
    for key, value in dict(override).items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
