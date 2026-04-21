"""
Persistent soundcheck profile storage and learned target corridor helpers.

Profiles are intentionally JSON-based and lightweight so they can be layered
onto the existing AutoSoundcheckEngine without changing the runtime model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

from autofoh_detectors import aggregate_analysis_features
from autofoh_models import AnalysisFeatures, TargetCorridor


PHASE_TARGET_DEFAULTS = {
    "SILENCE_CAPTURE": {
        "green_delta_db": 3.0,
        "yellow_delta_db": 6.0,
        "red_delta_db": 10.0,
        "channel_level_tolerance_db": 6.0,
        "stem_level_tolerance_db": 6.0,
    },
    "LINE_CHECK": {
        "green_delta_db": 2.5,
        "yellow_delta_db": 5.0,
        "red_delta_db": 8.0,
        "channel_level_tolerance_db": 4.5,
        "stem_level_tolerance_db": 4.5,
    },
    "SOURCE_LEARNING": {
        "green_delta_db": 2.0,
        "yellow_delta_db": 4.0,
        "red_delta_db": 7.0,
        "channel_level_tolerance_db": 3.5,
        "stem_level_tolerance_db": 3.5,
    },
    "STEM_LEARNING": {
        "green_delta_db": 1.75,
        "yellow_delta_db": 3.5,
        "red_delta_db": 6.5,
        "channel_level_tolerance_db": 3.0,
        "stem_level_tolerance_db": 3.0,
    },
    "FULL_BAND_LEARNING": {
        "green_delta_db": 1.5,
        "yellow_delta_db": 3.0,
        "red_delta_db": 6.0,
        "channel_level_tolerance_db": 2.5,
        "stem_level_tolerance_db": 2.5,
    },
    "SNAPSHOT_LOCK": {
        "green_delta_db": 1.25,
        "yellow_delta_db": 2.5,
        "red_delta_db": 5.0,
        "channel_level_tolerance_db": 2.0,
        "stem_level_tolerance_db": 2.0,
    },
}

LOW_END_SOURCE_ROLES = {
    "kick",
    "kick_in",
    "kick_out",
    "bass",
    "bass_di",
    "bass_mic",
    "synth_bass",
}
VOICE_SOURCE_ROLES = {
    "lead_vocal",
    "backing_vocal",
    "vocal",
    "main_vox",
}
BRIGHT_SOURCE_ROLES = {
    "hihat",
    "ride",
    "cymbals",
    "overhead",
    "oh_l",
    "oh_r",
    "click",
    "talkback",
    "fx_return",
    "reverb_return",
    "delay_return",
}
MIDRANGE_MUSIC_SOURCE_ROLES = {
    "guitar",
    "electric_guitar",
    "acoustic_guitar",
    "lead_guitar",
    "rhythm_guitar",
    "keys",
    "piano",
    "organ",
    "synth",
    "pad",
    "lead_synth",
    "playback",
    "tracks",
    "music",
    "tom",
    "rack_tom",
    "floor_tom",
    "snare",
    "snare_top",
    "snare_bottom",
}


@dataclass
class PhaseLearningSnapshot:
    phase_name: str
    runtime_state: str
    captured_at: float = field(default_factory=time.time)
    active_channel_ids: Sequence[int] = field(default_factory=list)
    channel_features: Dict[int, AnalysisFeatures] = field(default_factory=dict)
    stem_features: Dict[str, AnalysisFeatures] = field(default_factory=dict)
    master_features: AnalysisFeatures = field(default_factory=AnalysisFeatures)
    metadata: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase_name": self.phase_name,
            "runtime_state": self.runtime_state,
            "captured_at": float(self.captured_at),
            "active_channel_ids": [int(value) for value in self.active_channel_ids],
            "channel_features": {
                str(channel_id): features.to_dict()
                for channel_id, features in self.channel_features.items()
            },
            "stem_features": {
                str(stem_name): features.to_dict()
                for stem_name, features in self.stem_features.items()
            },
            "master_features": self.master_features.to_dict(),
            "metadata": dict(self.metadata),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "PhaseLearningSnapshot":
        payload = payload or {}
        return cls(
            phase_name=str(payload.get("phase_name", "")),
            runtime_state=str(payload.get("runtime_state", "")),
            captured_at=float(payload.get("captured_at", time.time())),
            active_channel_ids=[
                int(value)
                for value in list(payload.get("active_channel_ids", []))
            ],
            channel_features={
                int(channel_id): AnalysisFeatures.from_dict(channel_payload)
                for channel_id, channel_payload in dict(payload.get("channel_features", {})).items()
            },
            stem_features={
                str(stem_name): AnalysisFeatures.from_dict(stem_payload)
                for stem_name, stem_payload in dict(payload.get("stem_features", {})).items()
            },
            master_features=AnalysisFeatures.from_dict(payload.get("master_features")),
            metadata=dict(payload.get("metadata", {})),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class PhaseLearningTarget:
    phase_name: str
    runtime_state: str = ""
    target_corridor: TargetCorridor = field(default_factory=TargetCorridor.default_intergenre)
    expected_channel_rms_db: Dict[int, float] = field(default_factory=dict)
    expected_stem_rms_db: Dict[str, float] = field(default_factory=dict)
    expected_source_role_rms_db: Dict[str, float] = field(default_factory=dict)
    noise_floor_db_by_channel: Dict[int, float] = field(default_factory=dict)
    hpf_frequency_range_hz_by_channel: Dict[int, Dict[str, float]] = field(default_factory=dict)
    compressor_threshold_range_db_by_channel: Dict[int, Dict[str, float]] = field(default_factory=dict)
    compressor_ratio_range_by_channel: Dict[int, Dict[str, float]] = field(default_factory=dict)
    fx_send_level_range_db_by_channel: Dict[int, Dict[str, float]] = field(default_factory=dict)
    lead_masking_margin_db: float = 0.0
    channel_level_tolerance_db: float = 3.0
    stem_level_tolerance_db: float = 3.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase_name": self.phase_name,
            "runtime_state": self.runtime_state,
            "target_corridor": self.target_corridor.to_dict(),
            "expected_channel_rms_db": {
                str(channel_id): float(level_db)
                for channel_id, level_db in self.expected_channel_rms_db.items()
            },
            "expected_stem_rms_db": {
                str(stem_name): float(level_db)
                for stem_name, level_db in self.expected_stem_rms_db.items()
            },
            "expected_source_role_rms_db": {
                str(source_role): float(level_db)
                for source_role, level_db in self.expected_source_role_rms_db.items()
            },
            "noise_floor_db_by_channel": {
                str(channel_id): float(level_db)
                for channel_id, level_db in self.noise_floor_db_by_channel.items()
            },
            "hpf_frequency_range_hz_by_channel": {
                str(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in self.hpf_frequency_range_hz_by_channel.items()
            },
            "compressor_threshold_range_db_by_channel": {
                str(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in self.compressor_threshold_range_db_by_channel.items()
            },
            "compressor_ratio_range_by_channel": {
                str(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in self.compressor_ratio_range_by_channel.items()
            },
            "fx_send_level_range_db_by_channel": {
                str(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in self.fx_send_level_range_db_by_channel.items()
            },
            "lead_masking_margin_db": float(self.lead_masking_margin_db),
            "channel_level_tolerance_db": float(self.channel_level_tolerance_db),
            "stem_level_tolerance_db": float(self.stem_level_tolerance_db),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "PhaseLearningTarget":
        payload = payload or {}
        return cls(
            phase_name=str(payload.get("phase_name", "")),
            runtime_state=str(payload.get("runtime_state", "")),
            target_corridor=TargetCorridor.from_dict(payload.get("target_corridor")),
            expected_channel_rms_db={
                int(channel_id): float(level_db)
                for channel_id, level_db in dict(payload.get("expected_channel_rms_db", {})).items()
            },
            expected_stem_rms_db={
                str(stem_name): float(level_db)
                for stem_name, level_db in dict(payload.get("expected_stem_rms_db", {})).items()
            },
            expected_source_role_rms_db={
                str(source_role): float(level_db)
                for source_role, level_db in dict(payload.get("expected_source_role_rms_db", {})).items()
            },
            noise_floor_db_by_channel={
                int(channel_id): float(level_db)
                for channel_id, level_db in dict(payload.get("noise_floor_db_by_channel", {})).items()
            },
            hpf_frequency_range_hz_by_channel={
                int(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in dict(
                    payload.get("hpf_frequency_range_hz_by_channel", {})
                ).items()
            },
            compressor_threshold_range_db_by_channel={
                int(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in dict(
                    payload.get("compressor_threshold_range_db_by_channel", {})
                ).items()
            },
            compressor_ratio_range_by_channel={
                int(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in dict(
                    payload.get("compressor_ratio_range_by_channel", {})
                ).items()
            },
            fx_send_level_range_db_by_channel={
                int(channel_id): {
                    str(key): float(value)
                    for key, value in dict(bounds).items()
                }
                for channel_id, bounds in dict(
                    payload.get("fx_send_level_range_db_by_channel", {})
                ).items()
            },
            lead_masking_margin_db=float(payload.get("lead_masking_margin_db", 0.0)),
            channel_level_tolerance_db=float(payload.get("channel_level_tolerance_db", 3.0)),
            stem_level_tolerance_db=float(payload.get("stem_level_tolerance_db", 3.0)),
        )


@dataclass
class ChannelSoundcheckProfile:
    channel_id: int
    name: str
    source_role: str
    stem_roles: Sequence[str] = field(default_factory=list)
    allowed_controls: Sequence[str] = field(default_factory=list)
    priority: float = 0.0
    noise_floor_db: float = -100.0
    nominal_rms_db: float = -100.0
    max_rms_db: float = -100.0
    peak_db: float = -100.0
    crest_factor_range_db: Sequence[float] = field(default_factory=list)
    spectral_fingerprint: Dict[str, float] = field(default_factory=dict)
    activity_threshold_db: float = -60.0
    problem_bands: Sequence[str] = field(default_factory=list)
    analysis_features: AnalysisFeatures = field(default_factory=AnalysisFeatures)

    def to_dict(self) -> Dict[str, object]:
        return {
            "channel_id": int(self.channel_id),
            "name": self.name,
            "source_role": self.source_role,
            "stem_roles": list(self.stem_roles),
            "allowed_controls": list(self.allowed_controls),
            "priority": float(self.priority),
            "noise_floor_db": float(self.noise_floor_db),
            "nominal_rms_db": float(self.nominal_rms_db),
            "max_rms_db": float(self.max_rms_db),
            "peak_db": float(self.peak_db),
            "crest_factor_range_db": [float(value) for value in self.crest_factor_range_db],
            "spectral_fingerprint": {
                str(key): float(value)
                for key, value in self.spectral_fingerprint.items()
            },
            "activity_threshold_db": float(self.activity_threshold_db),
            "problem_bands": list(self.problem_bands),
            "analysis_features": self.analysis_features.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "ChannelSoundcheckProfile":
        payload = payload or {}
        return cls(
            channel_id=int(payload.get("channel_id", 0)),
            name=str(payload.get("name", "")),
            source_role=str(payload.get("source_role", "unknown")),
            stem_roles=list(payload.get("stem_roles", [])),
            allowed_controls=list(payload.get("allowed_controls", [])),
            priority=float(payload.get("priority", 0.0)),
            noise_floor_db=float(payload.get("noise_floor_db", -100.0)),
            nominal_rms_db=float(payload.get("nominal_rms_db", -100.0)),
            max_rms_db=float(payload.get("max_rms_db", -100.0)),
            peak_db=float(payload.get("peak_db", -100.0)),
            crest_factor_range_db=[
                float(value)
                for value in list(payload.get("crest_factor_range_db", []))
            ],
            spectral_fingerprint={
                str(key): float(value)
                for key, value in dict(payload.get("spectral_fingerprint", {})).items()
            },
            activity_threshold_db=float(payload.get("activity_threshold_db", -60.0)),
            problem_bands=list(payload.get("problem_bands", [])),
            analysis_features=AnalysisFeatures.from_dict(
                payload.get("analysis_features")
            ),
        )


@dataclass
class StemSoundcheckProfile:
    stem_name: str
    nominal_rms_db: float = -100.0
    peak_db: float = -100.0
    band_energy_distribution_db: Dict[str, float] = field(default_factory=dict)
    contribution_profile: Dict[str, float] = field(default_factory=dict)
    typical_correction_range_db: float = 0.0
    analysis_features: AnalysisFeatures = field(default_factory=AnalysisFeatures)

    def to_dict(self) -> Dict[str, object]:
        return {
            "stem_name": self.stem_name,
            "nominal_rms_db": float(self.nominal_rms_db),
            "peak_db": float(self.peak_db),
            "band_energy_distribution_db": {
                str(key): float(value)
                for key, value in self.band_energy_distribution_db.items()
            },
            "contribution_profile": {
                str(key): float(value)
                for key, value in self.contribution_profile.items()
            },
            "typical_correction_range_db": float(self.typical_correction_range_db),
            "analysis_features": self.analysis_features.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "StemSoundcheckProfile":
        payload = payload or {}
        return cls(
            stem_name=str(payload.get("stem_name", "UNKNOWN")),
            nominal_rms_db=float(payload.get("nominal_rms_db", -100.0)),
            peak_db=float(payload.get("peak_db", -100.0)),
            band_energy_distribution_db={
                str(key): float(value)
                for key, value in dict(payload.get("band_energy_distribution_db", {})).items()
            },
            contribution_profile={
                str(key): float(value)
                for key, value in dict(payload.get("contribution_profile", {})).items()
            },
            typical_correction_range_db=float(
                payload.get("typical_correction_range_db", 0.0)
            ),
            analysis_features=AnalysisFeatures.from_dict(
                payload.get("analysis_features")
            ),
        )


@dataclass
class MasterSoundcheckProfile:
    nominal_rms_db: float = -100.0
    peak_db: float = -100.0
    lead_channel_ids: Sequence[int] = field(default_factory=list)
    expected_stem_levels_db: Dict[str, float] = field(default_factory=dict)
    expected_lead_masking_margin_db: float = 0.0
    target_corridor: TargetCorridor = field(default_factory=TargetCorridor.default_intergenre)
    analysis_features: AnalysisFeatures = field(default_factory=AnalysisFeatures)

    def to_dict(self) -> Dict[str, object]:
        return {
            "nominal_rms_db": float(self.nominal_rms_db),
            "peak_db": float(self.peak_db),
            "lead_channel_ids": [int(value) for value in self.lead_channel_ids],
            "expected_stem_levels_db": {
                str(key): float(value)
                for key, value in self.expected_stem_levels_db.items()
            },
            "expected_lead_masking_margin_db": float(self.expected_lead_masking_margin_db),
            "target_corridor": self.target_corridor.to_dict(),
            "analysis_features": self.analysis_features.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "MasterSoundcheckProfile":
        payload = payload or {}
        return cls(
            nominal_rms_db=float(payload.get("nominal_rms_db", -100.0)),
            peak_db=float(payload.get("peak_db", -100.0)),
            lead_channel_ids=[int(value) for value in list(payload.get("lead_channel_ids", []))],
            expected_stem_levels_db={
                str(key): float(value)
                for key, value in dict(payload.get("expected_stem_levels_db", {})).items()
            },
            expected_lead_masking_margin_db=float(
                payload.get("expected_lead_masking_margin_db", 0.0)
            ),
            target_corridor=TargetCorridor.from_dict(payload.get("target_corridor")),
            analysis_features=AnalysisFeatures.from_dict(payload.get("analysis_features")),
        )


@dataclass
class AutoFOHSoundcheckProfile:
    version: int = 1
    created_at: float = field(default_factory=time.time)
    name: str = "autofoh_soundcheck"
    sample_rate: int = 48000
    channel_count: int = 0
    target_corridor: TargetCorridor = field(default_factory=TargetCorridor.default_intergenre)
    channels: Dict[int, ChannelSoundcheckProfile] = field(default_factory=dict)
    stems: Dict[str, StemSoundcheckProfile] = field(default_factory=dict)
    master: MasterSoundcheckProfile = field(default_factory=MasterSoundcheckProfile)
    phase_snapshots: Dict[str, PhaseLearningSnapshot] = field(default_factory=dict)
    phase_targets: Dict[str, PhaseLearningTarget] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": int(self.version),
            "created_at": float(self.created_at),
            "name": self.name,
            "sample_rate": int(self.sample_rate),
            "channel_count": int(self.channel_count),
            "target_corridor": self.target_corridor.to_dict(),
            "channels": {
                str(channel_id): profile.to_dict()
                for channel_id, profile in self.channels.items()
            },
            "stems": {
                str(stem_name): profile.to_dict()
                for stem_name, profile in self.stems.items()
            },
            "master": self.master.to_dict(),
            "phase_snapshots": {
                str(phase_name): snapshot.to_dict()
                for phase_name, snapshot in self.phase_snapshots.items()
            },
            "phase_targets": {
                str(phase_name): target.to_dict()
                for phase_name, target in self.phase_targets.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "AutoFOHSoundcheckProfile":
        payload = payload or {}
        channels_payload = dict(payload.get("channels", {}))
        stems_payload = dict(payload.get("stems", {}))
        phases_payload = dict(payload.get("phase_snapshots", {}))
        targets_payload = dict(payload.get("phase_targets", {}))
        return cls(
            version=int(payload.get("version", 1)),
            created_at=float(payload.get("created_at", time.time())),
            name=str(payload.get("name", "autofoh_soundcheck")),
            sample_rate=int(payload.get("sample_rate", 48000)),
            channel_count=int(payload.get("channel_count", len(channels_payload))),
            target_corridor=TargetCorridor.from_dict(payload.get("target_corridor")),
            channels={
                int(channel_id): ChannelSoundcheckProfile.from_dict(channel_payload)
                for channel_id, channel_payload in channels_payload.items()
            },
            stems={
                str(stem_name): StemSoundcheckProfile.from_dict(stem_payload)
                for stem_name, stem_payload in stems_payload.items()
            },
            master=MasterSoundcheckProfile.from_dict(payload.get("master")),
            phase_snapshots={
                str(phase_name): PhaseLearningSnapshot.from_dict(snapshot_payload)
                for phase_name, snapshot_payload in phases_payload.items()
            },
            phase_targets={
                str(phase_name): PhaseLearningTarget.from_dict(target_payload)
                for phase_name, target_payload in targets_payload.items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


def _phase_target_defaults(phase_name: str) -> Dict[str, float]:
    fallback = PHASE_TARGET_DEFAULTS["FULL_BAND_LEARNING"]
    return dict(PHASE_TARGET_DEFAULTS.get(phase_name, fallback))


def _mean_level(levels):
    if not levels:
        return -100.0
    return sum(float(value) for value in levels) / float(len(levels))


def _lead_masking_margin_for_snapshot(
    snapshot: PhaseLearningSnapshot,
    channel_metadata: Mapping[int, Mapping[str, object]],
) -> float:
    lead_levels = []
    accompaniment_levels = []

    for channel_id, features in snapshot.channel_features.items():
        channel_meta = channel_metadata.get(channel_id, {})
        stems = set(channel_meta.get("stem_roles", []))
        if channel_meta.get("source_role") == "lead_vocal" or "LEAD" in stems:
            lead_levels.append(
                float(features.named_band_levels_db.get("PRESENCE", -100.0))
            )

    for stem_name, features in snapshot.stem_features.items():
        if stem_name in {"MASTER", "LEAD"}:
            continue
        accompaniment_levels.append(
            float(features.named_band_levels_db.get("PRESENCE", -100.0))
        )

    if not lead_levels or not accompaniment_levels:
        return 0.0
    return max(lead_levels) - max(accompaniment_levels)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _derive_hpf_frequency_range_hz(
    source_role: str,
    features: AnalysisFeatures,
) -> Dict[str, float]:
    source_role = str(source_role or "unknown")
    bass_level = float(features.named_band_levels_db.get("BASS", -100.0))
    body_level = float(features.named_band_levels_db.get("BODY", -100.0))
    presence_level = float(features.named_band_levels_db.get("PRESENCE", -100.0))

    if source_role in LOW_END_SOURCE_ROLES:
        min_hz, max_hz = 20.0, 90.0
    elif source_role in BRIGHT_SOURCE_ROLES:
        min_hz, max_hz = 100.0, 350.0
    elif source_role in VOICE_SOURCE_ROLES:
        min_hz, max_hz = 60.0, 220.0
    elif source_role in MIDRANGE_MUSIC_SOURCE_ROLES:
        min_hz, max_hz = 35.0, 180.0
    else:
        min_hz, max_hz = 40.0, 200.0

    if bass_level < -45.0 and body_level < -40.0 and presence_level > body_level + 8.0:
        min_hz += 20.0
        max_hz += 40.0
    elif max(bass_level, body_level) > presence_level - 3.0:
        max_hz -= 20.0

    max_hz = _clamp(max_hz, 30.0, 400.0)
    min_hz = _clamp(min_hz, 20.0, max_hz)
    return {
        "min_hz": float(round(min_hz, 1)),
        "max_hz": float(round(max_hz, 1)),
    }


def _derive_compressor_threshold_range_db(
    source_role: str,
    features: AnalysisFeatures,
) -> Dict[str, float]:
    source_role = str(source_role or "unknown")
    rms_db = float(features.rms_db)
    crest_factor_db = max(0.0, float(features.crest_factor_db))

    if rms_db <= -80.0:
        min_db, max_db = -30.0, -12.0
    else:
        min_db = max(-50.0, rms_db + 2.0)
        max_db = min(-5.0, rms_db + min(14.0, 6.0 + (crest_factor_db * 0.5)))

    if source_role in LOW_END_SOURCE_ROLES:
        min_db -= 2.0
    elif source_role in VOICE_SOURCE_ROLES:
        min_db += 1.0
        max_db += 1.0
    elif source_role in BRIGHT_SOURCE_ROLES:
        min_db += 2.0
        max_db += 2.0

    min_db = _clamp(min_db, -50.0, -5.0)
    max_db = _clamp(max_db, -50.0, -5.0)
    if max_db < min_db + 2.0:
        max_db = min(-5.0, min_db + 2.0)

    return {
        "min_db": float(round(min_db, 1)),
        "max_db": float(round(max_db, 1)),
    }


def _derive_compressor_ratio_range(
    source_role: str,
    features: AnalysisFeatures,
) -> Dict[str, float]:
    source_role = str(source_role or "unknown")
    crest_factor_db = max(0.0, float(features.crest_factor_db))

    min_ratio = 1.5
    if source_role in LOW_END_SOURCE_ROLES:
        max_ratio = 8.0
    elif source_role in VOICE_SOURCE_ROLES:
        max_ratio = 6.0
    elif source_role in BRIGHT_SOURCE_ROLES:
        max_ratio = 4.0
    else:
        max_ratio = 5.0

    if crest_factor_db > 18.0:
        max_ratio += 1.0
    elif crest_factor_db < 8.0:
        max_ratio = max(3.0, max_ratio - 1.0)

    max_ratio = _clamp(max_ratio, min_ratio + 0.5, 10.0)
    return {
        "min_ratio": float(round(min_ratio, 1)),
        "max_ratio": float(round(max_ratio, 1)),
    }


def _derive_fx_send_level_range_db(
    source_role: str,
    features: AnalysisFeatures,
) -> Dict[str, float]:
    source_role = str(source_role or "unknown")
    rms_db = float(features.rms_db)

    if source_role == "lead_vocal":
        min_db, max_db = -30.0, -10.0
    elif source_role in {"backing_vocal", "vocal"}:
        min_db, max_db = -32.0, -12.0
    elif source_role in BRIGHT_SOURCE_ROLES or source_role in LOW_END_SOURCE_ROLES:
        min_db, max_db = -40.0, -18.0
    elif source_role in MIDRANGE_MUSIC_SOURCE_ROLES:
        min_db, max_db = -34.0, -10.0
    elif source_role in {"click", "talkback"}:
        min_db, max_db = -40.0, -25.0
    else:
        min_db, max_db = -36.0, -12.0

    if rms_db > -18.0:
        min_db -= 2.0
        max_db -= 2.0
    elif rms_db < -30.0:
        max_db += 2.0

    min_db = _clamp(min_db, -40.0, -5.0)
    max_db = _clamp(max_db, -40.0, -5.0)
    if max_db < min_db + 2.0:
        max_db = min(-5.0, min_db + 2.0)

    return {
        "min_db": float(round(min_db, 1)),
        "max_db": float(round(max_db, 1)),
    }


def _build_phase_learning_targets(
    phase_snapshots: Mapping[str, PhaseLearningSnapshot],
    channel_metadata: Mapping[int, Mapping[str, object]],
) -> Dict[str, PhaseLearningTarget]:
    if not phase_snapshots:
        return {}

    silence_snapshot = phase_snapshots.get("SILENCE_CAPTURE")
    silence_noise_floor = {
        int(channel_id): float(features.rms_db)
        for channel_id, features in getattr(silence_snapshot, "channel_features", {}).items()
    }

    phase_targets: Dict[str, PhaseLearningTarget] = {}
    for phase_name, snapshot in phase_snapshots.items():
        defaults = _phase_target_defaults(phase_name)

        if (
            snapshot.master_features.rms_db <= -80.0
            or not snapshot.master_features.slope_compensated_band_levels_db
        ):
            default_corridor = TargetCorridor.default_intergenre()
            target_corridor = TargetCorridor(
                name=f"phase_{phase_name.lower()}_fallback",
                source=f"phase:{phase_name.lower()}",
                target_median_db=dict(default_corridor.target_median_db),
                green_delta_db=float(defaults["green_delta_db"]),
                yellow_delta_db=float(defaults["yellow_delta_db"]),
                red_delta_db=float(defaults["red_delta_db"]),
            )
        else:
            target_corridor = learn_target_corridor(
                snapshot.master_features,
                name=f"learned_{phase_name.lower()}",
                source=f"phase:{phase_name.lower()}",
                green_delta_db=float(defaults["green_delta_db"]),
                yellow_delta_db=float(defaults["yellow_delta_db"]),
                red_delta_db=float(defaults["red_delta_db"]),
            )

        expected_channel_rms_db = {
            int(channel_id): float(features.rms_db)
            for channel_id, features in snapshot.channel_features.items()
        }
        expected_stem_rms_db = {
            str(stem_name): float(features.rms_db)
            for stem_name, features in snapshot.stem_features.items()
            if stem_name != "MASTER"
        }
        source_role_levels: Dict[str, list] = {}
        for channel_id, features in snapshot.channel_features.items():
            source_role = str(
                channel_metadata.get(channel_id, {}).get("source_role", "unknown")
            )
            source_role_levels.setdefault(source_role, []).append(float(features.rms_db))
        expected_source_role_rms_db = {
            source_role: _mean_level(levels)
            for source_role, levels in source_role_levels.items()
        }
        noise_floor_db_by_channel = dict(silence_noise_floor)
        if not noise_floor_db_by_channel:
            noise_floor_db_by_channel = {
                int(channel_id): max(-90.0, float(features.rms_db) - 12.0)
                for channel_id, features in snapshot.channel_features.items()
            }
        hpf_frequency_range_hz_by_channel = {}
        compressor_threshold_range_db_by_channel = {}
        compressor_ratio_range_by_channel = {}
        fx_send_level_range_db_by_channel = {}
        for channel_id, features in snapshot.channel_features.items():
            source_role = str(
                channel_metadata.get(channel_id, {}).get("source_role", "unknown")
            )
            hpf_frequency_range_hz_by_channel[int(channel_id)] = (
                _derive_hpf_frequency_range_hz(source_role, features)
            )
            compressor_threshold_range_db_by_channel[int(channel_id)] = (
                _derive_compressor_threshold_range_db(source_role, features)
            )
            compressor_ratio_range_by_channel[int(channel_id)] = (
                _derive_compressor_ratio_range(source_role, features)
            )
            fx_send_level_range_db_by_channel[int(channel_id)] = (
                _derive_fx_send_level_range_db(source_role, features)
            )

        phase_targets[phase_name] = PhaseLearningTarget(
            phase_name=phase_name,
            runtime_state=snapshot.runtime_state,
            target_corridor=target_corridor,
            expected_channel_rms_db=expected_channel_rms_db,
            expected_stem_rms_db=expected_stem_rms_db,
            expected_source_role_rms_db=expected_source_role_rms_db,
            noise_floor_db_by_channel=noise_floor_db_by_channel,
            hpf_frequency_range_hz_by_channel=hpf_frequency_range_hz_by_channel,
            compressor_threshold_range_db_by_channel=compressor_threshold_range_db_by_channel,
            compressor_ratio_range_by_channel=compressor_ratio_range_by_channel,
            fx_send_level_range_db_by_channel=fx_send_level_range_db_by_channel,
            lead_masking_margin_db=_lead_masking_margin_for_snapshot(
                snapshot,
                channel_metadata,
            ),
            channel_level_tolerance_db=float(defaults["channel_level_tolerance_db"]),
            stem_level_tolerance_db=float(defaults["stem_level_tolerance_db"]),
        )

    return phase_targets


def learn_target_corridor(
    master_features: AnalysisFeatures,
    *,
    name: str = "learned_soundcheck",
    source: str = "soundcheck_profile",
    green_delta_db: float = 1.5,
    yellow_delta_db: float = 3.0,
    red_delta_db: float = 6.0,
) -> TargetCorridor:
    target_median_db = {
        band_name: float(level_db)
        for band_name, level_db in master_features.slope_compensated_band_levels_db.items()
    }
    if not target_median_db:
        return TargetCorridor.default_intergenre()
    return TargetCorridor(
        name=name,
        source=source,
        target_median_db=target_median_db,
        green_delta_db=float(green_delta_db),
        yellow_delta_db=float(yellow_delta_db),
        red_delta_db=float(red_delta_db),
    )


def build_phase_learning_snapshot(
    *,
    phase_name: str,
    runtime_state: str,
    channel_features: Optional[Mapping[int, AnalysisFeatures]] = None,
    stem_features: Optional[Mapping[str, AnalysisFeatures]] = None,
    active_channel_ids: Optional[Sequence[int]] = None,
    metadata: Optional[Dict[str, object]] = None,
    notes: str = "",
) -> PhaseLearningSnapshot:
    channel_features = dict(channel_features or {})
    stem_features = dict(stem_features or {})
    master_features = stem_features.get("MASTER")
    if master_features is None and channel_features:
        master_features = aggregate_analysis_features(channel_features.values())
    master_features = master_features or AnalysisFeatures()

    return PhaseLearningSnapshot(
        phase_name=phase_name,
        runtime_state=runtime_state,
        active_channel_ids=list(active_channel_ids or sorted(channel_features.keys())),
        channel_features={
            int(channel_id): features
            for channel_id, features in channel_features.items()
        },
        stem_features={
            str(stem_name): features
            for stem_name, features in stem_features.items()
        },
        master_features=master_features,
        metadata=dict(metadata or {}),
        notes=notes,
    )


def build_soundcheck_profile(
    *,
    channel_features: Mapping[int, AnalysisFeatures],
    channel_metadata: Mapping[int, Mapping[str, object]],
    stem_features: Mapping[str, AnalysisFeatures],
    stem_contributions: Optional[Mapping[str, Mapping[str, float]]] = None,
    sample_rate: int = 48000,
    profile_name: str = "autofoh_soundcheck",
    phase_snapshots: Optional[Mapping[str, PhaseLearningSnapshot]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> AutoFOHSoundcheckProfile:
    stem_contributions = stem_contributions or {}
    phase_snapshots = dict(phase_snapshots or {})
    channel_profiles: Dict[int, ChannelSoundcheckProfile] = {}

    for channel_id, features in channel_features.items():
        channel_meta = channel_metadata.get(channel_id, {})
        rms_db = float(features.rms_db)
        crest_db = float(features.crest_factor_db)
        channel_profiles[int(channel_id)] = ChannelSoundcheckProfile(
            channel_id=int(channel_id),
            name=str(channel_meta.get("name", f"Ch {channel_id}")),
            source_role=str(channel_meta.get("source_role", "unknown")),
            stem_roles=list(channel_meta.get("stem_roles", [])),
            allowed_controls=list(channel_meta.get("allowed_controls", [])),
            priority=float(channel_meta.get("priority", 0.0)),
            noise_floor_db=max(-90.0, rms_db - 12.0),
            nominal_rms_db=rms_db,
            max_rms_db=rms_db,
            peak_db=float(features.peak_db),
            crest_factor_range_db=[crest_db, crest_db],
            spectral_fingerprint=dict(features.slope_compensated_band_levels_db),
            activity_threshold_db=rms_db - 6.0,
            problem_bands=[
                band_name
                for band_name, level_db in features.slope_compensated_band_levels_db.items()
                if level_db > 6.0
            ],
            analysis_features=features,
        )

    stem_profiles: Dict[str, StemSoundcheckProfile] = {}
    for stem_name, features in stem_features.items():
        if stem_name == "MASTER":
            continue
        contribution_profile = {
            band_name: float(row.get(stem_name, 0.0))
            for band_name, row in stem_contributions.items()
        }
        stem_profiles[str(stem_name)] = StemSoundcheckProfile(
            stem_name=str(stem_name),
            nominal_rms_db=float(features.rms_db),
            peak_db=float(features.peak_db),
            band_energy_distribution_db=dict(features.slope_compensated_band_levels_db),
            contribution_profile=contribution_profile,
            typical_correction_range_db=3.0,
            analysis_features=features,
        )

    master_features = stem_features.get("MASTER")
    if master_features is None and channel_features:
        master_features = aggregate_analysis_features(channel_features.values())
    master_features = master_features or AnalysisFeatures()
    target_corridor = learn_target_corridor(master_features)

    expected_stem_levels_db = {
        stem_name: float(profile.nominal_rms_db)
        for stem_name, profile in stem_profiles.items()
    }
    lead_channel_ids = [
        channel_id
        for channel_id, channel_meta in channel_metadata.items()
        if channel_meta.get("source_role") == "lead_vocal"
        or "LEAD" in list(channel_meta.get("stem_roles", []))
    ]
    lead_margin_db = 0.0
    if lead_channel_ids:
        lead_levels = [
            float(channel_features[channel_id].named_band_levels_db.get("PRESENCE", -100.0))
            for channel_id in lead_channel_ids
            if channel_id in channel_features
        ]
        accompaniment_levels = [
            float(features.named_band_levels_db.get("PRESENCE", -100.0))
            for stem_name, features in stem_features.items()
            if stem_name not in {"MASTER", "LEAD"}
        ]
        if lead_levels and accompaniment_levels:
            lead_margin_db = max(lead_levels) - max(accompaniment_levels)

    master_profile = MasterSoundcheckProfile(
        nominal_rms_db=float(master_features.rms_db),
        peak_db=float(master_features.peak_db),
        lead_channel_ids=lead_channel_ids,
        expected_stem_levels_db=expected_stem_levels_db,
        expected_lead_masking_margin_db=float(lead_margin_db),
        target_corridor=target_corridor,
        analysis_features=master_features,
    )
    phase_targets = _build_phase_learning_targets(phase_snapshots, channel_metadata)

    return AutoFOHSoundcheckProfile(
        version=1,
        created_at=time.time(),
        name=profile_name,
        sample_rate=int(sample_rate),
        channel_count=len(channel_profiles),
        target_corridor=target_corridor,
        channels=channel_profiles,
        stems=stem_profiles,
        master=master_profile,
        phase_snapshots=phase_snapshots,
        phase_targets=phase_targets,
        metadata=dict(metadata or {}),
    )


class AutoFOHSoundcheckProfileStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def save(self, profile: AutoFOHSoundcheckProfile):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(profile.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True)

    def load(self) -> AutoFOHSoundcheckProfile:
        with self.path.open("r", encoding="utf-8") as handle:
            return AutoFOHSoundcheckProfile.from_dict(json.load(handle))

    def exists(self) -> bool:
        return self.path.exists()
