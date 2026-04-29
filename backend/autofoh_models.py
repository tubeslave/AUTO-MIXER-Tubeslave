"""
Foundational data models for the AutoFOH analysis/safety pipeline.

These models are intentionally lightweight so they can be used by the
existing soundcheck engine without forcing a large architectural rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class SourceRole(str, Enum):
    KICK = "kick"
    KICK_IN = "kick_in"
    KICK_OUT = "kick_out"
    SNARE = "snare"
    SNARE_TOP = "snare_top"
    SNARE_BOTTOM = "snare_bottom"
    TOM = "tom"
    RACK_TOM = "rack_tom"
    FLOOR_TOM = "floor_tom"
    OVERHEAD = "overhead"
    OH_L = "oh_l"
    OH_R = "oh_r"
    HIHAT = "hihat"
    RIDE = "ride"
    CYMBALS = "cymbals"
    ROOM = "room"
    BASS = "bass"
    BASS_DI = "bass_di"
    BASS_MIC = "bass_mic"
    SYNTH_BASS = "synth_bass"
    GUITAR = "guitar"
    ELECTRIC_GUITAR = "electric_guitar"
    ACOUSTIC_GUITAR = "acoustic_guitar"
    LEAD_GUITAR = "lead_guitar"
    RHYTHM_GUITAR = "rhythm_guitar"
    KEYS = "keys"
    PIANO = "piano"
    ORGAN = "organ"
    SYNTH = "synth"
    PAD = "pad"
    LEAD_SYNTH = "lead_synth"
    ACCORDION = "accordion"
    LEAD_VOCAL = "lead_vocal"
    BACKING_VOCAL = "backing_vocal"
    PLAYBACK = "playback"
    TRACKS = "tracks"
    MUSIC = "music"
    CLICK = "click"
    TALKBACK = "talkback"
    FX_RETURN = "fx_return"
    REVERB_RETURN = "reverb_return"
    DELAY_RETURN = "delay_return"
    BUS_DRUMS = "bus_drums"
    BUS_VOCAL = "bus_vocal"
    BUS_INSTRUMENT = "bus_instrument"
    UNKNOWN = "unknown"


class StemRole(str, Enum):
    DRUMS = "DRUMS"
    KICK = "KICK"
    SNARE = "SNARE"
    TOMS = "TOMS"
    CYMBALS = "CYMBALS"
    BASS = "BASS"
    GUITARS = "GUITARS"
    KEYS = "KEYS"
    PLAYBACK = "PLAYBACK"
    LEAD = "LEAD"
    BGV = "BGV"
    FX = "FX"
    MUSIC = "MUSIC"
    MASTER = "MASTER"
    UNKNOWN = "UNKNOWN"


class ControlType(str, Enum):
    GAIN = "gain"
    HPF = "hpf"
    EQ = "eq"
    COMPRESSOR = "compressor"
    FADER = "fader"
    PAN = "pan"
    FX_SEND = "fx_send"
    FEEDBACK_NOTCH = "feedback_notch"
    EMERGENCY_FADER = "emergency_fader"


class RuntimeState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    SILENCE_CAPTURE = "SILENCE_CAPTURE"
    LINE_CHECK = "LINE_CHECK"
    SOURCE_LEARNING = "SOURCE_LEARNING"
    STEM_LEARNING = "STEM_LEARNING"
    FULL_BAND_LEARNING = "FULL_BAND_LEARNING"
    SNAPSHOT_LOCK = "SNAPSHOT_LOCK"
    PRE_SHOW_CHECK = "PRE_SHOW_CHECK"
    LOAD_SONG_SNAPSHOT = "LOAD_SONG_SNAPSHOT"
    SONG_START_STABILIZE = "SONG_START_STABILIZE"
    VERSE = "VERSE"
    CHORUS = "CHORUS"
    SOLO = "SOLO"
    SPEECH = "SPEECH"
    BETWEEN_SONGS = "BETWEEN_SONGS"
    EMERGENCY_FEEDBACK = "EMERGENCY_FEEDBACK"
    EMERGENCY_SPL = "EMERGENCY_SPL"
    EMERGENCY_SIGNAL_LOSS = "EMERGENCY_SIGNAL_LOSS"
    MANUAL_LOCK = "MANUAL_LOCK"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    low_hz: float
    high_hz: float

    @property
    def center_hz(self) -> float:
        return (self.low_hz * self.high_hz) ** 0.5


NAMED_FREQUENCY_BANDS: Tuple[FrequencyBand, ...] = (
    FrequencyBand("SUB", 30.0, 60.0),
    FrequencyBand("BASS", 60.0, 120.0),
    FrequencyBand("BODY", 120.0, 250.0),
    FrequencyBand("MUD", 250.0, 500.0),
    FrequencyBand("LOW_MID", 500.0, 1000.0),
    FrequencyBand("PRESENCE", 1500.0, 4000.0),
    FrequencyBand("HARSHNESS", 3000.0, 6000.0),
    FrequencyBand("SIBILANCE", 6000.0, 10000.0),
    FrequencyBand("AIR", 10000.0, 16000.0),
)

NAMED_FREQUENCY_BAND_MAP: Dict[str, FrequencyBand] = {
    band.name: band for band in NAMED_FREQUENCY_BANDS
}


@dataclass
class MixIndexSet:
    sub_index: float = 0.0
    bass_index: float = 0.0
    body_index: float = 0.0
    mud_index: float = 0.0
    presence_index: float = 0.0
    harshness_index: float = 0.0
    sibilance_index: float = 0.0
    air_index: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "SubIndex": self.sub_index,
            "BassIndex": self.bass_index,
            "BodyIndex": self.body_index,
            "MudIndex": self.mud_index,
            "PresenceIndex": self.presence_index,
            "HarshnessIndex": self.harshness_index,
            "SibilanceIndex": self.sibilance_index,
            "AirIndex": self.air_index,
        }


@dataclass
class TargetCorridor:
    name: str = "default_intergenre"
    source: str = "factory"
    target_median_db: Dict[str, float] = field(default_factory=dict)
    green_delta_db: float = 1.5
    yellow_delta_db: float = 3.0
    red_delta_db: float = 6.0

    @classmethod
    def default_intergenre(cls) -> "TargetCorridor":
        base = {band.name: 0.0 for band in NAMED_FREQUENCY_BANDS}
        base["SUB"] = -1.5
        base["AIR"] = -1.0
        return cls(target_median_db=base)

    def target_for_band(self, band_name: str) -> float:
        return float(self.target_median_db.get(band_name, 0.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "target_median_db": {
                str(key): float(value)
                for key, value in self.target_median_db.items()
            },
            "green_delta_db": float(self.green_delta_db),
            "yellow_delta_db": float(self.yellow_delta_db),
            "red_delta_db": float(self.red_delta_db),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "TargetCorridor":
        payload = payload or {}
        return cls(
            name=str(payload.get("name", "loaded_corridor")),
            source=str(payload.get("source", "profile")),
            target_median_db={
                str(key): float(value)
                for key, value in dict(payload.get("target_median_db", {})).items()
            },
            green_delta_db=float(payload.get("green_delta_db", 1.5)),
            yellow_delta_db=float(payload.get("yellow_delta_db", 3.0)),
            red_delta_db=float(payload.get("red_delta_db", 6.0)),
        )


@dataclass
class AnalysisFeatures:
    rms_db: float = -100.0
    peak_db: float = -100.0
    crest_factor_db: float = 0.0
    named_band_levels_db: Dict[str, float] = field(default_factory=dict)
    octave_band_levels_db: Dict[str, float] = field(default_factory=dict)
    slope_compensated_band_levels_db: Dict[str, float] = field(default_factory=dict)
    mix_indexes: MixIndexSet = field(default_factory=MixIndexSet)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "rms_db": float(self.rms_db),
            "peak_db": float(self.peak_db),
            "crest_factor_db": float(self.crest_factor_db),
            "named_band_levels_db": {
                str(key): float(value)
                for key, value in self.named_band_levels_db.items()
            },
            "octave_band_levels_db": {
                str(key): float(value)
                for key, value in self.octave_band_levels_db.items()
            },
            "slope_compensated_band_levels_db": {
                str(key): float(value)
                for key, value in self.slope_compensated_band_levels_db.items()
            },
            "mix_indexes": self.mix_indexes.as_dict(),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "AnalysisFeatures":
        payload = payload or {}
        mix_indexes = dict(payload.get("mix_indexes", {}))
        return cls(
            rms_db=float(payload.get("rms_db", -100.0)),
            peak_db=float(payload.get("peak_db", -100.0)),
            crest_factor_db=float(payload.get("crest_factor_db", 0.0)),
            named_band_levels_db={
                str(key): float(value)
                for key, value in dict(payload.get("named_band_levels_db", {})).items()
            },
            octave_band_levels_db={
                str(key): float(value)
                for key, value in dict(payload.get("octave_band_levels_db", {})).items()
            },
            slope_compensated_band_levels_db={
                str(key): float(value)
                for key, value in dict(payload.get("slope_compensated_band_levels_db", {})).items()
            },
            mix_indexes=MixIndexSet(
                sub_index=float(mix_indexes.get("SubIndex", 0.0)),
                bass_index=float(mix_indexes.get("BassIndex", 0.0)),
                body_index=float(mix_indexes.get("BodyIndex", 0.0)),
                mud_index=float(mix_indexes.get("MudIndex", 0.0)),
                presence_index=float(mix_indexes.get("PresenceIndex", 0.0)),
                harshness_index=float(mix_indexes.get("HarshnessIndex", 0.0)),
                sibilance_index=float(mix_indexes.get("SibilanceIndex", 0.0)),
                air_index=float(mix_indexes.get("AirIndex", 0.0)),
            ),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass
class ConfidenceRisk:
    problem_confidence: float = 0.0
    culprit_confidence: float = 0.0
    action_confidence: float = 0.0
    risk_score: float = 1.0


@dataclass
class DetectedProblem:
    problem_type: str
    description: str
    channel_id: Optional[int] = None
    stem: Optional[str] = None
    band_name: Optional[str] = None
    persistence_sec: float = 0.0
    features: Optional[AnalysisFeatures] = None
    confidence_risk: ConfidenceRisk = field(default_factory=ConfidenceRisk)
    expected_effect: str = ""


@dataclass
class CorrectionAction:
    action_type: str
    reason: str
    channel_id: Optional[int] = None
    stem: Optional[str] = None
    delta_db: float = 0.0
    freq_hz: Optional[float] = None
    q: Optional[float] = None
    ttl_seconds: Optional[float] = None
    confidence_risk: ConfidenceRisk = field(default_factory=ConfidenceRisk)


@dataclass
class ActionResult:
    action: CorrectionAction
    supported: bool = True
    sent: bool = False
    bounded: bool = False
    rollback_triggered: bool = False
    measured_effect: str = ""
    message: str = ""
