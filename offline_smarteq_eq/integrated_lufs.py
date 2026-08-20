from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from auto_fader_v2.core.integrated_lufs import RollingIntegratedLufs
from lufs_gain_staging import KWeightingFilter


AIAK_INTEGRATED_LUFS_METHOD_ID = "aiak_rolling_integrated_lufs_correction_v1"
AIAK_LEVEL_PLANE_METHOD_ID = "aiak_integrated_lufs_level_plane_correction_v1"
MIN_POWER = 1e-20
MAX_LUFS_SAMPLE_RATE = 192_000
MAX_LUFS_DURATION_SEC = 30 * 60
MAX_LUFS_CHANNELS = 64


LEVEL_PLANE_DEFAULT_BASE_LUFS = -24.0
LEVEL_PLANE_ALIASES = {
    "leadvocal": "lead_vocal",
    "lead_vocal": "lead_vocal",
    "vocal": "lead_vocal",
    "voice": "lead_vocal",
    "backvocal": "backing_vocal",
    "backingvocal": "backing_vocal",
    "back_vocal": "backing_vocal",
    "backing_vocal": "backing_vocal",
    "electricguitar": "guitar",
    "electric_guitar": "guitar",
    "acousticguitar": "acoustic_guitar",
    "acoustic_guitar": "acoustic_guitar",
    "tom": "toms",
    "overhead": "overheads",
}

DEFAULT_LEVEL_PLANE_OFFSETS_DB = {
    "lead_vocal": 2.0,
    "drums": 2.0,
    "toms": 2.0,
    "synth": 1.0,
    "kick": -1.0,
    "snare": -1.0,
    "bass": -1.0,
    "guitar": -1.0,
    "acoustic_guitar": -1.0,
    "piano": -1.0,
    "playback": -1.0,
    "accordion": -1.0,
    "strings": -1.0,
    "brass": -1.0,
    "sax": -1.0,
    "woodwinds": -1.0,
    "backing_vocal": -1.0,
    "keys": -2.0,
    "keyboard": -2.0,
    "pads": -2.0,
    "fx": -2.0,
    "percussion": -2.0,
    "hihat": -11.0,
    "ride": -11.0,
    "cymbals": -11.0,
    "overheads": -11.0,
    "room": -16.0,
}

LEVEL_PLANE_GROUPS = {
    "lead_vocal": "vocals",
    "backing_vocal": "vocals",
    "kick": "low_end",
    "bass": "low_end",
    "snare": "drums",
    "drums": "drums",
    "toms": "drums",
    "percussion": "drums",
    "hihat": "cymbals",
    "ride": "cymbals",
    "cymbals": "cymbals",
    "overheads": "cymbals",
    "room": "ambience",
    "guitar": "harmonic",
    "acoustic_guitar": "harmonic",
    "keys": "harmonic",
    "keyboard": "harmonic",
    "piano": "harmonic",
    "synth": "harmonic",
    "pads": "harmonic",
    "playback": "harmonic",
    "accordion": "harmonic",
    "strings": "harmonic",
    "brass": "harmonic",
    "sax": "harmonic",
    "woodwinds": "harmonic",
    "fx": "fx",
}

HIGH_BLEED_INSTRUMENTS = {"toms", "hihat", "ride", "overheads", "room", "drums"}


@dataclass(frozen=True)
class AIAKIntegratedLufsCorrectionReport:
    method_id: str
    target_lufs: float
    pre_integrated_lufs: float
    post_integrated_lufs: float
    requested_gain_db: float
    range_limited_gain_db: float
    applied_gain_db: float
    sample_peak_ceiling_dbfs: float
    pre_sample_peak_dbfs: float
    post_sample_peak_dbfs: float
    block_sec: float
    hop_sec: float
    block_count: int

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class AIAKLevelPlaneCorrectionReport:
    method_id: str
    channel: int
    role: str
    instrument_key: str
    level_plane_group: str
    integrated_lufs: float
    target_lufs: float
    ideal_fader_db: float
    range_shift_db: float
    correction_db: float
    active: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _amp_to_db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(value), 1e-12))


def _db_to_amp(value: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value) / 20.0)


def normalize_aiak_level_plane_role(role: str | None) -> str:
    if not role:
        return "custom"
    key = str(role).strip()
    if not key:
        return "custom"
    key = re.sub(r"(?<!^)(?=[A-Z])", "_", key)
    key = re.sub(r"[\s\-]+", "_", key).lower()
    key = re.sub(r"_+", "_", key).strip("_")
    compact = key.replace("_", "")
    return LEVEL_PLANE_ALIASES.get(compact, LEVEL_PLANE_ALIASES.get(key, key))


def resolve_aiak_level_plane_target_lufs(role: str | None, base_lufs: float = LEVEL_PLANE_DEFAULT_BASE_LUFS) -> Tuple[float, str, str]:
    instrument_key = normalize_aiak_level_plane_role(role)
    offset_db = DEFAULT_LEVEL_PLANE_OFFSETS_DB.get(instrument_key)
    if offset_db is None:
        return float(base_lufs), instrument_key, LEVEL_PLANE_GROUPS.get(instrument_key, "custom")
    return float(base_lufs) + float(offset_db), instrument_key, LEVEL_PLANE_GROUPS.get(instrument_key, "custom")


def _as_channels(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[1] > MAX_LUFS_CHANNELS or not np.all(np.isfinite(arr)):
        raise ValueError("audio channel or finite-sample limit violated")
    return arr


def _sample_peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if np.asarray(audio).size else 0.0
    return float(_amp_to_db(peak))


def _k_weight_channels(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    arr = _as_channels(audio)
    filtered = []
    for channel in range(arr.shape[1]):
        k_filter = KWeightingFilter(int(sample_rate))
        filtered.append(k_filter.process(arr[:, channel].astype(np.float32)))
    return np.stack(filtered, axis=1).astype(np.float32)


def aiak_block_lufs_samples(
    audio: np.ndarray,
    sample_rate: int,
    *,
    block_sec: float = 0.4,
    hop_sec: float = 0.1,
) -> Tuple[Tuple[float, float], ...]:
    if type(sample_rate) is not int or not 1 <= sample_rate <= MAX_LUFS_SAMPLE_RATE:
        raise ValueError("sample_rate is outside supported bounds")
    if not (0.05 <= float(block_sec) <= 10.0 and 0.01 <= float(hop_sec) <= 10.0):
        raise ValueError("LUFS block/hop duration is outside supported bounds")
    weighted = _k_weight_channels(audio, sample_rate)
    if weighted.shape[0] / float(sample_rate) > MAX_LUFS_DURATION_SEC:
        raise ValueError("audio duration exceeds LUFS limit")
    block_size = max(1, int(round(float(sample_rate) * float(block_sec))))
    hop_size = max(1, int(round(float(sample_rate) * float(hop_sec))))
    if weighted.shape[0] < block_size:
        padded = np.zeros((block_size, weighted.shape[1]), dtype=np.float32)
        padded[: weighted.shape[0], :] = weighted
        weighted = padded

    blocks = []
    last_start = max(0, weighted.shape[0] - block_size)
    for start in range(0, last_start + 1, hop_size):
        block = weighted[start : start + block_size].astype(np.float64, copy=False)
        mean_square = float(np.mean(np.square(block)))
        block_lufs = -100.0 if mean_square <= MIN_POWER else -0.691 + 10.0 * math.log10(mean_square)
        blocks.append((float(start) / float(sample_rate), float(block_lufs)))
    return tuple(blocks)


def measure_aiak_integrated_lufs(
    audio: np.ndarray,
    sample_rate: int,
    *,
    block_sec: float = 0.4,
    hop_sec: float = 0.1,
) -> Tuple[float, int]:
    blocks = aiak_block_lufs_samples(audio, sample_rate, block_sec=block_sec, hop_sec=hop_sec)
    if not blocks:
        return -100.0, 0

    duration = float(_as_channels(audio).shape[0]) / max(float(sample_rate), 1.0)
    meter = RollingIntegratedLufs(window_seconds=max(0.5, duration + float(block_sec) + float(hop_sec)))
    current = -100.0
    for now, block_lufs in blocks:
        current = meter.update(0, block_lufs, now)
    return float(meter.get(0, fallback=current)), len(blocks)


def compute_aiak_level_plane_corrections(
    measurements: Sequence[Mapping[str, Any]],
    *,
    base_lufs: float = LEVEL_PLANE_DEFAULT_BASE_LUFS,
    fader_ceiling_db: float = 0.0,
    fader_floor_db: float = -18.0,
    max_boost_db: float = 4.0,
    bleed_threshold_lufs: float = -50.0,
    high_bleed_max_boost_db: float = 4.0,
) -> Tuple[AIAKLevelPlaneCorrectionReport, ...]:
    active_rows = []
    inactive_reports = []

    for measurement in measurements:
        channel = int(measurement.get("channel", 0))
        role = str(measurement.get("role", "custom"))
        integrated_lufs = float(measurement.get("integrated_lufs", -100.0))
        target_lufs, instrument_key, group = resolve_aiak_level_plane_target_lufs(role, base_lufs)

        if integrated_lufs <= float(bleed_threshold_lufs) or integrated_lufs <= -99.0:
            inactive_reports.append(AIAKLevelPlaneCorrectionReport(
                method_id=AIAK_LEVEL_PLANE_METHOD_ID,
                channel=channel,
                role=role,
                instrument_key=instrument_key,
                level_plane_group=group,
                integrated_lufs=round(integrated_lufs, 3),
                target_lufs=round(target_lufs, 3),
                ideal_fader_db=0.0,
                range_shift_db=0.0,
                correction_db=0.0,
                active=False,
                reason=f"inactive below {float(bleed_threshold_lufs):.1f} LUFS",
            ))
            continue

        ideal_fader_db = float(target_lufs) - float(integrated_lufs)
        if instrument_key in HIGH_BLEED_INSTRUMENTS and ideal_fader_db > float(high_bleed_max_boost_db):
            ideal_fader_db = float(high_bleed_max_boost_db)
        active_rows.append({
            "channel": channel,
            "role": role,
            "instrument_key": instrument_key,
            "group": group,
            "integrated_lufs": integrated_lufs,
            "target_lufs": target_lufs,
            "ideal_fader_db": ideal_fader_db,
        })

    range_shift_db = 0.0
    if active_rows:
        ideal_sorted = sorted(float(row["ideal_fader_db"]) for row in active_rows)
        p90_index = min(int(len(ideal_sorted) * 0.9), len(ideal_sorted) - 1)
        p90_ideal = ideal_sorted[p90_index]
        min_ideal = ideal_sorted[0]
        if p90_ideal > float(fader_ceiling_db):
            range_shift_db = p90_ideal - float(fader_ceiling_db)
        elif min_ideal < float(fader_floor_db):
            range_shift_db = min_ideal - float(fader_floor_db)

    reports = []
    for row in active_rows:
        new_fader = float(row["ideal_fader_db"]) - range_shift_db
        new_fader = float(np.clip(new_fader, float(fader_floor_db), float(fader_ceiling_db)))
        correction_db = min(new_fader, float(max_boost_db))
        correction_db = round(correction_db * 10.0) / 10.0
        reason = "level-plane correction"
        if str(row["instrument_key"]) in HIGH_BLEED_INSTRUMENTS and correction_db >= float(high_bleed_max_boost_db):
            reason = "high-bleed boost capped"
        reports.append(AIAKLevelPlaneCorrectionReport(
            method_id=AIAK_LEVEL_PLANE_METHOD_ID,
            channel=int(row["channel"]),
            role=str(row["role"]),
            instrument_key=str(row["instrument_key"]),
            level_plane_group=str(row["group"]),
            integrated_lufs=round(float(row["integrated_lufs"]), 3),
            target_lufs=round(float(row["target_lufs"]), 3),
            ideal_fader_db=round(float(row["ideal_fader_db"]), 3),
            range_shift_db=round(float(range_shift_db), 3),
            correction_db=round(float(correction_db), 3),
            active=True,
            reason=reason,
        ))

    return tuple(sorted([*reports, *inactive_reports], key=lambda report: report.channel))


def apply_aiak_integrated_lufs_correction(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = -16.0,
    max_gain_db: float = 6.0,
    sample_peak_ceiling_dbfs: float = -1.0,
    block_sec: float = 0.4,
    hop_sec: float = 0.1,
) -> Tuple[np.ndarray, AIAKIntegratedLufsCorrectionReport]:
    arr = np.asarray(audio, dtype=np.float32)
    pre_lufs, block_count = measure_aiak_integrated_lufs(
        arr,
        sample_rate,
        block_sec=block_sec,
        hop_sec=hop_sec,
    )
    pre_peak_dbfs = _sample_peak_dbfs(arr)

    if pre_lufs <= -99.0:
        requested_gain_db = 0.0
    else:
        requested_gain_db = float(target_lufs) - float(pre_lufs)

    limited_gain_db = float(np.clip(requested_gain_db, -abs(float(max_gain_db)), abs(float(max_gain_db))))
    peak_limited_gain_db = min(limited_gain_db, float(sample_peak_ceiling_dbfs) - pre_peak_dbfs)
    corrected = (arr * _db_to_amp(peak_limited_gain_db)).astype(np.float32)

    post_lufs, _ = measure_aiak_integrated_lufs(
        corrected,
        sample_rate,
        block_sec=block_sec,
        hop_sec=hop_sec,
    )
    post_peak_dbfs = _sample_peak_dbfs(corrected)

    return corrected, AIAKIntegratedLufsCorrectionReport(
        method_id=AIAK_INTEGRATED_LUFS_METHOD_ID,
        target_lufs=round(float(target_lufs), 3),
        pre_integrated_lufs=round(float(pre_lufs), 3),
        post_integrated_lufs=round(float(post_lufs), 3),
        requested_gain_db=round(float(requested_gain_db), 3),
        range_limited_gain_db=round(float(limited_gain_db), 3),
        applied_gain_db=round(float(peak_limited_gain_db), 3),
        sample_peak_ceiling_dbfs=round(float(sample_peak_ceiling_dbfs), 3),
        pre_sample_peak_dbfs=round(float(pre_peak_dbfs), 3),
        post_sample_peak_dbfs=round(float(post_peak_dbfs), 3),
        block_sec=round(float(block_sec), 3),
        hop_sec=round(float(hop_sec), 3),
        block_count=int(block_count),
    )
