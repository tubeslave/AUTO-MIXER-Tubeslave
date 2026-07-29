"""Venue-калибровка (Ф5б мастер-плана, DSP-совет §4 п.4 2026-07-29).

Пороги feedback-детектора и цели громкости — константы уровня модуля,
откалиброванные «в среднем по залам». Реальный зал отличается шумовым
полом (вентиляция, гул толпы, наводки 50/100 Гц): пороги, снятые в тихой
студии, в шумном зале дают ложные срабатывания, а в очень тихом —
пропуски тихой заводки.

Калибровочный проход выполняется в Prep на стадии SILENCE_CAPTURE
(до звука на сцене): по захвату тишины строится профиль шума зала и
выводятся per-venue пороги. Модуль — чистые вычисления, записей нет.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Запасы над шумовым полом (дБ): пик-кандидат заводки обязан быть заметно
# громче пола зала, иначе детектор жуёт вентиляцию
FEEDBACK_HEIGHT_MARGIN_DB = 20.0
STEADY_STATE_MARGIN_DB = 26.0

# Клампы: калибровка не имеет права сделать детектор глухим или параноиком
HEIGHT_CLAMP_DB = (-60.0, -25.0)
STEADY_CLAMP_DB = (-45.0, -20.0)

HUM_FREQS_HZ = (50.0, 100.0, 60.0, 120.0)
HUM_PROMINENCE_DB = 12.0


@dataclass
class VenueNoiseProfile:
    channel_floor_dbfs: Dict[int, float] = field(default_factory=dict)
    overall_floor_dbfs: float = -90.0
    hum_detected: bool = False
    hum_freq_hz: Optional[float] = None
    hum_prominence_db: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_floor_dbfs": {
                str(k): round(v, 1) for k, v in self.channel_floor_dbfs.items()
            },
            "overall_floor_dbfs": round(self.overall_floor_dbfs, 1),
            "hum_detected": self.hum_detected,
            "hum_freq_hz": self.hum_freq_hz,
            "hum_prominence_db": round(self.hum_prominence_db, 1),
        }


@dataclass
class VenueCalibration:
    profile: VenueNoiseProfile
    peak_height_db: float
    steady_state_min_level_dbfs: float
    peak_prominence_bump_db: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "peak_height_db": round(self.peak_height_db, 1),
            "steady_state_min_level_dbfs": round(
                self.steady_state_min_level_dbfs, 1),
            "peak_prominence_bump_db": round(self.peak_prominence_bump_db, 1),
            "notes": list(self.notes),
        }


def _channel_noise_floor_dbfs(x: np.ndarray, sample_rate: int) -> Optional[float]:
    """Пол шума канала: RMS тихих 20% 100-мс блоков (робастно к случайным
    звукам на сцене во время «тишины»)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    block = max(256, int(0.1 * sample_rate))
    if len(x) < block * 3:
        return None
    rms_db: List[float] = []
    for i in range(0, len(x) - block, block):
        ms = float(np.mean(x[i:i + block] ** 2))
        rms_db.append(10.0 * np.log10(ms + 1e-14))
    if not rms_db:
        return None
    quiet = sorted(rms_db)[: max(1, len(rms_db) // 5)]
    return float(np.mean(quiet))


def _detect_hum(x: np.ndarray, sample_rate: int) -> tuple[Optional[float], float]:
    """Гул сети: узкий пик на 50/100 (или 60/120) Гц над соседними полосами."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = min(len(x), sample_rate * 2)
    if n < 8192:
        return None, 0.0
    seg = x[:n] * np.hanning(n)
    spec_db = 20.0 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    best_freq, best_prom = None, 0.0
    for f0 in HUM_FREQS_HZ:
        band = (freqs >= f0 - 2.0) & (freqs <= f0 + 2.0)
        neigh = ((freqs >= f0 - 15.0) & (freqs < f0 - 4.0)) | (
            (freqs > f0 + 4.0) & (freqs <= f0 + 15.0))
        if not band.any() or not neigh.any():
            continue
        prom = float(np.max(spec_db[band]) - np.median(spec_db[neigh]))
        if prom > best_prom:
            best_freq, best_prom = f0, prom
    if best_prom >= HUM_PROMINENCE_DB:
        return best_freq, best_prom
    return None, best_prom


def profile_venue_noise(
    buffers: Dict[int, np.ndarray], sample_rate: int
) -> VenueNoiseProfile:
    """Профиль шума зала по захвату тишины (стадия SILENCE_CAPTURE)."""
    profile = VenueNoiseProfile()
    hum_freq, hum_prom = None, 0.0
    for ch, x in buffers.items():
        floor = _channel_noise_floor_dbfs(x, sample_rate)
        if floor is None:
            continue
        profile.channel_floor_dbfs[int(ch)] = floor
        f, p = _detect_hum(x, sample_rate)
        if p > hum_prom:
            hum_freq, hum_prom = f, p
    if profile.channel_floor_dbfs:
        profile.overall_floor_dbfs = float(
            np.median(list(profile.channel_floor_dbfs.values()))
        )
    profile.hum_detected = hum_freq is not None
    profile.hum_freq_hz = hum_freq
    profile.hum_prominence_db = hum_prom
    return profile


def calibrate_feedback_thresholds(
    profile: VenueNoiseProfile,
    *,
    base_peak_height_db: float = -40.0,
    base_steady_dbfs: float = -30.0,
) -> VenueCalibration:
    """Per-venue пороги детектора заводки от фактического пола зала.

    Пороги только ПОДНИМАЮТСЯ относительно пола (шумный зал → выше порог,
    меньше ложных), но не опускаются ниже базовых значений детектора и
    зажаты клампами: калибровка не делает систему ни глухой, ни параноиком.
    """
    floor = profile.overall_floor_dbfs
    notes: List[str] = []

    height = max(base_peak_height_db, floor + FEEDBACK_HEIGHT_MARGIN_DB)
    height = float(np.clip(height, *HEIGHT_CLAMP_DB))
    steady = max(base_steady_dbfs, floor + STEADY_STATE_MARGIN_DB)
    steady = float(np.clip(steady, *STEADY_CLAMP_DB))

    if height > base_peak_height_db + 0.5:
        notes.append(
            f"шумный зал (пол {floor:.0f} dBFS): порог пика поднят до "
            f"{height:.0f} dBFS"
        )
    prominence_bump = 0.0
    if profile.hum_detected:
        prominence_bump = 2.0
        notes.append(
            f"гул сети {profile.hum_freq_hz:.0f} Гц "
            f"({profile.hum_prominence_db:.0f} дБ): prominence +2 дБ — "
            "стабильный пик гула не должен набирать persistence"
        )
    headroom = -1.0 - floor  # до 0 dBFS минус потолок −1 dBTP
    if headroom < 40.0:
        notes.append(
            f"запас пол→потолок всего {headroom:.0f} дБ — зал шумный, "
            "следить за gain staging"
        )
    return VenueCalibration(
        profile=profile,
        peak_height_db=height,
        steady_state_min_level_dbfs=steady,
        peak_prominence_bump_db=prominence_bump,
        notes=notes,
    )
