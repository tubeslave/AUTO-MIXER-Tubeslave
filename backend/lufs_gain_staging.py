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

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from collections import deque
from scipy import signal
from enum import Enum
from dataclasses import dataclass, field
from method_preset_loader import load_method_preset_base, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_GAIN_TARGETS_PATH = (
    Path(__file__).resolve().parents[1] / "presets" / "gain_staging_targets.json"
)
DEFAULT_GAIN_PRESET_BASE_PATH = (
    Path(__file__).resolve().parents[1] / "presets" / "method_presets_gain.json"
)
DEFAULT_CATEGORY_PEAKS = {
    "drums_close": -6.0,
    "drums_ambient": -18.0,
    "instruments": -12.0,
    "vocals": -6.0,
}
PRESET_ALIASES = {
    "tom": "tom_mid",
    "toms": "tom_mid",
    "kick_in": "kick",
    "kick_out": "kick",
    "kick_sub": "kick",
    "snare_top": "snare",
    "tom_floor": "tom_lo",
    "hihat": "hi_hat",
    "overheads": "overhead",
    "leadvocal": "vocal",
    "lead_vocal": "vocal",
    "vocal": "vocal",
    "back_vocal": "bgv",
    "backvocal": "bgv",
    "vocals": "vocal",
    "electric_guitar": "guitar",
    "electricguitar": "guitar",
    "acoustic_guitar": "guitar",
    "acousticguitar": "guitar",
    "playbackl": "playback_l",
    "playbackr": "playback_r",
}
EXPECTED_CENTROID_HZ = {
    "kick": (40.0, 180.0),
    "snare": (120.0, 5000.0),
    "snare_bottom": (150.0, 7000.0),
    "tom_hi": (100.0, 4000.0),
    "tom_mid": (80.0, 3000.0),
    "tom_lo": (60.0, 2200.0),
    "hi_hat": (3000.0, 16000.0),
    "ride": (2500.0, 15000.0),
    "overhead": (1500.0, 16000.0),
    "overhead_l": (1500.0, 16000.0),
    "overhead_r": (1500.0, 16000.0),
    "ambience": (200.0, 10000.0),
    "ambience_l": (200.0, 10000.0),
    "ambience_r": (200.0, 10000.0),
    "room": (200.0, 10000.0),
    "ambient": (200.0, 10000.0),
    "room_mic": (200.0, 10000.0),
    "bass": (35.0, 900.0),
    "guitar": (90.0, 6000.0),
    "keys": (80.0, 8000.0),
    "keyboard": (80.0, 9000.0),
    "synth": (80.0, 10000.0),
    "accordion": (120.0, 5000.0),
    "playback": (60.0, 15000.0),
    "playback_l": (60.0, 15000.0),
    "playback_r": (60.0, 15000.0),
    "tracks": (60.0, 15000.0),
    "vocal": (120.0, 8000.0),
    "vocal_2": (120.0, 8000.0),
    "bgv": (150.0, 9000.0),
}

# Channels with these presets should not use bleed-based rejection.
# Bypass only for: overhead, bass, guitar, playback, accordion.
DEFAULT_BLEED_BYPASS_PRESETS = {
    "playback",
    "playback_l",
    "playback_r",
    "guitar",  # includes Electric Guitar aliases
    "overhead",
    "overhead_l",
    "overhead_r",
    "bass",
    "accordion",
}

STEREO_PAIR_DRUM_EXCLUDED_PREFIXES = (
    "kick",
    "snare",
    "tom",
    "overhead",
    "drum_room",
)


def _normalize_preset_name(raw: Optional[str]) -> str:
    if not raw:
        return "custom"
    name = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return PRESET_ALIASES.get(name, name)


def _load_gain_targets(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = config or {}
    defaults = {
        "targets": {},
        "categories_defaults": DEFAULT_CATEGORY_PEAKS.copy(),
    }

    # Preferred source: unified method preset base schema.
    base_path_cfg = (
        cfg.get("automation", {})
        .get("preset_bases", {})
        .get("gain", {})
        .get("file", str(DEFAULT_GAIN_PRESET_BASE_PATH))
    )
    base_path = resolve_path(base_path_cfg, DEFAULT_GAIN_PRESET_BASE_PATH)
    base_data = load_method_preset_base(base_path, expected_method="gain")
    if base_data.get("instruments"):
        targets: Dict[str, Dict[str, Any]] = {}
        for item in base_data.get("instruments", []):
            if not isinstance(item, dict):
                continue
            preset_id = (
                str(item.get("id", ""))
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            params = (
                item.get("params", {}) if isinstance(item.get("params"), dict) else {}
            )
            if not preset_id or "peak_dbfs" not in params:
                continue
            cfg_entry = {
                "peak_dbfs": float(params.get("peak_dbfs")),
                "category": str(params.get("category", "instruments")),
            }
            targets[preset_id] = cfg_entry
            for alias in item.get("aliases", []) or []:
                alias_key = (
                    str(alias).strip().lower().replace("-", "_").replace(" ", "_")
                )
                if alias_key:
                    targets[alias_key] = cfg_entry

        categories = (
            base_data.get("defaults", {}).get("category_peak_dbfs", {})
            if isinstance(base_data.get("defaults"), dict)
            else {}
        )
        return {
            "targets": targets,
            "categories_defaults": {
                **DEFAULT_CATEGORY_PEAKS,
                **(categories if isinstance(categories, dict) else {}),
            },
        }

    # Backward compatibility with legacy gain_staging_targets.json.
    path_str = (
        cfg.get("automation", {})
        .get("safe_gain_calibration", {})
        .get("targets_file", str(DEFAULT_GAIN_TARGETS_PATH))
    )
    path = resolve_path(path_str, DEFAULT_GAIN_TARGETS_PATH)
    try:
        if not path.exists():
            logger.warning("Gain targets file not found: %s", path)
            return defaults
        data = json.loads(path.read_text(encoding="utf-8"))
        targets = data.get("targets", {})
        categories = data.get("categories_defaults", {})
        return {
            "targets": targets if isinstance(targets, dict) else {},
            "categories_defaults": {
                **DEFAULT_CATEGORY_PEAKS,
                **(categories if isinstance(categories, dict) else {}),
            },
        }
    except Exception as exc:
        logger.warning("Failed to load gain targets from %s: %s", path, exc)
        return defaults


class AnalysisState(Enum):
    """Analysis state for Safe Static Gain system."""

    IDLE = "idle"
    LEARNING = "learning"
    READY = "ready"
    APPLYING = "applying"


class ChannelPhase(Enum):
    """Per-channel learning phase inside Safe Gain analysis."""

    BLEED_LEARN = "bleed_learn"
    WAIT_FOR_OWN_SIGNAL = "wait_for_own_signal"
    CAPTURE_OWN_LEVEL = "capture_own_level"
    READY = "ready"


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

    noise_gate_threshold_db: float = -40.0

    crest_factor_db: float = 0.0

    suggested_gain_db: float = 0.0
    gain_limited_by: str = "none"
    own_source_samples: int = 0
    rejected_bleed_samples: int = 0
    rejected_mismatch_samples: int = 0
    rejected_out_of_window_count: int = 0
    last_source_confidence: float = 0.0
    last_bleed_ratio: float = 0.0
    last_bleed_confidence: float = 0.0
    last_bleed_method: str = "none"
    target_peak_dbfs: float = -12.0
    phase: ChannelPhase = ChannelPhase.BLEED_LEARN
    phase_started_at: Optional[float] = None
    bleed_blocks_collected: int = 0
    bleed_peak_samples: List[float] = field(default_factory=list)
    bleed_rms_samples: List[float] = field(default_factory=list)
    bleed_baseline_peak_dbfs: float = -100.0
    bleed_baseline_rms_db: float = -100.0
    reference_peak_dbfs: Optional[float] = None
    capture_started_at: Optional[float] = None
    accepted_count: int = 0
    required_own_events: int = 3
    global_max_true_peak_db: float = -100.0

    def update_sample(self, true_peak_db: float, lufs: float, sample_rms_db: float):
        """Update stats with new sample measurement."""
        self.total_samples += 1

        if true_peak_db > self.noise_gate_threshold_db:
            self.active_samples += 1
            self.peak_values.append(true_peak_db)
            self.rms_values.append(lufs)

            if true_peak_db > self.max_true_peak_db:
                self.max_true_peak_db = true_peak_db

        self.signal_presence_ratio = self.active_samples / max(self.total_samples, 1)

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

    def calculate_peak_gain(
        self,
        target_peak_dbfs: float = -12.0,
        min_signal_presence: float = 0.05,
        max_adjustment_db: float = 18.0,
        min_accepted_samples: int = 1,
    ):
        """Calculate gain correction from own-source peak only."""
        del min_signal_presence  # retained for backward compatibility
        self.target_peak_dbfs = float(target_peak_dbfs)
        if len(self.peak_values) < max(1, int(min_accepted_samples)):
            self.suggested_gain_db = 0.0
            self.gain_limited_by = "no_own_source"
            logger.info(
                "Ch%s: insufficient own-source captures (%s/%s), gain=0",
                self.channel_id,
                len(self.peak_values),
                max(1, int(min_accepted_samples)),
            )
            return

        self.calculate_integrated_lufs()
        self.calculate_crest_factor()
        delta_peak = self.target_peak_dbfs - self.max_true_peak_db
        self.suggested_gain_db = float(
            np.clip(delta_peak, -max_adjustment_db, max_adjustment_db)
        )
        self.gain_limited_by = "peak_target"
        logger.info(
            "Ch%s: peak=%.1f dBFS target=%.1f dBFS gain=%+.1f dB (own=%.1f%% bleed_rej=%s)",
            self.channel_id,
            self.max_true_peak_db,
            self.target_peak_dbfs,
            self.suggested_gain_db,
            (self.own_source_samples / max(self.total_samples, 1)) * 100.0,
            self.rejected_bleed_samples,
        )

    def calculate_safe_gain(
        self,
        target_lufs: float = -18.0,
        max_peak_limit: float = -3.0,
        min_signal_presence: float = 0.05,
    ):
        """Backward-compatible alias: uses peak-based method now."""
        del target_lufs  # legacy parameter
        self.calculate_peak_gain(
            target_peak_dbfs=max_peak_limit,
            min_signal_presence=min_signal_presence,
        )

    def get_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        return {
            "channel": self.channel_id,
            "peak_db": round(self.max_true_peak_db, 1),
            "lufs": round(self.integrated_lufs, 1),
            "crest_factor_db": round(self.crest_factor_db, 1),
            "signal_presence": round(self.signal_presence_ratio * 100, 1),
            "own_source_ratio": round(
                (self.own_source_samples / max(self.total_samples, 1)) * 100, 1
            ),
            "bleed_rejected": self.rejected_bleed_samples,
            "mismatch_rejected": self.rejected_mismatch_samples,
            "out_of_window_rejected": self.rejected_out_of_window_count,
            "source_confidence": round(self.last_source_confidence, 3),
            "bleed_ratio": round(self.last_bleed_ratio, 3),
            "bleed_confidence": round(self.last_bleed_confidence, 3),
            "bleed_method": self.last_bleed_method,
            "target_peak_dbfs": round(self.target_peak_dbfs, 1),
            "suggested_gain_db": round(self.suggested_gain_db, 1),
            "limited_by": self.gain_limited_by,
            "samples_analyzed": self.total_samples,
            "active_samples": self.active_samples,
            "phase": self.phase.value,
            "accepted_count": self.accepted_count,
            "required_own_events": self.required_own_events,
            "bleed_baseline_peak_dbfs": round(self.bleed_baseline_peak_dbfs, 1),
            "reference_peak_dbfs": (
                round(self.reference_peak_dbfs, 1)
                if self.reference_peak_dbfs is not None
                else None
            ),
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
        Vb = Vh**0.4996667741545416

        a0 = 1 + K / Q + K * K
        self.shelf_b = np.array(
            [
                (Vh + Vb * K / Q + K * K) / a0,
                2 * (K * K - Vh) / a0,
                (Vh - Vb * K / Q + K * K) / a0,
            ]
        )
        self.shelf_a = np.array([1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])

        # Stage 2: High-pass filter (срез ниже ~60Hz)
        f0 = 38.13547087602444
        Q = 0.5003270373238773

        K = np.tan(np.pi * f0 / fs)
        a0 = 1 + K / Q + K * K
        self.hp_b = np.array([1 / a0, -2 / a0, 1 / a0])
        self.hp_a = np.array([1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])

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
        upsampled[:: self.oversample_factor] = samples

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

    def __init__(
        self,
        sample_rate: int = 48000,
        attack_ms: float = 50.0,
        release_ms: float = 500.0,
        hold_ms: float = 200.0,
        update_interval_ms: float = 100.0,
    ):
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
        self.attack_coef = (
            1 - np.exp(-self.update_interval_ms / attack_ms) if attack_ms > 0 else 1.0
        )
        self.release_coef = (
            1 - np.exp(-self.update_interval_ms / release_ms) if release_ms > 0 else 1.0
        )

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

    def __init__(
        self,
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
        update_interval_ms: float = 100.0,
    ):
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
        self.envelope = AGCEnvelope(
            sample_rate, attack_ms, release_ms, hold_ms, update_interval_ms
        )

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

    def __init__(
        self,
        mixer_client=None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,
        config: Optional[Dict[str, Any]] = None,
        bleed_service=None,
        audio_capture=None,
    ):
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._audio_capture = audio_capture

        # Загрузка конфигурации
        self.config = config or {}
        self.bleed_service = bleed_service
        agc_config = self.config.get("automation", {}).get("lufs_gain_staging", {})
        peak_config = self.config.get("automation", {}).get("peak_gain_staging", {})

        # Параметры AGC
        self.target_lufs = agc_config.get("target_lufs", -23.0)
        self.true_peak_limit = agc_config.get("true_peak_limit", -1.0)
        self.max_gain_db = agc_config.get("max_gain_db", 12.0)
        self.min_gain_db = agc_config.get("min_gain_db", -12.0)
        self.ratio = agc_config.get("ratio", 4.0)
        self.attack_ms = agc_config.get("attack_ms", 50.0)
        self.release_ms = agc_config.get("release_ms", 500.0)
        self.hold_ms = agc_config.get("hold_ms", 200.0)
        self.gate_threshold_lufs = agc_config.get("gate_threshold_lufs", -50.0)
        self.update_interval = agc_config.get("update_interval_ms", 100) / 1000.0

        # Centralized bleed detection service
        self.bleed_service = bleed_service

        # Параметры Peak Gain Staging
        self.peak_staging_mode = False  # Режим работы: False = LUFS, True = Peak
        self.default_peak_threshold = peak_config.get("default_peak_threshold", -6.0)
        self.trim_reduction_step_db = peak_config.get("trim_reduction_step_db", 1.0)
        self.min_trim_db = peak_config.get("min_trim_db", -18.0)
        # Live mode: ограничение снижения trim (для концерта не глубже -8 dB, чтобы канал оставался слышимым)
        self.live_mode = agc_config.get("live_mode", False)
        self.min_trim_db_live = agc_config.get("min_trim_db_live", -8.0)
        # Пресеты для soundcheck / live (target_lufs, ratio, attack/release, gate)
        presets = agc_config.get("presets", {})
        self._presets = {
            "soundcheck": presets.get(
                "soundcheck",
                {
                    "target_lufs": -18.0,
                    "ratio": 2.0,
                    "attack_ms": 50.0,
                    "release_ms": 500.0,
                    "gate_threshold_lufs": -50.0,
                },
            ),
            "live": presets.get(
                "live",
                {
                    "target_lufs": -16.0,
                    "ratio": 1.5,
                    "attack_ms": 200.0,
                    "release_ms": 2000.0,
                    "gate_threshold_lufs": -40.0,
                },
            ),
        }

        # Каналы
        self.channels: Dict[int, ChannelAGC] = {}
        self.channel_mapping: Dict[int, int] = {}  # audio_ch -> mixer_ch
        self.channel_settings: Dict[int, Dict] = {}

        # Измеренные уровни (для совместимости с server.py)
        self.measured_levels: Dict[int, Dict] = {}

        # Отслеживание состояния коррекции для Peak Staging режима
        self.trim_correction_active: Dict[
            int, bool
        ] = {}  # mixer_ch -> активна ли коррекция
        self.last_trim_value: Dict[
            int, float
        ] = {}  # mixer_ch -> последнее значение Trim
        self.trim_reduction_count: Dict[
            int, int
        ] = {}  # mixer_ch -> количество снижений

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

        logger.info(
            f"LUFSGainStagingController initialized: target={self.target_lufs} LUFS, "
            f"ratio={self.ratio}:1, attack={self.attack_ms}ms, release={self.release_ms}ms"
        )

    def start(
        self,
        device_id: int,
        channels: List[int],
        channel_settings: Dict[int, Dict],
        channel_mapping: Dict[int, int],
        on_status_callback: Callable = None,
    ) -> bool:
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
            logger.info(
                f"Cleared old channels, initializing {len(channels)} new channels"
            )
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
                self._audio_capture.subscribe(
                    "lufs_gain_staging", self._audio_capture_poll
                )
                self.is_active = True
                self._stop_event.clear()
                logger.info(
                    f"Audio capture started via AudioCapture: {len(channels)} channels"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"AudioCapture integration failed, falling back to PyAudio: {e}"
                )
                self._audio_capture = None

        # Fallback: direct PyAudio stream
        try:
            import pyaudio

            self.pa = pyaudio.PyAudio()

            # Получаем информацию об устройстве
            device_info = self.pa.get_device_info_by_index(int(device_id))
            max_channels = int(device_info.get("maxInputChannels", 2))
            device_sample_rate = int(device_info.get("defaultSampleRate", 48000))

            logger.info(
                f"Audio device: {device_info.get('name')}, "
                f"max channels: {max_channels}, sample rate: {device_sample_rate}"
            )

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
                stream_callback=self._audio_callback,
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
                    logger.info(
                        f"Channel {mixer_ch}: Peak staging initialized, active=True, trim={current_trim:.1f} dB"
                    )
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
            target=self._correction_loop, daemon=True
        )
        self._correction_thread.start()

        mode_name = "Peak" if self.peak_staging_mode else "LUFS"
        logger.info(f"Real-time {mode_name} correction started")

        if self.on_status_update:
            self.on_status_update(
                {"type": "realtime_correction_started", "active": True}
            )

        return True

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
                    buffer_sizes = {
                        ch: len(buf) for ch, buf in self._audio_buffers.items()
                    }
                    logger.info(f"Iteration {iteration}: buffer sizes = {buffer_sizes}")

                for audio_ch, agc in list(self.channels.items()):
                    # Получаем накопленные аудио данные
                    if (
                        audio_ch in self._audio_buffers
                        and len(self._audio_buffers[audio_ch]) > 0
                    ):
                        # Объединяем буферы
                        chunks = list(self._audio_buffers[audio_ch])
                        self._audio_buffers[audio_ch].clear()

                        if chunks:
                            try:
                                audio_data = np.concatenate(chunks)

                                # Обрабатываем (включая тихие данные - AGC сам отметит gated/idle)
                                result = agc.process(audio_data)
                            except Exception as e:
                                logger.error(
                                    f"Error processing audio for channel {audio_ch}: {e}",
                                    exc_info=True,
                                )
                                continue

                            mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)

                            # Логируем первые несколько каналов каждые 50 итераций
                            if iteration % 50 == 0 and audio_ch <= 3:
                                logger.info(
                                    f"Ch{audio_ch}: LUFS={result['lufs']:.1f}, TruePeak={result['true_peak']:.1f}, "
                                    f"gain={result['gain']:.2f}, gated={result['is_gated']}, status={result['status']}"
                                )

                            # Обновляем measured_levels для совместимости
                            self.measured_levels[audio_ch] = {
                                "peak": result["true_peak"],
                                "signal_present": not result["is_gated"],
                                "lufs": result["lufs"],
                                "true_peak": result["true_peak"],
                                "gain": result["gain"],
                                "applied_gain": result["applied_gain"],
                                "status": result["status"],
                            }

                            # Bleed detection: optionally bypass for specific presets
                            bleed_info = None
                            bleed_ratio = 0.0
                            ch_settings = self.channel_settings.get(audio_ch, {})
                            raw_preset = (
                                ch_settings.get("preset")
                                or ch_settings.get("instrumentType")
                                or "custom"
                            )
                            normalized_preset = _normalize_preset_name(raw_preset)
                            bypass_bleed = (
                                normalized_preset in DEFAULT_BLEED_BYPASS_PRESETS
                            )

                            if (
                                self.bleed_service
                                and self.bleed_service.enabled
                                and not bypass_bleed
                            ):
                                bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                                if bleed_info:
                                    bleed_ratio = float(bleed_info.bleed_ratio)
                                    # Блокируем повышение gain при высоком bleed (>0.5)
                                    if (
                                        bleed_ratio > 0.5
                                        and result["gain"] > agc.current_gain
                                    ):
                                        # Не повышаем gain - оставляем текущий
                                        logger.debug(
                                            f"Ch{audio_ch}: High bleed (ratio={bleed_ratio:.2f}), blocking gain increase"
                                        )
                                        # Можно также компенсировать LUFS перед расчетом коррекции
                                    # Компенсируем LUFS для более точного расчета
                                    compensated_lufs = (
                                        self.bleed_service.get_compensated_level(
                                            audio_ch, result["lufs"]
                                        )
                                    )
                                    if compensated_lufs != result["lufs"]:
                                        # Пересчитываем gain с компенсированным LUFS
                                        error = agc.target_lufs - compensated_lufs
                                        compensated_gain = error / agc.ratio
                                        compensated_gain = max(
                                            agc.min_gain_db,
                                            min(agc.max_gain_db, compensated_gain),
                                        )
                                        # Используем компенсированный gain только если он меньше (меньше усиление при блиде)
                                        if compensated_gain < result["gain"]:
                                            agc.current_gain = compensated_gain
                                            result["gain"] = compensated_gain
                                            result["applied_gain"] = (
                                                agc.base_trim + compensated_gain
                                            )
                                            logger.debug(
                                                f"Ch{audio_ch}: Compensated LUFS {result['lufs']:.1f} -> {compensated_lufs:.1f}, gain {result['gain']:.2f} -> {compensated_gain:.2f}"
                                            )

                            self.measured_levels[audio_ch]["bleed_ratio"] = bleed_ratio
                            self.measured_levels[audio_ch]["bleed_source"] = (
                                int(bleed_info.bleed_source_channel)
                                if bleed_info and bleed_info.bleed_source_channel
                                else None
                            )

                            levels_update[audio_ch] = self.measured_levels[audio_ch]

                            # Peak Staging режим (пропуск при freeze)
                            if self.peak_staging_mode and not self.automation_frozen:
                                effective_min_trim = (
                                    self.min_trim_db_live
                                    if self.live_mode
                                    else self.min_trim_db
                                )
                                # Проверяем, активна ли коррекция для этого канала
                                if self.trim_correction_active.get(mixer_ch, False):
                                    # Получаем порог пика из channel_settings
                                    channel_setting = self.channel_settings.get(
                                        audio_ch, {}
                                    )
                                    peak_threshold = channel_setting.get(
                                        "peak_threshold", self.default_peak_threshold
                                    )

                                    # Проверяем превышение порога
                                    if result["true_peak"] > peak_threshold:
                                        logger.info(
                                            f"Peak threshold exceeded for channel {mixer_ch}: "
                                            f"{result['true_peak']:.1f} > {peak_threshold:.1f} dBTP"
                                        )

                                        # Получаем текущий Trim
                                        try:
                                            current_trim = (
                                                self.mixer_client.get_channel_gain(
                                                    mixer_ch
                                                )
                                                or 0.0
                                            )

                                            # Вычисляем новый Trim (снижаем на 1дБ)
                                            new_trim = (
                                                current_trim
                                                - self.trim_reduction_step_db
                                            )

                                            # Ограничиваем диапазон (в live_mode не глубже min_trim_db_live)
                                            new_trim = max(effective_min_trim, new_trim)

                                            # Применяем новый Trim
                                            logger.info(
                                                f"Reducing Trim for channel {mixer_ch}: "
                                                f"{current_trim:.1f} -> {new_trim:.1f} dB"
                                            )
                                            self.mixer_client.set_channel_gain(
                                                mixer_ch, new_trim
                                            )

                                            # Проверяем фактический Trim после применения
                                            actual_trim = (
                                                self.mixer_client.get_channel_gain(
                                                    mixer_ch
                                                )
                                                or new_trim
                                            )

                                            # Определяем остановку коррекции
                                            last_trim = self.last_trim_value.get(
                                                mixer_ch, current_trim
                                            )

                                            # Остановка если:
                                            # 1. Trim не изменился после попытки снижения (actual_trim == current_trim)
                                            #    Это означает, что микшер не принял изменение или достигнут минимум
                                            # 2. Trim достиг минимального значения
                                            # 3. Новый Trim равен предыдущему значению (Trim перестал уменьшаться)
                                            trim_unchanged = (
                                                abs(actual_trim - current_trim) < 0.01
                                            )
                                            trim_at_minimum = (
                                                actual_trim <= effective_min_trim + 0.01
                                            )
                                            trim_no_progress = (
                                                abs(actual_trim - last_trim) < 0.01
                                                and self.trim_reduction_count.get(
                                                    mixer_ch, 0
                                                )
                                                > 0
                                            )

                                            if (
                                                trim_unchanged
                                                or trim_at_minimum
                                                or trim_no_progress
                                            ):
                                                self.trim_correction_active[
                                                    mixer_ch
                                                ] = False
                                                if trim_at_minimum:
                                                    reason = "reached minimum limit"
                                                elif trim_unchanged:
                                                    reason = "trim did not change after reduction attempt"
                                                else:
                                                    reason = "no progress detected (trim stopped decreasing)"
                                                logger.info(
                                                    f"Trim correction stopped for channel {mixer_ch}: "
                                                    f"{reason} (trim={actual_trim:.1f} dB, was={current_trim:.1f} dB, "
                                                    f"last={last_trim:.1f} dB)"
                                                )
                                            else:
                                                # Обновляем состояние
                                                self.last_trim_value[mixer_ch] = (
                                                    actual_trim
                                                )
                                                self.trim_reduction_count[mixer_ch] = (
                                                    self.trim_reduction_count.get(
                                                        mixer_ch, 0
                                                    )
                                                    + 1
                                                )
                                                logger.debug(
                                                    f"Channel {mixer_ch}: Trim reduced, count={self.trim_reduction_count[mixer_ch]}"
                                                )

                                        except Exception as e:
                                            logger.error(
                                                f"Error processing peak staging for channel {mixer_ch}: {e}"
                                            )
                            else:
                                # LUFS режим (оригинальная логика)
                                # Если не gated и коррекция достаточно значительная
                                if not result["is_gated"] and abs(result["gain"]) > 0.1:
                                    trim_updates[mixer_ch] = result["applied_gain"]

                # Применяем TRIM коррекции (только для LUFS режима; пропуск при freeze)
                if (
                    not self.peak_staging_mode
                    and trim_updates
                    and self.mixer_client
                    and not self.automation_frozen
                ):
                    for mixer_ch, new_trim in trim_updates.items():
                        try:
                            # Ограничиваем диапазон
                            new_trim = max(-18.0, min(18.0, new_trim))
                            self.mixer_client.set_channel_gain(mixer_ch, new_trim)
                            logger.info(
                                f"Channel {mixer_ch}: TRIM -> {new_trim:.1f} dB"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error setting TRIM for channel {mixer_ch}: {e}"
                            )

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
            self.on_status_update(
                {"type": "realtime_correction_stopped", "active": False}
            )

    def stop(self):
        """Полная остановка контроллера."""
        self.stop_realtime_correction()

        self.is_active = False
        self._stop_event.set()

        if self._audio_capture is not None:
            try:
                self._audio_capture.unsubscribe("lufs_gain_staging")
            except Exception:
                pass

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
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
                key: bool(value)
                if isinstance(value, np.bool_)
                else float(value)
                if isinstance(
                    value, (np.floating, np.float_, np.float16, np.float32, np.float64)
                )
                else int(value)
                if isinstance(
                    value,
                    (
                        np.integer,
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                    ),
                )
                else value
                for key, value in data.items()
            }

        return {
            "active": bool(self.is_active),
            "realtime_enabled": bool(self.realtime_correction_enabled),
            "target_lufs": float(self.target_lufs),
            "true_peak_limit": float(self.true_peak_limit),
            "ratio": float(self.ratio),
            "channels": int(len(self.channels)),
            "levels": levels,
        }

    def update_parameters(
        self,
        target_lufs: Optional[float] = None,
        true_peak_limit: Optional[float] = None,
        max_gain_db: Optional[float] = None,
        min_gain_db: Optional[float] = None,
        ratio: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
        hold_ms: Optional[float] = None,
        gate_threshold_lufs: Optional[float] = None,
    ):
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

        logger.info(
            f"AGC parameters updated: target={self.target_lufs} LUFS, ratio={self.ratio}:1, gate={self.gate_threshold_lufs}"
        )

    def apply_preset(self, name: str) -> bool:
        """Применить пресет soundcheck или live (target_lufs, ratio, attack/release, gate)."""
        preset = self._presets.get(name)
        if not preset:
            logger.warning(f"Unknown preset: {name}")
            return False
        self.update_parameters(
            target_lufs=preset.get("target_lufs"),
            ratio=preset.get("ratio"),
            attack_ms=preset.get("attack_ms"),
            release_ms=preset.get("release_ms"),
            gate_threshold_lufs=preset.get("gate_threshold_lufs"),
        )
        self.live_mode = name == "live"
        logger.info(
            f"Applied preset '{name}': target={self.target_lufs} LUFS, ratio={self.ratio}:1, gate={self.gate_threshold_lufs}"
        )
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

    def __init__(
        self,
        mixer_client=None,
        sample_rate: int = 48000,
        config: Optional[Dict[str, Any]] = None,
        bleed_service=None,
    ):
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.bleed_service = bleed_service

        self.config = config or {}
        cal_config = self.config.get("automation", {}).get("safe_gain_calibration", {})

        self.target_lufs = cal_config.get("target_lufs", -18.0)
        self.max_peak_limit = cal_config.get("max_peak_limit", -3.0)
        self.default_target_peak_dbfs = cal_config.get(
            "default_target_peak_dbfs", -12.0
        )
        self.max_gain_adjustment_db = float(
            cal_config.get("max_gain_adjustment_db", 18.0)
        )
        self.noise_gate_threshold = cal_config.get("noise_gate_threshold", -40.0)
        self.min_signal_presence = cal_config.get("min_signal_presence", 0.05)
        self.own_source_threshold = float(cal_config.get("own_source_threshold", 0.40))
        self.bleed_reject_ratio = float(cal_config.get("bleed_reject_ratio", 0.50))
        self.bleed_learn_blocks = max(1, int(cal_config.get("bleed_learn_blocks", 12)))
        self.own_trigger_delta_db = float(cal_config.get("own_trigger_delta_db", 6.0))
        self.capture_window_db = float(cal_config.get("capture_window_db", 3.0))
        self.required_own_events = max(1, int(cal_config.get("required_own_events", 3)))
        self.capture_timeout_sec = max(
            0.5, float(cal_config.get("capture_timeout_sec", 8.0))
        )
        self.retrigger_delta_db = float(cal_config.get("retrigger_delta_db", 4.0))
        self.max_trigger_peak_dbfs = float(
            cal_config.get("max_trigger_peak_dbfs", -6.0)
        )
        self.bypass_trigger_peak_dbfs = float(
            cal_config.get("bypass_trigger_peak_dbfs", -20.0)
        )
        phase_overrides = cal_config.get("phase_overrides")
        if isinstance(phase_overrides, dict):
            self.phase_overrides = {
                _normalize_preset_name(preset): values
                for preset, values in phase_overrides.items()
                if isinstance(values, dict)
            }
        else:
            self.phase_overrides = {}
        self.low_confidence_source_threshold = float(
            cal_config.get("low_confidence_source_threshold", 0.50)
        )
        self.low_confidence_max_boost_db = float(
            cal_config.get("low_confidence_max_boost_db", 6.0)
        )
        self.gain_increase_true_peak_ceiling_dbtp = float(
            cal_config.get("gain_increase_true_peak_ceiling_dbtp", -1.0)
        )
        self.drums_close_max_boost_db = float(
            cal_config.get("drums_close_max_boost_db", 6.0)
        )
        self.exclude_bleed_from_own_capture = bool(
            cal_config.get("exclude_bleed_from_own_capture", True)
        )
        self.capture_bleed_guard_ratio = float(
            cal_config.get("capture_bleed_guard_ratio", 0.55)
        )
        self.capture_bleed_guard_confidence = float(
            cal_config.get("capture_bleed_guard_confidence", 0.45)
        )
        self.max_single_step_cut_db = float(
            cal_config.get("max_single_step_cut_db", -12.0)
        )
        self.trim_apply_limit_db = float(cal_config.get("trim_apply_limit_db", 12.0))
        self.auto_stop_when_ready = bool(cal_config.get("auto_stop_when_ready", True))
        self.min_total_samples_for_ready = int(
            cal_config.get("min_total_samples_for_ready", 20)
        )
        self.min_own_source_samples_for_ready = int(
            cal_config.get("min_own_source_samples_for_ready", 8)
        )
        self.max_learning_duration_sec = float(
            cal_config.get("max_learning_duration_sec", 0.0)
        )
        bypass_presets = cal_config.get("bleed_bypass_presets")
        if isinstance(bypass_presets, list) and bypass_presets:
            self.bleed_bypass_presets = {
                _normalize_preset_name(preset) for preset in bypass_presets
            }
        else:
            self.bleed_bypass_presets = set(DEFAULT_BLEED_BYPASS_PRESETS)
        self.learning_duration = cal_config.get(
            "learning_duration_sec", 30.0
        )  # По умолчанию 30 секунд
        logger.info(
            f"SafeGainCalibrator initialized with learning_duration={self.learning_duration} seconds"
        )

        # Channel settings for preset-based metrics
        self.channel_settings: Dict[int, Dict] = {}
        self.preset_targets = cal_config.get("preset_targets", {})
        self.targets_data = _load_gain_targets(self.config)
        self.state = AnalysisState.IDLE

        self.channels: Dict[int, SignalStats] = {}

        self.channel_mapping: Dict[int, int] = {}

        self.lufs_meters: Dict[int, LUFSMeter] = {}
        self.true_peak_meters: Dict[int, TruePeakMeter] = {}

        self.learning_start_time: Optional[float] = None
        self.learning_progress: float = 0.0
        self._last_progress_update_time: float = 0.0
        self._progress_update_interval: float = 0.5

        self.suggestions: Dict[int, Dict[str, Any]] = {}

        self.on_progress_update: Optional[Callable] = None
        self.on_suggestions_ready: Optional[Callable] = None
        self._frozen_ready_channels: set[int] = set()

        logger.info(
            "SafeGainCalibrator initialized: peak default=%.1f dBFS, gate=%.1f dB, own_th=%.2f",
            self.default_target_peak_dbfs,
            self.noise_gate_threshold,
            self.own_source_threshold,
        )

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
            channel_id=audio_channel, noise_gate_threshold_db=self.noise_gate_threshold
        )

        self.lufs_meters[audio_channel] = LUFSMeter(
            sample_rate=self.sample_rate, window_ms=400.0
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

    def _resolve_target_peak(self, channel_id: int) -> tuple[str, float, str]:
        setting = self.channel_settings.get(channel_id, {})
        raw = setting.get("preset") or setting.get("instrumentType") or "custom"
        preset = _normalize_preset_name(raw)

        targets = self.targets_data.get("targets", {})
        categories = self.targets_data.get(
            "categories_defaults", DEFAULT_CATEGORY_PEAKS
        )
        target_cfg = targets.get(preset)
        if target_cfg:
            category = target_cfg.get("category", "instruments")
            return (
                preset,
                float(target_cfg.get("peak_dbfs", self.default_target_peak_dbfs)),
                category,
            )

        category = "instruments"
        if "vocal" in preset or preset in {"bgv"}:
            category = "vocals"
        elif preset in {
            "kick",
            "snare",
            "snare_bottom",
            "tom",
            "tom_hi",
            "tom_mid",
            "tom_lo",
        }:
            category = "drums_close"
        elif preset in {"hi_hat", "ride", "overhead", "overhead_l", "overhead_r"}:
            category = "drums_ambient"
        peak = float(categories.get(category, self.default_target_peak_dbfs))
        return preset, peak, category

    def _spectral_centroid_hz(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        spectrum = np.abs(np.fft.rfft(samples))
        if spectrum.size == 0:
            return 0.0
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        den = float(np.sum(spectrum)) + 1e-10
        return float(np.sum(freqs * spectrum) / den)

    def _estimate_source_confidence(
        self,
        channel_id: int,
        preset: str,
        true_peak: float,
        spectral_centroid: float,
        bleed_ratio: float,
    ) -> float:
        score = 1.0 - max(0.0, min(1.0, bleed_ratio))
        if true_peak <= self.noise_gate_threshold:
            score *= 0.5
        low, high = EXPECTED_CENTROID_HZ.get(preset, (0.0, 24000.0))
        if spectral_centroid < low or spectral_centroid > high:
            score *= 0.5
        if preset == "custom":
            score = max(score, 0.5)
        return max(0.0, min(1.0, score))

    def _resolve_phase_settings(self, preset: str) -> Dict[str, float]:
        normalized = _normalize_preset_name(preset)
        override = self.phase_overrides.get(normalized, {})
        if not override and normalized.startswith("tom_"):
            override = self.phase_overrides.get("tom", {})
        return {
            "bleed_learn_blocks": max(
                1, int(override.get("bleed_learn_blocks", self.bleed_learn_blocks))
            ),
            "own_trigger_delta_db": float(
                override.get("own_trigger_delta_db", self.own_trigger_delta_db)
            ),
            "capture_window_db": max(
                0.5, float(override.get("capture_window_db", self.capture_window_db))
            ),
            "required_own_events": max(
                1, int(override.get("required_own_events", self.required_own_events))
            ),
            "capture_timeout_sec": max(
                0.5,
                float(override.get("capture_timeout_sec", self.capture_timeout_sec)),
            ),
            "retrigger_delta_db": float(
                override.get("retrigger_delta_db", self.retrigger_delta_db)
            ),
            "max_trigger_peak_dbfs": float(
                override.get("max_trigger_peak_dbfs", self.max_trigger_peak_dbfs)
            ),
            "bypass_trigger_peak_dbfs": float(
                override.get("bypass_trigger_peak_dbfs", self.bypass_trigger_peak_dbfs)
            ),
        }

    @staticmethod
    def _phase_status_text(
        phase: ChannelPhase, accepted: int = 0, required: int = 0
    ) -> str:
        if phase == ChannelPhase.BLEED_LEARN:
            return "Фиксация bleed..."
        if phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL:
            return "Подайте основной сигнал"
        if phase == ChannelPhase.CAPTURE_OWN_LEVEL:
            return f"Накопление данных {accepted}/{required}"
        return "Готов к применению"

    def _set_channel_phase(
        self,
        stats: SignalStats,
        phase: ChannelPhase,
        now: float,
    ) -> None:
        stats.phase = phase
        stats.phase_started_at = now

    def _accept_capture_event(
        self,
        stats: SignalStats,
        true_peak: float,
        lufs: float,
    ) -> None:
        stats.own_source_samples += 1
        stats.accepted_count += 1
        stats.peak_values.append(float(true_peak))
        stats.rms_values.append(float(lufs))
        if true_peak > stats.max_true_peak_db:
            stats.max_true_peak_db = float(true_peak)

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
                channel_id=ch_id, noise_gate_threshold_db=self.noise_gate_threshold
            )
            self.channels[ch_id].phase = ChannelPhase.BLEED_LEARN
            self.lufs_meters[ch_id].reset()
            self.true_peak_meters[ch_id].reset()

        self.suggestions.clear()
        self.learning_start_time = time.time()
        self.learning_progress = 0.0
        self._last_progress_update_time = 0.0
        self._frozen_ready_channels.clear()
        self.state = AnalysisState.LEARNING
        for stats in self.channels.values():
            stats.phase_started_at = self.learning_start_time

        logger.info(
            "Analysis started: auto_stop=%s, phase_cfg blocks=%s trigger=%.1fdB window=%.1fdB events=%s",
            self.auto_stop_when_ready,
            self.bleed_learn_blocks,
            self.own_trigger_delta_db,
            self.capture_window_db,
            self.required_own_events,
        )

        if self.on_progress_update:
            self.on_progress_update(
                {
                    "state": self.state.value,
                    "progress": 0.0,
                    "message": "Analyzing signal until all channels are ready",
                }
            )

        return True

    def _is_channel_ready_for_finalize(self, stats: SignalStats) -> bool:
        """A channel is ready when its phase machine reaches READY."""
        return (
            stats.phase == ChannelPhase.READY
            and stats.accepted_count >= max(1, stats.required_own_events)
            and len(stats.peak_values) > 0
        )

    def _ready_channels_count(self) -> int:
        return sum(
            1
            for stats in self.channels.values()
            if self._is_channel_ready_for_finalize(stats)
        )

    def _get_current_channel_stats(self) -> Dict[int, Dict[str, Any]]:
        """Return current per-channel metrics for frontend display during learning."""
        result: Dict[int, Dict[str, Any]] = {}
        for ch_id in self.channels:
            stats = self.channels[ch_id]
            preset, target_peak, _ = self._resolve_target_peak(ch_id)
            lufs = -60.0
            peak = -60.0
            if ch_id in self.lufs_meters:
                lufs = self.lufs_meters[ch_id].get_current_lufs()
            if ch_id in self.true_peak_meters:
                peak = self.true_peak_meters[ch_id].get_current_peak_dbtp()
            own_source_ratio = float(
                stats.own_source_samples / max(stats.total_samples, 1)
            )
            ready_for_finalize = self._is_channel_ready_for_finalize(stats)
            status = self._phase_status_text(
                stats.phase,
                accepted=stats.accepted_count,
                required=stats.required_own_events,
            )
            result[ch_id] = {
                "lufs": float(lufs),
                "true_peak": float(peak),
                "peak": float(peak),
                "signal_present": stats.signal_presence_ratio > 0.01,
                "signal_presence": float(stats.signal_presence_ratio),
                "status": status,
                "gain": 0.0,
                "preset": preset,
                "target_peak_dbfs": float(target_peak),
                "source_confidence": float(stats.last_source_confidence),
                "bleed_ratio": float(stats.last_bleed_ratio),
                "bleed_confidence": float(stats.last_bleed_confidence),
                "bleed_method": str(stats.last_bleed_method),
                "own_source_ratio": own_source_ratio,
                "phase": stats.phase.value,
                "accepted_count": int(stats.accepted_count),
                "required_own_events": int(stats.required_own_events),
                "phase_progress": float(
                    min(
                        1.0,
                        stats.accepted_count / max(1, stats.required_own_events),
                    )
                ),
                "reference_peak_dbfs": (
                    float(stats.reference_peak_dbfs)
                    if stats.reference_peak_dbfs is not None
                    else None
                ),
                "bleed_baseline_peak_dbfs": float(stats.bleed_baseline_peak_dbfs),
                "ready_for_finalize": ready_for_finalize,
                "frozen_after_ready": ch_id in self._frozen_ready_channels,
            }
        return result

    # dB above other channels to treat as dominant (constant loud signal = own source)
    DOMINANT_SIGNAL_MARGIN_DB = 8.0
    STEREO_MONO_SUM_COMPENSATION_DB = 6.0

    def process_audio(
        self,
        channel: int,
        samples: np.ndarray,
        bleed_ratio_override: Optional[float] = None,
        bleed_confidence_override: Optional[float] = None,
        bleed_method_override: Optional[str] = None,
        source_confidence_override: Optional[float] = None,
        spectral_centroid_override: Optional[float] = None,
        all_channel_levels_db: Optional[Dict[int, float]] = None,
    ):
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

        if channel in self._frozen_ready_channels:
            return

        stats = self.channels[channel]
        lufs_meter = self.lufs_meters[channel]
        peak_meter = self.true_peak_meters[channel]
        now = time.time()

        lufs = lufs_meter.process(samples)
        true_peak = peak_meter.process(samples)
        spectral_centroid = (
            float(spectral_centroid_override)
            if spectral_centroid_override is not None
            else self._spectral_centroid_hz(samples)
        )

        rms = np.sqrt(np.mean(samples**2) + 1e-10)
        rms_db = 20 * np.log10(rms) if rms > 0 else -100.0

        preset, _, _ = self._resolve_target_peak(channel)
        bleed_ratio = 0.0
        bleed_confidence = 0.0
        bleed_method = "none"
        bypass_bleed = preset in self.bleed_bypass_presets
        dominant_signal = False
        if all_channel_levels_db and channel in all_channel_levels_db:
            ch_level = all_channel_levels_db[channel]
            others = [
                v for k, v in all_channel_levels_db.items() if k != channel and v > -90
            ]
            if others and ch_level >= max(others) + self.DOMINANT_SIGNAL_MARGIN_DB:
                dominant_signal = True

        if bypass_bleed or dominant_signal:
            bleed_ratio = 0.0
            bleed_confidence = 0.0
            bleed_method = "bypass_or_dominant"
        elif bleed_ratio_override is not None:
            bleed_ratio = float(bleed_ratio_override)
            bleed_confidence = float(bleed_confidence_override or 0.0)
            bleed_method = str(bleed_method_override or "override")
        elif self.bleed_service and self.bleed_service.enabled:
            bleed_info = self.bleed_service.get_bleed_info(channel)
            if bleed_info:
                bleed_ratio = float(bleed_info.bleed_ratio)
                bleed_confidence = float(getattr(bleed_info, "confidence", 0.0))
                bleed_method = str(getattr(bleed_info, "method_used", "service"))

        source_confidence = (
            float(source_confidence_override)
            if source_confidence_override is not None
            else self._estimate_source_confidence(
                channel_id=channel,
                preset=preset,
                true_peak=true_peak,
                spectral_centroid=spectral_centroid,
                bleed_ratio=bleed_ratio,
            )
        )
        stats.last_bleed_ratio = bleed_ratio
        stats.last_bleed_confidence = bleed_confidence
        stats.last_bleed_method = bleed_method
        stats.last_source_confidence = source_confidence

        strong_bleed_block = (
            self.exclude_bleed_from_own_capture
            and not bypass_bleed
            and not dominant_signal
            and bleed_ratio >= self.capture_bleed_guard_ratio
            and bleed_confidence >= self.capture_bleed_guard_confidence
        )

        stats.total_samples += 1
        if true_peak > stats.noise_gate_threshold_db:
            stats.active_samples += 1
        if true_peak > stats.global_max_true_peak_db:
            stats.global_max_true_peak_db = float(true_peak)
        stats.signal_presence_ratio = stats.active_samples / max(stats.total_samples, 1)

        phase_cfg = self._resolve_phase_settings(preset)
        stats.required_own_events = int(phase_cfg["required_own_events"])

        if stats.phase == ChannelPhase.BLEED_LEARN:
            stats.bleed_blocks_collected += 1
            stats.bleed_peak_samples.append(float(true_peak))
            stats.bleed_rms_samples.append(float(rms_db))
            if stats.bleed_blocks_collected >= int(phase_cfg["bleed_learn_blocks"]):
                stats.bleed_baseline_peak_dbfs = float(
                    np.median(stats.bleed_peak_samples)
                )
                stats.bleed_baseline_rms_db = float(np.median(stats.bleed_rms_samples))
                self._set_channel_phase(stats, ChannelPhase.WAIT_FOR_OWN_SIGNAL, now)
                logger.info(
                    "Ch%s (%s): bleed baseline fixed peak=%.1f dBFS rms=%.1f dB",
                    channel,
                    preset,
                    stats.bleed_baseline_peak_dbfs,
                    stats.bleed_baseline_rms_db,
                )

        elif stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL:
            if bypass_bleed:
                bypass_trigger_peak = float(phase_cfg["bypass_trigger_peak_dbfs"])
                own_trigger = (
                    (true_peak >= bypass_trigger_peak) or dominant_signal
                ) and true_peak > stats.noise_gate_threshold_db
            else:
                trigger_peak = min(
                    stats.bleed_baseline_peak_dbfs
                    + float(phase_cfg["own_trigger_delta_db"]),
                    float(phase_cfg["max_trigger_peak_dbfs"]),
                )
                own_trigger = (
                    true_peak >= trigger_peak
                    and true_peak > stats.noise_gate_threshold_db
                )
            if own_trigger and not strong_bleed_block:
                self._set_channel_phase(stats, ChannelPhase.CAPTURE_OWN_LEVEL, now)
                stats.capture_started_at = now
                stats.reference_peak_dbfs = float(true_peak)
                stats.accepted_count = 0
                stats.peak_values.clear()
                stats.rms_values.clear()
                stats.max_true_peak_db = -100.0
                self._accept_capture_event(stats, true_peak, lufs)
                logger.info(
                    "Ch%s (%s): own signal detected. capture start ref=%.1f dBFS",
                    channel,
                    preset,
                    stats.reference_peak_dbfs,
                )
            elif strong_bleed_block or bleed_ratio > self.bleed_reject_ratio:
                stats.rejected_bleed_samples += 1

        elif stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL:
            if (
                stats.capture_started_at is not None
                and now - stats.capture_started_at
                > float(phase_cfg["capture_timeout_sec"])
            ):
                stats.rejected_mismatch_samples += 1
                self._set_channel_phase(stats, ChannelPhase.WAIT_FOR_OWN_SIGNAL, now)
                stats.capture_started_at = None
                stats.reference_peak_dbfs = None
                stats.accepted_count = 0
                stats.peak_values.clear()
                stats.rms_values.clear()
                stats.max_true_peak_db = -100.0
            else:
                reference = (
                    stats.reference_peak_dbfs
                    if stats.reference_peak_dbfs is not None
                    else true_peak
                )
                window_db = float(phase_cfg["capture_window_db"])
                if bypass_bleed:
                    own_capture_trigger = (
                        true_peak >= float(phase_cfg["bypass_trigger_peak_dbfs"])
                    ) or dominant_signal
                else:
                    own_capture_trigger = (
                        true_peak
                        >= min(
                            stats.bleed_baseline_peak_dbfs
                            + float(phase_cfg["own_trigger_delta_db"]),
                            float(phase_cfg["max_trigger_peak_dbfs"]),
                        )
                    ) or dominant_signal

                if strong_bleed_block:
                    stats.rejected_bleed_samples += 1
                elif (
                    abs(true_peak - reference) <= window_db
                    and true_peak > stats.noise_gate_threshold_db
                ):
                    self._accept_capture_event(stats, true_peak, lufs)
                elif own_capture_trigger and true_peak > stats.noise_gate_threshold_db:
                    # Loud own-source block with unstable dynamics:
                    # accept it and re-center the capture window around this level.
                    stats.reference_peak_dbfs = float(true_peak)
                    self._accept_capture_event(stats, true_peak, lufs)
                elif true_peak >= reference + float(phase_cfg["retrigger_delta_db"]):
                    # New stable level: restart capture around stronger reference.
                    stats.capture_started_at = now
                    stats.reference_peak_dbfs = float(true_peak)
                    stats.accepted_count = 0
                    stats.peak_values.clear()
                    stats.rms_values.clear()
                    stats.max_true_peak_db = -100.0
                    self._accept_capture_event(stats, true_peak, lufs)
                else:
                    stats.rejected_out_of_window_count += 1

                if stats.accepted_count >= stats.required_own_events:
                    self._set_channel_phase(stats, ChannelPhase.READY, now)

        if (
            channel not in self._frozen_ready_channels
            and self._is_channel_ready_for_finalize(stats)
        ):
            self._frozen_ready_channels.add(channel)
            logger.info(
                "Ch%s: capture complete (%s/%s), channel excluded from further analysis",
                channel,
                stats.accepted_count,
                stats.required_own_events,
            )

        elapsed = now - self.learning_start_time
        ready_channels = self._ready_channels_count()
        total_channels = max(len(self.channels), 1)
        self.learning_progress = min(ready_channels / total_channels, 1.0)

        # Periodic progress update with per-channel metrics for frontend display
        if (
            self.on_progress_update
            and elapsed - self._last_progress_update_time
            >= self._progress_update_interval
        ):
            self._last_progress_update_time = elapsed
            channel_stats = self._get_current_channel_stats()
            pending_channels = [
                ch_id
                for ch_id, st in self.channels.items()
                if not self._is_channel_ready_for_finalize(st)
            ]
            pending_hint = (
                (
                    " | Подайте основной сигнал в каналы: "
                    + ", ".join(str(ch) for ch in pending_channels)
                )
                if pending_channels
                else ""
            )
            self.on_progress_update(
                {
                    "state": self.state.value,
                    "progress": self.learning_progress,
                    "message": (
                        f"Анализ... готово каналов {ready_channels}/{len(self.channels)} "
                        f"({elapsed:.1f}s){pending_hint}"
                    ),
                    "channels": channel_stats,
                }
            )

        if self.state != AnalysisState.LEARNING:
            return

        if self.auto_stop_when_ready and ready_channels >= len(self.channels):
            logger.info(
                "All channels ready for correction (%s/%s). Finalizing analysis early.",
                ready_channels,
                len(self.channels),
            )
            self._finalize_analysis()
            return

        if (
            self.max_learning_duration_sec > 0
            and elapsed >= self.max_learning_duration_sec
        ):
            logger.info(
                "Max learning duration reached (%.1fs). Finalizing with current data.",
                elapsed,
            )
            self._finalize_analysis()

    def _finalize_analysis(self):
        """Завершение фазы анализа и расчет рекомендаций."""
        logger.info("Learning phase completed. Calculating suggestions...")

        for ch_id, stats in self.channels.items():
            preset, target_peak, category = self._resolve_target_peak(ch_id)
            logger.info(
                "Ch%s (%s/%s): target_peak=%.1f dBFS",
                ch_id,
                preset,
                category,
                target_peak,
            )
            if (
                category == "drums_close"
                and stats.global_max_true_peak_db > stats.max_true_peak_db
                and stats.global_max_true_peak_db > -90.0
            ):
                logger.info(
                    "Ch%s (%s): drums_close peak upgrade %.1f -> %.1f dBFS "
                    "(global max over entire analysis)",
                    ch_id,
                    preset,
                    stats.max_true_peak_db,
                    stats.global_max_true_peak_db,
                )
                stats.max_true_peak_db = stats.global_max_true_peak_db

            stats.calculate_peak_gain(
                target_peak_dbfs=target_peak,
                max_adjustment_db=self.max_gain_adjustment_db,
                min_accepted_samples=stats.required_own_events,
            )

            capture_confidence = stats.accepted_count / max(
                stats.required_own_events, 1
            )
            if (
                stats.suggested_gain_db > self.low_confidence_max_boost_db
                and capture_confidence < self.low_confidence_source_threshold
            ):
                logger.info(
                    "Ch%s: limiting boost %.1f -> %.1f dB (capture_confidence=%.1f%% < %.1f%%)",
                    ch_id,
                    stats.suggested_gain_db,
                    self.low_confidence_max_boost_db,
                    capture_confidence * 100.0,
                    self.low_confidence_source_threshold * 100.0,
                )
                stats.suggested_gain_db = self.low_confidence_max_boost_db
                stats.gain_limited_by = "low_confidence_boost_cap"

            if stats.suggested_gain_db > 0.0:
                max_safe_boost = (
                    self.gain_increase_true_peak_ceiling_dbtp - stats.max_true_peak_db
                )
                if stats.suggested_gain_db > max_safe_boost:
                    logger.info(
                        "Ch%s: limiting boost %.1f -> %.1f dB by true-peak ceiling %.1f dBTP",
                        ch_id,
                        stats.suggested_gain_db,
                        max_safe_boost,
                        self.gain_increase_true_peak_ceiling_dbtp,
                    )
                    stats.suggested_gain_db = max(0.0, max_safe_boost)
                    stats.gain_limited_by = "true_peak_ceiling"

            if category == "drums_close" and stats.suggested_gain_db > 0.0:
                if stats.suggested_gain_db > self.drums_close_max_boost_db:
                    logger.info(
                        "Ch%s (%s): limiting boost %.1f -> %.1f dB (drums_close cap)",
                        ch_id,
                        preset,
                        stats.suggested_gain_db,
                        self.drums_close_max_boost_db,
                    )
                    stats.suggested_gain_db = self.drums_close_max_boost_db
                    stats.gain_limited_by = "drums_close_max_boost"

            if stats.suggested_gain_db < self.max_single_step_cut_db:
                logger.info(
                    "Ch%s (%s): limiting cut %.1f -> %.1f dB (max_single_step_cut)",
                    ch_id,
                    preset,
                    stats.suggested_gain_db,
                    self.max_single_step_cut_db,
                )
                stats.suggested_gain_db = self.max_single_step_cut_db
                stats.gain_limited_by = "max_single_step_cut"

        self._apply_stereo_pair_rule()

        for ch_id, stats in self.channels.items():
            self.suggestions[ch_id] = stats.get_report()

        self.state = AnalysisState.READY

        logger.info(f"Analysis complete: {len(self.suggestions)} channels ready")

        if self.on_suggestions_ready:
            self.on_suggestions_ready(self.suggestions)

        if self.on_progress_update:
            self.on_progress_update(
                {
                    "state": self.state.value,
                    "progress": 1.0,
                    "message": "Analysis complete. Ready to apply.",
                }
            )

    @staticmethod
    def _is_stereo_excluded_preset(preset: str) -> bool:
        return any(
            preset == prefix or preset.startswith(f"{prefix}_")
            for prefix in STEREO_PAIR_DRUM_EXCLUDED_PREFIXES
        )

    def _apply_stereo_pair_rule(self) -> None:
        """Align duplicated non-drum presets as stereo pairs with configured mono compensation."""
        groups: Dict[str, List[int]] = {}
        for ch_id in self.channels:
            raw = (
                self.channel_settings.get(ch_id, {}).get("preset")
                or self.channel_settings.get(ch_id, {}).get("instrumentType")
                or "custom"
            )
            preset = _normalize_preset_name(raw)
            if preset == "custom" or self._is_stereo_excluded_preset(preset):
                continue
            groups.setdefault(preset, []).append(ch_id)

        for preset, pair_channels in groups.items():
            if len(pair_channels) != 2:
                continue

            left_ch, right_ch = pair_channels
            left_stats = self.channels.get(left_ch)
            right_stats = self.channels.get(right_ch)
            if not left_stats or not right_stats:
                continue

            left_corrected_peak = (
                left_stats.max_true_peak_db + left_stats.suggested_gain_db
            )
            right_corrected_peak = (
                right_stats.max_true_peak_db + right_stats.suggested_gain_db
            )
            stereo_target_peak = (
                left_corrected_peak + right_corrected_peak
            ) / 2.0 - self.STEREO_MONO_SUM_COMPENSATION_DB

            left_new_gain = stereo_target_peak - left_stats.max_true_peak_db
            right_new_gain = stereo_target_peak - right_stats.max_true_peak_db

            left_stats.target_peak_dbfs = stereo_target_peak
            right_stats.target_peak_dbfs = stereo_target_peak

            left_stats.suggested_gain_db = float(
                np.clip(
                    left_new_gain,
                    -self.max_gain_adjustment_db,
                    self.max_gain_adjustment_db,
                )
            )
            right_stats.suggested_gain_db = float(
                np.clip(
                    right_new_gain,
                    -self.max_gain_adjustment_db,
                    self.max_gain_adjustment_db,
                )
            )

            if left_stats.suggested_gain_db < self.max_single_step_cut_db:
                left_stats.suggested_gain_db = self.max_single_step_cut_db
            if right_stats.suggested_gain_db < self.max_single_step_cut_db:
                right_stats.suggested_gain_db = self.max_single_step_cut_db

            left_stats.gain_limited_by = "stereo_pair_balance"
            right_stats.gain_limited_by = "stereo_pair_balance"

            logger.info(
                "Stereo pair (%s): ch%s/ch%s target=%.1f dBFS "
                "(from corrected %.1f / %.1f dBFS, mono compensation -%.1f dB)",
                preset,
                left_ch,
                right_ch,
                stereo_target_peak,
                left_corrected_peak,
                right_corrected_peak,
                self.STEREO_MONO_SUM_COMPENSATION_DB,
            )

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

    def apply_corrections(self, channel_ids: Optional[List[int]] = None) -> bool:
        """
        Применение рекомендованных коррекций к микшеру.

        Args:
            channel_ids: Список каналов для применения (по умолчанию все готовые)

        Returns:
            True если хотя бы одна коррекция была применена
        """
        if self.state != AnalysisState.READY:
            logger.error(
                f"Cannot apply corrections: state is {self.state}, must be READY"
            )
            return False

        if not self.mixer_client:
            logger.error("Cannot apply corrections: no mixer client")
            return False

        channels_to_apply = (
            channel_ids if channel_ids else list(self.suggestions.keys())
        )

        self.state = AnalysisState.APPLYING

        applied_count = 0
        for ch_id in channels_to_apply:
            if ch_id not in self.suggestions:
                logger.warning(f"No suggestion for channel {ch_id}, skipping")
                continue

            suggestion = self.suggestions[ch_id]
            gain = suggestion["suggested_gain_db"]

            mixer_ch = self.channel_mapping.get(ch_id, ch_id)

            try:
                current_trim = self.mixer_client.get_channel_gain(mixer_ch) or 0.0
                new_trim = current_trim + gain

                limit = self.trim_apply_limit_db
                new_trim = float(np.clip(new_trim, -limit, limit))

                self.mixer_client.set_channel_gain(mixer_ch, new_trim)
                logger.info(
                    f"Channel {mixer_ch}: TRIM {current_trim:.1f} -> {new_trim:.1f} dB "
                    f"(gain={gain:+.1f}dB)"
                )
                applied_count += 1

            except Exception as e:
                logger.error(f"Failed to apply correction to channel {mixer_ch}: {e}")

        self.state = AnalysisState.IDLE

        logger.info(
            f"Corrections applied: {applied_count}/{len(channels_to_apply)} channels"
        )

        if self.on_progress_update:
            self.on_progress_update(
                {
                    "state": self.state.value,
                    "progress": 1.0,
                    "message": f"Applied {applied_count} corrections",
                }
            )

        return applied_count > 0

    def reset(self):
        """Сброс калибратора в IDLE состояние."""
        self.state = AnalysisState.IDLE
        for stats in self.channels.values():
            stats.total_samples = 0
            stats.active_samples = 0
            stats.own_source_samples = 0
            stats.rejected_bleed_samples = 0
            stats.rejected_mismatch_samples = 0
            stats.rejected_out_of_window_count = 0
            stats.last_source_confidence = 0.0
            stats.last_bleed_ratio = 0.0
            stats.last_bleed_confidence = 0.0
            stats.last_bleed_method = "none"
            stats.rms_values.clear()
            stats.peak_values.clear()
            stats.phase = ChannelPhase.BLEED_LEARN
            stats.phase_started_at = None
            stats.bleed_blocks_collected = 0
            stats.bleed_peak_samples.clear()
            stats.bleed_rms_samples.clear()
            stats.bleed_baseline_peak_dbfs = -100.0
            stats.bleed_baseline_rms_db = -100.0
            stats.reference_peak_dbfs = None
            stats.capture_started_at = None
            stats.accepted_count = 0
            stats.max_true_peak_db = -100.0
            stats.global_max_true_peak_db = -100.0
        self.suggestions.clear()
        self._frozen_ready_channels.clear()
        self.learning_start_time = None
        self.learning_progress = 0.0
        logger.info("Calibrator reset to IDLE state")

    def get_status(self) -> Dict[str, Any]:
        """Получить текущий статус калибратора."""
        return {
            "state": self.state.value,
            "learning_progress": self.learning_progress,
            "channels_count": len(self.channels),
            "ready_channels_count": self._ready_channels_count(),
            "suggestions_ready": len(self.suggestions) > 0,
            "target_lufs": self.target_lufs,  # legacy key
            "max_peak_limit": self.max_peak_limit,  # legacy key
            "default_target_peak_dbfs": self.default_target_peak_dbfs,
            "bleed_reject_ratio": self.bleed_reject_ratio,
            "own_source_threshold": self.own_source_threshold,
            "auto_stop_when_ready": self.auto_stop_when_ready,
            "min_total_samples_for_ready": self.min_total_samples_for_ready,  # legacy
            "min_own_source_samples_for_ready": self.min_own_source_samples_for_ready,  # legacy
            "max_learning_duration_sec": self.max_learning_duration_sec,
            "learning_duration": self.learning_duration,
            "bleed_learn_blocks": self.bleed_learn_blocks,
            "own_trigger_delta_db": self.own_trigger_delta_db,
            "capture_window_db": self.capture_window_db,
            "required_own_events": self.required_own_events,
            "capture_timeout_sec": self.capture_timeout_sec,
            "retrigger_delta_db": self.retrigger_delta_db,
            "max_trigger_peak_dbfs": self.max_trigger_peak_dbfs,
            "bypass_trigger_peak_dbfs": self.bypass_trigger_peak_dbfs,
            "drums_close_max_boost_db": self.drums_close_max_boost_db,
            "exclude_bleed_from_own_capture": self.exclude_bleed_from_own_capture,
            "capture_bleed_guard_ratio": self.capture_bleed_guard_ratio,
            "capture_bleed_guard_confidence": self.capture_bleed_guard_confidence,
            "max_single_step_cut_db": self.max_single_step_cut_db,
            "trim_apply_limit_db": self.trim_apply_limit_db,
        }

    def update_settings(self, settings: Dict[str, Any]):
        """Обновить настройки калибратора."""
        if "learning_duration_sec" in settings:
            new_duration = float(settings["learning_duration_sec"])
            if new_duration > 0:
                self.learning_duration = new_duration
                logger.info(
                    f"Updated learning_duration to {self.learning_duration} seconds"
                )
                # Если анализ уже идет, пересчитываем прогресс
                if self.state == AnalysisState.LEARNING and self.learning_start_time:
                    elapsed = time.time() - self.learning_start_time
                    self.learning_progress = min(
                        1.0, (elapsed / self.learning_duration)
                    )

        if "target_lufs" in settings:
            self.target_lufs = float(settings["target_lufs"])
            logger.info(f"Updated target_lufs to {self.target_lufs}")

        if "max_peak_limit" in settings:
            self.max_peak_limit = float(settings["max_peak_limit"])
            logger.info(f"Updated max_peak_limit to {self.max_peak_limit}")

        if "default_target_peak_dbfs" in settings:
            self.default_target_peak_dbfs = float(settings["default_target_peak_dbfs"])
            logger.info(
                "Updated default_target_peak_dbfs to %s", self.default_target_peak_dbfs
            )

        if "bleed_reject_ratio" in settings:
            self.bleed_reject_ratio = float(settings["bleed_reject_ratio"])
            logger.info("Updated bleed_reject_ratio to %s", self.bleed_reject_ratio)

        if "own_source_threshold" in settings:
            self.own_source_threshold = float(settings["own_source_threshold"])
            logger.info("Updated own_source_threshold to %s", self.own_source_threshold)

        if "auto_stop_when_ready" in settings:
            self.auto_stop_when_ready = bool(settings["auto_stop_when_ready"])
            logger.info("Updated auto_stop_when_ready to %s", self.auto_stop_when_ready)
        if "min_total_samples_for_ready" in settings:
            self.min_total_samples_for_ready = max(
                1, int(settings["min_total_samples_for_ready"])
            )
            logger.info(
                "Updated min_total_samples_for_ready to %s",
                self.min_total_samples_for_ready,
            )
        if "min_own_source_samples_for_ready" in settings:
            self.min_own_source_samples_for_ready = max(
                1, int(settings["min_own_source_samples_for_ready"])
            )
            logger.info(
                "Updated min_own_source_samples_for_ready to %s",
                self.min_own_source_samples_for_ready,
            )
        if "max_learning_duration_sec" in settings:
            self.max_learning_duration_sec = max(
                0.0, float(settings["max_learning_duration_sec"])
            )
            logger.info(
                "Updated max_learning_duration_sec to %s",
                self.max_learning_duration_sec,
            )
        if "bleed_learn_blocks" in settings:
            self.bleed_learn_blocks = max(1, int(settings["bleed_learn_blocks"]))
            logger.info("Updated bleed_learn_blocks to %s", self.bleed_learn_blocks)
        if "own_trigger_delta_db" in settings:
            self.own_trigger_delta_db = float(settings["own_trigger_delta_db"])
            logger.info("Updated own_trigger_delta_db to %s", self.own_trigger_delta_db)
        if "capture_window_db" in settings:
            self.capture_window_db = max(0.5, float(settings["capture_window_db"]))
            logger.info("Updated capture_window_db to %s", self.capture_window_db)
        if "required_own_events" in settings:
            self.required_own_events = max(1, int(settings["required_own_events"]))
            logger.info("Updated required_own_events to %s", self.required_own_events)
        if "capture_timeout_sec" in settings:
            self.capture_timeout_sec = max(0.5, float(settings["capture_timeout_sec"]))
            logger.info("Updated capture_timeout_sec to %s", self.capture_timeout_sec)
        if "retrigger_delta_db" in settings:
            self.retrigger_delta_db = float(settings["retrigger_delta_db"])
            logger.info("Updated retrigger_delta_db to %s", self.retrigger_delta_db)
        if "max_trigger_peak_dbfs" in settings:
            self.max_trigger_peak_dbfs = float(settings["max_trigger_peak_dbfs"])
            logger.info(
                "Updated max_trigger_peak_dbfs to %s", self.max_trigger_peak_dbfs
            )
        if "bypass_trigger_peak_dbfs" in settings:
            self.bypass_trigger_peak_dbfs = float(settings["bypass_trigger_peak_dbfs"])
            logger.info(
                "Updated bypass_trigger_peak_dbfs to %s", self.bypass_trigger_peak_dbfs
            )
        if "drums_close_max_boost_db" in settings:
            self.drums_close_max_boost_db = float(settings["drums_close_max_boost_db"])
            logger.info(
                "Updated drums_close_max_boost_db to %s", self.drums_close_max_boost_db
            )
        if "exclude_bleed_from_own_capture" in settings:
            self.exclude_bleed_from_own_capture = bool(
                settings["exclude_bleed_from_own_capture"]
            )
            logger.info(
                "Updated exclude_bleed_from_own_capture to %s",
                self.exclude_bleed_from_own_capture,
            )
        if "capture_bleed_guard_ratio" in settings:
            self.capture_bleed_guard_ratio = float(
                settings["capture_bleed_guard_ratio"]
            )
            logger.info(
                "Updated capture_bleed_guard_ratio to %s",
                self.capture_bleed_guard_ratio,
            )
        if "capture_bleed_guard_confidence" in settings:
            self.capture_bleed_guard_confidence = float(
                settings["capture_bleed_guard_confidence"]
            )
            logger.info(
                "Updated capture_bleed_guard_confidence to %s",
                self.capture_bleed_guard_confidence,
            )
        if "max_single_step_cut_db" in settings:
            self.max_single_step_cut_db = float(settings["max_single_step_cut_db"])
            logger.info(
                "Updated max_single_step_cut_db to %s", self.max_single_step_cut_db
            )
        if "trim_apply_limit_db" in settings:
            self.trim_apply_limit_db = float(settings["trim_apply_limit_db"])
            logger.info("Updated trim_apply_limit_db to %s", self.trim_apply_limit_db)
        if "phase_overrides" in settings and isinstance(
            settings["phase_overrides"], dict
        ):
            self.phase_overrides = {
                _normalize_preset_name(preset): values
                for preset, values in settings["phase_overrides"].items()
                if isinstance(values, dict)
            }
            logger.info(
                "Updated phase_overrides for %s presets", len(self.phase_overrides)
            )
