"""
Auto Fader Module - Automatic Fader Control System

Реализует автоматическое управление фейдерами выходного уровня каналов
с двумя режимами работы:

1. Real-Time Fader (Динамический режим):
   - Непрерывный мониторинг и автоматическая регулировка фейдеров
   - Анализ LUFS, спектрального центра, энергии каналов
   - Нечеткая логика (Fuzzy Logic) для плавных коррекций
   - Поддержка жанровых профилей

2. Auto Fader (Статический режим):
   - Сбор статистики за период (10-30 секунд)
   - Одноразовая установка начального баланса
   - Вокал как референс для баланса

Основано на документе "Архитектура автоматического микширования"
"""

import numpy as np
import threading
import logging
import time
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import deque
from scipy import signal
from enum import Enum
from dataclasses import dataclass, field

# Import LUFS components from existing module
from lufs_gain_staging import LUFSMeter, TruePeakMeter, KWeightingFilter, AGCEnvelope

logger = logging.getLogger(__name__)


class BalanceMode(Enum):
    """Режимы работы Auto Fader"""

    REALTIME = "realtime"  # Динамический режим
    STATIC = "static"  # Статический режим (Auto Balance)


class SongSection(Enum):
    """Секция песни для адаптации offsets (verse = вокал ниже, chorus = вокал выше)."""

    UNKNOWN = "unknown"
    VERSE = "verse"
    CHORUS = "chorus"


def detect_song_section(
    channel_lufs: Dict[int, float],
    vocal_channel_ids: List[int],
    threshold_db: float = -24.0,
) -> SongSection:
    """
    Простой детектор секции по энергии вокала (заглушка: по среднему LUFS вокала).
    Chorus обычно громче verse. В будущем: спектральная энергия + RMS по окну.
    """
    if not vocal_channel_ids or not channel_lufs:
        return SongSection.UNKNOWN
    vocal_lufs = [
        channel_lufs.get(ch, -100.0) for ch in vocal_channel_ids if ch in channel_lufs
    ]
    if not vocal_lufs:
        return SongSection.UNKNOWN
    avg = sum(vocal_lufs) / len(vocal_lufs)
    return SongSection.CHORUS if avg > threshold_db else SongSection.VERSE


class GenreProfile(Enum):
    """Жанровые профили баланса"""

    CUSTOM = "custom"
    POP_ROCK = "pop_rock"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    ACOUSTIC = "acoustic"
    CLASSICAL = "classical"


@dataclass
class BalanceProfile:
    """
    Жанровый профиль баланса микса.

    Определяет характеристики микширования для конкретного жанра:
    - Динамический диапазон
    - Приоритет вокала
    - Степень компрессии
    - Целевые уровни для групп инструментов
    """

    name: str
    genre: GenreProfile
    dynamic_range_db: Tuple[float, float]  # (min, max) dB
    vocal_priority: float  # 0.0-1.0, насколько вокал приоритетен
    compression_amount: float  # 0.0-1.0, степень компрессии

    # Абсолютные целевые уровни LUFS для каждого инструмента
    instrument_target_lufs: Dict[str, float] = field(default_factory=dict)

    # Параметры AGC
    attack_ms: float = 100.0
    release_ms: float = 1000.0
    hold_ms: float = 500.0
    ratio: float = 2.0

    @classmethod
    def get_preset(cls, genre: GenreProfile) -> "BalanceProfile":
        """Получить предустановленный профиль по жанру"""
        presets = {
            GenreProfile.POP_ROCK: cls(
                name="Pop/Rock",
                genre=GenreProfile.POP_ROCK,
                dynamic_range_db=(8.0, 12.0),
                vocal_priority=0.9,
                compression_amount=0.7,
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "piano": -25.0,
                    "pads": -26.0,
                    "fx": -26.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "strings": -25.0,
                    "percussion": -26.0,
                    "brass": -25.0,
                    "sax": -25.0,
                    "woodwinds": -25.0,
                },
                attack_ms=50.0,
                release_ms=500.0,
                hold_ms=200.0,
                ratio=3.0,
            ),
            GenreProfile.JAZZ: cls(
                name="Jazz",
                genre=GenreProfile.JAZZ,
                dynamic_range_db=(15.0, 20.0),
                vocal_priority=0.6,
                compression_amount=0.3,
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "piano": -25.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "brass": -25.0,
                    "sax": -25.0,
                    "woodwinds": -25.0,
                    "pads": -26.0,
                    "fx": -26.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "strings": -25.0,
                    "percussion": -26.0,
                },
                attack_ms=150.0,
                release_ms=2000.0,
                hold_ms=800.0,
                ratio=1.5,
            ),
            GenreProfile.ELECTRONIC: cls(
                name="Electronic",
                genre=GenreProfile.ELECTRONIC,
                dynamic_range_db=(10.0, 14.0),
                vocal_priority=0.7,
                compression_amount=0.6,
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "pads": -26.0,
                    "fx": -26.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "piano": -25.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "strings": -25.0,
                    "percussion": -26.0,
                    "brass": -25.0,
                    "sax": -25.0,
                    "woodwinds": -25.0,
                },
                attack_ms=30.0,
                release_ms=400.0,
                hold_ms=150.0,
                ratio=4.0,
            ),
            GenreProfile.ACOUSTIC: cls(
                name="Acoustic",
                genre=GenreProfile.ACOUSTIC,
                dynamic_range_db=(12.0, 18.0),
                vocal_priority=0.8,
                compression_amount=0.4,
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "piano": -25.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "strings": -25.0,
                    "percussion": -26.0,
                    "brass": -25.0,
                    "sax": -25.0,
                    "woodwinds": -25.0,
                    "pads": -26.0,
                    "fx": -26.0,
                },
                attack_ms=100.0,
                release_ms=1500.0,
                hold_ms=600.0,
                ratio=2.0,
            ),
            GenreProfile.CLASSICAL: cls(
                name="Classical",
                genre=GenreProfile.CLASSICAL,
                dynamic_range_db=(18.0, 25.0),
                vocal_priority=0.5,
                compression_amount=0.2,
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "piano": -25.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "strings": -25.0,
                    "woodwinds": -25.0,
                    "brass": -25.0,
                    "percussion": -26.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "sax": -25.0,
                    "pads": -26.0,
                    "fx": -26.0,
                },
                attack_ms=200.0,
                release_ms=3000.0,
                hold_ms=1000.0,
                ratio=1.2,
            ),
            GenreProfile.CUSTOM: cls(
                name="Custom",
                genre=GenreProfile.CUSTOM,
                dynamic_range_db=(10.0, 15.0),
                vocal_priority=0.7,
                compression_amount=0.5,
                # Абсолютные target LUFS для каждого инструмента
                instrument_target_lufs={
                    "leadVocal": -22.0,
                    "lead_vocal": -22.0,
                    "drums": -22.0,
                    "tom": -22.0,
                    "toms": -22.0,
                    "synth": -23.0,
                    "kick": -25.0,
                    "snare": -25.0,
                    "bass": -25.0,
                    "guitar": -25.0,
                    "electricGuitar": -25.0,
                    "electric_guitar": -25.0,
                    "backVocal": -25.0,
                    "backing_vocal": -25.0,
                    "back_vocal": -25.0,
                    "playback": -25.0,
                    "accordion": -25.0,
                    "keys": -26.0,
                    "piano": -25.0,
                    "synth": -23.0,
                    "pads": -26.0,
                    "fx": -26.0,
                    "hihat": -35.0,
                    "ride": -35.0,
                    "cymbals": -35.0,
                    "overheads": -35.0,
                    "overhead": -35.0,
                    "room": -40.0,
                    "acousticGuitar": -25.0,
                    "acoustic_guitar": -25.0,
                    "strings": -25.0,
                    "percussion": -26.0,
                    "brass": -25.0,
                    "sax": -25.0,
                    "woodwinds": -25.0,
                },
                attack_ms=100.0,
                release_ms=1000.0,
                hold_ms=500.0,
                ratio=2.0,
            ),
        }
        return presets.get(genre, presets[GenreProfile.CUSTOM])


class SpectralAnalyzer:
    """
    Анализатор спектра для определения характеристик сигнала.

    Вычисляет:
    - Спектральный центр (центр тяжести спектра)
    - Спектральный флэтнесс (тональность vs шум)
    - Энергия в частотных диапазонах
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = fft_size // 2

        # Частотные бины
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

        # Окно Ханна для FFT
        self.window = np.hanning(fft_size)

        # Диапазоны частот для анализа
        self.freq_bands = {
            "sub": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "high": (4000, 8000),
            "air": (8000, 20000),
        }

    def analyze(self, samples: np.ndarray) -> Dict[str, float]:
        """
        Анализ спектральных характеристик сигнала.

        Args:
            samples: Аудио сэмплы (float32)

        Returns:
            Словарь с метриками
        """
        if len(samples) < self.fft_size:
            # Дополняем нулями если недостаточно сэмплов
            samples = np.pad(samples, (0, self.fft_size - len(samples)))

        # Берем последний блок размера fft_size
        block = samples[-self.fft_size :] * self.window

        # FFT
        spectrum = np.abs(np.fft.rfft(block))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)

        # Спектральный центр (центроид)
        spectral_centroid = self._compute_centroid(spectrum)

        # Спектральный флэтнесс
        spectral_flatness = self._compute_flatness(spectrum)

        # Энергия по диапазонам
        band_energy = self._compute_band_energy(spectrum)

        # Спектральный наклон (rolloff)
        spectral_rolloff = self._compute_rolloff(spectrum)

        return {
            "centroid": spectral_centroid,
            "flatness": spectral_flatness,
            "rolloff": spectral_rolloff,
            "band_energy": band_energy,
            "total_energy": np.sum(spectrum**2),
        }

    def _compute_centroid(self, spectrum: np.ndarray) -> float:
        """Вычисление спектрального центроида (Hz)"""
        magnitude_sum = np.sum(spectrum)
        if magnitude_sum < 1e-10:
            return 0.0
        return float(np.sum(self.freqs * spectrum) / magnitude_sum)

    def _compute_flatness(self, spectrum: np.ndarray) -> float:
        """
        Вычисление спектрального флэтнесса.
        Близко к 1 = шум, близко к 0 = тональный сигнал
        """
        spectrum = spectrum + 1e-10  # Избегаем log(0)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        if arithmetic_mean < 1e-10:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    def _compute_rolloff(self, spectrum: np.ndarray, percentile: float = 0.85) -> float:
        """Вычисление спектрального rolloff (Hz)"""
        cumsum = np.cumsum(spectrum)
        total = cumsum[-1]
        if total < 1e-10:
            return 0.0
        threshold = percentile * total
        rolloff_idx = np.searchsorted(cumsum, threshold)
        return float(self.freqs[min(rolloff_idx, len(self.freqs) - 1)])

    def _compute_band_energy(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Вычисление энергии в частотных диапазонах"""
        band_energy = {}
        for band_name, (low, high) in self.freq_bands.items():
            mask = (self.freqs >= low) & (self.freqs < high)
            band_spectrum = spectrum[mask]
            if len(band_spectrum) > 0:
                energy = np.sum(band_spectrum**2)
                band_energy[band_name] = float(20 * np.log10(energy + 1e-10))
            else:
                band_energy[band_name] = -100.0
        return band_energy


class IntegratedLUFSMeter:
    """
    Измеритель Integrated LUFS по стандарту ITU-R BS.1770-4 с гейтингом.

    В отличие от Short-term LUFS (400ms окно), Integrated LUFS вычисляется
    за весь период измерения с применением двойного гейтинга:
    1. Абсолютный гейт: -70 LUFS (отсекаем тишину)
    2. Относительный гейт: -10 LU ниже среднего (отсекаем тихие блоки)

    Это даёт точную среднюю громкость сигнала, игнорируя паузы и тишину.
    Используется в алгоритме Auto Balance для определения уровня каждого канала.

    Поддерживает два режима:
    - Full integrated: накапливает все блоки (для Auto Balance)
    - Sliding window: скользящее окно ~3 секунды (для real-time)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        block_size_ms: float = 400.0,
        overlap_ms: float = 300.0,
        window_seconds: Optional[float] = None,
    ):
        """
        Args:
            sample_rate: Частота дискретизации
            block_size_ms: Размер блока в миллисекундах (400ms по стандарту)
            overlap_ms: Перекрытие блоков в миллисекундах
            window_seconds: Если задано, использовать скользящее окно указанной длительности.
                           None = накапливать все блоки (full integrated)
        """
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_size_ms / 1000)
        self.hop_size = int(sample_rate * (block_size_ms - overlap_ms) / 1000)
        self.window_seconds = window_seconds

        # K-weighting фильтр (отдельный экземпляр для integrated измерений)
        self.k_filter = KWeightingFilter(sample_rate)

        # Буфер для накопления сэмплов между вызовами process()
        self.buffer = np.array([], dtype=np.float32)

        # Список средних квадратов для каждого 400ms блока
        # Если window_seconds задан, используем deque с ограничением
        if window_seconds is not None:
            # Количество блоков для окна: window_seconds / (block_size_ms / 1000)
            # При block_size_ms=400ms и overlap_ms=300ms, hop_size = 100ms
            # Для 3 секунд: 3.0 / 0.1 = 30 блоков
            blocks_per_window = int(
                window_seconds * 1000 / (block_size_ms - overlap_ms)
            )
            self.block_loudnesses = deque(maxlen=blocks_per_window)
        else:
            # Full integrated: список без ограничения
            self.block_loudnesses: List[float] = []

    def process(self, samples: np.ndarray):
        """
        Накапливает аудио сэмплы и разбивает на перекрывающиеся блоки.

        Args:
            samples: Входные аудио сэмплы (float32)
        """
        # Применяем K-weighting фильтрацию
        filtered = self.k_filter.process(samples)

        # Добавляем в буфер
        self.buffer = np.concatenate([self.buffer, filtered])

        # Извлекаем полные блоки с перекрытием
        while len(self.buffer) >= self.block_size:
            block = self.buffer[: self.block_size]
            mean_sq = float(np.mean(block**2))
            # deque автоматически ограничивает размер при window_seconds задан
            self.block_loudnesses.append(mean_sq)
            self.buffer = self.buffer[self.hop_size :]

    def get_integrated_lufs(self) -> float:
        """
        Вычисление Integrated LUFS с двойным гейтингом по ITU-R BS.1770-4.

        Returns:
            Integrated LUFS в dB. Возвращает -70.0 если нет данных.
        """
        if not self.block_loudnesses:
            return -70.0

        # Преобразуем deque в список для обработки
        blocks = (
            list(self.block_loudnesses)
            if isinstance(self.block_loudnesses, deque)
            else self.block_loudnesses
        )

        if not blocks:
            return -70.0

        # Шаг 1: Абсолютный гейт при -70 LUFS
        # Переводим -70 LUFS в линейное значение mean_square
        abs_gate_threshold = 10 ** ((-70 + 0.691) / 10)
        above_abs = [ms for ms in blocks if ms > abs_gate_threshold]

        if not above_abs:
            return -70.0

        # Шаг 2: Относительный гейт при -10 LU ниже среднего
        ungated_mean = float(np.mean(above_abs))
        rel_gate_threshold = ungated_mean * 10 ** (-10 / 10)  # -10 LU = -10 dB
        above_rel = [ms for ms in above_abs if ms > rel_gate_threshold]

        if not above_rel:
            return -70.0

        # Шаг 3: Вычисляем Integrated LUFS из оставшихся блоков
        gated_mean = float(np.mean(above_rel))
        integrated_lufs = -0.691 + 10 * np.log10(max(gated_mean, 1e-10))

        return float(integrated_lufs)

    def get_block_count(self) -> int:
        """Количество накопленных блоков"""
        return (
            len(self.block_loudnesses)
            if isinstance(self.block_loudnesses, deque)
            else len(self.block_loudnesses)
        )

    def get_current_short_term_lufs(self) -> float:
        """
        Получить текущий Short-term LUFS (последний блок).
        Полезно для отображения в реальном времени во время сбора.
        """
        if not self.block_loudnesses:
            return -70.0
        last_ms = self.block_loudnesses[-1]
        if last_ms < 1e-10:
            return -70.0
        return float(-0.691 + 10 * np.log10(last_ms))

    def get_window_info(self) -> Dict[str, Any]:
        """Получить информацию об окне измерения"""
        blocks = (
            list(self.block_loudnesses)
            if isinstance(self.block_loudnesses, deque)
            else self.block_loudnesses
        )
        block_duration_ms = (self.block_size / self.sample_rate) * 1000
        hop_duration_ms = (self.hop_size / self.sample_rate) * 1000

        if self.window_seconds:
            actual_window_sec = len(blocks) * hop_duration_ms / 1000
            return {
                "mode": "sliding_window",
                "window_seconds": self.window_seconds,
                "actual_window_seconds": actual_window_sec,
                "block_count": len(blocks),
                "max_blocks": self.block_loudnesses.maxlen
                if isinstance(self.block_loudnesses, deque)
                else None,
            }
        else:
            total_duration_sec = len(blocks) * hop_duration_ms / 1000
            return {
                "mode": "full_integrated",
                "total_duration_seconds": total_duration_sec,
                "block_count": len(blocks),
            }

    def reset(self):
        """Сброс всех накопленных данных"""
        self.k_filter.reset()
        self.buffer = np.array([], dtype=np.float32)
        self.block_loudnesses.clear()


class FuzzyFaderController:
    """
    Нечеткий контроллер для плавного управления фейдерами.

    Использует нечеткую логику для принятия решений о коррекции уровня,
    что обеспечивает более естественные и плавные переходы.
    """

    def __init__(self):
        self._init_fuzzy_sets()

    def _init_fuzzy_sets(self):
        """Инициализация нечетких множеств (error ±10 dB для live, hard limit на выходе)."""
        # Вход: отклонение от целевого уровня (dB), сужено до ±10 для live
        self.error_range = np.arange(-10, 11, 1)

        # Нечеткие множества для ошибки
        self.error_nb = self._trimf(self.error_range, [-10, -10, -4])  # Negative Big
        self.error_ns = self._trimf(self.error_range, [-5, -2, 0])  # Negative Small
        self.error_z = self._trimf(self.error_range, [-2, 0, 2])  # Zero
        self.error_ps = self._trimf(self.error_range, [0, 2, 5])  # Positive Small
        self.error_pb = self._trimf(self.error_range, [4, 10, 10])  # Positive Big

        # Выход: коррекция фейдера (dB), hard limit ±10
        self.output_range = np.arange(-10, 11, 0.5)
        self.output_hard_limit_db = 10.0

        # Нечеткие множества для выхода
        self.out_nb = self._trimf(self.output_range, [-10, -10, -3])
        self.out_ns = self._trimf(self.output_range, [-5, -2, 0])
        self.out_z = self._trimf(self.output_range, [-1, 0, 1])
        self.out_ps = self._trimf(self.output_range, [0, 2, 5])
        self.out_pb = self._trimf(self.output_range, [3, 10, 10])

    def _trimf(self, x: np.ndarray, params: List[float]) -> np.ndarray:
        """Треугольная функция принадлежности"""
        a, b, c = params
        y = np.zeros_like(x, dtype=float)

        # Левый склон
        mask = (x >= a) & (x <= b)
        if b != a:
            y[mask] = (x[mask] - a) / (b - a)

        # Правый склон
        mask = (x > b) & (x <= c)
        if c != b:
            y[mask] = (c - x[mask]) / (c - b)

        return y

    def _interp_membership(
        self, x_val: float, x_range: np.ndarray, membership: np.ndarray
    ) -> float:
        """Интерполяция значения функции принадлежности"""
        return float(np.interp(x_val, x_range, membership))

    def compute_correction(self, error_db: float, rate_of_change: float = 0.0) -> float:
        """
        Вычисление коррекции фейдера на основе нечеткой логики.

        Args:
            error_db: Отклонение от целевого уровня в dB
            rate_of_change: Скорость изменения ошибки (dB/s)

        Returns:
            Рекомендуемая коррекция в dB
        """
        # Ограничиваем входное значение (±10 dB для live)
        error_db = np.clip(error_db, -10, 10)

        # Фаззификация - вычисляем степени принадлежности
        mu_nb = self._interp_membership(error_db, self.error_range, self.error_nb)
        mu_ns = self._interp_membership(error_db, self.error_range, self.error_ns)
        mu_z = self._interp_membership(error_db, self.error_range, self.error_z)
        mu_ps = self._interp_membership(error_db, self.error_range, self.error_ps)
        mu_pb = self._interp_membership(error_db, self.error_range, self.error_pb)

        # Правила (error = target - current):
        # Положительная ошибка = сигнал тише целевого → нужно УВЕЛИЧИТЬ фейдер
        # Отрицательная ошибка = сигнал громче целевого → нужно УМЕНЬШИТЬ фейдер
        #
        # Если ошибка PB (сигнал слишком тихий) -> выход PB (сильно увеличить)
        # Если ошибка PS (сигнал немного тихий) -> выход PS (немного увеличить)
        # Если ошибка Z (норма) -> выход Z (не менять)
        # Если ошибка NS (сигнал немного громкий) -> выход NS (немного уменьшить)
        # Если ошибка NB (сигнал слишком громкий) -> выход NB (сильно уменьшить)

        # Активация выходных множеств (прямая связь: знак ошибки = знак коррекции)
        out_activation_nb = np.minimum(
            mu_nb, self.out_nb
        )  # NB error → NB output (уменьшить)
        out_activation_ns = np.minimum(
            mu_ns, self.out_ns
        )  # NS error → NS output (уменьшить)
        out_activation_z = np.minimum(
            mu_z, self.out_z
        )  # Z error → Z output (не менять)
        out_activation_ps = np.minimum(
            mu_ps, self.out_ps
        )  # PS error → PS output (увеличить)
        out_activation_pb = np.minimum(
            mu_pb, self.out_pb
        )  # PB error → PB output (увеличить)

        # Агрегация
        aggregated = np.maximum.reduce(
            [
                out_activation_nb,
                out_activation_ns,
                out_activation_z,
                out_activation_ps,
                out_activation_pb,
            ]
        )

        # Дефаззификация (центр тяжести)
        if np.sum(aggregated) < 1e-10:
            return 0.0

        correction = float(np.sum(self.output_range * aggregated) / np.sum(aggregated))
        # Hard limiter на выходе
        correction = max(
            -self.output_hard_limit_db, min(self.output_hard_limit_db, correction)
        )
        return correction


class ChannelFaderState:
    """Состояние фейдера одного канала"""

    def __init__(
        self,
        channel_id: int,
        mixer_channel: int,
        sample_rate: int = 48000,
        instrument_type: str = "custom",
        lufs_window_sec: Optional[float] = None,
    ):
        """
        Args:
            channel_id: Audio channel ID
            mixer_channel: Mixer channel number
            sample_rate: Audio sample rate
            instrument_type: Instrument type name
            lufs_window_sec: Integrated LUFS window in seconds. None = full integrated (for Auto Balance),
                            3.0 = sliding window (for real-time). Default None.
        """
        self.channel_id = channel_id
        self.mixer_channel = mixer_channel
        self.instrument_type = instrument_type
        self.sample_rate = sample_rate

        # Анализаторы
        self.lufs_meter = LUFSMeter(sample_rate)
        self.true_peak_meter = TruePeakMeter(sample_rate)
        self.spectral_analyzer = SpectralAnalyzer(sample_rate)

        # Текущие метрики
        self.current_lufs = -60.0
        self.current_true_peak = -60.0
        self.current_fader = 0.0  # В dB (0 = unity)
        self.target_fader = 0.0
        self.spectral_centroid = 0.0
        self.band_energy: Dict[str, float] = {}
        self.band_energy_max: Dict[
            str, float
        ] = {}  # Peak band energy during collection (for bleed detection)

        # Статистика для Auto Balance
        self.lufs_history: deque = deque(maxlen=300)  # 30 сек при 100ms updates
        self.peak_history: deque = deque(maxlen=300)
        self.avg_lufs = -60.0
        self.avg_peak = -60.0

        # Integrated LUFS: используем окно 3 сек для real-time, full integrated для Auto Balance
        self.integrated_lufs_meter = IntegratedLUFSMeter(
            sample_rate, window_seconds=lufs_window_sec
        )
        self.integrated_lufs = -70.0
        self.locked = False  # Блокировка канала (не менять при повторных проходах)

        # Статус
        self.is_active = False
        self.status = "idle"

        # Время последней коррекции (для hold периода)
        self.last_adjustment_time = 0.0

        self.pre_fader_lufs = -60.0  # LUFS с компенсацией фейдера (pre-fader уровень)

        # Real-Time Fader Riding: Anchor position and running average
        self.initial_fader_db = 0.0  # Anchor position read from mixer at start
        self.lufs_ring_buffer: deque = deque(
            maxlen=50
        )  # Ring buffer for running average (will be resized based on avg_window_sec)
        self.avg_lufs = -60.0  # Running average LUFS (computed from ring buffer)

        # Calibration phase: baseline values and calibration state
        self.calibration_complete = False  # Фаза калибровки завершена
        self.calibration_samples: deque = (
            deque()
        )  # Буфер для накопления значений во время калибровки
        self.baseline_lufs = -60.0  # Базовое значение LUFS после калибровки
        self.baseline_peak = -60.0  # Базовое значение Peak после калибровки
        self.calibration_start_time = 0.0  # Время начала калибровки

    def process(self, samples: np.ndarray) -> Dict[str, Any]:
        """Обработка аудио сэмплов"""
        # LUFS (измеряется post-fader из USB-интерфейса микшера)
        measured_lufs = self.lufs_meter.process(samples)

        # КОМПЕНСАЦИЯ ФЕЙДЕРА: вычисляем pre-fader уровень
        # Проблема: USB-интерфейс микшера отправляет post-fader сигнал, что создает обратную связь:
        # когда мы корректируем фейдер, это влияет на измеряемый уровень.
        # Решение: компенсируем влияние фейдера, добавляя его значение к измеренному LUFS.
        # Если фейдер уменьшает уровень на X dB, то измеренный LUFS на X dB меньше реального pre-fader уровня.
        # Формула: pre_fader_lufs = measured_lufs + current_fader_db
        # Пример: если measured_lufs = -30 LUFS, а current_fader = -6 dB, то pre_fader_lufs = -24 LUFS
        self.pre_fader_lufs = measured_lufs + self.current_fader
        # Используем компенсированный уровень для дальнейших расчетов (это реальный уровень входного сигнала)
        self.current_lufs = self.pre_fader_lufs

        # True Peak
        self.current_true_peak = self.true_peak_meter.process(samples)

        # Спектральный анализ
        spectral = self.spectral_analyzer.analyze(samples)
        self.spectral_centroid = spectral["centroid"]
        self.band_energy = spectral["band_energy"]
        # Накопление peak band energy для bleed detection (используем максимум за период)
        for band, val in self.band_energy.items():
            self.band_energy_max[band] = max(self.band_energy_max.get(band, -100), val)

        # Определяем активность (используем компенсированный уровень)
        # Bleed detection теперь выполняется централизованно через bleed_service
        self.is_active = self.current_lufs > -50.0

        # Добавляем в историю
        if self.is_active:
            self.lufs_history.append(self.current_lufs)
            self.peak_history.append(self.current_true_peak)

            # Обновляем средние
            if len(self.lufs_history) > 0:
                self.avg_lufs = float(np.mean(list(self.lufs_history)))
                self.avg_peak = float(np.max(list(self.peak_history)))

        # Integrated LUFS: накапливаем все сэмплы (гейтинг применяется при вычислении)
        self.integrated_lufs_meter.process(samples)
        self.integrated_lufs = self.integrated_lufs_meter.get_integrated_lufs()

        return {
            "channel": self.channel_id,
            "mixer_channel": self.mixer_channel,
            "lufs": self.current_lufs,  # Pre-fader LUFS (с компенсацией)
            "pre_fader_lufs": self.pre_fader_lufs,  # Явно указываем pre-fader уровень
            "integrated_lufs": self.integrated_lufs,  # Integrated LUFS за весь период
            "true_peak": self.current_true_peak,
            "spectral_centroid": self.spectral_centroid,
            "current_fader": self.current_fader,
            "target_fader": self.target_fader,
            "is_active": self.is_active,
            "locked": self.locked,
            "status": self.status,
        }

    def reset_statistics(self):
        """Сброс накопленной статистики"""
        self.lufs_history.clear()
        self.peak_history.clear()
        self.avg_lufs = -60.0
        self.avg_peak = -60.0
        self.integrated_lufs_meter.reset()
        self.integrated_lufs = -70.0
        self.band_energy_max.clear()

    def reset(self):
        """Полный сброс состояния"""
        self.lufs_meter.reset()
        self.true_peak_meter.reset()
        self.reset_statistics()
        self.current_lufs = -60.0
        self.pre_fader_lufs = -60.0
        self.integrated_lufs = -70.0
        self.current_true_peak = -60.0
        self.current_fader = 0.0
        self.target_fader = 0.0
        self.is_active = False
        self.locked = False
        self.status = "idle"

        # Сброс калибровки
        self.calibration_complete = False
        self.calibration_samples.clear()
        self.baseline_lufs = -60.0
        self.baseline_peak = -60.0
        self.calibration_start_time = 0.0


class AutoFaderController:
    """
    Главный контроллер автоматического управления фейдерами.

    Поддерживает два режима:
    1. Real-Time Fader - непрерывная автоматическая балансировка
    2. Auto Fader (Static) - одноразовая установка баланса
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
        self.config = config or {}
        self.bleed_service = bleed_service

        # Загрузка конфигурации
        auto_fader_config = self.config.get("automation", {}).get("auto_fader", {})

        # Real-Time Fader Riding Parameters (new approach)
        self.fader_range_db = auto_fader_config.get(
            "fader_range_db", 3.0
        )  # Max deviation from initial position
        self.avg_window_sec = auto_fader_config.get(
            "avg_window_sec", 5.0
        )  # Window for running average LUFS
        self.sensitivity = auto_fader_config.get(
            "sensitivity", 0.5
        )  # Correction sensitivity (0.0-1.0)

        # Integrated LUFS window (3 seconds for real-time per document)
        self.lufs_window_sec = auto_fader_config.get("lufs_window_sec", 3.0)

        # Envelope parameters for smooth fader movement
        self.attack_ms = auto_fader_config.get("attack_ms", 100.0)
        self.release_ms = auto_fader_config.get("release_ms", 1000.0)
        self.gate_threshold = auto_fader_config.get("gate_threshold", -50.0)
        self.bleed_threshold = auto_fader_config.get(
            "bleed_threshold", -50.0
        )  # Порог блидинга для фильтрации неактивных каналов
        self.calibration_duration_sec = auto_fader_config.get(
            "calibration_duration_sec", 5.0
        )  # Длительность фазы калибровки
        self.update_interval = auto_fader_config.get("update_interval_ms", 100) / 1000.0

        # Safety: max fader ceiling from config (0 dBFS when enable_limits, else 10 dB)
        safety = self.config.get("safety", {})
        if safety.get("enable_limits", False):
            self._max_fader_db = safety.get("max_fader", 0)
        else:
            self._max_fader_db = 10.0  # Hardware max when safety disabled

        # Legacy parameters (kept for Auto Balance mode, not used in real-time mode)
        self.target_lufs = auto_fader_config.get(
            "target_lufs", -22.0
        )  # Only for Auto Balance
        self.max_adjustment_db = auto_fader_config.get(
            "max_adjustment_db", 6.0
        )  # Only for Auto Balance
        self.min_adjustment_db = auto_fader_config.get(
            "min_adjustment_db", -12.0
        )  # Only for Auto Balance
        self.ratio = auto_fader_config.get("ratio", 2.0)  # Only for Auto Balance
        self.hold_ms = auto_fader_config.get("hold_ms", 800.0)  # Only for Auto Balance

        # Профиль баланса
        profile_name = auto_fader_config.get("profile", "custom")
        try:
            self.profile = BalanceProfile.get_preset(GenreProfile(profile_name))
        except Exception:
            self.profile = BalanceProfile.get_preset(GenreProfile.CUSTOM)

        # Режим работы
        self.mode = BalanceMode.REALTIME

        # Каналы
        self.channels: Dict[int, ChannelFaderState] = {}
        self.channel_mapping: Dict[int, int] = {}  # audio_ch -> mixer_ch
        self.channel_settings: Dict[int, Dict] = {}

        # Нечеткий контроллер
        self.fuzzy_controller = FuzzyFaderController()

        # Bleed detection and compensation now handled by centralized bleed_service

        # Envelope для плавности
        self.envelopes: Dict[int, AGCEnvelope] = {}

        # PyAudio
        self.pa = None
        self.stream = None
        self.device_index = None
        self._num_channels = 0

        # Threading
        self.is_active = False
        self.realtime_enabled = False
        self.auto_balance_collecting = False
        self.auto_balance_duration = 15.0  # секунд
        self.bleed_threshold = -50.0  # порог блидинга в LUFS
        self.auto_balance_start_time = 0.0
        self._stop_event = threading.Event()
        # Freeze / manual override
        self.automation_frozen = False
        self.channel_freeze_until: Dict[
            int, float
        ] = {}  # mixer_ch -> unix time until resume
        self.freeze_cooldown_seconds = 10.0
        self._control_thread = None

        # Callbacks
        self.on_status_update: Optional[Callable[[Dict], None]] = None
        self.on_levels_updated: Optional[Callable[[Dict], None]] = None

        # Буферы аудио
        self._audio_buffers: Dict[int, deque] = {}

        # Результаты Auto Balance
        # Формат: {audio_ch: {'correction': float, 'integrated_lufs': float,
        #           'target_lufs': float, 'locked': bool}}
        self.auto_balance_result: Dict[int, Any] = {}

        self.auto_balance_pass = (
            0  # Номер текущего прохода (0 = не было, 1 = первый, 2 = второй)
        )

        logger.info(
            f"AutoFaderController initialized: fader_range={self.fader_range_db} dB, "
            f"avg_window={self.avg_window_sec}s, sensitivity={self.sensitivity}, "
            f"profile={self.profile.name}"
        )

    def set_automation_frozen(self, frozen: bool):
        """Freeze or unfreeze all automation (no fader commands when frozen)."""
        self.automation_frozen = frozen
        logger.info(f"Auto fader automation frozen: {frozen}")

    def set_channel_frozen(self, mixer_channel: int, seconds: float):
        """Exclude channel from auto-control for the given seconds."""
        self.channel_freeze_until[mixer_channel] = time.time() + max(0.0, seconds)
        logger.info(f"Channel {mixer_channel} frozen for {seconds:.1f}s")

    def get_freeze_status(self) -> Dict[str, Any]:
        """Return current freeze state for UI."""
        now = time.time()
        frozen_channels = [
            ch for ch, until in self.channel_freeze_until.items() if until > now
        ]
        return {
            "automation_frozen": self.automation_frozen,
            "frozen_channels": frozen_channels,
            "freeze_cooldown_seconds": self.freeze_cooldown_seconds,
        }

    def start(
        self,
        device_id: int,
        channels: List[int],
        channel_settings: Dict[int, Dict],
        channel_mapping: Dict[int, int],
        on_status_callback: Callable = None,
    ) -> bool:
        """
        Запуск захвата аудио.

        Args:
            device_id: Индекс аудио устройства
            channels: Список номеров каналов для анализа
            channel_settings: Настройки каналов (instrument_type и т.д.)
            channel_mapping: Маппинг audio_ch -> mixer_ch
            on_status_callback: Callback для обновлений статуса
        """
        if self.is_active:
            logger.warning("Controller already active")
            return False

        self.device_index = device_id
        self.channel_settings = channel_settings
        self.channel_mapping = channel_mapping
        self.on_status_update = on_status_callback

        def _init_channels(channels, channel_mapping, channel_settings):
            self.channels.clear()
            self._audio_buffers.clear()
            self.envelopes.clear()
            update_interval_ms = self.update_interval * 1000
            for audio_ch in channels:
                mixer_ch = channel_mapping.get(audio_ch, audio_ch)
                settings = channel_settings.get(audio_ch, {})
                instrument_type = settings.get("instrument_type", "custom")
                self.channels[audio_ch] = ChannelFaderState(
                    channel_id=audio_ch,
                    mixer_channel=mixer_ch,
                    sample_rate=self.sample_rate,
                    instrument_type=instrument_type,
                    lufs_window_sec=self.lufs_window_sec
                    if self.mode == BalanceMode.REALTIME
                    else None,
                )
                self._audio_buffers[audio_ch] = deque(maxlen=10)
                self.envelopes[audio_ch] = AGCEnvelope(
                    self.sample_rate,
                    self.attack_ms,
                    self.release_ms,
                    self.hold_ms,
                    update_interval_ms,
                )

        # Use unified AudioCapture if available
        if self._audio_capture is not None:
            try:
                self.sample_rate = self._audio_capture.sample_rate
                self._num_channels = max(channels) if channels else 2
                _init_channels(channels, channel_mapping, channel_settings)
                self._audio_capture.subscribe("auto_fader", self._audio_capture_poll)
                self.is_active = True
                self._stop_event.clear()
                logger.info(
                    f"Auto Fader started via AudioCapture: {len(channels)} channels"
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

            _init_channels(channels, channel_mapping, channel_settings)

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

            logger.info(f"Auto Fader started: {len(channels)} channels")

            return True

        except Exception as e:
            logger.error(f"Failed to start Auto Fader: {e}", exc_info=True)
            self.stop()
            return False

    def _audio_capture_poll(self):
        """Poll AudioCapture buffers and fill local _audio_buffers."""
        if not self._audio_capture:
            return
        for audio_ch in list(self.channels.keys()):
            if audio_ch <= self._num_channels:
                data = self._audio_capture.get_buffer(audio_ch, self.chunk_size)
                if data is not None and len(data) > 0:
                    self._audio_buffers[audio_ch].append(data.copy())

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback для обработки аудио"""
        import pyaudio

        if not self.is_active:
            return (None, pyaudio.paComplete)

        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            if self._num_channels > 1:
                audio_data = audio_data.reshape(-1, self._num_channels)
            else:
                audio_data = audio_data.reshape(-1, 1)

            for audio_ch in list(self.channels.keys()):
                if audio_ch <= self._num_channels:
                    channel_data = audio_data[:, audio_ch - 1]
                    self._audio_buffers[audio_ch].append(channel_data.copy())

        except Exception as e:
            logger.error(f"Audio callback error: {e}")

        return (None, pyaudio.paContinue)

    def start_realtime_fader(self) -> bool:
        """
        Запуск Real-Time Fader режима (Fader Riding approach).
        Читает начальные позиции фейдеров с микшера и делает небольшие коррекции
        вокруг них на основе среднего LUFS.
        """
        if not self.is_active:
            logger.error("Cannot start realtime fader: controller not active")
            return False

        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.error("Cannot start realtime fader: mixer not connected")
            return False

        if self.realtime_enabled:
            logger.warning("Realtime fader already enabled")
            return True

        self.mode = BalanceMode.REALTIME

        # Читаем текущие позиции фейдеров с микшера и устанавливаем их как якорные
        logger.info("Reading initial fader positions from mixer...")

        # Запрашиваем все фейдеры
        for audio_ch, state in list(self.channels.items()):
            try:
                self.mixer_client.send(f"/ch/{state.mixer_channel}/fdr")
            except Exception as e:
                logger.warning(
                    f"Could not query fader for channel {state.mixer_channel}: {e}"
                )

        # Ждем до 500ms для получения ответов от микшера
        max_wait_time = 0.5
        wait_start = time.time()
        while time.time() - wait_start < max_wait_time:
            time.sleep(0.05)  # Проверяем каждые 50ms

        # Читаем значения фейдеров и устанавливаем якорные позиции
        for audio_ch, state in list(self.channels.items()):
            try:
                current_fader = self.mixer_client.get_channel_fader(state.mixer_channel)
                if current_fader is not None:
                    # Устанавливаем якорную позицию
                    state.initial_fader_db = float(current_fader)
                    state.current_fader = state.initial_fader_db
                    state.target_fader = state.initial_fader_db

                    # Инициализируем ring buffer для среднего LUFS
                    # Размер буфера = avg_window_sec / update_interval
                    buffer_size = max(
                        10, int(self.avg_window_sec / self.update_interval)
                    )
                    state.lufs_ring_buffer = deque(maxlen=buffer_size)
                    state.avg_lufs = -60.0  # Начальное значение

                    # Инициализация калибровки
                    state.calibration_complete = False
                    state.calibration_samples.clear()
                    state.baseline_lufs = -60.0
                    state.baseline_peak = -60.0
                    state.calibration_start_time = time.time()

                    logger.info(
                        f"Channel {state.mixer_channel} (audio {audio_ch}): "
                        f"anchored at {state.initial_fader_db:.1f} dB, "
                        f"range: ±{self.fader_range_db:.1f} dB, "
                        f"avg_window: {self.avg_window_sec:.1f}s, "
                        f"instrument = {state.instrument_type}"
                    )
                else:
                    # Если значение не получено, используем 0 dB (unity) по умолчанию
                    state.initial_fader_db = 0.0
                    state.current_fader = 0.0
                    state.target_fader = 0.0
                    buffer_size = max(
                        10, int(self.avg_window_sec / self.update_interval)
                    )
                    state.lufs_ring_buffer = deque(maxlen=buffer_size)
                    state.avg_lufs = -60.0

                    # Инициализация калибровки
                    state.calibration_complete = False
                    state.calibration_samples.clear()
                    state.baseline_lufs = -60.0
                    state.baseline_peak = -60.0
                    state.calibration_start_time = time.time()

                    logger.warning(
                        f"Channel {state.mixer_channel} (audio {audio_ch}): "
                        f"fader value not received, using 0 dB default as anchor"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not get fader for channel {state.mixer_channel}: {e}"
                )
                state.initial_fader_db = 0.0
                state.current_fader = 0.0
                state.target_fader = 0.0
                buffer_size = max(10, int(self.avg_window_sec / self.update_interval))
                state.lufs_ring_buffer = deque(maxlen=buffer_size)
                state.avg_lufs = -60.0

                # Инициализация калибровки
                state.calibration_complete = False
                state.calibration_samples.clear()
                state.baseline_lufs = -60.0
                state.baseline_peak = -60.0
                state.calibration_start_time = time.time()

        self.realtime_enabled = True
        self._stop_event.clear()

        logger.info(
            f"Real-Time Fader Riding started: "
            f"fader_range=±{self.fader_range_db:.1f} dB, "
            f"avg_window={self.avg_window_sec:.1f}s, "
            f"sensitivity={self.sensitivity:.2f}"
        )

        # Запускаем поток управления в фоновом режиме
        self._control_thread = threading.Thread(
            target=self._realtime_control_loop,
            daemon=True,
            name="AutoFaderRealtimeLoop",
        )
        self._control_thread.start()

        logger.info("Real-time Fader control loop started")

        # Отправляем статус асинхронно
        if self.on_status_update:
            try:
                self.on_status_update(
                    {
                        "type": "realtime_fader_started",
                        "mode": "realtime",
                        "active": True,
                    }
                )
            except Exception as e:
                logger.error(f"Error in on_status_update callback: {e}")

        return True

    def _realtime_control_loop(self):
        """
        Основной цикл Real-Time Fader (Fader Riding approach).

        Алгоритм:
        1. Измеряем short-term LUFS для каждого канала
        2. Вычисляем pre-fader LUFS (с компенсацией текущего фейдера)
        3. Добавляем в ring buffer и вычисляем среднее LUFS
        4. Вычисляем отклонение от среднего: error = avg_lufs - short_term_lufs
        5. Применяем sensitivity: correction = error * sensitivity
        6. Вычисляем целевой фейдер: target = initial_fader + correction
        7. Ограничиваем диапазоном: clip(target, initial ± fader_range_db)
        8. Применяем envelope smoothing
        9. Отправляем в микшер
        """
        logger.info("Realtime control loop started (Fader Riding mode)")

        iteration = 0

        while self.realtime_enabled and not self._stop_event.is_set():
            try:
                levels_update = {}
                fader_updates = {}
                iteration += 1

                # Update bleed detection with current metrics
                if self.bleed_service and self.bleed_service.enabled:
                    all_channel_levels = {}
                    all_channel_centroids = {}
                    all_channel_metrics = {}
                    for ch, st in self.channels.items():
                        all_channel_levels[ch] = st.pre_fader_lufs
                        all_channel_centroids[ch] = st.spectral_centroid
                        # Create adapter for band_energy
                        band_dict = (
                            st.band_energy_max if st.band_energy_max else st.band_energy
                        )
                        if isinstance(band_dict, dict):

                            class _BandMetricsAdapter:
                                def __init__(self, band_dict, rms_level):
                                    self.band_energy_sub = float(
                                        band_dict.get("sub", -100)
                                    )
                                    self.band_energy_bass = float(
                                        band_dict.get("bass", -100)
                                    )
                                    self.band_energy_low_mid = float(
                                        band_dict.get("low_mid", -100)
                                    )
                                    self.band_energy_mid = float(
                                        band_dict.get("mid", -100)
                                    )
                                    self.band_energy_high_mid = float(
                                        band_dict.get("high_mid", -100)
                                    )
                                    self.band_energy_high = float(
                                        band_dict.get("high", -100)
                                    )
                                    self.band_energy_air = float(
                                        band_dict.get("air", -100)
                                    )
                                    self.rms_level = float(rms_level)

                            all_channel_metrics[ch] = _BandMetricsAdapter(
                                band_dict, st.pre_fader_lufs
                            )
                    self.bleed_service.update(
                        all_channel_levels, all_channel_centroids, all_channel_metrics
                    )

                for audio_ch, state in list(self.channels.items()):
                    # Получаем аудио данные
                    buffer_size = (
                        len(self._audio_buffers[audio_ch])
                        if audio_ch in self._audio_buffers
                        else 0
                    )

                    if buffer_size > 0:
                        chunks = list(self._audio_buffers[audio_ch])
                        self._audio_buffers[audio_ch].clear()

                        if chunks:
                            audio_data = np.concatenate(chunks)
                            result = state.process(audio_data)

                            if iteration % 50 == 0 and audio_ch <= 3:
                                logger.info(
                                    f"Ch{audio_ch}: buffer_size={buffer_size}, "
                                    f"audio_samples={len(audio_data)}, "
                                    f"LUFS={state.current_lufs:.1f}, "
                                    f"is_active={state.is_active}"
                                )
                        else:
                            if iteration % 50 == 0 and audio_ch <= 3:
                                logger.warning(
                                    f"Ch{audio_ch}: buffer had {buffer_size} chunks but chunks list is empty"
                                )
                    else:
                        if iteration % 50 == 0 and audio_ch <= 3:
                            logger.warning(
                                f"Ch{audio_ch}: no audio data in buffer (buffer_size=0)"
                            )

                    # Pre-fader LUFS (с компенсацией текущего фейдера)
                    pre_fader_lufs = (
                        state.pre_fader_lufs
                    )  # Уже вычислено в state.process()
                    current_peak = state.current_true_peak

                    # Проверка на bleed threshold (порог блидинга) - фильтрация неактивных каналов
                    if pre_fader_lufs <= self.bleed_threshold:
                        state.status = "inactive_bleed"
                        if iteration % 50 == 0:
                            logger.debug(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): INACTIVE (pre-fader={pre_fader_lufs:.1f} LUFS <= bleed_threshold {self.bleed_threshold:.1f} LUFS)"
                            )
                        levels_update[audio_ch] = {
                            "lufs": float(state.current_lufs),
                            "pre_fader_lufs": float(pre_fader_lufs),
                            "avg_lufs": float(state.avg_lufs),
                            "true_peak": float(current_peak),
                            "current_fader": float(state.current_fader),
                            "target_fader": float(state.current_fader),
                            "initial_fader": float(state.initial_fader_db),
                            "correction": 0.0,
                            "fader_range": float(self.fader_range_db),
                            "is_active": False,
                            "bleed_ratio": 0.0,
                            "bleed_source": None,
                            "status": "inactive_bleed",
                            "instrument_type": str(state.instrument_type),
                            "calibration_complete": bool(state.calibration_complete),
                            "baseline_lufs": float(state.baseline_lufs),
                            "baseline_peak": float(state.baseline_peak),
                        }
                        continue

                    # Пропускаем неактивные каналы (ниже gate threshold)
                    if not state.is_active:
                        state.status = "idle"
                        levels_update[audio_ch] = {
                            "lufs": float(state.current_lufs),
                            "pre_fader_lufs": float(pre_fader_lufs),
                            "avg_lufs": float(state.avg_lufs),
                            "true_peak": float(current_peak),
                            "current_fader": float(state.current_fader),
                            "target_fader": float(state.target_fader),
                            "initial_fader": float(state.initial_fader_db),
                            "correction": 0.0,
                            "fader_range": float(self.fader_range_db),
                            "is_active": False,
                            "bleed_ratio": 0.0,
                            "bleed_source": None,
                            "status": "idle",
                            "instrument_type": str(state.instrument_type),
                            "calibration_complete": bool(state.calibration_complete),
                            "baseline_lufs": float(state.baseline_lufs),
                            "baseline_peak": float(state.baseline_peak),
                        }
                        continue

                    state.status = "active"

                    # Get bleed info from bleed_service
                    bleed_info = None
                    bleed_ratio = 0.0
                    bleed_source = None
                    if self.bleed_service:
                        bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                        if bleed_info:
                            bleed_ratio = float(bleed_info.bleed_ratio)
                            bleed_source = (
                                int(bleed_info.bleed_source_channel)
                                if bleed_info.bleed_source_channel
                                else None
                            )

                    # High bleed ratio - skip correction (bleed compensation handled elsewhere)
                    if bleed_info and bleed_ratio > 0.7:
                        state.status = "high_bleed"
                        if iteration % 50 == 0:
                            logger.warning(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): HIGH BLEED DETECTED "
                                f"(ratio={bleed_ratio:.2f}, source=Ch{bleed_source}) - skipping correction"
                            )
                        levels_update[audio_ch] = {
                            "lufs": float(state.current_lufs),
                            "pre_fader_lufs": float(pre_fader_lufs),
                            "avg_lufs": float(state.avg_lufs),
                            "true_peak": float(current_peak),
                            "current_fader": float(state.current_fader),
                            "target_fader": float(state.current_fader),
                            "initial_fader": float(state.initial_fader_db),
                            "correction": 0.0,
                            "fader_range": float(self.fader_range_db),
                            "is_active": False,
                            "bleed_ratio": bleed_ratio,
                            "bleed_source": bleed_source,
                            "status": "high_bleed",
                            "instrument_type": str(state.instrument_type),
                            "calibration_complete": bool(state.calibration_complete),
                            "baseline_lufs": float(state.baseline_lufs),
                            "baseline_peak": float(state.baseline_peak),
                        }
                        continue

                    # ФАЗА КАЛИБРОВКИ: накопление значений LUFS и Peak
                    elapsed_time = time.time() - state.calibration_start_time
                    if (
                        not state.calibration_complete
                        and elapsed_time < self.calibration_duration_sec
                    ):
                        # Накапливаем значения во время калибровки
                        if (
                            pre_fader_lufs > self.gate_threshold
                        ):  # Только активные значения
                            state.calibration_samples.append(
                                {"lufs": pre_fader_lufs, "peak": current_peak}
                            )

                        state.status = "calibrating"
                        calibration_progress = (
                            elapsed_time / self.calibration_duration_sec
                        ) * 100.0

                        if iteration % 50 == 0 and audio_ch <= 3:
                            logger.info(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): CALIBRATING "
                                f"({calibration_progress:.0f}%, samples={len(state.calibration_samples)}, "
                                f"elapsed={elapsed_time:.1f}s/{self.calibration_duration_sec:.1f}s)"
                            )

                        levels_update[audio_ch] = {
                            "lufs": float(state.current_lufs),
                            "pre_fader_lufs": float(pre_fader_lufs),
                            "avg_lufs": float(state.avg_lufs),
                            "true_peak": float(current_peak),
                            "current_fader": float(state.current_fader),
                            "target_fader": float(state.current_fader),
                            "initial_fader": float(state.initial_fader_db),
                            "correction": 0.0,
                            "fader_range": float(self.fader_range_db),
                            "is_active": bool(state.is_active),
                            "bleed_ratio": 0.0,
                            "bleed_source": None,
                            "status": "calibrating",
                            "instrument_type": str(state.instrument_type),
                            "calibration_complete": False,
                            "calibration_progress": float(calibration_progress),
                            "baseline_lufs": float(state.baseline_lufs),
                            "baseline_peak": float(state.baseline_peak),
                        }
                        continue

                    # ЗАВЕРШЕНИЕ КАЛИБРОВКИ: установка базовых значений
                    if not state.calibration_complete:
                        if len(state.calibration_samples) > 0:
                            # Вычисляем средние значения из накопленных данных
                            lufs_values = [s["lufs"] for s in state.calibration_samples]
                            peak_values = [s["peak"] for s in state.calibration_samples]

                            state.baseline_lufs = float(np.mean(lufs_values))
                            state.baseline_peak = float(np.mean(peak_values))

                            logger.info(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): CALIBRATION COMPLETE - "
                                f"baseline_lufs={state.baseline_lufs:.1f} LUFS, "
                                f"baseline_peak={state.baseline_peak:.1f} dBTP, "
                                f"samples={len(state.calibration_samples)}"
                            )
                        else:
                            # Если нет данных, используем текущие значения
                            state.baseline_lufs = pre_fader_lufs
                            state.baseline_peak = current_peak
                            logger.warning(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): CALIBRATION COMPLETE but no samples, "
                                f"using current values: baseline_lufs={state.baseline_lufs:.1f}, "
                                f"baseline_peak={state.baseline_peak:.1f}"
                            )

                        state.calibration_complete = True

                    # УПРАВЛЕНИЕ ФЕЙДЕРОМ ОТНОСИТЕЛЬНО БАЗОВЫХ ЗНАЧЕНИЙ
                    # Вычисляем отклонение от базовых значений
                    lufs_error = (
                        pre_fader_lufs - state.baseline_lufs
                    )  # Положительное = выше базового
                    peak_error = (
                        current_peak - state.baseline_peak
                    )  # Положительное = выше базового

                    # Комбинируем ошибки LUFS и Peak (можно настроить веса)
                    lufs_weight = 0.7
                    peak_weight = 0.3
                    combined_error = (lufs_error * lufs_weight) + (
                        peak_error * peak_weight
                    )

                    # Применяем sensitivity для плавной коррекции
                    correction_lufs = combined_error * self.sensitivity

                    # Преобразуем коррекцию LUFS в коррекцию фейдера (dB)
                    lufs_to_fader_ratio = 1.5
                    correction_fader_db = correction_lufs / lufs_to_fader_ratio

                    # Вычисляем целевой фейдер относительно якорной позиции
                    target_fader_db = state.initial_fader_db + correction_fader_db

                    # Ограничиваем диапазоном вокруг якорной позиции
                    min_fader = state.initial_fader_db - self.fader_range_db
                    max_fader = state.initial_fader_db + self.fader_range_db
                    target_fader_db = np.clip(target_fader_db, min_fader, max_fader)

                    # Также ограничиваем абсолютными пределами (safety.max_fader при enable_limits)
                    target_fader_db = np.clip(
                        target_fader_db, -60.0, self._max_fader_db
                    )

                    # Применяем envelope для плавности
                    if audio_ch in self.envelopes:
                        smoothed_target_fader = self.envelopes[audio_ch].process(
                            target_fader_db
                        )
                        target_fader_db = smoothed_target_fader

                    # Вычисляем фактическую коррекцию
                    actual_correction = target_fader_db - state.current_fader

                    # Применяем коррекцию только если она значительна (минимум 0.1 dB)
                    min_correction_threshold = 0.1
                    if abs(actual_correction) > min_correction_threshold:
                        state.target_fader = target_fader_db
                        fader_updates[state.mixer_channel] = target_fader_db
                        state.status = "adjusting"

                        if iteration % 50 == 0 and audio_ch <= 3:
                            logger.info(
                                f"Ch{audio_ch} (mixer {state.mixer_channel}): "
                                f"pre_fader_lufs={pre_fader_lufs:.1f} (baseline={state.baseline_lufs:.1f}, error={lufs_error:+.1f}), "
                                f"peak={current_peak:.1f} (baseline={state.baseline_peak:.1f}, error={peak_error:+.1f}), "
                                f"combined_error={combined_error:+.1f}LUFS, correction={correction_fader_db:+.2f}dB, "
                                f"fader={state.current_fader:.1f}->{target_fader_db:.1f}dB "
                                f"(anchor={state.initial_fader_db:.1f}dB, range=±{self.fader_range_db:.1f}dB)"
                            )
                    else:
                        state.target_fader = state.current_fader
                        state.status = "active"

                    # Формируем данные для отправки
                    levels_update[audio_ch] = {
                        "lufs": float(state.current_lufs),
                        "pre_fader_lufs": float(pre_fader_lufs),
                        "avg_lufs": float(state.avg_lufs),
                        "true_peak": float(current_peak),
                        "current_fader": float(state.current_fader),
                        "target_fader": float(state.target_fader),
                        "initial_fader": float(state.initial_fader_db),
                        "correction": float(actual_correction),
                        "fader_range": float(self.fader_range_db),
                        "is_active": bool(state.is_active),
                        "bleed_ratio": bleed_ratio
                        if "bleed_ratio" in locals()
                        else 0.0,
                        "bleed_source": bleed_source
                        if "bleed_source" in locals()
                        else None,
                        "status": str(state.status),
                        "instrument_type": str(state.instrument_type),
                        "calibration_complete": bool(state.calibration_complete),
                        "baseline_lufs": float(state.baseline_lufs),
                        "baseline_peak": float(state.baseline_peak),
                    }

                # Применяем обновления фейдеров (пропуск при freeze)
                if fader_updates and self.mixer_client and not self.automation_frozen:
                    now = time.time()
                    for ch in list(self.channel_freeze_until.keys()):
                        if self.channel_freeze_until[ch] <= now:
                            del self.channel_freeze_until[ch]
                    for mixer_ch, new_fader_db in fader_updates.items():
                        if (
                            mixer_ch in self.channel_freeze_until
                            and self.channel_freeze_until[mixer_ch] > time.time()
                        ):
                            continue
                        try:
                            if iteration % 50 == 0:
                                logger.info(
                                    f"Setting fader: mixer_ch={mixer_ch}, dB={new_fader_db:.2f}"
                                )

                            self.mixer_client.set_channel_fader(mixer_ch, new_fader_db)

                            # Обновляем состояние для всех каналов с этим mixer_channel
                            for audio_ch, state in self.channels.items():
                                if state.mixer_channel == mixer_ch:
                                    state.current_fader = new_fader_db

                        except Exception as e:
                            logger.error(
                                f"Error setting fader for channel {mixer_ch}: {e}",
                                exc_info=True,
                            )

                # Отправляем обновление
                if levels_update and self.on_levels_updated:
                    self.on_levels_updated(levels_update)

            except Exception as e:
                logger.error(f"Realtime control loop error: {e}", exc_info=True)

            self._stop_event.wait(self.update_interval)

        logger.info("Realtime control loop stopped")

    def stop_realtime_fader(self):
        """Остановка Real-Time Fader режима"""
        if not self.realtime_enabled:
            return

        self.realtime_enabled = False
        self._stop_event.set()

        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)

        logger.info("Real-time Fader stopped")

        if self.on_status_update:
            self.on_status_update({"type": "realtime_fader_stopped", "active": False})

    def start_auto_balance(
        self, duration: float = 15.0, bleed_threshold: float = -50.0
    ) -> bool:
        """
        Запуск сбора статистики для Auto Balance (LEARN фаза в стиле MIX MONOLITH).

        Args:
            duration: Длительность сбора в секундах (LEARN период)
            bleed_threshold: Порог блидинга в LUFS (каналы с pre-fader ниже этого значения считаются неактивными)
        """
        if not self.is_active:
            logger.error("Cannot start auto balance: controller not active")
            return False

        self.mode = BalanceMode.STATIC
        self.auto_balance_collecting = True
        self.auto_balance_duration = duration
        self.bleed_threshold = bleed_threshold
        self.auto_balance_start_time = time.time()
        self.auto_balance_result.clear()

        # Увеличиваем номер прохода (Pass 1, Pass 2, ...)
        self.auto_balance_pass += 1

        # Сбрасываем статистику каналов (кроме заблокированных)
        for state in self.channels.values():
            if not state.locked:
                state.reset_statistics()

        # Запускаем поток сбора
        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._auto_balance_collect_loop, daemon=True
        )
        self._control_thread.start()

        logger.info(
            f"Auto Balance LEARN started: pass={self.auto_balance_pass}, "
            f"duration={duration}s, bleed_threshold={bleed_threshold} LUFS"
        )

        if self.on_status_update:
            self.on_status_update(
                {
                    "type": "auto_balance_started",
                    "mode": "static",
                    "duration": duration,
                    "collecting": True,
                    "pass_number": self.auto_balance_pass,
                }
            )

        return True

    def _auto_balance_collect_loop(self):
        """Цикл сбора статистики для Auto Balance"""
        logger.info("Auto balance collection loop started")

        while self.auto_balance_collecting and not self._stop_event.is_set():
            elapsed = time.time() - self.auto_balance_start_time

            if elapsed >= self.auto_balance_duration:
                # Завершаем сбор
                self.auto_balance_collecting = False
                self._compute_auto_balance()
                break

            try:
                levels_update = {}

                for audio_ch, state in list(self.channels.items()):
                    if (
                        audio_ch in self._audio_buffers
                        and len(self._audio_buffers[audio_ch]) > 0
                    ):
                        chunks = list(self._audio_buffers[audio_ch])
                        self._audio_buffers[audio_ch].clear()

                        if chunks:
                            audio_data = np.concatenate(chunks)
                            state.process(audio_data)

                            levels_update[audio_ch] = {
                                "lufs": float(state.current_lufs),
                                "avg_lufs": float(state.avg_lufs),
                                "integrated_lufs": float(state.integrated_lufs),
                                "true_peak": float(state.current_true_peak),
                                "is_active": bool(state.is_active),
                                "locked": bool(state.locked),
                                "samples_collected": int(len(state.lufs_history)),
                                "blocks_collected": int(
                                    state.integrated_lufs_meter.get_block_count()
                                ),
                                "progress": float(elapsed / self.auto_balance_duration),
                            }

                if levels_update and self.on_levels_updated:
                    self.on_levels_updated(levels_update)

            except Exception as e:
                logger.error(f"Auto balance collection error: {e}")

            self._stop_event.wait(self.update_interval)

        logger.info("Auto balance collection loop stopped")

    def _compute_auto_balance(self):
        """
        Вычисление баланса на основе Integrated LUFS.

        Алгоритм:
        1. Измеряем Integrated LUFS каждого канала (с гейтингом по ITU-R BS.1770-4)
        2. Читаем текущие позиции фейдеров из микшера
        3. Вычисляем pre-fader уровень: pre_fader = measured - current_fader
        4. Вычисляем идеальный фейдер: ideal = target_LUFS - pre_fader
        5. Если какой-то фейдер выходит за пределы [-60, +10], сдвигаем ВСЕ
           фейдеры пропорционально, сохраняя относительный баланс
        6. Коррекция = ideal_fader - current_fader
        """
        logger.info(f"Computing auto balance (pass {self.auto_balance_pass})...")

        FADER_MAX = self._max_fader_db  # dB — safety ceiling when enable_limits
        FADER_MIN = -60.0  # dB — минимум фейдера микшера

        # === Шаг 1: Получаем Integrated LUFS для каждого канала ===
        channel_integrated = {}
        for audio_ch, state in self.channels.items():
            if state.locked:
                continue
            integrated = state.integrated_lufs_meter.get_integrated_lufs()
            channel_integrated[audio_ch] = integrated
            logger.info(
                f"Channel {audio_ch} ({state.instrument_type}): "
                f"Integrated LUFS = {integrated:.1f}, blocks = {state.integrated_lufs_meter.get_block_count()}"
            )

        # === Шаг 1.5: Bleed detection — обновляем централизованный сервис ===
        all_channel_levels = {
            ch: channel_integrated.get(ch, -70)
            for ch in self.channels
            if not self.channels[ch].locked
        }
        all_channel_centroids = {
            ch: self.channels[ch].spectral_centroid
            for ch in self.channels
            if not self.channels[ch].locked
        }
        all_channel_metrics = {}
        instrument_types_for_bleed = {}
        for ch, state in self.channels.items():
            if state.locked:
                continue
            # Используем peak band energy (накопленный за период) для более надёжного bleed detection
            band_dict = (
                state.band_energy_max
                if state.band_energy_max
                else (state.band_energy if isinstance(state.band_energy, dict) else {})
            )
            # Create adapter for band_energy
            if isinstance(band_dict, dict):

                class _BandMetricsAdapter:
                    def __init__(self, band_dict, rms_level):
                        self.band_energy_sub = float(band_dict.get("sub", -100))
                        self.band_energy_bass = float(band_dict.get("bass", -100))
                        self.band_energy_low_mid = float(band_dict.get("low_mid", -100))
                        self.band_energy_mid = float(band_dict.get("mid", -100))
                        self.band_energy_high_mid = float(
                            band_dict.get("high_mid", -100)
                        )
                        self.band_energy_high = float(band_dict.get("high", -100))
                        self.band_energy_air = float(band_dict.get("air", -100))
                        self.rms_level = float(rms_level)

                all_channel_metrics[ch] = _BandMetricsAdapter(
                    band_dict, state.integrated_lufs
                )
            # Map 'toms' -> 'tom' for BleedDetector INSTRUMENT_BANDS
            it = state.instrument_type or "custom"
            instrument_types_for_bleed[ch] = "tom" if it == "toms" else it

        if self.bleed_service:
            self.bleed_service.configure(instrument_types_for_bleed)
            self.bleed_service.update(
                all_channel_levels, all_channel_centroids, all_channel_metrics
            )

        # === Шаг 2: Читаем текущие фейдеры и вычисляем pre-fader + ideal fader ===
        active_channels = {}  # {audio_ch: {'target': float, 'pre_fader': float, 'ideal_fader': float, ...}}

        for audio_ch, state in self.channels.items():
            # Заблокированные каналы
            if state.locked:
                if audio_ch not in self.auto_balance_result:
                    self.auto_balance_result[audio_ch] = {
                        "correction": 0.0,
                        "integrated_lufs": state.integrated_lufs,
                        "target_lufs": self.target_lufs,
                        "locked": True,
                    }
                else:
                    self.auto_balance_result[audio_ch]["locked"] = True
                continue

            integrated = channel_integrated.get(audio_ch, -70.0)

            # Получаем прямой target LUFS из профиля инструмента
            # Все расчеты основаны на integrated LUFS
            target_lufs = self.profile.instrument_target_lufs.get(
                state.instrument_type,
                self.target_lufs,  # fallback на базовый target если инструмент не найден
            )

            # Читаем текущий фейдер из микшера
            current_fader = 0.0
            if self.mixer_client and self.mixer_client.is_connected:
                fader_val = self.mixer_client.get_channel_fader(state.mixer_channel)
                if fader_val is not None and float(fader_val) > -100:
                    current_fader = float(fader_val)

            # Pre-fader уровень = измеренный post-fader - текущий фейдер
            pre_fader = integrated - current_fader

            # Проверяем активность по PRE-FADER уровню (не post-fader!)
            # Порог блидинга — каналы ниже этого уровня считаются неактивными
            bleed_thresh = getattr(self, "bleed_threshold", -50)
            if pre_fader <= bleed_thresh:
                self.auto_balance_result[audio_ch] = {
                    "correction": 0.0,
                    "integrated_lufs": float(integrated),
                    "target_lufs": float(target_lufs),
                    "locked": False,
                }
                logger.info(
                    f"Channel {audio_ch} ({state.instrument_type}): "
                    f"INACTIVE (pre-fader={pre_fader:.1f} LUFS <= threshold {bleed_thresh} LUFS, fader={current_fader:.1f})"
                )
                continue

            # Bleed compensation: при блидинге используем компенсированный уровень
            bleed_info = None
            bleed_applied = False
            if self.bleed_service and self.bleed_service.enabled:
                bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                if bleed_info and bleed_info.bleed_ratio > 0:
                    # Используем компенсированный уровень из bleed_service
                    pre_fader_for_balance = self.bleed_service.get_compensated_level(
                        audio_ch, pre_fader
                    )
                    bleed_applied = True
                    logger.info(
                        f"Channel {audio_ch} ({state.instrument_type}): "
                        f"BLEED detected (ratio={bleed_info.bleed_ratio:.2f}, source=Ch{bleed_info.bleed_source_channel}), "
                        f"raw={pre_fader:.1f} -> compensated={pre_fader_for_balance:.1f} LUFS"
                    )
                else:
                    pre_fader_for_balance = pre_fader
            else:
                pre_fader_for_balance = pre_fader

            # Идеальная абсолютная позиция фейдера (с учётом компенсации блидинга)
            ideal_fader = target_lufs - pre_fader_for_balance

            # Консервативный лимит для инструментов с естественным блидингом (tom, ride, hihat, overhead, room)
            # — детектор может не сработать, поэтому жёстко ограничиваем усиление
            HIGH_BLEED_INSTRUMENTS = {
                "toms",
                "tom",
                "ride",
                "hihat",
                "overhead",
                "room",
                "drums",
            }
            max_boost_high_bleed = (
                4.0  # dB — макс. усиление для каналов с ожидаемым блидингом
            )
            if state.instrument_type in HIGH_BLEED_INSTRUMENTS:
                if ideal_fader > max_boost_high_bleed:
                    logger.info(
                        f"Channel {audio_ch} ({state.instrument_type}): high-bleed instrument, capping ideal_fader {ideal_fader:+.1f} -> {max_boost_high_bleed} dB"
                    )
                    ideal_fader = max_boost_high_bleed
            # При сильном блидинге (если детектор сработал) — дополнительное ограничение
            elif bleed_applied and bleed_info.bleed_ratio > 0.5:
                max_boost_bleed = 6.0  # dB — макс. усиление для каналов с >50% bleed
                ideal_fader = min(ideal_fader, max_boost_bleed)
                logger.info(
                    f"Channel {audio_ch}: bleed_ratio={bleed_info.bleed_ratio:.2f} > 0.5, capping ideal_fader to {max_boost_bleed} dB"
                )

            active_channels[audio_ch] = {
                "integrated": integrated,
                "target_lufs": target_lufs,
                "current_fader": current_fader,
                "pre_fader": pre_fader_for_balance,
                "ideal_fader": ideal_fader,
                "state": state,
                "bleed_ratio": float(bleed_info.bleed_ratio)
                if bleed_applied and bleed_info
                else 0.0,
                "bleed_source": int(bleed_info.bleed_source_channel)
                if bleed_applied
                and bleed_info
                and bleed_info.bleed_source_channel is not None
                else None,
            }

            log_pre = (
                f"compensated={pre_fader_for_balance:.1f}"
                if bleed_applied
                else f"pre-fader={pre_fader:.1f}"
            )
            logger.info(
                f"Channel {audio_ch} ({state.instrument_type}): "
                f"Integrated={integrated:.1f} LUFS, fader={current_fader:.1f}, "
                f"{log_pre}, target={target_lufs:.1f}, "
                f"ideal_fader={ideal_fader:+.1f} dB"
            )

        # === Шаг 4: Сдвигаем фейдеры чтобы поместиться в диапазон ===
        # Используем 90-й перцентиль ideal_fader для вычисления shift,
        # чтобы единичные аутлайеры (очень тихие каналы) не утягивали весь микс вниз.
        # Аутлайеры просто клипуются на +10 dB.
        if active_channels:
            ideal_faders_sorted = sorted(
                ch["ideal_fader"] for ch in active_channels.values()
            )
            n = len(ideal_faders_sorted)

            # 90-й перцентиль (или максимум если каналов мало)
            p90_index = min(int(n * 0.9), n - 1)
            p90_ideal = ideal_faders_sorted[p90_index]
            max_ideal = ideal_faders_sorted[-1]
            min_ideal = ideal_faders_sorted[0]

            shift = 0.0
            if p90_ideal > FADER_MAX:
                shift = p90_ideal - FADER_MAX
                logger.info(
                    f"Fader range adjustment: p90 ideal={p90_ideal:+.1f} dB > {FADER_MAX} dB. "
                    f"Shifting all faders down by {shift:.1f} dB. "
                    f"(max={max_ideal:+.1f}, min={min_ideal:+.1f}, active={n})"
                )
            elif min_ideal < FADER_MIN:
                shift = min_ideal - FADER_MIN
                logger.info(
                    f"Fader range adjustment: min ideal={min_ideal:+.1f} dB < {FADER_MIN} dB. "
                    f"Shifting all faders up by {-shift:.1f} dB."
                )
            else:
                logger.info(
                    f"Fader range OK: min={min_ideal:+.1f}, max={max_ideal:+.1f}, "
                    f"p90={p90_ideal:+.1f}, active={n}"
                )

            # Применяем коррекции
            for audio_ch, ch_data in active_channels.items():
                state = ch_data["state"]
                new_fader = ch_data["ideal_fader"] - shift
                new_fader = float(np.clip(new_fader, FADER_MIN, FADER_MAX))
                correction = new_fader - ch_data["current_fader"]

                # Округляем до 0.1 dB
                correction = round(correction * 10) / 10.0

                self.auto_balance_result[audio_ch] = {
                    "correction": float(correction),
                    "integrated_lufs": float(ch_data["integrated"]),
                    "target_lufs": float(ch_data["target_lufs"]),
                    "locked": False,
                    "bleed_ratio": float(ch_data.get("bleed_ratio", 0)),
                    "bleed_source": ch_data.get("bleed_source"),
                }

                logger.info(
                    f"Channel {audio_ch} ({state.instrument_type}): "
                    f"fader {ch_data['current_fader']:.1f} -> {new_fader:.1f} dB "
                    f"(correction={correction:+.1f}, shift={shift:+.1f})"
                )

        logger.info(
            f"Auto balance computed: pass={self.auto_balance_pass}, "
            f"channels={len(self.auto_balance_result)}"
        )

        if self.on_status_update:
            result_dict = {}
            for k, v in self.auto_balance_result.items():
                result_dict[int(k)] = {
                    "correction": float(v["correction"]),
                    "integrated_lufs": float(v["integrated_lufs"]),
                    "target_lufs": float(v["target_lufs"]),
                    "locked": bool(v["locked"]),
                    "bleed_ratio": float(v.get("bleed_ratio", 0)),
                    "bleed_source": v.get("bleed_source"),
                }
            self.on_status_update(
                {
                    "type": "auto_balance_ready",
                    "mode": "static",
                    "collecting": False,
                    "pass_number": self.auto_balance_pass,
                    "result": result_dict,
                }
            )

    def apply_auto_balance(self, **kwargs) -> bool:
        """
        Применение вычисленного Auto Balance к микшеру.

        Применяет точные коррекции по LUFS к фейдерам.
        """
        if not self.auto_balance_result:
            logger.error("No auto balance result to apply")
            return False

        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.error("Mixer not connected")
            return False

        logger.info(f"Applying auto balance (pass {self.auto_balance_pass})...")

        applied_count = 0
        applied_details = {}

        for audio_ch, result in self.auto_balance_result.items():
            if audio_ch not in self.channels:
                continue

            state = self.channels[audio_ch]

            # Пропускаем заблокированные каналы
            if state.locked or result.get("locked", False):
                logger.info(
                    f"Channel {audio_ch} ({state.instrument_type}): LOCKED, skipping"
                )
                continue

            correction = result.get("correction", 0.0)

            try:
                # Получаем текущий фейдер из микшера (в dB)
                current_db_value = self.mixer_client.get_channel_fader(
                    state.mixer_channel
                )

                if current_db_value is None:
                    if state.current_fader != 0.0 and abs(state.current_fader) < 100:
                        current_db = state.current_fader
                        logger.warning(
                            f"Channel {state.mixer_channel}: Could not read fader, using saved value {current_db:.1f} dB"
                        )
                    else:
                        current_db = 0.0
                        logger.warning(
                            f"Channel {state.mixer_channel}: Could not read fader, using default 0 dB"
                        )
                else:
                    current_db = float(current_db_value)
                    if current_db < -100:
                        if (
                            state.current_fader != 0.0
                            and abs(state.current_fader) < 100
                        ):
                            current_db = state.current_fader
                            logger.warning(
                                f"Channel {state.mixer_channel}: Fader read as {current_db_value:.1f} dB (invalid), using saved value {current_db:.1f} dB"
                            )
                        else:
                            current_db = 0.0
                            logger.warning(
                                f"Channel {state.mixer_channel}: Fader read as {current_db_value:.1f} dB (invalid), using default 0 dB"
                            )

                new_db = current_db + correction
                new_db = float(np.clip(new_db, -60.0, self._max_fader_db))

                # Округляем до 0.05 dB
                new_db = round(new_db * 20) / 20.0

                if self.automation_frozen:
                    continue
                if (
                    state.mixer_channel in self.channel_freeze_until
                    and self.channel_freeze_until[state.mixer_channel] > time.time()
                ):
                    continue
                self.mixer_client.set_channel_fader(state.mixer_channel, new_db)

                state.current_fader = new_db
                applied_count += 1

                applied_details[int(audio_ch)] = {
                    "mixer_channel": int(state.mixer_channel),
                    "previous_db": float(current_db),
                    "new_db": float(new_db),
                    "applied_correction": float(correction),
                }

                logger.info(
                    f"Channel {state.mixer_channel} ({state.instrument_type}): "
                    f"{current_db:.2f} -> {new_db:.2f} dB "
                    f"(correction={correction:+.1f})"
                )

            except Exception as e:
                logger.error(
                    f"Error applying balance to channel {state.mixer_channel}: {e}"
                )

        logger.info(
            f"Auto balance applied to {applied_count} channels (pass {self.auto_balance_pass})"
        )

        if self.on_status_update:
            self.on_status_update(
                {
                    "type": "auto_balance_applied",
                    "applied_count": applied_count,
                    "total_count": len(self.auto_balance_result),
                    "pass_number": self.auto_balance_pass,
                    "details": applied_details,
                }
            )

        # Очищаем результат после применения
        self.auto_balance_result = {}
        logger.info(
            "Auto balance result cleared after apply (re-run LEARN for next pass)"
        )

        return True

    def cancel_auto_balance(self):
        """Отмена сбора Auto Balance"""
        self.auto_balance_collecting = False
        self._stop_event.set()
        self.auto_balance_result.clear()
        self.auto_balance_pass = 0

        # Сбрасываем блокировки каналов
        for state in self.channels.values():
            state.locked = False

        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)

        logger.info("Auto balance cancelled, pass counter reset")

        if self.on_status_update:
            self.on_status_update({"type": "auto_balance_cancelled"})

    # === Level-Plane управление (MIX MONOLITH style) ===

    def lock_channel(self, audio_ch: int) -> bool:
        """
        Заблокировать канал — его коррекция не будет изменена при повторных проходах.

        Args:
            audio_ch: Номер аудио-канала
        """
        if audio_ch in self.channels:
            self.channels[audio_ch].locked = True
            logger.info(f"Channel {audio_ch} locked")
            if self.on_status_update:
                self.on_status_update(
                    {
                        "type": "channel_lock_changed",
                        "channel": int(audio_ch),
                        "locked": True,
                    }
                )
            return True
        return False

    def unlock_channel(self, audio_ch: int) -> bool:
        """
        Разблокировать канал — его коррекция будет пересчитана при следующем проходе.

        Args:
            audio_ch: Номер аудио-канала
        """
        if audio_ch in self.channels:
            self.channels[audio_ch].locked = False
            logger.info(f"Channel {audio_ch} unlocked")
            if self.on_status_update:
                self.on_status_update(
                    {
                        "type": "channel_lock_changed",
                        "channel": int(audio_ch),
                        "locked": False,
                    }
                )
            return True
        return False

    def get_auto_balance_status(self) -> Dict[str, Any]:
        """Получить текущий статус Auto Balance"""
        return {
            "collecting": self.auto_balance_collecting,
            "pass_number": self.auto_balance_pass,
            "has_result": bool(self.auto_balance_result),
            "result": {
                int(k): {
                    "correction": float(v.get("correction", 0)),
                    "integrated_lufs": float(v.get("integrated_lufs", -70)),
                    "target_lufs": float(v.get("target_lufs", -18)),
                    "locked": bool(v.get("locked", False)),
                }
                for k, v in self.auto_balance_result.items()
            }
            if self.auto_balance_result
            else {},
        }

    def stop(self):
        """Полная остановка контроллера"""
        self.stop_realtime_fader()
        self.cancel_auto_balance()

        self.is_active = False
        self._stop_event.set()

        if self._audio_capture is not None:
            try:
                self._audio_capture.unsubscribe("auto_fader")
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
        self.envelopes.clear()

        logger.info("AutoFaderController stopped")

    def set_profile(self, genre: str):
        """Установка жанрового профиля"""
        try:
            self.profile = BalanceProfile.get_preset(GenreProfile(genre))

            # Обновляем параметры envelope
            for envelope in self.envelopes.values():
                envelope.set_times(
                    self.profile.attack_ms,
                    self.profile.release_ms,
                    self.profile.hold_ms,
                )

            logger.info(f"Profile set to: {self.profile.name}")
        except Exception as e:
            logger.error(f"Invalid profile: {genre}, error: {e}")

    def handle_external_fader_change(self, mixer_channel: int, new_fader_db: float):
        """
        Обработка внешнего изменения фейдера (от микшера или ручного управления).
        Если это референсный канал, быстро адаптируем сглаженное значение референса.

        Args:
            mixer_channel: Номер канала микшера
            new_fader_db: Новое значение фейдера в dB
        """
        if not self.realtime_enabled:
            return

        # Находим канал по mixer_channel
        for audio_ch, state in self.channels.items():
            if state.mixer_channel == mixer_channel:
                old_fader = state.current_fader
                fader_change = (
                    abs(new_fader_db - old_fader) if old_fader is not None else 0.0
                )

                # Обновляем значение фейдера
                state.current_fader = new_fader_db

                break

    def update_settings(
        self,
        fader_range_db: Optional[float] = None,
        avg_window_sec: Optional[float] = None,
        sensitivity: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        # Legacy parameters (for Auto Balance mode only)
        target_lufs: Optional[float] = None,
        max_adjustment_db: Optional[float] = None,
        ratio: Optional[float] = None,
        hold_ms: Optional[float] = None,
        **kwargs,
    ):
        """Обновление параметров контроллера"""
        # Real-Time Fader Riding parameters
        if fader_range_db is not None:
            self.fader_range_db = float(fader_range_db)
        if avg_window_sec is not None:
            self.avg_window_sec = float(avg_window_sec)
            # Обновляем размер ring buffer для всех каналов
            buffer_size = max(10, int(self.avg_window_sec / self.update_interval))
            for state in self.channels.values():
                state.lufs_ring_buffer = deque(maxlen=buffer_size)
        if sensitivity is not None:
            self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        if attack_ms is not None:
            self.attack_ms = float(attack_ms)
        if release_ms is not None:
            self.release_ms = float(release_ms)
        if gate_threshold is not None:
            self.gate_threshold = float(gate_threshold)

        # Legacy parameters (for Auto Balance mode)
        if target_lufs is not None:
            self.target_lufs = target_lufs
        if max_adjustment_db is not None:
            self.max_adjustment_db = max_adjustment_db
        if ratio is not None:
            self.ratio = ratio
        if hold_ms is not None:
            self.hold_ms = hold_ms

        # Обновляем envelopes
        for envelope in self.envelopes.values():
            envelope.set_times(
                self.attack_ms, self.release_ms, self.hold_ms if self.hold_ms else 0.0
            )

        logger.info(
            f"Settings updated: fader_range=±{self.fader_range_db:.1f}dB, "
            f"avg_window={self.avg_window_sec:.1f}s, sensitivity={self.sensitivity:.2f}"
        )

    def get_status(self) -> Dict:
        """Получение текущего статуса"""
        # Преобразуем результат Auto Balance в JSON-совместимый формат
        auto_balance_result = None
        if self.auto_balance_result:
            auto_balance_result = {}
            for k, v in self.auto_balance_result.items():
                if isinstance(v, dict):
                    auto_balance_result[int(k)] = {
                        "correction": float(v.get("correction", 0)),
                        "integrated_lufs": float(v.get("integrated_lufs", -70)),
                        "target_lufs": float(v.get("target_lufs", -18)),
                        "locked": bool(v.get("locked", False)),
                        "bleed_ratio": float(v.get("bleed_ratio", 0)),
                        "bleed_source": v.get("bleed_source"),
                    }
                else:
                    auto_balance_result[int(k)] = {
                        "correction": float(v),
                        "integrated_lufs": -70.0,
                        "target_lufs": float(self.target_lufs),
                        "locked": False,
                    }

        return {
            "active": bool(self.is_active),
            "mode": str(self.mode.value),
            "realtime_enabled": bool(self.realtime_enabled),
            "auto_balance_collecting": bool(self.auto_balance_collecting),
            "profile": str(self.profile.name),
            "target_lufs": float(self.target_lufs),
            "ratio": float(self.ratio),
            "channels_count": int(len(self.channels)),
            "auto_balance_result": auto_balance_result,
            "auto_balance_pass": int(self.auto_balance_pass),
        }

    def _db_to_normalized(self, db: float) -> float:
        """
        Конвертация dB в нормализованное значение (0-1).
        Wing fader: -144dB to +10dB mapped to 0.0-1.0
        """
        # Линейная интерполяция для упрощения
        # 0.75 = 0dB (unity), 1.0 = +10dB, 0.0 = -inf
        if db <= -144:
            return 0.0
        elif db >= 10:
            return 1.0
        else:
            # Приблизительная формула для Wing
            # При 0dB -> 0.75, при +10dB -> 1.0, при -10dB -> ~0.625
            return 0.75 + (db / 40.0)

    def _normalized_to_db(self, normalized: float) -> float:
        """
        Конвертация нормализованного значения (0-1) в dB.
        """
        if normalized <= 0:
            return -144.0
        elif normalized >= 1.0:
            return 10.0
        else:
            # Обратная формула
            return (normalized - 0.75) * 40.0
