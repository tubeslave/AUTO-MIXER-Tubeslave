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
from typing import Dict, List, Callable, Optional, Any
from collections import deque
from scipy import signal
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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

    noise_gate_threshold_db: float = -40.0

    crest_factor_db: float = 0.0

    suggested_gain_db: float = 0.0
    gain_limited_by: str = "none"

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
        """Calculate integrated LUFS from collected RMS values."""
        if len(self.rms_values) == 0:
            self.integrated_lufs = -100.0
            return

        valid_lufs = [l for l in self.rms_values if l > -70.0]
        if len(valid_lufs) == 0:
            self.integrated_lufs = -100.0
            return

        lufs_linear = [10 ** (l / 10.0) for l in valid_lufs]
        mean_linear = np.mean(lufs_linear)
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
                 bleed_service=None):
        
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
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
            
            # Очищаем старые каналы перед добавлением новых
            self.channels.clear()
            self._audio_buffers.clear()
            self.measured_levels.clear()
            logger.info(f"Cleared old channels, initializing {len(channels)} new channels")
            
            # Инициализируем AGC для каждого канала
            update_interval_ms = self.update_interval * 1000  # Конвертируем секунды в мс
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
        self.noise_gate_threshold = cal_config.get('noise_gate_threshold', -40.0)
        self.min_signal_presence = cal_config.get('min_signal_presence', 0.05)
        self.learning_duration = cal_config.get('learning_duration_sec', 30.0)  # По умолчанию 30 секунд
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

    def _finalize_analysis(self):
        """Завершение фазы анализа и расчет рекомендаций."""
        logger.info("Learning phase completed. Calculating suggestions...")

        for ch_id, stats in self.channels.items():
            # Получаем пресет для канала
            channel_setting = self.channel_settings.get(ch_id, {})
            preset = channel_setting.get('preset', 'custom')
            
            # Используем метрики для пресета или значения по умолчанию
            preset_metrics = self.preset_targets.get(preset, {})
            target_lufs = preset_metrics.get('target_lufs', self.target_lufs)
            max_peak_limit = preset_metrics.get('max_peak_limit', self.max_peak_limit)
            
            logger.info(f"Ch{ch_id} ({preset}): using target_lufs={target_lufs:.1f}, max_peak_limit={max_peak_limit:.1f}")
            
            stats.calculate_safe_gain(
                target_lufs=target_lufs,
                max_peak_limit=max_peak_limit,
                min_signal_presence=self.min_signal_presence
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
            gain = suggestion['suggested_gain_db']

            mixer_ch = self.channel_mapping.get(ch_id, ch_id)

            try:
                current_trim = self.mixer_client.get_channel_gain(mixer_ch) or 0.0
                new_trim = current_trim + gain

                new_trim = np.clip(new_trim, -24.0, 24.0)

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
            'learning_duration': self.learning_duration
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
