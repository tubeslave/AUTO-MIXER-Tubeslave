"""
ИСПРАВЛЕННЫЙ Gain Staging Module

Ключевые исправления:
1. Правильная логика HOLD в AGCEnvelope
2. Thread-safe audio buffers
3. Rate limiting для OSC
4. Deadband filtering
5. Graceful error handling
"""

import numpy as np
import threading
import logging
import time
from typing import Dict, List, Callable, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AnalysisState(Enum):
    """Analysis state for Safe Static Gain system."""
    IDLE = "idle"
    LEARNING = "learning"
    READY = "ready"
    APPLYING = "applying"


@dataclass
class SignalStats:
    """Signal statistics с ограничением размера буферов."""
    channel_id: int
    max_true_peak_db: float = -100.0
    integrated_lufs: float = -100.0
    signal_presence_ratio: float = 0.0
    total_samples: int = 0
    active_samples: int = 0
    
    _max_buffer_size: int = field(default=10000, repr=False)
    
    # ИСПРАВЛЕНИЕ: Ограниченные буферы для предотвращения утечки памяти
    # Инициализируем в __post_init__ с правильным maxlen
    
    noise_gate_threshold_db: float = -40.0
    crest_factor_db: float = 0.0
    suggested_gain_db: float = 0.0
    gain_limited_by: str = "none"
    
    def __post_init__(self):
        """Инициализация deque с правильным maxsize."""
        self.rms_values = deque(maxlen=self._max_buffer_size)
        self.peak_values = deque(maxlen=self._max_buffer_size)
    
    def update_sample(self, true_peak_db: float, lufs: float, sample_rms_db: float):
        """Update stats с ограниченными буферами."""
        self.total_samples += 1
        
        if true_peak_db > self.noise_gate_threshold_db:
            self.active_samples += 1
            self.peak_values.append(true_peak_db)
            self.rms_values.append(lufs)
            
            if true_peak_db > self.max_true_peak_db:
                self.max_true_peak_db = true_peak_db
        
        # ИСПРАВЛЕНИЕ: Проверка на деление на ноль
        denominator = max(self.total_samples, 1)
        self.signal_presence_ratio = self.active_samples / denominator
    
    def calculate_integrated_lufs(self):
        """Calculate integrated LUFS с BS.1770 two-stage gating."""
        if len(self.rms_values) == 0:
            self.integrated_lufs = -100.0
            return
        
        rms_list = list(self.rms_values)
        
        # Stage 1: absolute gate at -70 LUFS
        abs_gated = [l for l in rms_list if l > -70.0]
        if len(abs_gated) == 0:
            self.integrated_lufs = -100.0
            return
        
        abs_linear = [10 ** (l / 10.0) for l in abs_gated]
        abs_mean = np.mean(abs_linear)
        abs_mean_lufs = 10 * np.log10(abs_mean + 1e-10)
        
        # Stage 2: relative gate at 10 dB below the stage-1 average
        rel_threshold = abs_mean_lufs - 10.0
        rel_gated = [l for l in abs_gated if l > rel_threshold]
        if len(rel_gated) == 0:
            self.integrated_lufs = abs_mean_lufs
            return
        
        rel_linear = [10 ** (l / 10.0) for l in rel_gated]
        self.integrated_lufs = 10 * np.log10(np.mean(rel_linear) + 1e-10)
    
    def calculate_crest_factor(self):
        """Calculate crest factor."""
        if len(self.peak_values) == 0 or self.integrated_lufs < -70.0:
            self.crest_factor_db = 0.0
            return
        
        self.crest_factor_db = self.max_true_peak_db - self.integrated_lufs
    
    def calculate_safe_gain(self,
                           target_lufs: float = -18.0,
                           max_peak_limit: float = -3.0,
                           min_signal_presence: float = 0.05):
        """Calculate safe gain с проверками."""
        # ИСПРАВЛЕНИЕ: Проверка min_signal_presence > 0
        if min_signal_presence <= 0:
            min_signal_presence = 0.01
        
        if self.signal_presence_ratio < min_signal_presence:
            self.suggested_gain_db = 0.0
            self.gain_limited_by = "silent_channel"
            return
        
        self.calculate_integrated_lufs()
        self.calculate_crest_factor()
        
        delta_lufs = target_lufs - self.integrated_lufs
        delta_peak = max_peak_limit - self.max_true_peak_db
        
        # Smooth blend между LUFS и Peak приоритетом
        crest_norm = np.clip((self.crest_factor_db - 6.0) / 12.0, 0.0, 1.0)
        weight_peak = crest_norm
        weight_lufs = 1.0 - crest_norm
        
        gain_correction = (weight_lufs * delta_lufs) + (weight_peak * delta_peak)
        
        # ИСПРАВЛЕНИЕ: Правильное ограничение
        if delta_peak < 0:
            gain_correction = max(gain_correction, delta_peak)
        else:
            gain_correction = min(gain_correction, delta_peak)
        
        self.gain_limited_by = "blended_peak_lufs"
        self.suggested_gain_db = np.clip(gain_correction, -24.0, 24.0)
    
    def get_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        return {
            'channel': int(self.channel_id),
            'peak_db': float(round(self.max_true_peak_db, 1)),
            'lufs': float(round(self.integrated_lufs, 1)),
            'crest_factor_db': float(round(self.crest_factor_db, 1)),
            'signal_presence': float(round(self.signal_presence_ratio * 100, 1)),
            'suggested_gain_db': float(round(self.suggested_gain_db, 1)),
            'limited_by': str(self.gain_limited_by),
            'samples_analyzed': int(self.total_samples),
            'active_samples': int(self.active_samples)
        }


class AGCEnvelope:
    """
    ИСПРАВЛЕННЫЙ Envelope generator.
    
    Исправление: Hold теперь работает ПОСЛЕ attack, а не вместо него.
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 attack_ms: float = 50.0,
                 release_ms: float = 500.0,
                 hold_ms: float = 200.0,
                 update_interval_ms: float = 100.0):
        self.sample_rate = sample_rate
        self.update_interval_ms = update_interval_ms
        
        self.set_times(attack_ms, release_ms, hold_ms)
        
        # Состояние
        self._current_gain = 0.0
        self._hold_counter = 0
        self._is_holding = False
        self._last_target_gain = 0.0
        self._was_attacking = False
    
    def set_times(self, attack_ms: float, release_ms: float, hold_ms: float):
        """Установка временных констант."""
        self.attack_ms = max(attack_ms, 1.0)  # Минимум 1ms
        self.release_ms = max(release_ms, 1.0)
        self.hold_ms = max(hold_ms, 0.0)
        
        # Коэффициенты для экспоненциального сглаживания
        self.attack_coef = 1 - np.exp(-self.update_interval_ms / self.attack_ms)
        self.release_coef = 1 - np.exp(-self.update_interval_ms / self.release_ms)
        
        # Hold в итерациях
        self.hold_iterations = max(int(hold_ms / self.update_interval_ms), 0)
    
    def process(self, target_gain: float) -> float:
        """
        ИСПРАВЛЕННАЯ обработка целевого gain.
        
        Логика state machine:
        1. HOLD - если активен hold таймер, удерживаем текущее значение
        2. ATTACK - если target < current, быстро снижаем и запускаем hold
        3. RELEASE - если target > current, медленно повышаем
        """
        # Определяем направление изменения
        decreasing = target_gain < self._current_gain - 0.01
        increasing = target_gain > self._current_gain + 0.01
        
        if self._is_holding:
            # Hold phase - удерживаем текущее значение
            self._hold_counter -= 1
            if self._hold_counter <= 0:
                self._is_holding = False
            # В hold gain не меняется
            
        elif decreasing:
            # Attack phase (быстрое снижение)
            self._current_gain += self.attack_coef * (target_gain - self._current_gain)
            # Запускаем hold после attack
            if self.hold_iterations > 0:
                self._is_holding = True
                self._hold_counter = self.hold_iterations
            
        elif increasing:
            # Release phase (медленное повышение)
            self._current_gain += self.release_coef * (target_gain - self._current_gain)
        
        self._last_target_gain = target_gain
        return float(self._current_gain)
    
    def get_current_gain(self) -> float:
        """Возврат текущего gain."""
        return float(self._current_gain)
    
    def reset(self, initial_gain: float = 0.0):
        """Сброс envelope."""
        self._current_gain = float(initial_gain)
        self._hold_counter = 0
        self._is_holding = False
        self._was_attacking = False
        self._last_target_gain = 0.0


class OSCRateLimiter:
    """
    Rate limiter для OSC сообщений с deadband filtering.
    """
    
    def __init__(self, 
                 normal_rate_hz: float = 100.0,
                 emergency_rate_hz: float = 1000.0,
                 deadband_db: float = 0.5):
        self.normal_rate_hz = normal_rate_hz
        self.emergency_rate_hz = emergency_rate_hz
        self.deadband_db = deadband_db
        
        self._last_send_time: Dict[int, float] = {}
        self._last_values: Dict[int, float] = {}
        self._message_count = 0
        self._last_second = time.time()
    
    def should_send(self, channel: int, value: float, is_emergency: bool = False) -> bool:
        """
        Проверка нужно ли отправлять OSC сообщение.
        
        Returns:
            True если сообщение нужно отправить
        """
        current_time = time.time()
        rate_hz = self.emergency_rate_hz if is_emergency else self.normal_rate_hz
        min_interval = 1.0 / rate_hz
        
        # Deadband check (сначала проверяем значимость изменения)
        last_value = self._last_values.get(channel)
        if last_value is not None:
            if abs(value - last_value) < self.deadband_db:
                return False
        
        # Rate limiting check (только если не первое сообщение)
        last_send = self._last_send_time.get(channel)
        if last_send is not None:
            if current_time - last_send < min_interval:
                return False
        
        # Update tracking (только если будем отправлять)
        self._last_send_time[channel] = current_time
        self._last_values[channel] = value
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Возврат статистики rate limiting."""
        current_time = time.time()
        if current_time - self._last_second >= 1.0:
            self._message_count = 0
            self._last_second = current_time
        
        return {
            'messages_per_second': self._message_count,
            'tracked_channels': len(self._last_values)
        }


class ThreadSafeAudioBuffer:
    """
    Thread-safe аудио буфер для использования в callback и correction loop.
    """
    
    def __init__(self, max_chunks: int = 10):
        self._buffer: deque = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
    
    def append(self, data: np.ndarray):
        """Добавление данных (из callback)."""
        with self._lock:
            self._buffer.append(data.copy())
    
    def get_and_clear(self) -> Optional[np.ndarray]:
        """Получение всех данных и очистка буфера."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            
            chunks = list(self._buffer)
            self._buffer.clear()
        
        if chunks:
            try:
                return np.concatenate(chunks)
            except ValueError:
                return None
        return None
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# Экспорт для обратной совместимости
__all__ = [
    'AnalysisState',
    'SignalStats', 
    'AGCEnvelope',
    'OSCRateLimiter',
    'ThreadSafeAudioBuffer'
]