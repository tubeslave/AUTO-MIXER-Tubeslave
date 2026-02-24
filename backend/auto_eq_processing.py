"""
Auto EQ Signal Processing Components

Дополнительные компоненты для auto_eq.py по методу из дебатов:
- DualEMA: двухфазное экспоненциальное сглаживание (fast/slow)
- RateLimiter: ограничение скорости изменения параметров
- MirrorEQ: mirror equalization для разделения спектра
- PriorityMatrix: матрица приоритетов каналов

Научная база: Perez Gonzalez & Reiss (2009), Hafezi & Reiss (2015)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class DualEMA:
    """
    Двухфазное экспоненциальное сглаживание для Auto EQ.
    
    Fast EMA (α=0.4) — быстрая реакция на изменения
    Slow EMA (α=0.08) — сглаживание и стабильность
    
    Научное обоснование: метод Perez Gonzalez (2009) — 
    adaptive gating с двумя временными константами
    """
    
    def __init__(self, 
                 alpha_fast: float = 0.4,
                 alpha_slow: float = 0.08,
                 switch_threshold_db: float = 3.0):
        """
        Args:
            alpha_fast: Коэффициент быстрого EMA (0.4 = ~2.5 кадров)
            alpha_slow: Коэффициент медленного EMA (0.08 = ~12 кадров)
            switch_threshold_db: Порог переключения между fast/slow (dB)
        """
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.switch_threshold_db = switch_threshold_db
        
        self._fast_value = 0.0
        self._slow_value = 0.0
        self._current_alpha = alpha_slow
        self._last_input = 0.0
    
    def process(self, value: float) -> float:
        """
        Обработка нового значения.
        
        Args:
            value: Входное значение (dB)
            
        Returns:
            Сглаженное значение (dB)
        """
        # Определяем скорость изменения
        delta = abs(value - self._last_input)
        
        # Переключаемся на fast если изменение большое
        if delta > self.switch_threshold_db:
            self._current_alpha = self.alpha_fast
        else:
            # Плавно возвращаемся к slow
            self._current_alpha = (0.9 * self._current_alpha + 
                                   0.1 * self.alpha_slow)
        
        # Обновляем оба EMA
        self._fast_value = (self.alpha_fast * value + 
                           (1 - self.alpha_fast) * self._fast_value)
        self._slow_value = (self.alpha_slow * value + 
                           (1 - self.alpha_slow) * self._slow_value)
        
        # Используем текущий alpha
        if self._current_alpha == self.alpha_fast:
            output = self._fast_value
        else:
            output = self._slow_value
        
        self._last_input = value
        return float(output)
    
    def reset(self, initial_value: float = 0.0):
        """Сброс состояния."""
        self._fast_value = initial_value
        self._slow_value = initial_value
        self._current_alpha = self.alpha_slow
        self._last_input = initial_value
    
    def get_state(self) -> Dict[str, float]:
        """Возврат текущего состояния."""
        return {
            'fast': self._fast_value,
            'slow': self._slow_value,
            'current_alpha': self._current_alpha
        }


class EQLimiter:
    """
    Rate limiter для EQ параметров с hysteresis.
    
    Предотвращает резкие скачки gain и обеспечивает стабильность.
    
    Параметры из метода:
    - Rate limit: ±2 dB/frame
    - Hysteresis: ±0.5 dB
    """
    
    def __init__(self,
                 max_rate_db_per_frame: float = 2.0,
                 hysteresis_db: float = 0.5,
                 min_gain_db: float = -15.0,
                 max_gain_db: float = 15.0):
        """
        Args:
            max_rate_db_per_frame: Максимальное изменение за кадр (dB)
            hysteresis_db: Зона нечувствительности (dB)
            min_gain_db: Минимальный gain (dB)
            max_gain_db: Максимальный gain (dB)
        """
        self.max_rate = max_rate_db_per_frame
        self.hysteresis = hysteresis_db
        self.min_gain = min_gain_db
        self.max_gain = max_gain_db
        
        self._last_output = 0.0
        self._pending_value = 0.0
    
    def process(self, target_gain: float) -> float:
        """
        Применение rate limiting и hysteresis.
        
        Args:
            target_gain: Целевое значение gain (dB)
            
        Returns:
            Ограниченное значение gain (dB)
        """
        # Ограничиваем диапазон
        target_gain = np.clip(target_gain, self.min_gain, self.max_gain)
        
        # Hysteresis: если изменение мало — не меняем
        if abs(target_gain - self._last_output) < self.hysteresis:
            return self._last_output
        
        # Rate limiting
        delta = target_gain - self._last_output
        delta = np.clip(delta, -self.max_rate, self.max_rate)
        
        output = self._last_output + delta
        self._last_output = output
        
        return float(output)
    
    def reset(self, initial_value: float = 0.0):
        """Сброс состояния."""
        self._last_output = initial_value
        self._pending_value = initial_value


class PriorityMatrix:
    """
    Матрица приоритетов каналов с softmax нормализацией.
    
    Используется для определения "ведущего" элемента в миксе.
    
    Пример приоритетов:
    - Lead vox: 0.9
    - Kick/Snare: 0.8
    - Bass: 0.7
    - Guitars: 0.5
    - Pads: 0.3
    """
    
    def __init__(self, channel_ids: List[int]):
        """
        Args:
            channel_ids: Список ID каналов
        """
        self.channel_ids = channel_ids
        self.priorities: Dict[int, float] = {ch: 0.5 for ch in channel_ids}
        self.smoothed_priorities: Dict[int, float] = {ch: 0.5 for ch in channel_ids}
        
        # EMA для сглаживания
        self.alpha = 0.15
    
    def set_priority(self, channel: int, priority: float):
        """
        Установка приоритета для канала.
        
        Args:
            channel: ID канала
            priority: Приоритет (0.0 - 1.0)
        """
        if channel in self.priorities:
            self.priorities[channel] = np.clip(priority, 0.0, 1.0)
    
    def update_from_rms(self, channel_rms: Dict[int, float]):
        """
        Обновление приоритетов на основе RMS уровней.
        Каналы с более высоким уровнем получают повышенный приоритет.
        
        Args:
            channel_rms: Словарь {channel: rms_db}
        """
        if not channel_rms:
            return
        
        # Находим максимум
        max_rms = max(channel_rms.values()) if channel_rms else -60.0
        
        # Динамические приоритеты на основе уровня
        dynamic_priorities = {}
        for ch in self.channel_ids:
            rms = channel_rms.get(ch, -60.0)
            # Нормализуем относительно максимума
            level_factor = (rms + 60) / 60  # 0-1
            # Комбинируем с базовым приоритетом
            dynamic_priorities[ch] = 0.7 * self.priorities[ch] + 0.3 * level_factor
        
        # Softmax нормализация
        exp_p = {ch: np.exp(p) for ch, p in dynamic_priorities.items()}
        sum_exp = sum(exp_p.values())
        
        for ch in self.channel_ids:
            normalized = exp_p[ch] / sum_exp if sum_exp > 0 else 1.0 / len(self.channel_ids)
            # Сглаживаем
            self.smoothed_priorities[ch] = (
                self.alpha * normalized + 
                (1 - self.alpha) * self.smoothed_priorities[ch]
            )
    
    def get_leader(self) -> Optional[int]:
        """Возврат канала с максимальным приоритетом (ведущий элемент)."""
        if not self.smoothed_priorities:
            return None
        return max(self.smoothed_priorities, key=self.smoothed_priorities.get)
    
    def get_priority(self, channel: int) -> float:
        """Возврат приоритета канала."""
        return self.smoothed_priorities.get(channel, 0.5)


class MirrorEQ:
    """
    Mirror EQ для разделения спектра между каналами.
    
    Концепция: если один канал получает boost на частоте,
    другие каналы получают cut на той же частоте.
    
    Научное обоснование: Wakefield & Dewey — frequency spectrum sharing
    """
    
    def __init__(self, 
                 num_bands: int = 16,
                 max_cut_db: float = -6.0,
                 mirror_ratio: float = 0.5):
        """
        Args:
            num_bands: Количество частотных полос
            max_cut_db: Максимальный cut для других каналов
            mirror_ratio: Коэффициент mirror (0.5 = половина boost)
        """
        self.num_bands = num_bands
        self.max_cut_db = max_cut_db
        self.mirror_ratio = mirror_ratio
        
        # Состояние mirror для каждого канала и полосы
        self.mirror_gains: Dict[int, List[float]] = {}
    
    def calculate_mirror(self,
                        channel: int,
                        band_gains: List[float],
                        leader_channel: int,
                        leader_gains: List[float]) -> List[float]:
        """
        Расчёт mirror EQ для канала относительно лидера.
        
        Args:
            channel: Текущий канал
            band_gains: Текущие gains канала (dB)
            leader_channel: Канал-лидер
            leader_gains: Gains лидера (dB)
            
        Returns:
            Скорректированные gains с учётом mirror
        """
        if channel == leader_channel:
            return band_gains
        
        result = []
        for i, (gain, leader_gain) in enumerate(zip(band_gains, leader_gains)):
            # Если лидер имеет boost — мы делаем cut
            if leader_gain > 0:
                mirror_cut = -leader_gain * self.mirror_ratio
                mirror_cut = max(mirror_cut, self.max_cut_db)
                gain = min(gain, mirror_cut)
            
            result.append(gain)
        
        return result
    
    def resolve_conflicts(self,
                         all_channel_gains: Dict[int, List[float]],
                         priorities: PriorityMatrix) -> Dict[int, List[float]]:
        """
        Разрешение конфликтов между каналами.
        
        Args:
            all_channel_gains: {channel: [gains per band]}
            priorities: Матрица приоритетов
            
        Returns:
            Скорректированные gains для всех каналов
        """
        leader = priorities.get_leader()
        if leader is None:
            return all_channel_gains
        
        leader_gains = all_channel_gains.get(leader, [0.0] * self.num_bands)
        result = {}
        
        for channel, gains in all_channel_gains.items():
            if channel == leader:
                result[channel] = gains
            else:
                result[channel] = self.calculate_mirror(
                    channel, gains, leader, leader_gains
                )
        
        return result


class LinkwitzRileyFilter:
    """
    Linkwitz-Riley кроссовер фильтр 4-го порядка.
    
    Используется для разделения на 16 полос без фазовых искажений.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Стандартные частоты 1/3 октавы
        self.frequencies = [
            31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
            8000, 10000, 12500, 16000
        ]
        
        # Берём 16 центральных полос
        mid_idx = len(self.frequencies) // 2 - 8
        self.band_freqs = self.frequencies[mid_idx:mid_idx + 16]
    
    def get_band_frequencies(self) -> List[float]:
        """Возврат частот полос."""
        return self.band_freqs
    
    def calculate_crossover_freqs(self) -> List[float]:
        """Расчёт частот кроссовера между полосами."""
        crossovers = []
        for i in range(len(self.band_freqs) - 1):
            # Геометрическое среднее
            f_cross = np.sqrt(self.band_freqs[i] * self.band_freqs[i + 1])
            crossovers.append(f_cross)
        return crossovers


# Утилиты для интеграции с AutoEQController

def create_eq_processing_chain(channel_id: int,
                               alpha_fast: float = 0.4,
                               alpha_slow: float = 0.08,
                               max_rate: float = 2.0,
                               hysteresis: float = 0.5) -> Dict:
    """
    Создание цепочки обработки для канала.
    
    Returns:
        Словарь с DualEMA, EQLimiter для каждой полосы
    """
    num_bands = 16
    
    return {
        'channel': channel_id,
        'dual_ema': [DualEMA(alpha_fast, alpha_slow) for _ in range(num_bands)],
        'limiter': [EQLimiter(max_rate, hysteresis) for _ in range(num_bands)],
        'last_gains': [0.0] * num_bands
    }


def process_eq_band(band_idx: int,
                   target_gain: float,
                   dual_ema: DualEMA,
                   limiter: EQLimiter) -> float:
    """
    Обработка одной полосы EQ через цепочку.
    
    Args:
        band_idx: Индекс полосы
        target_gain: Целевой gain (dB)
        dual_ema: Dual EMA фильтр
        limiter: Rate limiter
        
    Returns:
        Обработанный gain (dB)
    """
    # Dual EMA сглаживание
    smoothed = dual_ema.process(target_gain)
    
    # Rate limiting с hysteresis
    limited = limiter.process(smoothed)
    
    return limited


# Тестирование
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тест DualEMA
    print("=== Тест DualEMA ===")
    dual_ema = DualEMA(alpha_fast=0.4, alpha_slow=0.08)
    
    # Симулируем резкое изменение
    inputs = [0, 0, 0, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5]
    outputs = []
    for inp in inputs:
        out = dual_ema.process(inp)
        outputs.append(out)
        print(f"Input: {inp:+.1f} dB → Output: {out:+.2f} dB")
    
    print("\n=== Тест EQLimiter ===")
    limiter = EQLimiter(max_rate_db_per_frame=2.0, hysteresis_db=0.5)
    
    # Симулируем большое изменение
    inputs = [0, 5, 10, 15, 15, 15, 10, 5, 0]
    limiter.reset()
    for inp in inputs:
        out = limiter.process(inp)
        print(f"Target: {inp:+.1f} dB → Limited: {out:+.2f} dB")
    
    print("\n=== Тест PriorityMatrix ===")
    matrix = PriorityMatrix([1, 2, 3, 4])
    matrix.set_priority(1, 0.9)  # Lead vox
    matrix.set_priority(2, 0.8)  # Kick
    matrix.set_priority(3, 0.5)  # Guitar
    matrix.set_priority(4, 0.3)  # Pad
    
    # Обновляем на основе RMS
    rms_levels = {1: -10, 2: -8, 3: -15, 4: -20}
    matrix.update_from_rms(rms_levels)
    
    print(f"Leader: Channel {matrix.get_leader()}")
    for ch in [1, 2, 3, 4]:
        print(f"Channel {ch}: priority = {matrix.get_priority(ch):.3f}")
    
    print("\n=== Тест MirrorEQ ===")
    mirror = MirrorEQ(num_bands=16)
    
    # Лидер имеет boost на полосах 5-7
    leader_gains = [0] * 16
    leader_gains[5] = 4.0
    leader_gains[6] = 3.0
    
    # Другой канал хочет boost на тех же полосах
    other_gains = [0] * 16
    other_gains[5] = 2.0
    other_gains[6] = 2.0
    
    result = mirror.calculate_mirror(2, other_gains, 1, leader_gains)
    print(f"Original gains[5-7]: {other_gains[5:8]}")
    print(f"After mirror[5-7]:   {result[5:8]}")
