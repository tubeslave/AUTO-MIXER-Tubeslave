"""
Enhanced Gain Agent с инструмент-специфичными профилями
для AUTO-MIXER Tubeslave v2.0
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Типы инструментов с разными настройками"""
    VOCALS = "vocals"
    KICK = "kick"
    SNARE = "snare"
    TOM = "tom"
    HI_HAT = "hi_hat"
    BASS = "bass"
    GUITAR = "guitar"
    KEYS = "keys"
    PLAYBACK = "playback"
    LINE = "line"
    UNKNOWN = "unknown"


# Профили инструментов (из книг по звукорежиссуре)
INSTRUMENT_PROFILES = {
    InstrumentType.VOCALS: {
        "target_lufs": -18.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 80,
        "comp_threshold": -18.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 5.0,
        "comp_release_ms": 150.0,
        "smoothing_fast": 0.5,
        "smoothing_slow": 0.1,
    },
    InstrumentType.KICK: {
        "target_lufs": -15.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -40.0,
        "hpf_freq": 40,
        "comp_threshold": -12.0,
        "comp_ratio": 4.0,
        "comp_attack_ms": 2.0,
        "comp_release_ms": 80.0,
        "smoothing_fast": 0.7,
        "smoothing_slow": 0.2,
    },
    InstrumentType.SNARE: {
        "target_lufs": -16.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -40.0,
        "hpf_freq": 100,
        "comp_threshold": -15.0,
        "comp_ratio": 4.0,
        "comp_attack_ms": 3.0,
        "comp_release_ms": 100.0,
        "smoothing_fast": 0.6,
        "smoothing_slow": 0.15,
    },
    InstrumentType.TOM: {
        "target_lufs": -16.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -40.0,
        "hpf_freq": 60,
        "comp_threshold": -15.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 5.0,
        "comp_release_ms": 150.0,
        "smoothing_fast": 0.6,
        "smoothing_slow": 0.15,
    },
    InstrumentType.HI_HAT: {
        "target_lufs": -18.0,
        "true_peak_limit": -6.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 400,
        "comp_threshold": -20.0,
        "comp_ratio": 2.0,
        "comp_attack_ms": 1.0,
        "comp_release_ms": 50.0,
        "smoothing_fast": 0.6,
        "smoothing_slow": 0.2,
    },
    InstrumentType.BASS: {
        "target_lufs": -15.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -40.0,
        "hpf_freq": 30,
        "comp_threshold": -15.0,
        "comp_ratio": 6.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 100.0,
        "smoothing_fast": 0.5,
        "smoothing_slow": 0.15,
    },
    InstrumentType.GUITAR: {
        "target_lufs": -20.0,
        "true_peak_limit": -6.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 80,
        "comp_threshold": -20.0,
        "comp_ratio": 4.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 200.0,
        "smoothing_fast": 0.4,
        "smoothing_slow": 0.1,
    },
    InstrumentType.KEYS: {
        "target_lufs": -18.0,
        "true_peak_limit": -6.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 40,
        "comp_threshold": -20.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 15.0,
        "comp_release_ms": 250.0,
        "smoothing_fast": 0.4,
        "smoothing_slow": 0.1,
    },
    InstrumentType.PLAYBACK: {
        "target_lufs": -16.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -60.0,
        "hpf_freq": 20,
        "comp_threshold": -18.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 200.0,
        "smoothing_fast": 0.4,
        "smoothing_slow": 0.1,
    },
    InstrumentType.LINE: {
        "target_lufs": -18.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 40,
        "comp_threshold": -20.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 200.0,
        "smoothing_fast": 0.4,
        "smoothing_slow": 0.1,
    },
    InstrumentType.UNKNOWN: {
        "target_lufs": -18.0,
        "true_peak_limit": -3.0,
        "gate_threshold_lufs": -45.0,
        "hpf_freq": 40,
        "comp_threshold": -20.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 200.0,
        "smoothing_fast": 0.4,
        "smoothing_slow": 0.1,
    },
}


class EnhancedGainAgent:
    """
    Улучшенный Gain Agent с профилями инструментов
    и адаптивным сглаживанием
    """
    
    def __init__(self, osc_client=None):
        self.osc_client = osc_client
        
        # Конфигурация каналов
        self.channel_config = {}
        
        # Состояние каналов
        self.channel_states = {}
        
        # Статистика
        self.stats = {
            'total_corrections': 0,
            'peak_warnings': 0,
            'gate_triggers': 0
        }
        
        # Настройки по умолчанию
        self.default_profile = INSTRUMENT_PROFILES[InstrumentType.UNKNOWN]
        
        logger.info("EnhancedGainAgent initialized")
    
    def configure_channel(self, channel: int, instrument_type: str):
        """Настроить канал с типом инструмента"""
        try:
            instrument = InstrumentType(instrument_type.lower())
        except ValueError:
            instrument = InstrumentType.UNKNOWN
        
        self.channel_config[channel] = instrument
        
        # Инициализируем состояние
        self.channel_states[channel] = {
            'instrument': instrument.value,
            'current_trim': 0.0,
            'target_trim': 0.0,
            'smoothed_trim': 0.0,
            'last_lufs': -60.0,
            'last_peak': -60.0,
            'peak_hold': -60.0,
            'is_active': False,
            'is_limited': False,
            'gain_reduction': 0.0,
            'corrections_count': 0
        }
        
        logger.info(f"Channel {channel} configured as {instrument.value}")
    
    def get_profile(self, channel: int) -> Dict[str, Any]:
        """Получить профиль канала"""
        instrument = self.channel_config.get(channel, InstrumentType.UNKNOWN)
        return INSTRUMENT_PROFILES[instrument]
    
    def process(self, channel: int, lufs: float, peak: float) -> Dict[str, Any]:
        """
        Обработать канал и вернуть рекомендуемые настройки
        
        Returns:
            dict с ключами: trim_db, hpf_freq, comp_settings, warnings
        """
        state = self.channel_states.get(channel)
        if not state:
            return {'error': 'Channel not configured'}
        
        profile = self.get_profile(channel)
        
        # Обновляем данные
        state['last_lufs'] = lufs
        state['last_peak'] = peak
        
        # Peak hold (удерживаем пик)
        if peak > state['peak_hold']:
            state['peak_hold'] = peak
        else:
            state['peak_hold'] -= 0.5  # decay
        
        # Gate (активен ли канал)
        was_active = state['is_active']
        state['is_active'] = lufs > profile['gate_threshold_lufs']
        
        if not was_active and state['is_active']:
            self.stats['gate_triggers'] += 1
        
        # Проверка на клиппинг
        if peak >= profile['true_peak_limit']:
            state['is_limited'] = True
            self.stats['peak_warnings'] += 1
        else:
            state['is_limited'] = False
        
        # Расчёт коррекции gain
        if state['is_active']:
            target_lufs = profile['target_lufs']
            lufs_error = target_lufs - lufs
            
            # Ограничение по пику
            max_gain = profile['true_peak_limit'] - state['peak_hold'] - 1.0
            
            # Применяем коррекцию
            correction = min(lufs_error, max_gain)
            
            # Сглаживание
            smoothing = profile['smoothing_fast'] if lufs_error > 0 else profile['smoothing_slow']
            new_trim = state['smoothed_trim'] + smoothing * (correction - state['smoothed_trim'])
            
            state['target_trim'] = correction
            state['smoothed_trim'] = new_trim
            
            if abs(new_trim - state['current_trim']) > 0.5:
                state['corrections_count'] += 1
                self.stats['total_corrections'] += 1
            
            state['current_trim'] = new_trim
        
        # Формируем результат
        result = {
            'channel': channel,
            'instrument': state['instrument'],
            'trim_db': state['current_trim'],
            'target_lufs': profile['target_lufs'],
            'current_lufs': lufs,
            'peak': peak,
            'is_active': state['is_active'],
            'is_limited': state['is_limited'],
            'hpf_freq': profile['hpf_freq'],
            'comp_threshold': profile['comp_threshold'],
            'comp_ratio': profile['comp_ratio'],
            'comp_attack': profile['comp_attack_ms'],
            'comp_release': profile['comp_release_ms'],
            'warnings': []
        }
        
        # Добавляем предупреждения
        if state['is_limited']:
            result['warnings'].append('Peak limiting active!')
        
        if not state['is_active']:
            result['warnings'].append('Channel below gate threshold')
        
        if lufs > -10:
            result['warnings'].append('Very hot signal!')
        
        return result
    
    def get_all_channels_status(self) -> Dict[str, Any]:
        """Получить статус всех каналов"""
        return {
            'channels': self.channel_states,
            'stats': self.stats,
            'profiles': {
                ch: self.get_profile(ch) 
                for ch in self.channel_config.keys()
            }
        }
    
    def apply_to_mixer(self, channel: int, osc_client):
        """Применить настройки к микшеру через OSC"""
        state = self.channel_states.get(channel)
        if not state:
            return
        
        # Trim/Gain
        osc_client.send(f"/ch/{channel}/trim", state['current_trim'])
        
        # HPF
        profile = self.get_profile(channel)
        osc_client.send(f"/ch/{channel}/eq/hp/freq", profile['hpf_freq'])


class DuganAutomixer:
    """
    Dugan Speech System - классический алгоритм gain sharing
    для автоматического микширования нескольких микрофонов
    """
    
    def __init__(self, num_channels: int = 8, gain_limit_db: float = -6.0):
        self.num_channels = num_channels
        self.gain_limit_db = gain_limit_db
        self.master_gain = 1.0
        self.channel_gains = [1.0] * num_channels
        
    def calculate(self, channel_levels_db: List[float]) -> List[float]:
        """
        Расчёт gain для каждого канала по Dugan алгоритму
        
        Формула:
        Gain = Target / (Sum of all channel levels)
        
        Args:
            channel_levels_db: список уровней каналов в dB
            
        Returns:
            Список корректировок gain для каждого канала
        """
        # Конвертируем dB в линейные значения
        linear_levels = [10**(level/20) for level in channel_levels_db]
        
        # Сумма всех каналов
        sum_linear = sum(linear_levels)
        
        # Целевой уровень
        target_linear = 10**(self.gain_limit_db/20)
        
        # Dugan формула
        if sum_linear > 0.0001:  # избегаем деления на ноль
            new_master = target_linear / sum_linear
            
            # Ограничиваем максимальный gain (+12dB)
            max_gain = 10**(12/20)
            new_master = min(new_master, max_gain)
            
            # Плавное изменение (smoothing)
            self.master_gain = self.master_gain * 0.3 + new_master * 0.7
        
        # Рассчитываем gain для каждого канала
        output_gains = []
        for i, level_db in enumerate(channel_levels_db):
            gain_db = 20 * np.log10(self.master_gain + 0.0001)
            output_gains.append(gain_db)
            self.channel_gains[i] = self.master_gain
        
        return output_gains
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус Dugan микшера"""
        return {
            'master_gain': self.master_gain,
            'channel_gains': self.channel_gains,
            'gain_limit_db': self.gain_limit_db
        }


# Пример использования
if __name__ == "__main__":
    # Создаём агент
    gain_agent = EnhancedGainAgent()
    
    # Настраиваем каналы
    gain_agent.configure_channel(1, "kick")
    gain_agent.configure_channel(2, "snare")
    gain_agent.configure_channel(3, "bass")
    gain_agent.configure_channel(4, "vocals")
    gain_agent.configure_channel(5, "guitar")
    
    # Обрабатываем сигнал
    for ch in range(1, 6):
        result = gain_agent.process(ch, lufs=-20, peak=-8)
        print(f"Channel {ch}: trim={result['trim_db']:.1f}dB, warnings={result['warnings']}")
    
    # Dugan пример
    dugan = DuganAutomixer(num_channels=4)
    levels = [-20, -25, -30, -60]  # 4 канала, последний молчит
    gains = dugan.calculate(levels)
    print(f"Dugan gains: {gains}")
