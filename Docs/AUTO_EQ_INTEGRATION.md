# Auto EQ Integration Guide

## Обзор

Этот документ описывает интеграцию новых компонентов (`auto_eq_processing.py`) с существующей системой (`auto_eq.py`) согласно методу из дебатов Kimi.

## Архитектура системы

### Новые компоненты

```
auto_eq_processing.py
├── DualEMA              # Двухфазное сглаживание (fast/slow)
├── EQLimiter            # Rate limiter + hysteresis
├── PriorityMatrix       # Матрица приоритетов каналов
├── MirrorEQ             # Mirror equalization
└── LinkwitzRileyFilter  # 16-полосный кроссовер
```

### Цепочка обработки

```
┌─────────────────────────────────────────────────────────────┐
│                    Auto EQ Processing Chain                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input → [16-band LR4] → [Analysis] → [Dual EMA]           │
│                                        (fast/slow)          │
│                                         ↓                   │
│                              [Hysteresis ±0.5dB]            │
│                                         ↓                   │
│                              [Rate Limiter ±2dB/frame]      │
│                                         ↓                   │
│                              [Priority Matrix]              │
│                                         ↓                   │
│                              [Mirror EQ]                    │
│                                         ↓                   │
│                              [OSC Output]                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Параметры системы

| Параметр | Значение | Описание |
|----------|----------|----------|
| Полосы | 16 | Linkwitz-Riley 4-го порядка |
| Частоты | 31.5Hz - 16kHz | 1/3 октавные |
| EMA fast α | 0.4 | ~2.5 кадра реакция |
| EMA slow α | 0.08 | ~12 кадров реакция |
| Hysteresis | ±0.5 dB | Зона нечувствительности |
| Rate limit | ±2 dB/frame | Максимальное изменение |
| Latency | <20ms | Общая задержка |
| CPU | ≤6% | Целевая нагрузка |

## Использование

### 1. Базовое использование DualEMA

```python
from auto_eq_processing import DualEMA

# Создание
ema = DualEMA(alpha_fast=0.4, alpha_slow=0.08)

# Обработка
target_gain = 10.0  # dB
smoothed_gain = ema.process(target_gain)
```

### 2. Rate Limiting

```python
from auto_eq_processing import EQLimiter

# Создание
limiter = EQLimiter(
    max_rate_db_per_frame=2.0,
    hysteresis_db=0.5
)

# Обработка
limited_gain = limiter.process(target_gain)
```

### 3. Полная цепочка для полосы

```python
from auto_eq_processing import process_eq_band

# Создание цепочки
chain = create_eq_processing_chain(channel_id=1)

# Обработка одной полосы
band_idx = 5
target_gain = 8.0  # dB
output_gain = process_eq_band(
    band_idx,
    target_gain,
    chain['dual_ema'][band_idx],
    chain['limiter'][band_idx]
)
```

### 4. Многоканальная обработка с приоритетами

```python
from auto_eq_processing import PriorityMatrix, MirrorEQ

# Матрица приоритетов
channels = [1, 2, 3, 4]
priority_matrix = PriorityMatrix(channels)

# Установка базовых приоритетов
priority_matrix.set_priority(1, 0.9)  # Lead vox
priority_matrix.set_priority(2, 0.8)  # Kick
priority_matrix.set_priority(3, 0.5)  # Guitar
priority_matrix.set_priority(4, 0.3)  # Pad

# Обновление на основе RMS
rms_levels = {1: -10, 2: -8, 3: -15, 4: -20}
priority_matrix.update_from_rms(rms_levels)

leader = priority_matrix.get_leader()  # Channel 1
```

### 5. Mirror EQ

```python
from auto_eq_processing import MirrorEQ

mirror = MirrorEQ(num_bands=16)

# Лидер имеет boost
leader_gains = [0.0] * 16
leader_gains[5] = 4.0
leader_gains[6] = 3.0

# Другой канал
other_gains = [0.0] * 16
other_gains[5] = 2.0

# Применение mirror
result_gains = mirror.calculate_mirror(
    channel=2,
    band_gains=other_gains,
    leader_channel=1,
    leader_gains=leader_gains
)
# result_gains[5] будет отрицательным (cut)
```

## Интеграция с auto_eq.py

### Модификация AutoEQController

```python
# В __init__ добавить:
from auto_eq_processing import DualEMA, EQLimiter

class AutoEQController:
    def __init__(self, ...):
        # ... существующий код ...
        
        # Новые компоненты
        self.dual_ema = None
        self.eq_limiter = None
        self._init_processing_chain()
    
    def _init_processing_chain(self):
        """Инициализация цепочки обработки."""
        num_bands = 16
        self.dual_ema = [DualEMA() for _ in range(num_bands)]
        self.eq_limiter = [EQLimiter() for _ in range(num_bands)]
```

### Модификация расчёта коррекций

```python
def _calculate_and_notify_corrections(self, spectral_data: SpectralData):
    # ... существующий код для расчёта ...
    
    # Применяем цепочку обработки
    processed_bands = []
    for i, band in enumerate(corrections):
        # Dual EMA
        smoothed = self.dual_ema[i].process(band.gain)
        
        # Rate limiter с hysteresis
        limited = self.eq_limiter[i].process(smoothed)
        
        band.gain = limited
        processed_bands.append(band)
```

### Интеграция Priority и Mirror

```python
# В MultiChannelAutoEQController
def __init__(self, ...):
    # ... существующий код ...
    self.priority_matrix = PriorityMatrix([])
    self.mirror_eq = MirrorEQ()

def start_multi_channel(self, channels, ...):
    # Инициализация priority matrix
    self.priority_matrix = PriorityMatrix(channels)
    # ...

def _on_channel_spectrum(self, channel: int, spectral_data: SpectralData):
    # ... расчёт коррекций ...
    
    # Обновление приоритетов
    rms_levels = self._get_channel_rms()
    self.priority_matrix.update_from_rms(rms_levels)
    
    # Mirror EQ
    all_gains = self._get_all_channel_gains()
    resolved_gains = self.mirror_eq.resolve_conflicts(
        all_gains, self.priority_matrix
    )
```

## Жанровые пресеты

Пресеты находятся в `config/eq_presets/`:

```
config/eq_presets/
├── pop.json   # Bright vocals, punchy lows
├── rock.json  # Forward mids, tight bass
├── edm.json   # Heavy sub, wide stereo
└── jazz.json  # Natural balance, warm
```

### Загрузка пресета

```python
import json

with open('config/eq_presets/pop.json') as f:
    preset = json.load(f)

# Применение
controller.set_profile(preset['name'])
```

## OSC Команды

```
/ch/{01-32}/eq/1/g      # Gain полосы 1
/ch/{01-32}/eq/1/f      # Частота полосы 1
/ch/{01-32}/eq/1/q      # Q полосы 1
/ch/{01-32}/dyn/thr     # Compressor threshold
/ch/{01-32}/mix/fader   # Fader level
/xremote                 # Подписка на обновления
```

## Тестирование

### Unit тесты

```bash
cd /Users/dmitrijvolkov/AUTO\ MIXER\ Tubeslave/backend
python auto_eq_processing.py
```

### Интеграционный тест

```python
# Тест полной цепочки
from auto_eq_processing import *

# Создание компонентов
chain = create_eq_processing_chain(channel_id=1)

# Тестовый сигнал: резкий скачок
inputs = [0, 0, 0, 10, 10, 10, 10, 10]
for inp in inputs:
    out = process_eq_band(0, inp, chain['dual_ema'][0], chain['limiter'][0])
    print(f"{inp:+.1f} → {out:+.2f} dB")
```

## Дополнительные ресурсы

### Научная база

- Perez Gonzalez & Reiss (2009) — Automatic EQ balancing
- Hafezi & Reiss (2015) — Multitrack EQ with masking
- De Man & Reiss (2013) — Analysis of professional mixes

### Документация

- `auto_eq.py` — Основной модуль
- `auto_eq_processing.py` — Новые компоненты
- `config/eq_presets/` — Жанровые пресеты

## TODO

- [ ] Интегрировать DualEMA в AutoEQController
- [ ] Добавить PriorityMatrix в MultiChannelAutoEQController
- [ ] Реализовать MirrorEQ для разделения спектра
- [ ] Тестирование с реальным микшером
- [ ] Настройка параметров по жанрам
