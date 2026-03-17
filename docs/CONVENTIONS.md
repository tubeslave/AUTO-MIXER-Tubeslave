# Соглашения по коду и DSP

## Python

### Общие правила
- Python 3.10+, type hints обязательны для всех public функций
- Docstrings: Google style
- Максимальная длина строки: 100 символов
- Linter: Ruff
- Импорты: `stdlib` → `third-party` → `local`, разделённые пустой строкой

### Именование
- Классы: `CamelCase` (`WingClient`, `AutoEQController`, `ThreadSafeMixerState`)
- Функции/методы: `snake_case` (`set_fader`, `get_lufs_level`)
- Константы: `UPPER_SNAKE_CASE` (`MAX_NOTCH_DEPTH_DB`, `FADER_0DB`)
- Приватные: `_prefix` (`_stop_receiver`, `_osc_throttle_hz`)
- Файлы модулей: `snake_case.py` (`auto_compressor.py`, `wing_client.py`)

### Async
- `asyncio` для всего I/O (WebSocket, HTTP, таймеры)
- `threading` ТОЛЬКО для блокирующего audio I/O (PyAudio callback)
- `asyncio.Lock` для shared state (НЕ `threading.Lock` в async контексте)
- При работе с mixer state — через `ThreadSafeMixerState` (copy-on-read)

### Обработка ошибок
```python
# Правильно: конкретные исключения, логирование
try:
    await self.wing_client.set_fader(channel, value_db)
except ConnectionError:
    logger.error("Wing connection lost", channel=channel)
    await self._reconnect()
except ValueError as e:
    logger.warning("Invalid fader value", channel=channel, value=value_db, error=str(e))

# Неправильно: голый except, подавление ошибок
try:
    self.wing_client.set_fader(channel, value_db)
except:
    pass
```

### JSON сериализация
Перед `json.dumps()` — всегда `convert_numpy_types()`:
```python
from server import convert_numpy_types
data = convert_numpy_types({"level": np.float64(-12.3)})
json.dumps(data)  # OK
```

## DSP стандарты

### LUFS (ITU-R BS.1770-4)
- K-Weighting: два biquad каскада
  - High-shelf: +4 dB, fc = 1681.974 Hz, Q = 0.7071
  - HPF: fc = 38.1355 Hz, Q = 0.5003
- Gating:
  - Absolute gate: -70 LUFS (блоки < -70 LUFS отбрасываются)
  - Relative gate: -10 LU от ungated LUFS (блоки ниже порога отбрасываются)
- Block size: 400 мс с 75% overlap (шаг = 100 мс)
- Формула: `LUFS = -0.691 + 10 * log10(sum(Gi * zi))` где Gi — channel weights

### True Peak (ITU-R BS.1770-4)
- 4x oversampling (48 kHz → 192 kHz)
- Lowpass FIR перед ресэмплингом
- True Peak limit: -1.0 dBTP (настраивается в config)

### EQ Biquads (Audio EQ Cookbook)
Все EQ-фильтры по Robert Bristow-Johnson:
```
b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

Peaking EQ:
  A  = sqrt(10^(dBgain/20))
  w0 = 2*pi*f0/Fs
  alpha = sin(w0)/(2*Q)
  b0 =   1 + alpha*A
  b1 =  -2*cos(w0)
  b2 =   1 - alpha*A
  a0 =   1 + alpha/A
  a1 =  -2*cos(w0)
  a2 =   1 - alpha/A
```

### GCC-PHAT (фазовое выравнивание)
- Аргументы: `gcc_phat(target_signal, reference_signal)` — **НЕ наоборот**
- Параболическая интерполяция для суб-сэмплной точности
- Максимальная задержка: ±10 мс (настраивается)
- Применение: delay compensation между микрофонами

### Компрессия
```python
# Правильная формула ratio
ratio = 1 + (max_ratio - 1) * factor  # factor ∈ [0, 1]

# НЕПРАВИЛЬНО (инвертированная):
ratio = max_ratio - (max_ratio - 1) * factor  # Перкуссия получит мягкую компрессию
```

### Gate
- Hysteresis: если объявлен (напр. 3 дБ), ОБЯЗАН применяться
- Порог открытия: threshold
- Порог закрытия: threshold - hysteresis
- Hold time: минимальное время удержания открытого состояния

### Единицы измерения
| Параметр | Единица | Пример |
|----------|---------|--------|
| Fader level | dBFS | -12.0 dBFS |
| Gain | dB | +24.0 dB |
| LUFS | LUFS | -18.0 LUFS |
| True Peak | dBTP | -1.0 dBTP |
| Частота EQ | Hz | 1000.0 Hz |
| Gain EQ | dB | +3.0 dB |
| Q factor | безразмерный | 1.41 |
| Ratio | X:1 | 4.0:1 |
| Attack/Release | мс | 10.0 мс |
| Threshold | dBFS | -20.0 dBFS |
| Pan | -1.0 … +1.0 | 0.0 = центр |
| Задержка | мс | 2.5 мс |
| Sample rate | Hz | 48000 Hz |

### Целевые LUFS по инструментам (live sound)
| Инструмент | Target LUFS | Допуск |
|------------|-------------|--------|
| Kick | -18 | ±2 |
| Snare | -20 | ±2 |
| Hi-Hat | -26 | ±2 |
| Toms | -22 | ±2 |
| Overhead | -24 | ±2 |
| Bass | -18 | ±2 |
| Electric Guitar | -20 | ±2 |
| Acoustic Guitar | -22 | ±2 |
| Keys/Synth | -22 | ±2 |
| Lead Vocal | -16 | ±1 |
| Backing Vocal | -20 | ±2 |

## Frontend (React)

- React 18, functional components + hooks
- CSS: отдельные .css файлы для каждого компонента
- WebSocket: единый сервис в `services/`
- State: локальный, синхронизация через WS messages
- Файлы: `CamelCase.js` / `CamelCase.css` для компонентов

## Git

- Ветка: `main` (единственная)
- Commit messages: на английском, императив (`Fix GCC-PHAT argument order`, `Add feedback detector`)
- Каждый коммит должен проходить `pytest`
- Не коммитить: `__pycache__`, `node_modules`, `venv`, `.DS_Store`, модели ML (> 100 MB)
