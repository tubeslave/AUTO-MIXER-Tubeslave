# Safe Static Gain Calibration - README

## Обзор

Новый метод **Safe Static Gain Calibration** решает проблемы текущего realtime gain staging подхода:

### Проблемы старого метода (непрерывная коррекция):

1. **Bleed Problem**: Микрофон вокалиста ловит барабаны в паузах → алгоритм усиливает фон → при вступлении перегруз
2. **Игнорирование Crest Factor**: Барабаны имеют высокие пики но низкий RMS → попытка натянуть RMS до -18 LUFS загоняет пики в клиппинг
3. **Нестабильность**: Постоянные микро-подстройки (+0.2dB, -0.3dB) слышны как модуляция громкости

### Решение - "Safe Static Gain":

Анализ сигнала за период (10-20 сек) → расчет безопасного gain → применение один раз

## Ключевые компоненты

### 1. Фильтр полезного сигнала (Anti-Bleed/Noise Gate)
```python
noise_gate_threshold = -40.0 dBFS
```
Не учитываем моменты когда сигнал ниже порога. Если за весь период канал не превысил порог "уверенного сигнала" (-30dB), помечаем как "Silent/Bleed Only" и не меняем gain.

### 2. Двойной лимитер (LUFS + TruePeak)
```python
Delta_LUFS = Target_LUFS (-18) - Measured_Integrated_LUFS
Delta_Peak = Max_Allowed_Peak (-3) - Measured_Max_TruePeak
Final_Correction = min(Delta_LUFS, Delta_Peak)
```

**Пример**: Малый барабан: -24 LUFS, пики -6dB
- По LUFS: +6dB
- По Пикам: только +3dB
- **Итог**: +3dB (сигнал будет -21 LUFS, но не клипанет)

### 3. Учет Crest Factor (Динамики)
```python
Crest Factor = Peak - RMS
```

- **Crest > 12dB** (барабаны, перкуссия): Приоритет пиковому запасу
- **Crest < 6dB** (дисторшн, синты): Приоритет LUFS
- **6-12dB**: Берем минимум

## Использование

### Базовый пример
```python
from lufs_gain_staging import SafeGainCalibrator

calibrator = SafeGainCalibrator(
    mixer_client=your_mixer,
    sample_rate=48000,
    config={
        'automation': {
            'safe_gain_calibration': {
                'target_lufs': -18.0,          # Целевой LUFS
                'max_peak_limit': -3.0,        # Максимальный пик
                'noise_gate_threshold': -40.0, # Порог шума
                'min_signal_presence': 0.05,   # Минимум 5% активности
                'learning_duration_sec': 15.0  # Длительность анализа
            }
        }
    }
)

calibrator.add_channel(audio_channel=1, mixer_channel=1)
calibrator.start_analysis()
```

### Workflow
```
1. Start Check      → calibrator.start_analysis()
2. Listening (15s)  → музыканты играют громкую часть
                     → система только читает метрики
3. Calculation      → автоматически после завершения анализа
4. Get Suggestions  → calibrator.get_suggestions()
5. Apply            → calibrator.apply_corrections()
```

### Обработка аудио
```python
while calibrator.get_status()['state'] == 'learning':
    audio_data = capture_audio()
    calibrator.process_audio(channel_id, audio_data)
```

### Получение результатов
```python
suggestions = calibrator.get_suggestions()

# Формат результата:
{
    1: {
        'channel': 1,
        'peak_db': -5.0,
        'lufs': -22.0,
        'crest_factor_db': 17.0,
        'signal_presence': 85.0,
        'suggested_gain_db': 4.0,
        'limited_by': 'peak',  # или 'lufs', 'lufs_priority', 'silent_channel'
        'samples_analyzed': 72000,
        'active_samples': 61200
    }
}
```

### Применение коррекций
```python
calibrator.apply_corrections()  # Применить ко всем каналам

calibrator.apply_corrections([1, 3, 5])  # Только к выбранным
```

## State Machine

Состояния системы:
- **IDLE**: Готов к запуску анализа
- **LEARNING**: Идет сбор статистики
- **READY**: Анализ завершен, рекомендации готовы
- **APPLYING**: Применение коррекций

## Преимущества перед realtime подходом

| Проблема | Realtime | Safe Static |
|----------|----------|-------------|
| Bleed усиления | ✗ Усиливает фон | ✓ Фильтрует через gate |
| Клиппинг перкуссии | ✗ Игнорирует пики | ✓ Двойной лимитер |
| Модуляция звука | ✗ Постоянные подстройки | ✓ Один раз |
| Тип инструмента | ✗ Один подход для всех | ✓ Учет Crest Factor |
| Контроль | ✗ Автопилот | ✓ Ассистент (предлагает) |

## Интеграция с server.py

SafeGainCalibrator может быть интегрирован как альтернативный режим к LUFSGainStagingController:

```python
if mode == 'realtime':
    controller = LUFSGainStagingController(...)
    controller.start_realtime_correction()
elif mode == 'safe_static':
    calibrator = SafeGainCalibrator(...)
    calibrator.start_analysis()
    # wait...
    suggestions = calibrator.get_suggestions()
    calibrator.apply_corrections()
```

## Конфигурация в config.yaml

```yaml
automation:
  safe_gain_calibration:
    target_lufs: -18.0
    max_peak_limit: -3.0
    noise_gate_threshold: -40.0
    min_signal_presence: 0.05
    learning_duration_sec: 15.0
```

## Тестирование

Запуск примера:
```bash
python backend/example_safe_gain.py
```

Ожидаемый результат:
- Прогресс анализа 0-100%
- Отчет по каждому каналу с рекомендациями
- Корректные значения Crest Factor для разных типов сигналов
