# Virtual Wing Mixer

Виртуальный микшер Behringer Wing для тестирования Auto Mixer Tubeslave.

## 🎯 Назначение

- Тестирование Auto Mixer без реального оборудования
- Отладка OSC команд
- Симуляция различных сценариев (drums, vocals, full mix)
- Логирование всех команд

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install python-osc websockets

# Запуск виртуального микшера
cd virtual_mixer
python test_integration.py
```

## 📡 Подключение

### Настройка Auto Mixer:
- **Mixer IP**: `localhost`
- **OSC Port**: `2222`
- **WebSocket**: `ws://localhost:8766`

## 🎛️ Компоненты

### 1. `virtual_mixer.py`
Симуляция Wing OSC API:
- 32 input channels
- 16 group busses
- 16 matrix busses
- Master LR
- Полная поддержка OSC команд

### 2. `test_signal_generator.py`
Генератор тестовых сигналов:
- Sine waves
- Pink noise
- Drum hits (kick, snare)
- Vocal simulation
- Mixed signals

### 3. `test_integration.py`
Интеграционный тест:
- Запускает виртуальный микшер
- Генерирует сигналы
- WebSocket bridge для мониторинга
- Логирование команд

## 📊 OSC Команды

### Input Channels
```
/ch/{01-32}/mix/fader    # Fader 0.0-1.0
/ch/{01-32}/mix/on       # On/Off 0/1
/ch/{01-32}/mix/pan      # Pan -1.0 to 1.0
/ch/{01-32}/mix/mute     # Mute 0/1
/ch/{01-32}/preamp/gain  # Gain -12 to +60 dB
/ch/{01-32}/eq/on        # EQ On/Off
/ch/{01-32}/eq/{1-4}/gain # EQ band gain
/ch/{01-32}/dyn/on       # Compressor On/Off
/ch/{01-32}/dyn/thr      # Compressor threshold
/ch/{01-32}/gate/on      # Gate On/Off
/ch/{01-32}/gate/thr     # Gate threshold
```

### Busses
```
/bus/{01-16}/mix/fader   # Group fader
/mtx/{01-16}/mix/fader   # Matrix fader
/main/st/mix/fader       # Master fader
```

## 🧪 Режимы тестирования

### Через WebSocket:
```json
{"type": "set_test_mode", "mode": "silence"}   # Все каналы тихие
{"type": "set_test_mode", "mode": "drums"}     # Только ударные
{"type": "set_test_mode", "mode": "full_mix"}  # Полный микс
{"type": "start_song", "bpm": 120}             # Запустить "песню"
```

## 📋 Структура проекта

```
virtual_mixer/
├── virtual_mixer.py          # Основной класс микшера
├── test_signal_generator.py  # Генератор сигналов
├── test_integration.py       # Интеграционный тест
├── osc_log.txt              # Лог OSC команд (создаётся автоматически)
└── README.md                # Этот файл
```

## 🎮 Пример использования

```python
from virtual_mixer import VirtualWingMixer
import asyncio

async def test():
    # Создаём микшер
    mixer = VirtualWingMixer(osc_port=2222)
    
    # Запускаем OSC сервер
    transport = await mixer.start()
    
    # Устанавливаем fader
    mixer.input_channels[1].fader = 0.75  # CH01 fader
    mixer.input_channels[1].on = True
    
    # Получаем состояние
    state = mixer.get_state()
    print(f"CH01 fader: {state['inputs']['1']['fader_db']:.1f} dB")
    
    # Останавливаем
    transport.close()

asyncio.run(test())
```

## 🔍 Отладка

### Просмотр OSC команд:
```bash
tail -f osc_log.txt
```

### WebSocket мониторинг:
```javascript
const ws = new WebSocket('ws://localhost:8766');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Mixer state:', data);
};
```

## ⚠️ Ограничения

- Нет реального аудио (только уровни в dB)
- Нет задержек обработки (instant response)
- Упрощённые алгоритмы EQ/Dynamics

## ✅ Интеграция с Auto Mixer

1. Запусти виртуальный микшер:
   ```bash
   python test_integration.py
   ```

2. Настрой Auto Mixer:
   - IP: `localhost`
   - Port: `2222`

3. Тестируй все модули без реального Wing!

## 📝 Лог изменений

- v1.0 - Базовая OSC симуляция
- v1.1 - Добавлен генератор сигналов
- v1.2 - WebSocket bridge для мониторинга

## 🤝 Ссылки

- Auto Mixer: `/Users/dmitrijvolkov/AUTO MIXER Tubeslave`
- Документация Wing OSC: https://behringer.com/wing/osc
