# AUTO-MIXER Tubeslave — Инструкция по установке и запуску

## Целевая конфигурация

- **Компьютер**: MacBook Pro с чипом Apple M3 Max (macOS 14+)
- **Микшерный пульт**: Allen & Heath dLive (IP: `192.168.3.70`, порт: `51328`)
- **Аудиоинтерфейс**: Waves SoundGrid (для многоканального приёма аудио)
- **Протокол управления**: MIDI over TCP/IP (dLive)
- **Аудиопротокол**: SoundGrid / CoreAudio

---

## 1. Подготовка оборудования

### Сеть

1. Подключите MacBook к той же сети, что и dLive.
2. Настройте статический IP на MacBook в подсети `192.168.3.x` (например, `192.168.3.100`).
3. Убедитесь, что пульт dLive доступен: `ping 192.168.3.70`.
4. Порт `51328` (MIDI over TCP) должен быть открыт на dLive.

### SoundGrid

1. Установите **Waves SoundGrid Studio** на MacBook.
2. Подключите SoundGrid-сервер (например, SoundGrid Server One) к сети.
3. В SoundGrid Studio назначьте входы с I/O стейджбокса dLive на виртуальные каналы.
4. Убедитесь, что SoundGrid-драйвер видится в системных настройках звука macOS как аудиоустройство.

### dLive

1. В настройках dLive (Utility → MIDI) включите **TCP MIDI** (порт 51328).
2. Назовите каналы в dLive осмысленными именами — система использует их для определения инструментов:
   - `Kick`, `Snare`, `HiHat`, `Tom 1`, `Tom 2`, `OH L`, `OH R`
   - `Bass`, `E.Gtr`, `A.Gtr`, `Keys`
   - `Lead Vox`, `BVox 1`, `BVox 2`
   - Русские имена тоже поддерживаются: `Бочка`, `Малый`, `Бас`, `Вокал`

---

## 2. Установка на MacBook M3 Max

### 2.1. Установка Python

```bash
# Проверьте версию Python (нужен 3.10+)
python3 --version

# Если Python не установлен:
brew install python@3.12
```

### 2.2. Клонирование проекта

```bash
git clone <repository-url> AUTO-MIXER-Tubeslave
cd AUTO-MIXER-Tubeslave
```

### 2.3. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.4. Установка зависимостей

```bash
# Основные зависимости
pip install -r backend/requirements.txt

# Для macOS M3: PyTorch с поддержкой MPS (Apple Silicon GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Опционально: PortAudio для PyAudio (если нужен PyAudio вместо sounddevice)
brew install portaudio
pip install pyaudio
```

### 2.5. Проверка установки

```bash
# Проверка, что все модули импортируются:
cd AUTO-MIXER-Tubeslave
PYTHONPATH=backend python3 -c "
from dlive_client import DLiveClient
from audio_capture import AudioCapture, list_audio_devices
from auto_soundcheck_engine import AutoSoundcheckEngine
print('All modules loaded successfully!')
print('Available audio devices:')
for d in list_audio_devices():
    print(f'  [{d[\"index\"]}] {d[\"name\"]} ({d[\"max_input_channels\"]}ch)')
"
```

---

## 3. Настройка конфигурации

Конфигурация находится в `config/automixer.yaml`. Основные параметры уже настроены для dLive + SoundGrid:

```yaml
mixer:
  type: "dlive"
  ip: "192.168.3.70"      # IP адрес dLive
  port: 51328              # MIDI over TCP порт
  protocol: "midi_tcp"
  tls: false

audio:
  sample_rate: 48000
  block_size: 1024
  channels: 48             # Количество входных каналов
  source: "soundgrid"      # Тип аудиоисточника
  device_name: "soundgrid" # Паттерн для поиска устройства
```

Если IP пульта или аудиоустройство отличаются — отредактируйте этот файл.

---

## 4. Запуск

### 4.1. Быстрый запуск — автообнаружение (рекомендуемый)

```bash
cd AUTO-MIXER-Tubeslave
source venv/bin/activate

# Полный автомат: сканирует сеть → находит пульт → подключается → работает
python3 start_soundcheck.py
```

При запуске без параметров система:
1. Отправляет UDP broadcast `WING?` на порт 2222 (для обнаружения Behringer WING)
2. Пробует TCP-подключение к типичным IP на порт 51328 (для обнаружения dLive)
3. Подключается к первому найденному пульту
4. Определяет аудиоустройство (SoundGrid > Dante > системное)
5. Запускает автоматический саундчек

### 4.2. Только сканирование сети

```bash
# Показать все найденные пульты без подключения
python3 start_soundcheck.py --scan-only

# Полное сканирование подсети (медленнее, но тщательнее)
python3 start_soundcheck.py --scan-only --full-scan
```

### 4.3. Запуск с указанием IP

```bash
# Прямое подключение к конкретному IP (без сканирования)
python3 start_soundcheck.py --ip 192.168.3.70 --no-discover

# Указать IP как предпочтительный (сначала проверит его, потом сканирует)
python3 start_soundcheck.py --ip 192.168.3.70

# Указать IP и аудиоустройство явно
python3 start_soundcheck.py --ip 192.168.3.70 --audio-device soundgrid --channels 48

# Только анализ без применения (dry run)
python3 start_soundcheck.py --no-apply

# Подробный вывод
python3 start_soundcheck.py --log-level DEBUG
```

### 4.3. Запуск с веб-интерфейсом

```bash
# Терминал 1: Backend сервер
cd AUTO-MIXER-Tubeslave
source venv/bin/activate
PYTHONPATH=backend python3 backend/server.py

# Терминал 2: Frontend (Electron)
cd AUTO-MIXER-Tubeslave/frontend
npm install
npm start
```

### 4.5. Полный список параметров CLI

```
python3 start_soundcheck.py --help

Опции:
  --mixer {dlive,wing}    Тип пульта (по умолчанию: авто-определение)
  --ip IP                 IP адрес пульта (по умолчанию: авто-обнаружение)
  --port PORT             Порт (по умолчанию: авто)
  --tls                   Использовать TLS (для dLive)
  --no-discover           Пропустить сканирование сети, подключаться к --ip напрямую
  --full-scan             Полное сканирование подсети /24 (медленнее)
  --scan-only             Только сканирование, без подключения
  --audio-device NAME     Имя аудиоустройства (например: soundgrid, dante)
  --channels N            Количество каналов (по умолчанию: 48)
  --sample-rate SR        Частота дискретизации (по умолчанию: 48000)
  --no-apply              Только анализ, без применения корректировок
  --log-level LEVEL       Уровень логирования (DEBUG/INFO/WARNING/ERROR)
```

---

## 5. Как это работает

### Автоматический процесс (без участия пользователя)

При запуске система выполняет следующие шаги полностью автоматически:

1. **Сканирование сети** — отправляет UDP broadcast `WING?` на порт 2222 и пробует
   TCP-подключение к известным IP на портах 51328/51329 для обнаружения dLive.
   Находит пульт, определяет его тип, IP и порт.
2. **Подключение к пульту** — соединяется с найденным пультом.
3. **Сканирование аудиоустройств** — перечисляет все доступные аудио-входы через
   sounddevice/PortAudio, классифицирует каждое устройство по протоколу
   (SoundGrid, Dante, MADI, AVB, USB, Thunderbolt, CoreAudio), подсчитывает
   каналы, и автоматически выбирает лучшее многоканальное устройство по приоритету:
   **SoundGrid > Dante > MADI > AVB > Thunderbolt > USB > ASIO > системное**.
3. **Сканирование каналов** — считывает имена каналов с пульта.
4. **Распознавание инструментов** — по имени канала определяет тип инструмента
   (Kick, Snare, Vocal и т.д.). Если имя не распознано — классифицирует
   по спектральному анализу аудиосигнала.
5. **Ожидание сигнала** — ждёт появления аудиосигнала на каналах.
6. **Анализ** — измеряет уровни (peak, RMS, LUFS), спектральный центроид.
7. **Применение корректировок** (для каждого канала с сигналом):
   - **HPF** — высокочастотный фильтр в зависимости от типа инструмента
   - **4-полосный параметрический EQ** — пресет для инструмента
   - **Gain staging** — коррекция усиления до целевого LUFS
   - **Фейдер** — начальная позиция для баланса
8. **Мониторинг** — непрерывное отслеживание:
   - Обнаружение обратной связи (feedback) с автоматическим notch EQ
   - Детекция новых сигналов на ранее тихих каналах

### Распознаваемые инструменты

| Категория | Ключевые слова (EN/RU) |
|-----------|----------------------|
| Kick | kick, bd, bass drum, бочка, кик |
| Snare | snare, sd, sn, малый, снейр |
| Hi-Hat | hi-hat, hh, хай-хэт, хэт |
| Ride | ride, райд |
| Tom | tom, том, floor, флор |
| Cymbals | crash, splash, china, тарелки |
| Overheads | oh, overhead, оверхэд |
| Room | room, рум |
| Bass | bass, бас, sub |
| Electric Guitar | electric, egtr, gtr, гитара |
| Acoustic Guitar | acoustic, акустик, agtr |
| Accordion | accordion, баян, аккордеон |
| Synth/Keys | synth, keys, keyboard, piano, клавиши |
| Playback | playback, pb, track, минус |
| Lead Vocal | lead vox, vox, вокал + имена |
| Backing Vocal | back vox, bvox, бэк-вок, хор |

### Целевые уровни LUFS по инструментам

| Инструмент | Целевой LUFS | Начальный фейдер |
|-----------|-------------|------------------|
| Lead Vocal | -20.0 | -3.0 dB |
| Synth/Keys | -22.0 | -8.0 dB |
| Bass | -23.0 | -5.0 dB |
| Kick | -25.0 | -5.0 dB |
| Snare | -25.0 | -5.0 dB |
| Electric Guitar | -23.0 | -8.0 dB |
| Acoustic Guitar | -25.0 | -8.0 dB |
| Overheads | -30.0 | -10.0 dB |
| Hi-Hat | -35.0 | -12.0 dB |
| Room | -35.0 | -15.0 dB |

---

## 6. Устранение проблем

### Пульт не подключается

```bash
# Проверьте сетевое соединение
ping 192.168.3.70

# Проверьте, что порт 51328 открыт
nc -zv 192.168.3.70 51328

# Попробуйте подключиться вручную:
PYTHONPATH=backend python3 -c "
from dlive_client import DLiveClient
c = DLiveClient(ip='192.168.3.70')
print('Connected:', c.connect())
c.disconnect()
"
```

### Аудиоустройство не найдено

```bash
# Полное сканирование аудиоустройств с классификацией:
PYTHONPATH=backend python3 backend/audio_device_scanner.py

# Или в формате JSON:
PYTHONPATH=backend python3 backend/audio_device_scanner.py --json

# Выбрать устройство по имени:
PYTHONPATH=backend python3 backend/audio_device_scanner.py --prefer soundgrid

# Выбрать по протоколу:
PYTHONPATH=backend python3 backend/audio_device_scanner.py --protocol dante
```

Система распознаёт следующие типы аудиоустройств:

| Протокол | Примеры устройств | Приоритет |
|----------|------------------|-----------|
| SoundGrid | Waves SoundGrid, SG Driver | Высший |
| Dante | Dante Virtual Soundcard, DVS, Audinate | Высокий |
| MADI | RME MADIface, Digiface | Высокий |
| AVB | PreSonus AVB, MOTU AVB | Средний |
| Thunderbolt | Universal Audio Apollo | Средний |
| USB | Focusrite, MOTU, Behringer UMC, dLive USB | Обычный |
| ASIO | ASIO4ALL | Обычный |
| CoreAudio | Built-in Microphone | Низкий |

Убедитесь, что:
- Waves SoundGrid Studio запущена (для SoundGrid)
- Dante Controller и DVS установлены (для Dante)
- Драйвер аудиоустройства установлен и виден в System Preferences → Sound
- Устройство не захвачено другим приложением

### Ошибки импорта модулей

```bash
# Проверьте, что виртуальное окружение активировано:
which python3  # Должен показывать путь в venv/

# Переустановите зависимости:
pip install -r backend/requirements.txt --force-reinstall
```

### sounddevice не работает на macOS

```bash
# На macOS M-серии sounddevice использует CoreAudio напрямую.
# Если есть проблемы — разрешите доступ к микрофону:
# System Preferences → Privacy & Security → Microphone → Terminal

# Также можно попробовать:
pip install sounddevice --force-reinstall
```

---

## 7. Тестирование

### Запуск тестов

```bash
cd AUTO-MIXER-Tubeslave
source venv/bin/activate
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

### Тест без оборудования (виртуальный микшер)

```bash
PYTHONPATH=backend python3 virtual_mixer/virtual_mixer.py
```

---

## 8. Структура проекта

```
AUTO-MIXER-Tubeslave/
├── start_soundcheck.py          ← Точка входа для автоматического саундчека
├── backend/
│   ├── server.py                ← WebSocket сервер (для работы с UI)
│   ├── auto_soundcheck_engine.py ← Движок автоматического саундчека
│   ├── dlive_client.py          ← Клиент Allen & Heath dLive (MIDI/TCP)
│   ├── wing_client.py           ← Клиент Behringer WING (OSC)
│   ├── audio_capture.py         ← Захват аудио (SoundGrid/Dante/CoreAudio)
│   ├── feedback_detector.py     ← Детектор обратной связи
│   ├── channel_recognizer.py    ← Распознавание инструментов по именам
│   ├── auto_eq.py               ← Автоматический параметрический EQ
│   ├── auto_fader.py            ← Автоматическое управление фейдерами
│   ├── auto_compressor.py       ← Автоматический компрессор
│   ├── handlers/                ← WebSocket обработчики
│   ├── agents/                  ← Мультиагентная система
│   ├── ml/                      ← ML модели (классификатор, стиль)
│   └── ai/                      ← LLM интеграция (RAG, ChromaDB)
├── frontend/                    ← React + Electron UI
├── config/
│   ├── automixer.yaml           ← Основной конфиг
│   └── default_config.json      ← Дефолтные параметры автоматизации
├── tests/                       ← Тесты (pytest)
└── Docs/                        ← Техническая документация
```

---

## 9. Безопасность

⚠️ **Это система управления живым звуком. Ошибки могут привести к повреждению слуха.**

- Фейдер никогда не устанавливается выше 0 dBFS без явного запроса
- True peak всегда проверяется < -1.0 dBTP перед повышением gain
- Feedback detector имеет абсолютный приоритет и может снизить фейдер мгновенно
- При любых сомнениях система снижает уровень, а не повышает
- Максимальная коррекция gain: ±12 dB

---

## 10. Лицензия и авторы

AUTO-MIXER Tubeslave — AI-управляемый автоматический микшер.
