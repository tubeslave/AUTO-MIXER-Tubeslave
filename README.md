# Auto Mixer Tubeslave

AI-управляемый автоматический микшер для live sound. Поддержка **Behringer WING** (OSC), **Allen & Heath dLive** (MIDI/TCP), **Mixing Station**.

## Структура проекта

```
AUTO MIXER Tubeslave/
├── backend/              # Python-бэкенд (WebSocket, OSC, MIDI)
│   ├── server.py         # WebSocket-сервер, точка входа
│   ├── wing_client.py   # OSC-клиент Behringer WING
│   ├── dlive_client.py  # MIDI/TCP клиент Allen & Heath dLive
│   ├── handlers/        # Обработчики WebSocket-сообщений
│   ├── wing_addresses.py
│   └── requirements.txt
├── frontend/             # React + Electron
│   └── src/components/  # Вкладки: EQ, Compressor, Fader, Gate, Voice Control и др.
├── config/
│   └── default_config.json  # Конфиг (safety, automation, wing)
├── docs/                 # Инженерная документация для разработки
├── Docs/                 # PDF, технические гайды
└── tests/                # pytest
```

## Возможности

### Реализовано:
- ✅ OSC-клиент с синхронизацией WING Rack (40 каналов)
- ✅ WebSocket-сервер real-time
- ✅ Поддержка dLive, Mixing Station
- ✅ 40 канальных полос: фейдеры, gain, 4-полосный EQ, компрессор, gate, routing
- ✅ **LUFS Gain Staging** — автоматическая калибровка по LUFS и true peak
- ✅ **Auto-EQ** — спектральный анализ и коррекция
- ✅ **Auto Fader** — real-time балансировка каналов
- ✅ **Auto Compressor** — подбор параметров по типу инструмента
- ✅ **Phase Alignment** — GCC-PHAT выравнивание фазы
- ✅ **Auto Soundcheck** — последовательный саундчек
- ✅ **Voice Control** — голосовое управление (Whisper, Sherpa)
- ✅ **Bleed Service** — компенсация блидинга
- ✅ **Safety Limits** — fader ≤ 0 dBFS, true peak check (см. `config/default_config.json`)

### Документация:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — архитектура
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md) — DSP, стиль кода
- [docs/TOOLS.md](docs/TOOLS.md) — OSC/MIDI протоколы
- [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) — план доработок

## Требования

### Backend:
- Python 3.9+
- python-osc
- websockets
- numpy, scipy (для модулей автоматизации)

### Frontend:
- Node.js 16+
- React 18

## Установка

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

## Запуск

### 1. Запуск Backend-сервера

```bash
cd backend
python server.py
```

Сервер запустится на `ws://localhost:8765`

### 2. Запуск Frontend

```bash
cd frontend
npm start
```

Приложение откроется в браузере на `http://localhost:3000`

## Подключение к Wing Rack

1. Подключите Wing Rack к той же сети, что и компьютер
2. Найдите IP-адрес пульта в настройках Wing (Setup → Network)
3. В интерфейсе приложения:
   - Выберите модель: **Wing Rack**
   - Введите IP-адрес (по умолчанию: `192.168.1.100`)
   - Send Port: `2222` (стандартный для Wing)
   - Receive Port: `2223`
   - Нажмите **Connect to Wing**

4. При успешном подключении:
   - Статус изменится на "Wing: Connected"
   - GUI синхронизируется с текущим состоянием пульта
   - Все изменения на пульте отобразятся в реальном времени

## OSC-адреса Wing

Примеры адресов согласно спецификации Wing OSC API:

- Фейдер канала: `/ch/01/mix/fader`
- Gain канала: `/ch/01/preamp/trim`
- EQ полоса 1 частота: `/ch/01/eq/1/f`
- Компрессор threshold: `/ch/01/dyn/thr`
- Подписка на обновления: `/xremote`

Полный справочник в файле `backend/wing_addresses.py`

## Настройка конфигурации

Отредактируйте `config/default_config.json`:
- `wing` — IP, порты
- `automation` — Auto-EQ, Auto Fader, Gain Staging, Bleed и др.
- **`safety`** — защита live sound:
  - `enable_limits: true` — включить лимиты
  - `max_fader: 0` — максимум фейдера в dBFS (0 = не выше unity)
  - `max_gain: 18` — максимум gain в dB

## Тестирование OSC-связи

Используйте тестовый скрипт для проверки подключения:

```bash
cd backend
python test_wing_connection.py
```

Скрипт выведет все входящие OSC-сообщения. Покрутите ручки на пульте и убедитесь, что изменения отображаются.

## Тестирование

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Ссылки на документацию

- [Behringer Wing OSC Implementation](https://wiki.munichmakerlab.de/wiki/Behringer_Wing)
- [Mixing Secrets for the Small Studio - Mike Senior](https://www.cambridge-mt.com/ms/mix-book/)
- [Sound Systems: Design and Optimization - Bob McCarthy](https://bobmccarthy.com/)

## Поддержка

При возникновении проблем:
1. Проверьте сетевое подключение Wing и компьютера
2. Убедитесь, что порты 2222/2223 не заняты
3. Проверьте версию прошивки Wing (должна быть 3.0.5)
4. Проверьте логи backend-сервера

---

**Firmware:** Wing Rack fw 3.0.5  
**Protocol:** OSC (Open Sound Control)  
**License:** MIT
