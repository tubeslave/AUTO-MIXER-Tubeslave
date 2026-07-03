# AUTO-MIXER-Tubeslave

AI-управляемый автоматический микшер для саундчеков и концертов.
Behringer WING Rack (OSC) + Allen & Heath dLive (MIDI/TCP) + Dante/SoundGrid.

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt && python server.py

# Frontend
cd frontend && npm install && npm start

# Tests
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Структура (TOC)

```
backend/
  server.py              — WebSocket координатор, точка входа
  wing_client.py         — Behringer WING OSC client (MixerClientBase)
  dlive_client.py        — Allen & Heath dLive MIDI/TCP client (MixerClientBase)
  mixer_client_base.py   — ABC для всех mixer clients
  audio_capture.py       — Единый аудио-сервис (ring buffer, subscribe)
  feedback_detector.py   — Обнаружение обратной связи, notch EQ, fader fallback
  thread_safety.py       — ThreadSafeMixerState (asyncio.Lock, copy-on-read)
  config_manager.py      — YAML config с hot-reload (watchdog)
  handlers/              — 14 WebSocket handler-модулей (см. docs/ARCHITECTURE.md)
  agents/                — Multi-agent система (coordinator → eq/fader/gain agents)
  ai/                    — LLM orchestrator + ChromaDB/RAG knowledge base
  ml/                    — PyTorch ML модели (classifier, predictor, style transfer)
  osc/                   — Enhanced OSC client с retry и батчингом
frontend/
  src/components/        — React компоненты (табы: EQ, Compressor, Fader, Gate, etc.)
config/
  automixer.yaml         — Главный конфиг (mixer, audio, websocket, agent, ai, safety)
tests/                   — pytest (test_ml/, test_ai/, test_infrastructure/)
virtual_mixer/           — Эмулятор микшера для тестов без оборудования
Docs/                    — PDF документация WING, технические документы
```

Подробная архитектура: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md)
Соглашения по коду и DSP: [Docs/CONVENTIONS.md](Docs/CONVENTIONS.md)
Протоколы и инструменты: [Docs/TOOLS.md](Docs/TOOLS.md)
WebSocket-контракт фронт↔бэк: [Docs/PROTOCOL.md](Docs/PROTOCOL.md)
Архив неиспользуемого кода: [backend/archive/README.md](backend/archive/README.md) — НЕ импортировать оттуда

## Критические ограничения (SAFETY)

Это система управления живым звуком. Ошибки = физический вред слуху.

1. **Fader ceiling**: никогда не устанавливать fader > 0 dBFS без явного запроса оператора
2. **True peak**: всегда проверять < -1.0 dBTP перед повышением gain
3. **Feedback first**: feedback_detector.py имеет абсолютный приоритет — может снизить фейдер без одобрения других агентов
4. **Snap safety**: find_snap_by_name() читает только кэш имён. Загрузка снапшота перезаписывает настройки пульта
5. **При сомнениях — снижай, не повышай**

## Зафиксированные правила (обновляй при обнаружении нового)

### Баги, которые повторялись
- `gcc_phat(target, reference)` — НЕ `gcc_phat(reference, target)`. Инверсия = фаза корректируется в обратную сторону
- GCC-PHAT знак ПРИМЕНЕНИЯ: `delay_ms > 0` = target ОТСТАЁТ. Задерживать target только при `delay_ms < 0` (когда он опережает), иначе положительная обратная связь до предела delay
- Agent `while self.state == RUNNING` завершается при PAUSED. Использовать `asyncio.Event.wait()`
- Agent `start()` на PAUSED обязан только resume — второй `_run_loop` = дублирующиеся OSC-записи
- `find_snap_by_name()` — КЭШИРОВАТЬ имена. Загрузка = деструкция mixer state на концерте
- `ratio_float_to_wing()` — константы WING_RATIO_VALUES/WING_RATIO_STRINGS должны быть на уровне модуля
- NumPy типы (np.float64) не сериализуются в JSON → `convert_numpy_types()` перед `json.dumps()`
- `set_channel_fader()` принимает dB — НЕ конвертировать в линейный коэффициент (`_db_to_linear(-10)=0.316` ≈ unity → аварийное снижение не срабатывает)
- Pre-fader компенсация: `pre_fader = measured - fader_db` — НЕ плюс (плюс = положительная обратная связь контура Fader Riding)
- Readback вернул None → НЕ подставлять 0.0 и НЕ писать в пульт (запись «0 − шаг» при реальном trim −10 dB = повышение gain при перегрузе). Пропустить цикл/канал
- Notch-слоты feedback_detector: max 7 (4 EQ + 3 PEQ бэнда), наименьший свободный слот, слот↔бэнд взаимно однозначно. `(slot % 4) + 1` перезаписывал чужие бэнды
- `signal.lfilter` в блочном DSP (TruePeakMeter и т.п.) — ВСЕГДА с `zi=`, иначе межблочные пики занижаются
- Клэмп по пиковому запасу: всегда `min(gain, delta_peak)`, в т.ч. при `delta_peak < 0`
- MIDI поверх TCP (dlive_client): накопительный RX-буфер обязателен — TCP не сохраняет границы сообщений, partial read без буфера искажает NRPN-state фейдеров
- dLive mute — ПАРА NoteOn(vel состояния) + NoteOn(vel=0): хвост vel=0 не несёт состояния, читать отдельный mute-ключ, не сырой note
- Пер-сэмпловые константы (peak_decay, attack/release/hold в сэмплах) при вызове раз на КАДР — умножать/возводить в степень на размер кадра
- Ratio-строки WING: только "1.1".."100" (формат "2.0:1"/"inf:1" пульт не принимает); таблицы signal_analysis и compressor_adaptation синхронны (тест)
- Reconnect клиентов: потоки receiver/renewal привязывать к поколению подключения (иначе disconnect→connect плодит дубликаты, читающие новый сокет)
- `asyncio.create_task` из потоков — только через `loop.call_soon_threadsafe(self._create_broadcast_task, payload)` (сильная ссылка + payload фиксируется сразу)

### Архитектурные решения
- Потолок автоматики для фейдера — 0 dBFS (клэмпы/`FADER_MAX`/`MAX_GAIN_DB` = 0.0); +10 dB доступен только оператору вручную
- Синхронные вызовы mixer-клиента с `time.sleep` (load_snap/save_snap) из async-хендлеров — только через `asyncio.to_thread`
- Fader Riding: канал без прочитанного с пульта якоря (`anchor_valid=False`) не управляется вообще
- Единый AudioCapture: `server._get_shared_audio_capture(device_id)` — модули получают его через `audio_capture=` и подписку; PyAudio-fallback в модулях только когда общий захват недоступен. Индексы PyAudio ≠ индексы sounddevice — резолвить через имя устройства. В тестах `AUTOMIXER_SHARED_AUDIO_CAPTURE=false` (conftest) — не открывать реальный микрофон
- Потеря WING по UDP видна только по тишине в приёме: heartbeat в renewal-цикле (`_last_rx_time`, активный запрос `/ch/1/fdr`) + `on_connection_lost` → reconnect EnhancedOSCClient. UDP-send «успешен» даже на выключенный пульт
- Применение DSP-коррекций на пульт — через верифицированные записи: `set_eq_band_verified()` / `apply_supervised_bundle` (pre-read → запись → readback с допуском → rollback). Вызывающий обязан проверять `status == "applied"`; notch против заводки с неподтверждённым readback = заводка не подавлена
- dLive: `set_fader_verified()` — best-effort по MIDI-эху (applied/mismatch/unconfirmed, БЕЗ отката); `unconfirmed` ≠ отказ (MIDI send на пульте может быть выключен), но и не подтверждение
- Конфигурация: default_config.json — база `self.config`, automixer.yaml поверх через ConfigManager (deep-merge, hot-reload через watchdog запускается в `server.start()`); клиенты получают `config_reloaded` broadcast
- AudioCapture — единый сервис. НЕ создавать PyAudio потоки в отдельных модулях
- ScenarioDetector — кэшировать экземпляр, НЕ создавать на каждый аудиоблок
- server.py — координатор. Логика в handlers/. НЕ добавлять новые маршруты в server.py напрямую
- ML classifier — использовать полноценный из backend/ml/channel_classifier.py (25 классов), НЕ заглушку

### DSP стандарты
- LUFS: ITU-R BS.1770-4, double gating (absolute -70 LUFS + relative -10 LU)
- True Peak: 4x oversampling
- K-Weighting: high-shelf +4dB @ 1681 Hz + HPF 38 Hz (два biquad каскада)
- EQ biquads: по Audio EQ Cookbook (Robert Bristow-Johnson)
- Компрессия ratio: `1 + (max_ratio - 1) * factor` — НЕ инвертировать
- Gate: объявленный hysteresis ОБЯЗАН применяться в GateProcessor

## Тестирование

```bash
# Все тесты
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q

# Только ML
PYTHONPATH=backend python -m pytest tests/test_ml/ -v

# Только AI
PYTHONPATH=backend python -m pytest tests/test_ai/ -v

# Только инфраструктура
PYTHONPATH=backend python -m pytest tests/test_infrastructure/ -v

# Virtual mixer (без оборудования)
python virtual_mixer/virtual_mixer.py
```

CI: GitHub Actions — Python 3.10, 3.11, 3.12. Все PR должны пройти `pytest`.

## Workflow для агента

1. Перед началом работы — прочитать этот файл и `.cursorrules`
2. При изменении DSP — написать тест с синтетическими сигналами
3. При изменении handlers — обновить регистрацию в `handlers/__init__.py`
4. При добавлении OSC/MIDI команд — сверяться с `Docs/WING Remote Protocols v3.0.5.pdf`
5. При обнаружении бага или паттерна — добавить в раздел «Зафиксированные правила» выше
6. Перед коммитом — `PYTHONPATH=backend python -m pytest tests/ -x`
7. Не добавлять зависимости без явной необходимости
