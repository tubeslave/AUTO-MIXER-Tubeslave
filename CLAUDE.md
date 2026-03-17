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

Подробная архитектура: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
Соглашения по коду и DSP: [docs/CONVENTIONS.md](docs/CONVENTIONS.md)
Протоколы и инструменты: [docs/TOOLS.md](docs/TOOLS.md)

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
- Agent `while self.state == RUNNING` завершается при PAUSED. Использовать `asyncio.Event.wait()`
- `find_snap_by_name()` — КЭШИРОВАТЬ имена. Загрузка = деструкция mixer state на концерте
- `ratio_float_to_wing()` — константы WING_RATIO_VALUES/WING_RATIO_STRINGS должны быть на уровне модуля
- NumPy типы (np.float64) не сериализуются в JSON → `convert_numpy_types()` перед `json.dumps()`

### Архитектурные решения
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
