# Архитектура AUTO-MIXER-Tubeslave

## Обзор

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React/Electron)                  │
│  Components: EQ, Compressor, Fader, Gate, Panner, Reverb,   │
│              Soundcheck, GainStaging, CrossAdaptiveEQ        │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket (ws://localhost:8765)
┌──────────────────────▼──────────────────────────────────────┐
│                   server.py (координатор)                     │
│  - Регистрирует handlers из backend/handlers/                │
│  - WebSocket маршрутизация сообщений                         │
│  - convert_numpy_types() для JSON сериализации               │
└───┬──────────┬──────────┬──────────┬────────────────────────┘
    │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼────┐ ┌───▼──────────────┐
│Handlers│ │Agents │ │   AI   │ │    ML Pipeline   │
│ (18)   │ │       │ │        │ │                  │
└───┬────┘ └───┬───┘ └───┬────┘ └───┬──────────────┘
    │          │          │          │
┌───▼──────────▼──────────▼──────────▼────────────────────────┐
│              Core Services Layer                              │
│  ThreadSafeMixerState │ AudioCapture │ FeedbackDetector      │
│  ConfigManager        │ SessionManager│ Auth + TLS           │
└───┬──────────────────────────────────┬──────────────────────┘
    │                                  │
┌───▼──────────┐              ┌────────▼──────────┐
│  WingClient  │              │   DLiveClient     │
│  (OSC/UDP)   │              │   (MIDI/TCP)      │
│  port 2223   │              │   port 51328      │
└──────────────┘              └───────────────────┘
```

## Слои

### 1. Frontend Layer
- **React 18** + **Electron** для desktop
- Компоненты в `frontend/src/components/` — по табу на каждый модуль автоматизации
- Связь с backend через единый WebSocket на порт 8765
- State management: локальный React state, синхронизация через WS messages

### 2. WebSocket Handlers (`backend/handlers/`)
Каждый handler — отдельный модуль, регистрируется через `register_all_handlers()`:

| Handler | Файл | Ответственность |
|---------|------|-----------------|
| Agent | `agent_handlers.py` | AI mixing agent, pending actions, approval queue |
| Audio | `audio_handlers.py` | Аудио устройства, capture, уровни |
| Automation | `automation_handlers.py` | Управление модулями автоматизации |
| Channel Scan | `channel_scan_handlers.py` | Сканирование каналов, распознавание |
| Compressor | `compressor_handlers.py` | Auto-compressor параметры |
| Connection | `connection_handlers.py` | Подключение к пульту |
| EQ | `eq_handlers.py` | Auto-EQ, cross-adaptive EQ |
| Fader | `fader_handlers.py` | Auto-fader, LUFS-based |
| FX | `fx_handlers.py` | FX slots, inserts, send/return state |
| Gain Staging | `gain_staging_handlers.py` | LUFS gain staging |
| Measurement | `measurement_handlers.py` | System/master measurement and correction |
| Mixer | `mixer_handlers.py` | Общие операции с пультом |
| Phase | `phase_handlers.py` | GCC-PHAT фазовое выравнивание |
| Routing | `routing_handlers.py` | Dante/аналоговый маршрутизация |
| Snapshot | `snapshot_handlers.py` | Загрузка/сохранение снапшотов |
| Soundcheck | `soundcheck_handlers.py` | Автоматический саундчек |
| Training | `training_handlers.py` | Online/offline training service controls |
| Voice | `voice_handlers.py` | Голосовое управление |

### 3. Agent System (`backend/agents/`)
Multi-agent с координатором:

```
Coordinator
├── GainAgent      — LUFS gain staging, trim
├── EQAgent        — Параметрический EQ, HPF, cross-adaptive
└── FaderAgent     — Авто-фейдеры, баланс, приоритеты
```

- `base_agent.py` — ABC с жизненным циклом (IDLE → RUNNING → PAUSED → STOPPED)
- `coordinator.py` — оркестратор: разрешает конфликты между агентами, лимит действий/цикл
- Каждый агент работает в своём asyncio task, общается через ThreadSafeMixerState

### 4. AI Layer (`backend/ai/`)
- `agent.py` — LLM Orchestrator: получает состояние микса, вызывает function calling
- `knowledge_base.py` — ChromaDB + sentence-transformers для RAG
- `llm_client.py` — абстракция над Ollama (локальный llama3) и Perplexity API
- `rule_engine.py` — детерминированный fallback, когда LLM недоступен
- Knowledge: `ai/knowledge/` — instrument_profiles, mixing_rules, troubleshooting, wing_osc_reference

### 5. ML Pipeline (`backend/ml/`)
PyTorch модели для intelligent mixing:

| Модуль | Назначение |
|--------|-----------|
| `channel_classifier.py` | Классификация инструментов (25 классов) по спектру |
| `gain_pan_predictor.py` | SE-based предиктор gain/pan по спектральным фичам |
| `style_transfer.py` | Перенос стиля микса с референса |
| `neural_mix_extractor.py` | Извлечение параметров (gain, EQ, comp) из dry/wet |
| `differentiable_console.py` | Дифференцируемая модель микшерного пульта |
| `processing_graph.py` | Модульный граф обработки с gradient optimization |
| `losses.py` | Аудио loss-функции (spectral, LUFS, multi-resolution STFT) |
| `lufs_targets.py` | Per-instrument LUFS targets |
| `drc_onset.py` | DRC onset peak statistics |
| `mix_quality.py` | Perceptual mix quality metric |
| `eq_normalization.py` | EQ normalization pipeline |
| `subgroup_mixer.py` | Subgroup mixing pipeline |
| `reference_profiles.py` | Референсные профили по жанрам |

### 6. Core Services

#### ThreadSafeMixerState (`thread_safety.py`)
- `asyncio.Lock` для всех мутаций
- Copy-on-read: `get_snapshot()` возвращает глубокую копию
- `StateSnapshot` — immutable dataclass с timestamp

#### AudioCapture (`audio_capture.py`)
- Единый сервис захвата аудио (Dante, SoundDevice, тестовые генераторы)
- Ring buffer на N секунд
- Subscribe pattern: модули подписываются на аудио-блоки
- **Все модули используют AudioCapture. НЕ создавать свои PyAudio потоки.**

#### FeedbackDetector (`feedback_detector.py`)
- FFT peak tracking для обнаружения обратной связи (< 50 мс)
- Автоматический notch EQ (до 8 фильтров/канал, глубина до -12 дB)
- Fallback: снижение фейдера если notch недостаточен
- Абсолютный приоритет над другими агентами

#### Runtime Config
- `server.py` загружает `config/default_config.json` и накладывает runtime-секции из `config/automixer.yaml`
- `config_manager.py` отвечает за YAML config/hot-reload для сервисов, которым нужен watchdog
- Пользовательские UI-настройки сохраняются отдельно в `config/user_config.json` и не являются каноническим default config

### 7. Mixer Clients

#### MixerClientBase (`mixer_client_base.py`)
ABC определяет интерфейс: connect, disconnect, send, subscribe, get_state, set_fader, get_fader, set_eq, set_compressor, и др.

#### WingClient (`wing_client.py`)
- OSC/UDP на порт 2223 (инициация через 'WING?' на 2222)
- Keepalive: XREMOTE каждые 5 сек
- Throttle: ≤ 10 OSC/сек на адрес
- Fader mapping: 0.0 = -∞, 0.7498 = 0 dB, 1.0 = +10 dB

#### DLiveClient (`dlive_client.py`)
- MIDI over TCP на порт 51328/51329
- NRPN 14-bit для faders, EQ, compressor
- SysEx для расширенных параметров
- Fader mapping: 0x0000 = -∞, 0x2AAA = 0 dB, 0x3FFF = +10 dB

## Потоки данных

### Саундчек (автоматический)
```
AudioCapture → ChannelClassifier → GainAgent (LUFS) → EQAgent (HPF + parametric)
    → AutoCompressor → FeedbackDetector → MixerClient (OSC/MIDI) → Console
```

### Концерт (real-time)
```
AudioCapture → FeedbackDetector (приоритет!) → Coordinator
    → [GainAgent | EQAgent | FaderAgent] → ThreadSafeMixerState
    → MixerClient → Console
```

### AI-управляемый режим
```
User query → AI Agent → Knowledge Base (RAG) → LLM (function calling)
    → Agents/Handlers → MixerClient → Console
```
