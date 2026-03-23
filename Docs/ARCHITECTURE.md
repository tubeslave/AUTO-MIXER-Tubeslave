# Архитектура AUTO-MIXER-Tubeslave (кратко)

Компактный ориентир для разработки. Подробные материалы — в `Docs/`.

## Высокоуровневая схема

- **Frontend** (React + Electron): UI, WebSocket клиент, вкладки (EQ, Compressor, Fader, Voice Control и др.)
- **Backend** (Python): WebSocket сервер, handlers, mixer clients (WING, dLive, Mixing Station)
- **Mixer**: WING Rack (OSC), dLive (MIDI/TCP) или Mixing Station

## Backend

- `backend/server.py` — точка входа, WebSocket, регистрация handlers
- `backend/handlers/` — mixer, connection, eq, fader, gain_staging, voice, snapshot, routing и др.
- `backend/wing_client.py` — OSC WING (safety_limits при enable_limits)
- `backend/dlive_client.py` — Allen & Heath dLive MIDI/TCP
- `backend/osc/enhanced_osc_client.py` — обёртка WingClient с auto-reconnect
- `backend/feedback_detector.py` — детекция обратной связи (FFT, notch EQ, fader fallback)
- `backend/handlers/feedback_handlers.py` — start/stop feedback detection
- `backend/controller_lifecycle.py` — cleanup_all_controllers (вынесено из server.py)
- `backend/services/` — сервисы с логикой start_*: fader_service, gain_staging_service, feedback_service

## Конфигурация

- `config/default_config.json` — safety (max_fader, max_gain, enable_limits), automation, wing

## Тесты

- `tests/` — pytest
- `tests/test_infrastructure/test_mixer_handlers_safety.py` — проверка safety limits
- `tests/integration/test_server_handlers.py` — интеграционные тесты (WebSocket + mock mixer)

## Документация

- **docs/** (инженерные правила): ARCHITECTURE, CONVENTIONS, TOOLS, IMPROVEMENT_PLAN
- **Docs/** (подробные материалы):
  - `WING Remote Protocols v3.0.5.pdf` — протокол OSC WING
  - TECHNICAL.md, CODE_REVIEW.md, AUTO_EQ_INTEGRATION.md, STAGE1_COMPLETE.md
