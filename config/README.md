# Конфигурация AUTO-MIXER-Tubeslave

## Канонический конфиг (используется server.py)

**`default_config.json`** — основной конфиг приложения. Загружается в `backend/server.py` при старте.

Секции:
- `wing` — IP, порты, модели WING
- `server` — WebSocket host/port
- `automation` — Auto-EQ, Auto Fader, Gain Staging, Bleed, Phase, Soundcheck, Compressor
- **`safety`** — `enable_limits`, `max_fader`, `max_gain` (см. docs/IMPROVEMENT_PLAN.md)

## automixer.yaml (экспериментальный)

`automixer.yaml` используется **ConfigManager** (`backend/config_manager.py`), который пока **не подключён** к server.py. ConfigManager поддерживает YAML, hot-reload и переменные окружения. Используется в тестах (`tests/test_infrastructure/test_config_manager.py`).

Для production используйте `default_config.json`.
