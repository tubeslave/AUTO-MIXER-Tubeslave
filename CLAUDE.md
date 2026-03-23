# AUTO-MIXER-Tubeslave

AI-управляемый автоматический микшер для live sound. Behringer WING (OSC), Allen & Heath dLive (MIDI/TCP), Mixing Station.

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt && python server.py

# Frontend
cd frontend && npm install && npm start

# Tests
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

**Логи Auto-EQ (параметры после анализа):** дублируются в `logs/automixer-backend.log`.
Просмотр только строк EQ: `./scripts/tail_automixer_eq_logs.sh` или `tail -f logs/automixer-backend.log`.

## Структура (TOC)

```
backend/
  server.py            — WebSocket сервер, точка входа
  wing_client.py       — OSC клиент WING (safety_limits)
  dlive_client.py      — dLive MIDI/TCP
  handlers/            — обработчики WebSocket
frontend/
  src/                 — React UI
config/
  default_config.json  — safety, automation, wing
tests/                 — pytest
docs/                  — ARCHITECTURE, CONVENTIONS, TOOLS, IMPROVEMENT_PLAN
Docs/                  — PDF, технические гайды
```

## Критические ограничения (SAFETY)

Это система управления живым звуком. Ошибки могут причинить физический вред слуху.

1. **Fader ceiling**: никогда не устанавливать fader > 0 dBFS без явного запроса оператора
2. **True peak**: всегда проверять < -1.0 dBTP перед повышением gain
3. **При сомнениях — снижай, не повышай**

## Workflow для агента

1. Перед началом работы — прочитать `.cursorrules` и этот файл
2. При изменении DSP — свериться с `docs/CONVENTIONS.md` и добавить/обновить тесты
3. При добавлении OSC команд — свериться с `docs/TOOLS.md` и `Docs/WING Remote Protocols v3.0.5.pdf`
4. Перед коммитом — `PYTHONPATH=backend python -m pytest tests/ -x`
