# Backend Documentation

Date: 2026-05-18

Backend source lives in `backend/`. Backend documentation lives here.

## Canonical Runtime

| Path | Role |
| --- | --- |
| `backend/server.py` | Main WebSocket coordinator. |
| `backend/wing_client.py` | WING OSC client and supervised write gate. |
| `backend/handlers/` | WebSocket message handlers. |
| `backend/operator_*.py` | Product Layer state, analysis, proposals, and mode policy. |
| `backend/auto_soundcheck_engine.py` | Soundcheck analysis and recommendation engine. |

## Optional Or Lab Runtime

| Path | Role |
| --- | --- |
| `backend/auto_*.py` | DSP and automation experiments or proposal candidates. |
| `backend/voice_*.py` | Voice input experiments. Voice must not bypass Product Layer safety. |
| `backend/ml/`, `backend/perceptual/`, `backend/evaluation/` | Research, scoring, and evaluation support. |
| `backend/lab_only/` | Quarantined or manual tooling. |

## Documentation Groups

| Path | Contents |
| --- | --- |
| `Docs/backend/` | Backend setup, routing, Dante, voice, safe gain, and test notes. |
| `Docs/reports/tub/` | Historical TUB reports and live-readiness runbooks. |
| `Docs/manuals/wing/` | WING manuals and protocol PDFs. |
| `Docs/archive/` | Old root notes and generated legacy snapshots. |

## Manual Checks

Manual backend probes are listed in `Docs/backend/MANUAL_PROBES.md`. Many still
live in `backend/` for import-path compatibility. New manual probes should go
under `backend/lab_only/` unless they become tested product code.
