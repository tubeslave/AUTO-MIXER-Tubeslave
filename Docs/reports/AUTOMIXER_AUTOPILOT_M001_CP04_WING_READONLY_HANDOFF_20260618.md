# M001-CP04: WING read-only handoff

Generated: 2026-06-18T20:49:40Z

## Итог

CP04 подготовлен без контакта с WING и без изменения runtime product behavior.
Свежий expected plan создан в:

`ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`

Проверка real-run inputs прошла через `--validate-run-inputs-only` и завершилась
до открытия WebSocket.

## Доказательства

- Baseline tests: `4027 passed, 1 skipped`.
- No-WING quick artifact:
  `artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check`.
- Prestart summary:
  `artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check/backend_port_prestart/summary.json`.
- Expected plan:
  `ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`.
- First validate-only summary, expected mismatch safely blocked:
  `artifacts/wing_readonly_preflight/20260618T204909Z_wing_readonly_preflight/summary.json`.
- Second validate-only summary, inputs ready for WebSocket:
  `artifacts/wing_readonly_preflight/20260618T204929Z_wing_readonly_preflight/summary.json`.
- Validate-only status: `ready_for_websocket`.
- Safety flags: `websocket_opened: false`, `write_request_sent: false`,
  `contacts_wing: false`, `contacts_backend: false`,
  `live_authority_granted: false`, `readiness_closure_allowed: false`.

## Важная находка

Первый validate-only запуск безопасно заблокировался с
`expected_plan_mismatch`. Причина: использовался plan из no-WING bundle, который
не совпадал с текущими runtime arguments/path. После генерации свежего plan с
теми же аргументами validate-only прошел.

## Команда проверки без WebSocket

```bash
PYTHONPATH=backend python tools/wing_readonly_preflight_runner.py \
  --url ws://127.0.0.1:8765 \
  --channels 3 \
  --value-types fader,gain,pan \
  --require-physical-readback \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --expected-plan ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json \
  --prestart-summary artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check/backend_port_prestart/summary.json \
  --validate-run-inputs-only \
  --artifact-root artifacts/wing_readonly_preflight
```

## Реальный read-only запуск

Только после того, как оператор подтвердит, что intended backend/WING session
готова, можно выполнить тот же runner без `--validate-run-inputs-only`.

```bash
PYTHONPATH=backend python tools/wing_readonly_preflight_runner.py \
  --url ws://127.0.0.1:8765 \
  --channels 3 \
  --value-types fader,gain,pan \
  --require-physical-readback \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --expected-plan ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json \
  --prestart-summary artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check/backend_port_prestart/summary.json \
  --artifact-root artifacts/wing_readonly_preflight
```

Эта команда должна оставаться read-only. Она не разрешает supervised fader/gain
proof, readiness closure или live authority.

## После реального запуска

```bash
python tools/review_wing_readonly_preflight_artifacts.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md \
  --decision-template-output artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json

python tools/review_wing_readonly_preflight_decisions.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --decision-root artifacts/wing_readonly_preflight_review_decisions \
  --output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md
```

## Environment note

В этой Codex session `python3` из PATH завершался с кодом 137 даже на
`--version`. Проверки выполнялись через рабочий `python` 3.11.7. Runbook still
mentions `python3`; on this machine use `python` if `python3` repeats the same
environment failure.
