# Runbook оператора: разблокировка записей на WING и attended-саундчек

**Версия:** 2026-07-28 · **Инструмент:** `tools/manual_write_recovery_operator.py` · **Аудитория:** оператор у пульта (человек), агенты Codex / Claude Code / Kimi

Этот runbook закрывает 4 рельса, блокирующих применение коррекций на пульт:
1) WAL emergency-stop, 2) пустой allowlist записей, 3) незакрытые HW01–HW05, 4) dirty tree / CP05.

---

## 0. Снимок состояния на 2026-07-28 (перед сеансом перепроверить!)

Из `ops/autopilot/STATE.json` и git (HEAD `ea5f5d737`):

- **Железо верифицировано** route-артефактом `4937d06623084a2d98e7bc3f9b8f8a17`: точный серийник WING, прошивка 3.0.5, serial-bound USB/CoreAudio 48x48/48 kHz, два полных прохода Main 47/48 MAIN L/R, живой сигнал −12.318 dBTP, 0 записей в пульт.
- **WAL на диске чист** (74 строки, незакрытых транзакций нет) — ночной прогон m138 довёл записи до терминальных фаз. Защёлка emergency-stop живёт в процессе backend → живую проверку всё равно делать (шаг 2).
- **HW01 не закрыт**: блокер — `proof_owned_master_route_teardown_failed:wing_disconnect_failed:RuntimeError` (teardown считал broadcast `readiness_invalidated` фатальным). M120-fix готов (142 теста runner'а зелёные), нужен **один чистый повтор route-only с verified teardown**. HW02–HW05 pending.
- **`approved_live_write_kinds: []`** — штатный UI-apply закрыт; записи только через attended runner.
- Reset / corrections / live writes на пульте за всю историю: **0**. Strict real-live readiness: **88/100**.
- Codex за ночь сделал merge `soundcheck trustworthy-apply + per-channel mode` и два rebind маркеров CP05; tracked-дерево чистое.

## 1. Координация с Codex / Claude Code (за 10 минут до сеанса)

Проект правят несколько агентов параллельно — перед сеансом зафиксируйте базовую линию:

```bash
cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
git log --oneline -5          # появились ли новые коммиты Codex/Claude?
git status --short | grep -v '^??'   # tracked-дерево должно быть ЧИСТЫМ
python3 -c "import json; print(json.load(open('ops/autopilot/STATE.json'))['current_blocker'][:400])"
```

- Быстрый release-gate: `PYTHONPATH=backend backend/venv/bin/python -m pytest tests/test_wing_mvp_rails.py -q` — должно быть зелёно (ранее падало на `m110_cp05_tracked_tree_not_clean`, после rebind-коммитов Codex — проверить).
- **Контракт WS мог измениться.** Сверьте, что команды на месте:
  `grep -n "reconcile_console_write_recovery\|reset_manual_write_emergency_stop\|get_manual_write_supervision" backend/handlers/automation_handlers.py`
  Если переименованы — сначала обновите `tools/manual_write_recovery_operator.py`.
- **Не коммитьте** `ops/autopilot/*`, `reports/*`, `~/.automixer/**` — это рабочие файлы автопилота; их правки ломают CP05.
- Во время сеанса с пультом попросите паузу у фоновых агентов, пишущих в пульт (machine-supervisor прогоны m13x), чтобы не смешать WAL-транзакции.

## 2. Железный чек-лист

- [ ] WING Rack, прошивка **3.0.5** (уже верифицировано артефактом, перепроверить в Setup → Firmware)
- [ ] L2-сеть, OSC-порты **2222/2223**, IP из RFC1918 (unicast)
- [ ] USB serial-bound; CoreAudio-устройство ровно **«WING»**, 48 in / 48 out / 48 kHz
- [ ] Dante Virtual Soundcard **выключен** (детектор его не принимает)
- [ ] **Не-тихий фид** на Main 47/48 (музыка/тон; прошлая попытка падала на цифровом нуле)
- [ ] **PA и мониторы заглушены** до окончания всех записей

## 3. Backend с токеном управления (обязательно)

Без токена backend переводится в observation-only и `reconcile`/`reset` отклоняются с `control_authentication_required`.

```bash
cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
export AUTOMIXER_WS_AUTH_TOKEN=$(backend/venv/bin/python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=').decode())")
echo "token: $AUTOMIXER_WS_AUTH_TOKEN"   # 43 символа, сохраните для шагов 4–6
export AUTOMIXER_WS_CLIENT_ID=operator-recovery
export AUTOMIXER_WS_HOST=127.0.0.1 AUTOMIXER_WS_PORT=8765
PYTHONPATH=backend backend/venv/bin/python backend/server.py
```

Затем в UI (frontend) выполните **observation-connect** к WING — без живого подключения reconcile невозможен (нужен физический readback).

## 4. Шаг 1 — диагностика (read-only, безопасно)

```bash
backend/venv/bin/python tools/manual_write_recovery_operator.py status
```

Ожидаем: `emergency_stopped: False`, `wal_unresolved_transaction_ids: []` → рельс №1 уже снят, идите к шагу 6.
Если `emergency_stopped: True` или список не пуст — шаги 5–6.

## 5. Шаг 2 — reconcile WAL (нужен живой пульт и ЧЕЛОВЕК)

`--approval-id` — это **аттестация оператора**: перед запуском посмотрите на пульт глазами и убедитесь, что фактическое состояние адресов из WAL-чекпоинтов вас устраивает (для прошлых прогонов это `/main/1/mute` = muted после teardown). Approval-id пишется в терминальную строку WAL.

```bash
backend/venv/bin/python tools/manual_write_recovery_operator.py \
  --token "$AUTOMIXER_WS_AUTH_TOKEN" \
  reconcile --approval-id operator-<имя>-20260728 \
  --report ~/.automixer/runtime/operator_reconcile_$(date +%Y%m%dT%H%M%S).json
```

Успех: по каждому id `OK: reason=reconciled_committed | reconciled_no_change | reconciled_rolled_back` + readback-строки. При `BLOCKED` — читайте подсказку и таблицу в разделе 9.

## 6. Шаг 3 — reset emergency-stop

```bash
backend/venv/bin/python tools/manual_write_recovery_operator.py \
  --token "$AUTOMIXER_WS_AUTH_TOKEN" \
  reset --approval-id operator-<имя>-20260728
```

Успех: `RESET OK` и финальный `emergency_stopped: False`.
Внимание: `automation_remains_frozen=true` — автоматика остаётся замороженной (`automation_frozen_after_emergency_stop`), это отдельный следующий шаг, не входящий в runbook.

## 7. Шаг 4 — attended-саундчек HW01 → HW05 (единственный легальный путь записей)

`approved_live_write_kinds: []` — UI-apply закрыт. Записи делает только интерактивный runner (нужен TTY):

```bash
# HW01: чистый M120 route-only повтор с verified teardown (текущий блокер)
backend/venv/bin/python tools/wing_attended_soundcheck_runner.py \
  --verify-route-only \
  --approval-id operator-<имя>-20260728 \
  --confirmation CAPTURE_WING_MAIN_USB_47_48_ROUTE_PROOF \
  --wing-ip <ip WING> --audio-device-index <index>

# HW02–HW05: полный attended-прогон
backend/venv/bin/python tools/wing_attended_soundcheck_runner.py \
  --execute \
  --approval-id operator-<имя>-20260728 \
  --confirmation EXECUTE_M110_ATTENDED_WING_SOUNDCHECK \
  --wing-ip <ip WING> --audio-device-index <index>
```

Runner сам поднимает proof-owned backend с токеном (через FD), своим портом (default 18765) — основной backend на 8765 не конфликтует.

## 8. После сеанса

- Артефакты runner'а и транскрипты скрипта остаются в `~/.automixer/runtime/` и `artifacts/` — **не коммитить**.
- `git status --short | grep -v '^??'` — tracked-дерево чистое? Если агенты снова сдвинули маркеры — нужен rebind-коммит CP05 (как `ea5f5d737`).
- Обновите `ops/autopilot/STATE.json` только через сам автопилот — не правьте вручную.
- Повторный прогон `tests/test_wing_mvp_rails.py -q` перед любыми заявлениями о готовности.

## 9. Troubleshooting (reason → действие)

| reason | Действие |
|---|---|
| `control_authentication_required` | Backend без токена → шаг 3, передайте `--token` |
| `operator_approval_required` | Нет/пустой `--approval-id` |
| `unresolved_transaction_not_found` | id не из списка → шаг 4, возьмите актуальные id |
| `physical_readback_transport_unavailable` | Нет observation-connect к WING → подключите в UI |
| `physical_recovery_read_failed` / `_unverified` | OSC-связь: L2-сеть, порт 2223, повтор |
| `recovery_state_mixed_or_changed` | Пульт не совпадает ни с target, ни с preimage → решить ВРУЧНУЮ, выставить верное состояние на пульте, повторить |
| `console_write_recovery_pending` | Reset раньше reconcile → шаг 5 |
| `emergency_stop_teardown_in_progress` / `_incomplete` | Подождать / смотреть teardown_errors в ответе |
| `restore_only_proof_target_cannot_be_reconciled_committed` | Ожидается откат к preimage, не committed |

## Приложение A — сырые WS-payload'ы (если скрипт недоступен)

```json
{"type": "authenticate", "token": "<43 chars>", "client_id": "operator-recovery"}
{"type": "get_manual_write_supervision"}
{"type": "reconcile_console_write_recovery", "transaction_id": "<id>", "approved": true, "approval_id": "<operator>"}
{"type": "reset_manual_write_emergency_stop", "approved": true, "approval_id": "<operator>", "reason": "operator_requested_emergency_reset"}
```

Ответы: `authentication_status`, `manual_write_supervision_status`, `console_write_recovery_result`, `manual_write_emergency_reset_result`. Корреляции по request-id **нет** — ждите ответ по `type`, пропуская broadcast'ы `mixer_update`/`state_update`.
