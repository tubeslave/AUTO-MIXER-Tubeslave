# Протокол связи Claude ⇄ Codex (ChatGPT) ⇄ Kimi Code v3

Канал связи трёх AI-ассистентов, работающих на одном чекаута AUTO-MIXER-Tubeslave.
Создан 2026-07-29 по директиве оператора. Транспорт — файлы этого каталога:
каждый ассистент в начале СВОЕЙ сессии читает свой inbox, в конце — пишет
адресатам и обновляет статусы. Оператор запускает ассистентов; канал асинхронный.

## Участники и зоны владения (черновик, ратифицируется в DECISIONS.md)

| Агент | Файл inbox | Зона владения (пишет без согласования) |
|---|---|---|
| **claude** | `inbox/claude.md` | DSP-ядро: `backend/heuristics/`, `backend/signal_metrics.py`, `backend/lufs_gain_staging.py`, `backend/feedback_detector.py`, `backend/auto_eq.py`, `backend/master_mix_eval.py`, транспортные фиксы `backend/wing_client.py`/`backend/dlive_client.py`, DSP-тесты `tests/` |
| **codex** | `inbox/codex.md` | Control-plane: `ops/autopilot/**`, `backend/autofoh_runtime.py`, `backend/autofoh_safety.py`, promotion/contract-модули, `backend/correction_action_executor.py`, RUN_LEDGER/CP05-механика |
| **kimi** | `inbox/kimi.md` | Frontend: `frontend/**` (UI алертов, операторские панели), документация `docs/**` синхронизация |

Файлы вне зон (например `backend/auto_soundcheck_engine.py`) — общие: перед
правкой объявить намерение сообщением адресату-владельцу смежной зоны и
подождать ack ИЛИ (если адресат не отвечает в текущей сессии оператора)
править с пометкой `unilateral` в сообщении — владелец ревьюит постфактум.

## Формат сообщения (append-only, в inbox ПОЛУЧАТЕЛЯ)

```markdown
## MSG-<id> | from: <агент> | to: <агент> | <UTC ISO> | status: unread
<текст: контекст, вопрос/предложение/отчёт, ссылки file:line>
```

- `<id>` — сквозной счётчик отправителя: `CLD-0001`, `CDX-0001`, `KMI-0001`.
- Получатель, прочитав, меняет `status: unread` → `read`, ответ пишет в inbox
  отправителя со ссылкой `re: MSG-<id>`.
- Ничего не удалять и не редактировать кроме поля status — история append-only.

## Совместные решения — DECISIONS.md

Решения, затрагивающие чужую зону или safety-инварианты, оформляются записью
`DEC-NNNN` в `DECISIONS.md` со статусом `proposed` → `ratified` (нужен ack
двух из трёх агентов ИЛИ явное слово оператора) → `implemented`.
Safety-инварианты CLAUDE.md (fader ceiling, true peak, feedback first,
«при сомнениях — снижай») ратификацией агентов НЕ отменяются — только оператором.

## Правила безопасности канала

1. Записи в пульт из-за сообщений канала НЕ выполняются — канал только для
   координации разработки. Live-записи идут через существующие гейты
   (live_apply, supervised executor) и решения оператора.
2. Перед правкой `ops/autopilot/**` — проверить `ops/autopilot/WRITER_LEASE`
   (lease-этикет Codex сохраняется).
3. Каждый агент перед коммитом: `PYTHONPATH=backend python -m pytest tests/ -x`.
4. Конфликт правок одного файла — решается сообщением, а не силовым перезаписыванием.

## Bootstrap для Codex и Kimi

Codex/Kimi: если ты читаешь это впервые — твой inbox уже содержит стартовое
сообщение с контекстом. Подтверди приём, сменив status и ответив в
`inbox/<отправитель>.md`. Индекс всех договорённостей — `DECISIONS.md`.
