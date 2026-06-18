# Контракт автономного выполнения

Status: active control-plane contract

## Область

Этот контракт управляет разработкой Automixer от текущего verified state до
Project Completion Gate. Он не дает live authority и не разрешает unattended
WING/OSC writes.

## Минимальное чтение на цикл

Каждый autonomous wake-up читает только нужное для продолжения:

- `AGENTS.md`
- `CLAUDE.md`
- `Docs/PROJECT_GOAL.md`
- `Docs/PROJECT_EXECUTION_CONTRACT.md`
- `Docs/PROJECT_COMPLETION_GATE.md`
- `Docs/CURRENT_STATE.md`
- `Docs/ROADMAP.md`
- `Docs/ACTIVE_MILESTONE.md`
- нужные строки `Docs/CAPABILITY_MATRIX.md`
- нужные строки `Docs/CONSOLE_CAPABILITY_MATRIX.md`
- `Docs/SAFETY_CONTRACT.md`
- active plan in `Docs/plans/`
- `ops/autopilot/STATE.json`
- `ops/autopilot/BLOCKERS.md`
- recent git diff and commits

Полный репозиторий перечитывается только при смене milestone, непонятном
regression, конфликте состояния или доказанном drift.

## Single-writer lease

Один пишущий агент на checkout. Lease атомарно берется через создание
директории:

`ops/autopilot/WRITER_LEASE/`

Внутри lease записывается `owner.json` со schema, pid/thread id when available,
started_at_utc, heartbeat_utc, intended_action и checkout path. Если директория
уже существует и heartbeat не старше 30 минут, новый writer не начинает работу.
Если heartbeat старше 30 минут, следующий run может пометить lease stale,
записать это в `RUN_LEDGER.jsonl`, удалить stale lease и продолжить с первого
непроверенного checkpoint.

Существующий `.codex_autoloop.lock` из `tools/codex_autoloop/run_codex_cycle.sh`
остается совместимым legacy lock для CLI autoloop. Новый autopilot обязан
уважать оба lock-механизма.

## STOP file

`ops/autopilot/STOP` вручную останавливает новую автономную разработку на
следующем safe checkpoint. При обнаружении STOP automation:

- не начинает product changes;
- завершает или безопасно откатывает незавершенную atomic operation;
- освобождает writer lease;
- ставит `program_status` в `STOPPED`;
- обновляет `HEARTBEAT.md`, `STATE.json`, `RUN_LEDGER.jsonl` и `LAST_REPORT.md`.

Setup run не создает `ops/autopilot/STOP`.

## Цикл выполнения

1. Проверить STOP, lease, heartbeat, thread state и active Goal.
2. Если другой non-stale run работает, не писать файлы и выйти.
3. Восстановить состояние из committed files, `STATE.json`, ledger и git.
4. Продолжить с первого checkpoint без verified evidence.
5. Сделать минимальный defensible diff.
6. Добавить или обновить тесты, если меняется исполняемое поведение.
7. Запустить targeted checks после checkpoint.
8. На milestone boundary запустить полный validation contract milestone.
9. Провести independent regression and safety review.
10. Проверить diff на unrelated changes.
11. Записать concrete evidence.
12. Обновить current state, matrices, decisions, active plan and ledger.
13. Сделать один atomic commit для verified logical checkpoint.

Checkpoint нельзя закрывать при failing required tests. Success нельзя
декларировать только по statement агента.

## Политика ошибок

После трех materially different failed approaches по одному item:

- записать все попытки и evidence;
- пометить item blocked;
- выбрать другой safe independent checkpoint или milestone;
- не повторять тот же подход бесконечно.

При regression агент остается внутри текущего milestone, диагностирует и
чинит или безопасно откатывает regression. Roadmap не продвигается на broken
baseline.

## Goal transition

Native thread Goal management является предпочтительным механизмом. В текущей
Codex app среде доступен набор goal tools для текущей thread session, но
scheduled heartbeat automation должен каждый раз проверять наличие этих tools
перед заменой Goal. Если native Goal update unavailable, automation не
притворяется, что Goal установлен: она обновляет repo state, печатает точную
ручную `/goal` команду в `LAST_REPORT.md` и продолжает только безопасные state
checks.

Fallback controller, если когда-либо будет добавлен, может управлять только
Codex threads/goals. Он не может менять Automixer product files, обращаться к
hardware или обходить single-writer serialization.

## Разрешения

- Работать в workspace-scoped checkout.
- Network disabled by default.
- Не использовать `danger-full-access` или `--yolo` как требуемый режим.
- Не push, merge, force-push или rewrite history без явной политики.
- Не трогать secrets, `.env`, tokens или API keys.

