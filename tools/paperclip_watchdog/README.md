# Automixer Paperclip Watchdog

Локальный watchdog каждые 3 минуты проверяет Paperclip и создает одну безопасную задачу по Automixer только когда агенты свободны: `running_agents=0` и `in_progress_tasks=0`. По умолчанию включен `dry-run`, поэтому реальные задачи не создаются.

Watchdog не подключается к WING RACK, не запускает audio capture и не меняет production-код Automixer. Он только читает Paperclip API и, после явного отключения dry-run, создает Paperclip issue.

## Настройка env

Скопируй пример:

```bash
cp tools/paperclip_watchdog/.env.example tools/paperclip_watchdog/.env
```

Заполни:

```bash
PAPERCLIP_API_URL=http://127.0.0.1:3100
PAPERCLIP_COMPANY_ID=<company-id>
PAPERCLIP_TOKEN=<token>
```

Токен не нужно добавлять в код или лог. Для `http://127.0.0.1:3100`, `http://localhost:3100` и `http://[::1]:3100` токен необязателен. Для внешнего Paperclip URL `PAPERCLIP_TOKEN` обязателен.

## Один проход

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py once
```

В `dry-run` режиме команда только покажет, какую задачу она создала бы.

## Цикл

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py loop
```

Цикл запускает `once` каждые `AUTOMIXER_WATCHDOG_INTERVAL_SECONDS` секунд. Остановка: `Ctrl+C`.

## Статус

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py status
```

Команда печатает нормализованные счетчики: running agents, in-progress tasks, error agents, blocked tasks и open backlog.

## Dispatch preview

Watchdog не является директором и не запускает агентов по умолчанию. Dispatch по умолчанию выключен:

```bash
AUTOMIXER_WATCHDOG_DISPATCH_ENABLED=false
AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN=true
AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS=
```

Dry-run preview показывает, какой `PATCH /api/issues/<issue-id>` был бы отправлен, но не меняет Paperclip state и не вызывает wakeup:

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch-preview
```

Для preview/dispatch нужны:

```bash
AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID=<issue-id>
AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID=<agent-id>
AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS=<agent-id>
```

Если `AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID` не задан, watchdog попробует взять последний `last_issue_id` из `.paperclip/watchdog_state.json`.

## Opt-in real dispatch

Реальный dispatch разрешен только при явном opt-in:

```bash
AUTOMIXER_WATCHDOG_DISPATCH_ENABLED=true
AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN=false
AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS=<agent-id>
AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID=<agent-id>
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch
```

Перед отправкой `PATCH` watchdog проверяет:

- `running_agents <= 0`;
- `in_progress_tasks <= 0`;
- issue status равен `todo`;
- active runs для issue отсутствуют;
- agent есть в allowlist;
- idempotency key еще не использован;
- dispatch method равен `patch_issue_assignment`;
- direct wakeup, live OSC и WING runtime writes запрещены policy-флагами.

Единственное действие при успешном real dispatch:

```http
PATCH /api/issues/<issue-id>
{"assigneeAgentId":"<agent-id>","status":"todo"}
```

`POST /api/agents/:id/wakeup` и legacy heartbeat invoke не используются как основной путь.

## Pause и resume

Поставить паузу:

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py pause
```

Снять паузу:

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py resume
```

Pause-файл по умолчанию: `.paperclip/watchdog_paused`. Пока он существует, задачи не создаются.

## Логи и state

Лог по умолчанию:

```text
logs/automixer_paperclip_watchdog.log
```

State-файл по умолчанию:

```text
.paperclip/watchdog_state.json
```

State хранит дневной счетчик, последнюю созданную задачу и список task keys, уже использованных за сутки. Это защищает от повторов и спама.

Для dispatch state дополнительно хранит последний attempt, `issue_id`, `selected_agent_id`, mode, result, reason, blocked reason и использованные idempotency keys. Повторный запуск с тем же key не отправляет повторный `PATCH`.

## Отключить dry-run

В `tools/paperclip_watchdog/.env` измени:

```bash
AUTOMIXER_WATCHDOG_DRY_RUN=false
```

Перед этим нужно подтвердить `PAPERCLIP_API_URL`, `PAPERCLIP_COMPANY_ID`, `PAPERCLIP_TOKEN` и реальные Paperclip endpoints. Дополнительно можно проверить mock:

```bash
AUTOMIXER_WATCHDOG_MOCK=true python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py once
```

## launchd

Plist уже создан:

```text
tools/paperclip_watchdog/com.tubeslave.automixer-paperclip-watchdog.plist
```

Установка вручную:

```bash
cp tools/paperclip_watchdog/com.tubeslave.automixer-paperclip-watchdog.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tubeslave.automixer-paperclip-watchdog.plist
launchctl enable gui/$(id -u)/com.tubeslave.automixer-paperclip-watchdog
```

Остановка:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.tubeslave.automixer-paperclip-watchdog.plist
```

Plist не устанавливается автоматически. Перед установкой проверь `.env`; launchd не должен хранить токены в plist.

## Paperclip endpoints

Дефолтный адаптер использует:

- `GET /api/health`
- `GET /api/companies/:companyId/dashboard`
- `GET /api/companies/:companyId/agents`
- `GET /api/companies/:companyId/issues`
- `GET /api/companies/:companyId/heartbeat-runs`
- `POST /api/companies/:companyId/issues`
- `GET /api/issues/:issueId`
- `GET /api/issues/:issueId/live-runs`
- fallback `GET /api/companies/:companyId/heartbeat-runs?issueId=:issueId`
- opt-in only `PATCH /api/issues/:issueId`

TODO перед отключением `dry-run`: подтвердить форму ответов `dashboard`, `agents`, `issues` и `heartbeat-runs` на живом Paperclip для компании Automixer.
