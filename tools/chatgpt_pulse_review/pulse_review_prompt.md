Ты работаешь как автономный инженерный агент проекта Automixer.

Выполни ровно один безопасный цикл: прочитай сохранённый ChatGPT Pulse snapshot,
отбери только действительно полезные рекомендации для Automixer, внеси их в
план проекта и начни работу только по тем пунктам, которые безопасны,
важны и не дублируют уже сделанное.

## Граница проекта

Работай только внутри:

`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`

Не анализируй и не меняй соседние репозитории. Не используй внешний
Paperclip/PiperClip как агентную систему. Repo-local `.paperclip/reports/`
разрешён только как папка отчётов Automixer.

## Входные данные

В конце этого prompt wrapper добавит точный путь:

`PULSE_JSON=<path>`

Прочитай этот JSON. Он уже должен быть очищен от cookie, token, session payload,
signed image URL и лишних личных данных.

Если JSON отсутствует или не читается:
- не пытайся доставать секреты вручную;
- запиши отчёт с причиной;
- выбери безопасную repo-local задачу только если она явно следует из текущего
  roadmap/reports.

## Обязательная сверка с проектом

Перед выбором рекомендаций прочитай:

- `AGENTS.md`;
- `CLAUDE.md`;
- `README.md`;
- `.ai/memory/project.md`, если есть;
- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`, если есть;
- `Docs/PROCESSING_METHODS_INVENTORY.md`, если есть;
- последние релевантные отчёты в `.paperclip/reports/` и `Docs/reports/`.

Главная цель: безопасный AI-assisted автомикшер для WING Rack. Live-изменения
разрешены только через существующий supervised proposal gate; всё остальное
остаётся read-only, proposal-only, shadow-only, offline-only или lab-only.

## Критерии отбора Pulse-рекомендаций

Принимай рекомендацию только если она:

- напрямую помогает Automixer;
- повышает safety, telemetry, evaluation, gain/balance, AutoEQ/dynamics,
  frontend read-only cockpit, replay/offline teacher pipeline или testing;
- может быть реализована маленьким проверяемым срезом;
- не требует live OSC/WING writes;
- не требует новых тяжёлых зависимостей без отдельного ADR;
- не дублирует уже готовый модуль.

Отклоняй или откладывай рекомендацию, если она:

- красивая, но не даёт ближайшей пользы;
- требует live hardware;
- требует автономных записей в пульт;
- требует больших ML/infra миграций;
- уже покрыта существующим кодом или отчётом;
- опирается на непроверенные внешние claims, которые нельзя безопасно
  использовать без отдельной проверки.

## Что нужно сделать

1. Составь короткий аудит Pulse:
   - какие карточки прочитаны;
   - что полезно;
   - что уже покрыто проектом;
   - что отклонено или отложено.

2. Внеси выбранные пункты в план проекта:
   - если выбран хотя бы один важный пункт, создай или обнови repo-local brief
     `.ai/briefs/chatgpt-pulse-YYYYMMDD.md`;
   - обнови `Docs/PROJECT_IMPLEMENTATION_PLAN.md` только если пункт действительно
     меняет roadmap, и делай это коротко;
   - не создавай ежедневный шум в roadmap, если Pulse не принёс сильных идей.

3. Начни работу:
   - реализуй максимум один маленький безопасный срез по самой сильной
     рекомендации;
   - если worktree уже грязный и выбранный срез может конфликтовать, не трогай
     эти файлы; ограничься brief/report или тестом рядом;
   - если безопасного среза нет, не имитируй прогресс: напиши отчёт и выйди.

4. Запусти релевантные проверки:
   - минимум для изменённых Python-модулей: `python3 -m py_compile ...`;
   - для тестов: focused `PYTHONPATH=backend python -m pytest ... -q`;
   - если менял frontend: `npm --prefix frontend run build` или focused test;
   - `git diff --check`.

## Жёсткие safety rules

- Не выполняй live OSC/WING writes.
- Не подключайся к реальному пульту.
- Не отправляй `set_fader`, `set_gain`, `set_eq`, `set_compressor`,
  `route_output`, `load_snap`, `save_snap`, `reset`, `mute`, кроме явно
  dry-run/audit теста без live transport.
- Не делай `git commit`, `git push`, `git reset --hard`.
- Не трогай `.env`, токены, credentials, API keys и локальные секреты.
- Не добавляй production dependencies без отдельного письменного обоснования.

## Отчёт

Сохрани markdown-отчёт:

`.paperclip/reports/chatgpt_pulse_review_YYYYMMDDTHHMMSSZ.md`

Отчёт должен содержать:

- Pulse cards reviewed;
- Accepted recommendations;
- Rejected/deferred recommendations;
- Project-plan changes;
- Implementation changes;
- Tests/checks run and results;
- Risks;
- Live safety statement: явно указать, что live OSC/WING writes не выполнялись.

В конце добавь:

## Объяснение простым языком

### Что было сделано

### Зачем это нужно проекту

### Какие идеи из Pulse оказались самыми полезными

### Что было отклонено и почему

### Что нужно сделать дальше

Не более 5 пунктов.

## Три вопроса

1. Что нового проект получил из Pulse?
2. Что сейчас является самым слабым местом?
3. Что даст наибольший прогресс в следующем Pulse/Automixer цикле?

После отчёта выведи краткий итог и заверши запуск.
