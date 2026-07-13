# Цель проекта Automixer

## Авторитетная цель

Automixer должен стать профессиональным AI mixing engineer для live и studio
работы. Система должна анализировать многоканальное аудио, понимать источники и
роли в аранжировке, формировать безопасные решения по gain staging, балансу,
EQ, dynamics, de-essing, panning, frequency separation, unmasking, phase,
effects, sends, master bus и адаптации во время выступления.

Итоговая автономия сохраняется как долгосрочная North Star, но она не является
одной задачей реализации. Исполняемая единица разработки всегда должна быть
конечным milestone с независимой проверкой, rollback и evidence.

## Текущий приоритет

Текущий проверенный приоритет репозитория уже сужен до supervised WING Rack MVP.
Это не отменяет долгосрочную цель, но запрещает перескакивать к широкой live
автономии без доказанной безопасной hardware-петли.

Полный рабочий маршрут взаимодействия с реальным WING Rack зафиксирован в
`Docs/plans/M001-real-wing-interaction-plan.md`. Этот маршрут является частью
текущей цели: сначала read-only evidence и operator decision, затем узкий
supervised `fader`/`gain-down` proof с physical readback, rollback и
emergency-stop evidence, и только после этого последующее расширение
soundcheck-автоматизации отдельными hardware-backed решениями.

## Текущий план доведения soundcheck pipeline до рабочего состояния

План принят владельцем проекта 2026-07-10. Его цель — получить безопасный,
воспроизводимый supervised soundcheck на работающем WING Rack и строгую оценку
не ниже 90/100 только после подтвержденных runtime и hardware evidence.

1. Закрыть сетевой доступ к live-командам: loopback по умолчанию, а для
   удаленного WebSocket — обязательная аутентификация и fail-closed startup.
2. Удалить runtime-обходы Safety Kernel: `force_auto_apply`, headless
   AutoSoundcheck `auto_apply=True` и engine-owned live transport.
3. Сделать один системный emergency-stop контур, который останавливает manual,
   agent и AutoSoundcheck writers и проверяется непосредственно перед каждой
   отправкой и повторной попыткой.
4. Разделить commanded и physically observed WING state. Статус `applied`
   разрешен только после коррелированного physical readback; timeout,
   disconnect и mismatch блокируют дальнейшие записи.
5. Сделать rollback транзакционным: exact transaction id, fresh physical
   compare-and-swap и запрет перезаписывать более свежее ручное движение.
6. Сохранить live scope узким: только operator-approved `fader` и `gain-down`.
   EQ, dynamics, gate, pan, delay, polarity, send, DCA, routing, scenes, FX и
   master остаются proposal-only до отдельных ADR и hardware proof.
7. Пройти targeted regression, полный backend/frontend/E2E/no-WING контур,
   затем read-only WING/audio smoke и операторские M005/M006 проверки.
8. Обновить readiness и оценку только из принятых evidence artifacts. Зеленые
   unit tests или заявления агента не могут сами дать оценку 90+ или live
   authority.

Исполняемый архитектурный контракт этого плана фиксируется в
`Docs/adr/soundcheck-live-safety-kernel-20260710.md`.

## Результат реализации 2026-07-10

План реализован для рабочего supervised soundcheck scope. Canonical
AutoSoundcheck теперь измеряет реальные WING/USB сигналы и формирует evidence в
`proposal-only` режиме; физическая запись остается только за узким ручным gate.

- WebSocket всегда слушает только loopback; bearer token не разрешает
  non-loopback bind. Remote browser возможен только через отдельно проверенный
  same-origin HTTPS/WSS reverse proxy к `127.0.0.1:8765`.
- `force_auto_apply` удален из frontend и отвергается backend.
- AutoSoundcheck принудительно использует `observe_only=true` и
  `auto_apply=false`; observe-only цикл не сохраняет tracked learning profile.
- Emergency stop останавливает AutoSoundcheck, MixingAgent и активные
  контроллеры, а generation latch проверяется перед каждым retry.
- WING `commanded_state` отделен от физически наблюдаемого `state`; статус
  `applied` требует `verified_from_physical_readback`.
- Rollback требует transaction id и fresh physical compare-and-swap.
- Положительный live gain без true-peak evidence и readiness без свежего master
  true peak закрываются fail-closed.

Проверка: `5848 passed, 1 skipped`, frontend `98 passed`, production build и
Playwright `9 passed`; реальный WING Rack `192.168.1.102`, fw `3.0.5`, дал три
physical readback на CH3 и 43 активных USB meter channel. Canonical engine
дошел до `running` на CH3/9/10, на всех трех каналах обнаружен сигнал,
`dry_run=true`, `live_apply_enabled=false`.

Строгая оценка этого supervised soundcheck scope: **92/100**. Остаток до 100 —
физический M005 disconnect/timeout/stale-state fault proof, M006 owner review и
отдельные hardware-backed ADR для любых live-семейств шире fader/gain-down.
Подробности: `Docs/reports/SOUNDCHECK_PIPELINE_HARDENING_20260710.md`.

## Цель frontend live cockpit — 2026-07-10

Довести frontend Automixer до production-grade операторского cockpit с оценкой
не ниже 95/100. В нормальной работе интерфейс обязан автоматически подключаться
к Product Layer, показывать подтвержденный WING/audio/readback и не подменять
отсутствующие live-данные mock-значениями. Каждая активная кнопка должна вызывать
канонический безопасный backend-контракт; неподтвержденные live-ручки и фейдеры
остаются read-only или явно блокируются с причиной. Desktop предназначен для
звукорежиссера, mobile — для быстрого контроля музыкантом или оператором.

### Результат

- Исходная строгая оценка: **62/100**.
- Итоговая строгая оценка: **96/100**.
- Реальный WING `192.168.1.102` отображается в observation-only режиме.
- Реальный audio capture: 48 каналов, 48 kHz; тишина отображается как
  `NO SIGNAL`, а не как выдуманные LUFS/peak/spectrum.
- Refresh, Analyze, read-only Connect WING и desktop/mobile Start/Stop meters
  подключены к backend.
- Full Autonomous блокируется, когда backend не дает apply authority.
- Fake transport, system network/clock placeholders и local-only proposal
  actions удалены с основных операторских поверхностей.
- Проверка: frontend 102 passed, Playwright 9 passed, backend meter/audio 9
  passed, production build passed, browser console 0 errors/warnings.

Evidence и шкала: `Docs/reports/FRONTEND_COCKPIT_AUDIT_20260710.md`.
Архитектурный контракт: `Docs/adr/frontend-live-cockpit-20260710.md`.

MVP закрыт только когда есть реальные, принятые evidence artifacts для:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based only on real proof.

## Неприкосновенные границы

- Offline, shadow, replay, mock и simulator evidence не являются hardware proof.
- Full Autonomous Mixing Mode не может быть объявлен готовым по документации,
  отчету, зеленому unit test или LLM statement.
- Любая live WING запись до отдельного ADR и hardware proof ограничена только
  supervised `fader`/`gain` rail с явным approval, readback и rollback.
- EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA и full auto
  live writes остаются запрещенными для unattended automation.

## Источники текущей цели

- `.ai/project-layer/automixer_goal.md`
- `.ai/project-layer/autonomous_mvp_and_model_routing.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_mode_policy.py`
