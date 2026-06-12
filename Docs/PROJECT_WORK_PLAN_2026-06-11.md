# AUTO-MIXER-Tubeslave: план работы
Дата: 2026-06-11
Ответственный контекст: текущая ветка `offline-smarteq-eq-song-repa-20260602`
Цель: привести проект к стабильному концертному MVP с безопасным live-путем и понятной offline-зоной.

## Принципы
- Безопасность live операций важнее удобства.
- Воспринимать `enhanced_osc_client.py` как уже готовое решение и интегрировать, а не писать заново.
- Каждое значимое изменение фиксировать滴滴 ADR в `Docs/adr/<task-id>.md`.
- Минимально необходимый дифф: тесты, дифф, запись решения.
- Не вносить production зависимости без обоснования.
- Хранить `Docs/PROJECT_STRUCTURE_MAP.md` и `Docs/CONVENTIONS.md` синхронно с кодом.

## Целевое состояние MVP
- Live-путь: аудио -> анализ -> безопасная корректировка -> WING/DLive с восстановлением связи и явными write-policy.
- Offline-путь: один канонический офлайн-путь, остальные маркированы `legacy`/`optional`.
- Наблюдаемость: логи, health-check, метрики минимума для оператора.
- Тестирование: базовые защитные сценарии для критических bug классов.

## Работа по фазам

### Фаза A — Стабилизация критической инфраструктуры
Цель: убрать риски, которые уже известны и не требуют дизайна новых функций.

A1. Исправить подтверждённые критические баги
- B1: `backend/auto_phase_gcc_phat.py` — исправить порядок аргументов `gcc_phat(target, reference)`.
- B3: `backend/agents/base_agent.py` — заменить условие выхода из цикла на поддержку `RUNNING` и `PAUSED`, внутри `PAUSED` использовать `await asyncio.sleep(...)`.
- B2: `backend/signal_analysis.py` — исправить `ratio_float_to_wing()`: либо вернуть корректные `WING_RATIO_VALUES`/`WING_RATIO_STRINGS` на уровне модуля, либо вынести соответствие в словарь.

A2. Включить существующий enhanced OSC путь
- Назначить `enhanced_osc_client.py` как основной путь к WING, оставив `wing_client.py` временно для обратной совместимости.
- Добавить явный write-policy: read-only / proposal-only / supervised live write / blocked.
- Покрыть простейшими тестами reconnect и fallback при недоступном микшере.

A3. Выделить единый AudioCapture-сервис
- Убрать локальные PyAudio-потоки из DSP модулей.
- Создать единый сервис с одним подписковым буфером.

A4. Добавить gate和安全-защиты в write path
- Ограничить fader gain повышение при наличии признаков bleed или недостатка true-peak headroom.
- Вынести правило в `Docs/SAFETY_RULES.md`, чтобы оно присутствовало в документации независимо от версии CLAUDE.md.

### Фаза B — Архитектура и наблюдаемост
Цель: сделать live-путь поддерживаемым и прозрачным в работе.

B1. Декомпозиция `server.py`
- Выделить WebSocket endpoint, routing, orchestration.
- Сохранить существующий контракт handlers на период миграции.

B2. Добавить thread safety в агентах
- Ввести явные блокировки shared state в `gain_agent.py`, `eq_agent.py`, `fader_agent.py`.
- Добавить coverage тесты на параллельный доступ к `ThreadSafeMixerState`.

B3. Ввести ограничения роста runtime-данных
- Ограничить историю конфликтов в coordinator, калибровочные данные в dynamic_mixer.
- Подключить periodic cleanup.

B4. Добавить базовую представление о состоянии системы
- Health endpoint / structured log / метрики времени выполнения.
- Для starts: graceful shutdown через существующий coordinator.

### Фаза C — Канонизация кодовой базы
Цель: сделать ясным, что canonical, что experimental, что legacy.

C1. Классификация путей offline
- `Docs/PROJECT_STRUCTURE_MAP.md` обновить с метками: canonical, optional, legacy, quarantine.
- Offline: назначить `automixer/production_mix_v1/` entrypoint.

C2. Организация backend root
- Перенести legacy test/manual-скрипты в явный namespace.
- Весь runtime source идентифицируем в пределах 5 минут.

C3. Удаление dead/duplicated code где возможно
- Удалить дубли LUFS-измерений, убрать устаревшие WING-скрипты из runtime path.

### Фаза D — Качество и расширяемость
Цель: поднять зрелость ML/AI и страховочных сценариев.

D1. Написать DSL unit-тесты
- LUFS measurement, GCC-PHAT задержка, EQ biquad, compressor ratio mapping.
- Цель по DSP: 80% критического пути покрыто базовыми сценариями.

D2. Связать модули masking и cross-adaptive EQ
- Вместо двух параллельных реализаций сделать единый anti-masking pipeline.

D3. Feedback-тракт как часть runtime
- Перенести feedback detection из best-effort в активный live guard.
- Добавить явный fallback и логирование инцидентов.

D4. Операционные сценарии
- Написать runbook для:
  - саундчека с WING offline режимом и Dante маппингом;
  - live режима с ограничениями по headroom/feedback;
  - восстановления OSC-соединения и rollback snapshot при критическом изменении.

## Acceptance criteria MVP
- Pass: `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` проходит стабильно на чистой машине.
- Pass: live write path проходит readonly preflight при недоступном микшере.
- Pass: после потери OSC-соединения и восстановления, fader values отражают актуальное console state или blocked fallback, а не устаревший кэш.
- Pass: найдено и зафиксировано в ADR, какой модуль отвечает за LUFS gain staging, и только он.
- Pass: удаление/перемещение legacy скрипта не ломает start_backend.sh.
- Pass: есть runbook на 2 сценария в `Docs/runbooks/`.

## Рекомендуемый порядок спринтов
1. Спринт 1: A1 + A2
2. Спринт 2: A3 + A4
3. Спринт 3: B1 + B2 + B3
4. Спринт 4: C1 + C2 + C3
5. Спринт 5: D1 + D2 + D3
6. Спринт 6: D4 + финальная проверка критериев

## Следующие действия сейчас
- Подтвердить, нужен ли максимально агрессивный scope к концертному MVP, или расширять ML/AI часть параллельно.
- Если нужен узкий MVP, начать с A1 и A2.
- Если нужен рост возможностей, начать с B2 и D1.
