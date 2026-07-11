# Proposal: Soundcheck Workflow 95

## 1. Short thesis

Разделить измерение, построение рекомендаций и запись в пульт; сделать первые
два этапа полноценными и наблюдаемыми, а третий оставить только за существующим
supervised authority gate.

## 2. What I understood

Пользователю нужен реально полезный режим саундчека, а не экран с пустой
очередью. При этом текущий M001 запрещает расширять live authority без M005/M006.
Следовательно, честная цель этой итерации — production-quality proposal-only
workflow с доказанным zero-write поведением.

## 3. Proposed solution

- Всегда выполнять отдельную dry-run derivation фазу после анализа.
- Fail-close отключить прямой CLI live apply и YAML promotion как источник
  полномочий.
- Внедрять shared mixer/audio services и учитывать ownership lifecycle.
- Добавить Soundcheck view в реально используемый TypeScript frontend.
- Добавить абсолютные baseline/target значения, stable recommendation ids и
  long-run тест отсутствия накопительного drift.
- Публиковать раздельные оценки local workflow и live release authority.

## 4. Likely files to touch

- `backend/auto_soundcheck_engine.py`
- `backend/handlers/soundcheck_handlers.py`
- `backend/server.py`
- `frontend/src/*`
- `tests/test_soundcheck_handlers.py`
- `tests/test_auto_soundcheck_engine_safety.py`
- frontend component tests
- `Docs/adr/`, `Docs/plans/`, `ops/autopilot/*`

## 5. Alternatives considered

- Вызвать существующий apply прямо из handler: отклонено без явного dry-run
  контракта.
- Разрешить CLI apply только флагом: отклонено, потому что это второй writer
  path вне canonical authority gate.
- Оживить legacy `AutoSoundcheckTab.js`: отклонено, потому что production entry
  использует другой TypeScript cockpit.

## 6. Risks

- Старые тесты могут считать analysis-only корректным proposal-only режимом.
- Shared runtime может иметь несовместимый lifecycle.
- UI может показать локальную готовность как разрешение live apply.

## 7. Test plan

- Zero-write proposal derivation test.
- CLI/YAML bypass rejection tests.
- Shared-service ownership tests.
- 24-cycle idempotence/drift test.
- Frontend start/stop/status/error tests.
- Standard backend suite, frontend suite/build, no-WING release gate.

## 8. Where another reviewer may disagree

Можно предпочесть отдельный recommendation engine вместо reuse существующих
correction builders. На этой итерации безопаснее выделить явный dry-run contract
в существующем engine и покрыть его границы тестами, сохраняя меньший diff.

Независимый Kimi review не запускается: в этом checkout действует single-writer
режим, а доступного отдельного council runner в текущей сессии нет. Решение
поэтому считается Codex proposal с обязательным итоговым diff review.

