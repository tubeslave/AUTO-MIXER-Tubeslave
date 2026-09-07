---
description: Полная верификация перед коммитом — pytest, safety sweep, секреты, diff
---

# /verify — верификация текущего состояния

Выполни фазы строго по порядку. Если фаза падает — сначала разберись с ней, потом продолжай.

## Фаза 1: Тесты

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Если тесты падают — СТОП: почини или явно объясни оператору, почему падение ожидаемо.

## Фаза 2: Safety sweep изменённых файлов

```bash
python3 .claude/hooks/py_safety_check.py --sweep
```

Проверяет все изменённые/новые .py: синтаксис, fader > 0 dBFS, json.dumps+NumPy без
convert_numpy_types, вызовы gcc_phat (порядок аргументов), load_snap (snap safety), секреты.

## Фаза 3: Скан секретов по diff

```bash
git diff HEAD | grep -nEi "sk-[A-Za-z0-9]{16,}|(api_key|apikey|token|secret|password)\s*=\s*[\"'][A-Za-z0-9_-]{12,}" || echo "секретов в diff нет"
```

В истории проекта уже был инцидент с неротированным OpenAI-ключом — эта фаза обязательна.

## Фаза 4: Обзор diff

```bash
git diff --stat HEAD
```

Просмотри каждый изменённый файл на предмет: непреднамеренных изменений, отсутствующей
обработки ошибок, нарушений «Зафиксированных правил» CLAUDE.md (thread safety через
ThreadSafeMixerState, приоритет feedback_detector, Ayaic-плоскости для gain staging).

## Фаза 5: Отчёт

```
VERIFICATION: [PASS/FAIL]

Tests:    [X passed / Y failed]
Safety:   [OK / N находок]
Secrets:  [OK / найдены]
Diff:     [N файлов]

Готово к коммиту: [ДА/НЕТ]

Что исправить:
1. ...
```

Если изменения затрагивают DSP или управление пультом (backend/wing_client.py,
dlive_client.py, feedback_detector.py, agents/, handlers/) — предложи оператору
дополнительно запустить агента dsp-safety-reviewer.
