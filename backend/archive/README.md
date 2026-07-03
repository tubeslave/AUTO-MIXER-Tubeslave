# Архив неиспользуемого кода

Модули, которые на момент ревью 2026-07-03 **не импортируются** ни боевым
кодом (backend/, handlers/), ни тестами (tests/). Проверка: grep по
`import <module>` / `from <module>` по всему репозиторию.

Перенесены сюда, а не удалены, чтобы сохранить историю и возможность
вернуть. Прежде чем возвращать модуль в backend/, добавьте тесты и
регистрацию в handlers.

## Что здесь и почему

| Модуль | Причина |
|--------|---------|
| `auto_fader_hybrid.py` | Третья реализация авто-фейдера; вытеснена `auto_fader.py` (боевая) и `auto_fader_v2/` |
| `auto_eq_processing.py` | Компоненты DualEMA/MirrorEQ/PriorityMatrix — не подключены к auto_eq.py |
| `auto_effects.py` | Скелет без реализации |
| `ableton_client.py` | Ableton Link — интеграция не завершена |
| `list_snapshots.py`, `list_all_snapshots.py` | Отладочные утилиты снапшотов |
| `check_*.py`, `find_*.py`, `query_*.py`, `route_*.py` | Одноразовые отладочные скрипты маршрутизации/проверок |
| `example_*.py` | Примеры использования API |
| `monitor_osc_routing.py`, `disable_modules_set_faders.py` | Отладочные скрипты |
| `analyze_test_results.py`, `auto_test_routing.py` | Вспомогательные скрипты тестирования на железе |

## Что осталось в backend/, хотя не подключено к server.py

Эти модули НЕ архивированы, потому что покрыты тестами и являются
рабочими библиотеками для будущей интеграции:

- `auto_gate_caig.py` — CAIG-гейт (tests/test_gate.py, test_safety_regressions.py)
- `auto_panner.py`, `auto_panner_adaptive.py` — панорамирование (test_safety_regressions.py)
- `tub345_supervised_write_runner.py` — раннер supervised-записей (собственный тест)
- `wing_telemetry_compare.py` — сравнение телеметрии (собственный тест)
