# Archive — дубликаты и неиспользуемые модули

Перенесено при сборке единой версии (2026-03-17).

## Дубликаты load_snap

- `load_snap.py` — базовая версия
- `load_snap_v2.py` — альтернативная реализация
- `find_and_load_snap.py` — поиск + загрузка
- `scan_and_load_snap.py` — сканирование + загрузка

**Используется:** `load_snap_final.py` + `wing_client.load_snap()`

## Альтернативные реализации (не в основном flow)

- `auto_fader_hybrid.py` — 6-level hybrid (Kimi), не подключён к server
- `auto_compressor_cf.py` — CF-based adaptive compressor, не подключён

**Основные:** `auto_fader.py` (AutoFaderController), `auto_compressor.py` (AutoCompressorController)

## Agents (ODA pattern, не интегрированы)

- `agents/gain_agent.py` — базовый GainAgent, superseded by SafeGainCalibrator
