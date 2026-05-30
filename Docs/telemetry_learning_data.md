# Telemetry Learning Data

## Зачем это нужно

Telemetry Learning Data - это безопасный журнал концертов, саундчеков,
репетиций и offline-тестов. Он нужен, чтобы Automixer постепенно накапливал
обучающие данные: что происходило на каналах, какие были уровни, какие
действия делал звукорежиссер, какие были SPL-измерения и какие метки качества
были поставлены.

Этот слой ничего не меняет в пульте. Он только читает уже доступные данные,
нормализует их и пишет локальные файлы.

## Safety

По умолчанию:

- `telemetry.enabled: false`;
- `telemetry.spl.enabled: false`;
- `telemetry.safety.allow_live_writes: false`;
- `telemetry.safety.telemetry_read_only: true`.

Модуль не отправляет OSC/MIDI/WING-команды. Интеграция с WING разрешена только
как чтение cached state или meter/audio-capture данных.

## Где лежат данные

По умолчанию:

`data/telemetry/sessions/<session_id>/`

Файлы сессии:

- `session.json` - описание сессии;
- `scene_metadata.json` - площадка, PA, мониторы, заметки инженера;
- `channel_telemetry.jsonl` - состояние каналов;
- `bus_telemetry.jsonl` - bus/master измерения;
- `spl_telemetry.jsonl` - SPL-измерения;
- `engineer_actions.jsonl` - действия человека или Automixer;
- `labels.jsonl` - ручные/автоматические метки качества;
- `manifest.json` - список файлов, счетчики, версия схемы и safety mode;
- `summary_report.md` - отчет, если запущен exporter.

## Как запустить тестовую сессию

```bash
PYTHONPATH=. python3 -m backend.telemetry.session_manager \
  --storage-dir data/telemetry/sessions \
  start \
  --mode soundcheck \
  --venue "Test Venue" \
  --artist "Test Artist"
```

Команда `start` является явным действием оператора. Автоматический runtime
сбор по конфигу остается выключенным, пока `telemetry.enabled` не включен
отдельно.

Остановить сессию:

```bash
PYTHONPATH=. python3 -m backend.telemetry.session_manager \
  --storage-dir data/telemetry/sessions \
  stop \
  --session-id <session_id> \
  --reason "soundcheck complete"
```

## SPL CSV

Можно подготовить CSV:

```csv
timestamp_ms,laeq_1s,lcpeak,lafmax,source,calibration_notes
1710000000000,86.2,103.5,91.0,external_meter,A-weighted calibrated meter
```

`FileSPLProvider` читает CSV или JSONL. Если SPL-файла нет, можно использовать
`MockSPLProvider` для тестов.

## Export training timeline

Собрать единый timeline по timestamp:

```bash
PYTHONPATH=. python3 -m backend.telemetry.learning_export \
  --storage-dir data/telemetry/sessions \
  timeline \
  --session-id <session_id> \
  --output data/telemetry/sessions/<session_id>/training_timeline.jsonl
```

Сформировать summary:

```bash
PYTHONPATH=. python3 -m backend.telemetry.learning_export \
  --storage-dir data/telemetry/sessions \
  summarize \
  --session-id <session_id>
```

## Как это поможет AI-агентам

Будущие агенты смогут использовать эти данные так:

- Balance Agent: учиться по действиям фейдеров и уровню каналов;
- Gain Agent: сравнивать trim/fader решения с peak, RMS, LUFS и headroom;
- AutoEQ Agent: сопоставлять instrument labels, EQ summary и метки качества;
- Dynamics Agent: анализировать compressor/gate snapshots и действия инженера;
- Safety Supervisor: проверять, какие решения были безопасны, заблокированы или
  подтверждены;
- Offline Evaluator: сравнивать сессии, метки качества, SPL и master metrics.

## Что пока не реализовано

- Автоматический runtime polling в backend server.
- Parquet export.
- Настоящая SPL-интеграция с измерителем.
- Автоматическая разметка accepted/rejected решений инженера.
- Live WING write integration. Это намеренно не добавлено.
