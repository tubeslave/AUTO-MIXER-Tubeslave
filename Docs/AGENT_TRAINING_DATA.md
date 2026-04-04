# Данные для обучения агентов (звукорежиссура / звукоинженерия)

Спецификация сбора, хранения и проверки данных для трёх направлений: **RAG/LLM**, **supervised ML** (`backend/ml`), **логи наблюдение–действие** (imitation / анализ). Согласовано с безопасностью live sound (см. `.cursorrules`, `config/default_config.json` → `safety`).

## Файлы и схемы

| Артефакт | Назначение |
|----------|------------|
| [schemas/mix_training_event_v1.schema.json](schemas/mix_training_event_v1.schema.json) | JSON Schema строки события (JSONL) |
| [schemas/dataset_manifest_v1.schema.json](schemas/dataset_manifest_v1.schema.json) | Манифест пакета файлов для обучения |
| [backend/ml/dataset_safety.py](../backend/ml/dataset_safety.py) | Проверки safety перед обучением |
| [backend/ml/training_dataset_io.py](../backend/ml/training_dataset_io.py) | Чтение JSONL + дисковый `Dataset` для классификатора |

### Локальный корпус на машине разработчика

Структурированный индекс книг (PDF + `extracted_text`), документа «LUFS → OSC», PDF по гейн-стейджингу, WING OSC, планов в `.cursor` и смежных материалов:  
`/Users/dmitrijvolkov/Documents/AUTOMIXER_AGENT_TRAINING_CORPUS.md`  
(не в git — персональные пути; при переносе на другую машину обновить индекс).

---

## 1. Событие микса v1 (`schema_version: "1.0"`)

Каждая строка JSONL — один таймстемп решения: метрики → предложение автоматики → (опционально) действие оператора.

**Обязательные поля:** `schema_version`, `event_id`, `recorded_at`, `source`.

**Рекомендуемые:** `session_id` (анонимный), `license`, `consent_training_use`, `channel`, `observation`, `automation`, `operator`, `safety`, `backend_git_sha`, `config_digest`.

Полная структура — в JSON Schema. Пример минимальной строки:

```json
{
  "schema_version": "1.0",
  "event_id": "01hz8k3e9qwerty7890abcd",
  "recorded_at": "2026-04-04T12:00:00Z",
  "source": "shadow_mode",
  "session_id": "sess_anon_7f3a",
  "license": "synthetic_only",
  "consent_training_use": true,
  "channel": { "index": 3, "instrument": "lead_vocal" },
  "observation": { "peak_dbfs": -12.4, "lufs_momentary": -20.1 },
  "automation": {
    "component": "gain_staging",
    "recommended_action": "trim_gain_db",
    "parameters": { "delta_db": -1.5 },
    "confidence": 0.72
  },
  "operator": { "approved": true },
  "safety": {
    "explicit_override_positive_fader": false,
    "max_fader_dbfs_ceiling_used": 0.0,
    "true_peak_limit_dbtp_used": -1.0
  }
}
```

### Политика анонимизации и лицензий

- **Не включать** в датасет: имена артистов, названия площадок, точные даты шоу (достаточно квартала или полностью убрать), IP консолей, геолокацию.
- **`session_id`:** случайный идентификатор на сессию, без ПИБ.
- **`license`:** явная метка происхождения (`proprietary_rehearsal`, `cc_by_4`, `synthetic_only`, `internal_runbook` и т.д.).
- **`consent_training_use`:** для прод-логов — только `true` при письменном согласии.
- **Аудиофайлы:** отдельный правовой режим; не смешивать в один корпус с логами без документа.

---

## 2. RAG: расширение `backend/ai/knowledge/`

### Источники (по возрастанию риска)

1. Внутренние runbooks и конспекты (ваша лицензия).
2. Протоколы консолей: `Docs/WING Remote Protocols`, [TOOLS.md](TOOLS.md).
3. Переводы/пересказы публичных статей — **с указанием источника и лицензии** в начале файла или секции.

### Чеклист качества перед добавлением `.md`

- [ ] Секции разбиты заголовками `##` (так парсит [KnowledgeBase](../backend/ai/knowledge_base.py)).
- [ ] Нет противоречий с [CONVENTIONS.md](CONVENTIONS.md) (LUFS ITU-R BS.1770-4, true peak, единицы dBFS/dBTP).
- [ ] Нет инструкций «поднимай фейдер выше 0 dBFS» без контекста безопасности и явного сценария мониторинга.
- [ ] Для live: приоритет формулировок «снижай при сомнении».
- [ ] Внешние цитаты: URL или библиография + лицензия.

---

## 3. Инвентаризация логов (текущее состояние и пробелы)

### Уже есть

| Компонент | Что пишется | Где смотреть |
|-----------|-------------|--------------|
| **Auto-EQ** | Старт/стоп, смена канала/профиля, применение коррекций, multi-channel прогресс | `logger` в [auto_eq.py](../backend/auto_eq.py), дубли в `logs/automixer-backend.log` (см. CLAUDE.md) |
| **Gain staging** | Состояние фаз LEARNING/APPLYING, коррекции (через контроллер) | [lufs_gain_staging.py](../backend/lufs_gain_staging.py), статус через server |
| **Снапшоты** | JSON файлы `presets/channel_backup_*.json`: `timestamp`, `channels` → результат `backup_channel()` | [server.py](../backend/server.py) `create_snapshot` |
| **MixingAgent** | Пока **не** интегрирован в server; в коде есть поля `AgentAction` (reason, source, confidence) | [agent.py](../backend/ai/agent.py) |

### Чего не хватает для пар «наблюдение → действие»

- Единый **структурированный** поток (JSONL) в формате события v1, а не только человекочитаемые `logger.info`.
- Явная связка: входные метрики автоматики → **рекомендация** → **факт применения** / одобрение UI.
- Режим **shadow_mode**: записывать «что бы сделала модель/агент» без отправки на FOH.

**Рекомендация по внедрению (следующий этап разработки):** небольшой модуль логирования (например `backend/training_telemetry.py`), вызываемый из handlers после расчёта и до/после `wing_client`, с опцией в `config` `training_telemetry.enabled` и путём файла.

---

## 4. Дисковый формат для `backend/ml`

### Общий манифест

Файл `manifest.json` рядом с данными, валидируемый [dataset_manifest_v1.schema.json](schemas/dataset_manifest_v1.schema.json): имя датасета, версия, `task`, список шардов `rows[].rel_path`, `safety_filter_applied`, `license`.

### Задача: классификатор канала (`task: instrument_class_mel`)

- **JSONL shard:** одна строка = один обучающий пример.
- Поля: `class` (строка из [INSTRUMENT_CLASSES](../backend/ml/channel_classifier.py)), `mel_npy` (путь относительно JSONL к файлу `.npy` формы `(1, n_mels, n_frames)` float32), опционально `schema_version: "1.0"`.
- Загрузчик: [training_dataset_io.py](../backend/ml/training_dataset_io.py) → `DiskInstrumentMelDataset`.

### Задача: gain/pan predictor (`task: gain_pan_multitrack`)

- **NPZ на пример:** ключи `channels` shape `(n_ch, n_samples)`, `target_gains` shape `(n_ch,)`, `target_pans` shape `(n_ch,)`, метаданные `sample_rate`, `schema_version`.
- Синтетический задел: [train_gain_predictor.py](../backend/ml/train_gain_predictor.py) `SyntheticMixDataset`.

### Задача: differentiable mix console (`task: differentiable_mix_console`)

- Как в [train_mix_console.py](../backend/ml/train_mix_console.py): мультитрек + эталонный микс; для диска — NPZ с теми же ключами + `reference_mix`.

### Смешанный корпус событий

- `task: mix_events_jsonl` — шардовые файлы в формате события v1; фильтрация через `dataset_safety.filter_events_for_training`; построчное чтение — `training_dataset_io.iter_jsonl_dicts`.

---

## 5. Safety-фильтры и тесты

Правила по умолчанию (см. `dataset_safety.SafetyLimits`):

- Целевой **fader в dBFS** ≤ 0, если нет `safety.explicit_override_positive_fader == true`.
- Рекомендуемый **true peak** для целей «boost»: ≤ -1.0 dBTP, если в событии задан `observation.true_peak_dbtp` и действие увеличивает уровень (эвристика по имени действия / знаку `delta_db`).
- **Trim gain:** ограничить по `max_gain_trim_db` (например 12…18, как в конфиге).

Юнит-тесты: `tests/test_ml/test_dataset_safety.py`.

---

## 6. Протокол оценки (offline и shadow)

### Offline (без звука в зал)

- **Классификатор:** accuracy / macro-F1 по hold-out; калибровка по `INSTRUMENT_CLASSES`.
- **Gain/pan:** MSE по gains и pans; отдельно — доля предсказаний, отклонённых `filter_events_for_training`.
- **Консольный STFT-loss:** как в `train_mix_console` на валидационном шарде.

### Shadow mode (FOH не трогаем)

1. Поднять backend с флагом/конфигом: телеметрия пишет `recommended_*`, команды на консоль не идут.
2. Параллельно логировать ручные действия оператора (или парсить снапшоты до/после).
3. Метрики: совпадение знака коррекции, MAE по dB, доля случаев когда агент и оператор оба в пределах safety.

### Критерий готовности к ограниченному AUTO

- Достигнуты пороги offline **и** shadow на N репетициях; включён жёсткий ceiling в [wing_client.py](../backend/wing_client.py) независимо от модели.

---

## Связь с кодом

- RAG: [knowledge_base.py](../backend/ai/knowledge_base.py), каталог [knowledge/](../backend/ai/knowledge/).
- ODA-агент: [agent.py](../backend/ai/agent.py).
- Обучение: [train_classifier.py](../backend/ml/train_classifier.py), [train_gain_predictor.py](../backend/ml/train_gain_predictor.py), [train_mix_console.py](../backend/ml/train_mix_console.py).
