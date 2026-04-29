# Auto Mixer Tubeslave

Приложение для автоматического микширования с использованием микшерного пульта **Behringer Wing Rack** (fw 3.0.5) через протокол OSC.

## Mix Agent Offline / Backend Facade

Для исследовательско-инженерного offline сведения и безопасной интеграции с
реальным пультом добавлен пакет `mix_agent/`.

```bash
python -m mix_agent analyze --stems ./stems --mix ./mix.wav --reference ./ref.wav --genre pop --out ./report.md
python -m mix_agent suggest --stems ./stems --genre rock --out ./suggestions.json
python -m mix_agent apply --stems ./stems --suggestions ./suggestions.json --out ./renders/conservative_mix.wav
```

Фасад не заменяет микс-инженера и не считает одну метрику признаком качества.
Он строит dashboard из независимых оценок, объясняет каждую рекомендацию,
пишет Markdown/JSON отчёты и применяет offline только малые обратимые операции.
Для live/backend работы рекомендации переводятся в AutoFOH typed actions через
`MixAgentBackendBridge` и могут быть отправлены на пульт только через
`AutoFOHSafetyController`.

## Структура проекта

```
AUTO MIXER Tubeslave/
├── backend/              # Python-бэкенд с OSC-клиентом
│   ├── wing_client.py    # Класс для управления Wing через OSC
│   ├── wing_addresses.py # Справочник OSC-адресов Wing
│   ├── server.py         # WebSocket-сервер для связи с фронтендом
│   └── requirements.txt  # Зависимости Python
├── frontend/             # React-фронтенд
│   ├── public/
│   ├── src/
│   │   ├── components/   # Компоненты GUI
│   │   ├── services/     # WebSocket-сервис
│   │   └── App.js
│   └── package.json
├── config/               # Конфигурационные файлы
│   └── default_config.json
├── presets/              # Сохраненные пресеты микса
└── Docs/                 # Документация
```

## Возможности (Этап 1)

### Реализовано:
- ✅ OSC-клиент с двусторонней синхронизацией с Wing Rack
- ✅ WebSocket-сервер для real-time коммуникации
- ✅ GUI с полным отображением параметров пульта:
  - Настройки подключения (IP, порты, модель)
  - 48 канальных полос с фейдерами и gain
  - Индикаторы уровней
  - 4-полосный параметрический EQ на канал
  - Компрессор с настройками threshold, ratio, attack, release
  - Интерфейс для Gate и Routing (в разработке)

### В разработке:
- ⏳ Модули автоматизации (Auto-Gain, Auto-EQ, Auto-Mix)
- ⏳ Система пресетов
- ⏳ Real-time анализ аудио с FFT

## Требования

### Backend:
- Python 3.9+
- python-osc
- websockets
- numpy, scipy (для модулей автоматизации)

### Frontend:
- Node.js 16+
- React 18

## Установка

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

## Запуск

### 1. Запуск Backend-сервера

```bash
cd backend
python server.py
```

Сервер запустится на `ws://localhost:8765`

### 2. Запуск Frontend

```bash
cd frontend
npm start
```

Приложение откроется в браузере на `http://localhost:3000`

## Подключение к Wing Rack

1. Подключите Wing Rack к той же сети, что и компьютер
2. Найдите IP-адрес пульта в настройках Wing (Setup → Network)
3. В интерфейсе приложения:
   - Выберите модель: **Wing Rack**
   - Введите IP-адрес (по умолчанию: `192.168.1.100`)
   - Send Port: `2222` (стандартный для Wing)
   - Receive Port: `2223`
   - Нажмите **Connect to Wing**

4. При успешном подключении:
   - Статус изменится на "Wing: Connected"
   - GUI синхронизируется с текущим состоянием пульта
   - Все изменения на пульте отобразятся в реальном времени

## OSC-адреса Wing

Примеры адресов согласно спецификации Wing OSC API:

- Фейдер канала: `/ch/01/mix/fader`
- Gain канала: `/ch/01/preamp/trim`
- EQ полоса 1 частота: `/ch/01/eq/1/f`
- Компрессор threshold: `/ch/01/dyn/thr`
- Подписка на обновления: `/xremote`

Полный справочник в файле `backend/wing_addresses.py`

## Настройка конфигурации

### AutoFOH session artifacts

Если включен `autofoh.logging.enabled`, движок пишет:

- JSONL event log: `sessions/autofoh_actions.jsonl`
- session summary report: `sessions/autofoh_session_report.json`

Пути можно переопределить через `config/automixer.yaml`:

```yaml
autofoh:
  logging:
    enabled: true
    path: "sessions/autofoh_actions.jsonl"
    write_session_report_on_stop: true
    report_path: "sessions/autofoh_session_report.json"
```

После завершения soundcheck краткая человекочитаемая summary также доступна:

- в `AutoSoundcheckEngine.get_status()` как `autofoh_session_report_summary`
- в WebSocket `auto_soundcheck_status` payload

Это позволяет быстро увидеть, сколько коррекций было отправлено, сколько заблокировано, и сколько действий система не применила из-за `phase_target_guard`, не открывая JSON report вручную.

### MuQ-Eval quality layer

В проект добавлен опциональный слой оценки качества итогового master mix на базе
[MuQ-Eval](https://github.com/dgtql/MuQ-Eval). Он оценивает 5-10 секунд master-reference
аудио и пишет reward/feedback для агента, но сам не управляет пультом и не отправляет OSC.
Все реальные команды по-прежнему проходят через `AutoFOHSafetyController` и mixer clients.

Конфиг: `config/muq_eval.yaml`.

```yaml
enabled: true
device: auto
window_sec: 10
hop_sec: 5
sample_rate: 24000
min_improvement_threshold: 0.03
rollback_on_quality_drop: true
fallback_enabled: true
log_scores: true
shadow_mode: true
```

По умолчанию `shadow_mode: true`: слой только оценивает уже предложенные/примененные решения
и пишет JSONL, не блокируя live OSC. Если перевести `shadow_mode` в `false`, агрессивные
коррекции требуют уверенного A/B результата; без candidate/dry-run audio они будут отклонены
как небезопасные.

MuQ-Eval зависимости не обязательны. Для полноценной модели установите/склонируйте MuQ-Eval
и его зависимости, затем укажите путь:

```bash
git clone https://github.com/dgtql/MuQ-Eval external/MuQ-Eval
export MUQ_EVAL_ROOT="$PWD/external/MuQ-Eval"
pip install torch omegaconf huggingface_hub
```

Сервис пытается найти checkpoint `zhudi2825/MuQ-Eval-A1` локально. Если MuQ-Eval или checkpoint
недоступны, программа не падает: включается fallback-оценка по clipping, low-end, harshness
2-5 kHz, spectral balance, crest factor и RMS/LUFS sanity.

Offline-артефакты сведения теперь разделены по папкам на рабочем столе:

- аудио-рендеры WAV/MP3: `~/Desktop/Ai MIXING/`
- отчёты, JSONL decision/reward logs и графики: `~/Desktop/Ai LOGS/`

Логи:

- `~/Desktop/Ai LOGS/muq_eval_decisions.jsonl` — evaluation cycle: action, scores before/after, delta,
  accepted/rejected, reason, OSC commands if accepted.
- `~/Desktop/Ai LOGS/muq_eval_rewards.jsonl` — training reward rows:
  `reward = delta_quality_score - safety_penalty - excessive_change_penalty`.

Пример строки:

```json
{"timestamp":123.0,"session_id":"song-1","current_scene":"verse","proposed_action":{"action_type":"ChannelEQMove","delta_db":0.2},"score_before":{"quality_score":0.0,"model_status":"unavailable"},"score_after":{"quality_score":0.56,"model_status":"unavailable"},"delta":0.56,"accepted":true,"rejection_reason":"","osc_commands":[{"address":"/ch/2/eq/1g","value":-1.0}],"reward":0.56}
```

MuQ-Eval не является единственным критерием: он добавляет perceptual reward к существующим
правилам баланса, фазировки, headroom, feedback safety и знаниям по инструментам.

### MuQ-Eval Director Offline Test

`MUQ_EVAL_DIRECTOR_OFFLINE_TEST` — экспериментальный offline-only режим, где MuQ-Eval
становится главным decision/reward engine для сведения мультитрека. Он не подключается к
пульту, не создает mixer client и жестко запрещает `send_osc: true`.

Конфиг: `config/muq_director_test.yaml`. Все изменения применяются только к offline-render
копиям: WAV-кандидаты и `best_mix.wav` пишутся в `~/Desktop/Ai MIXING/muq_director/<session_id>/`,
а `report.json`, `steps.jsonl`, best-state JSON и графики — в `~/Desktop/Ai LOGS/muq_director/<session_id>/`.

Запуск:

```bash
PYTHONPATH=backend python -m src.experiments.muq_eval_director \
  --input "/path/to/multitrack_dir" \
  --config config/muq_director_test.yaml \
  --mode muq_safe
```

Dry-run без рендера и MuQ-оценки:

```bash
PYTHONPATH=backend python -m src.experiments.muq_eval_director \
  --input "/path/to/multitrack_dir" \
  --config config/muq_director_test.yaml \
  --mode muq_safe \
  --dry-run
```

Режимы:

- `pure_muq` — выбор только по среднему MuQ quality score.
- `muq_safe` — `0.70 * MuQ + 0.10 * loudness + 0.10 * peak headroom + 0.05 * phase/mono + 0.05 * anti-overprocessing`.
- `muq_genre` — `muq_safe` плюс небольшой жанровый prior.
- `muq_vs_existing_agent` — тот же safety score, но report пытается сравнить director result с обычным automixer render, если путь задан в config.

Артефакты:

- `report.json` — итоговый score, action sequence, stopping reason, warnings, финальные LUFS/peak/phase.
- `steps.jsonl` — каждый A/B candidate: action, before/after параметры, MuQ before/after, delta, safety score, accepted/rejected.
- `best_mix.wav` — лучший найденный offline-render.
- `candidate_renders/` — WAV-кандидаты, если `save_every_candidate: true`.
- `best_states/` — JSON-снимки лучших beam states.
- `score_curve.png`, `action_timeline.png`, `channel_change_summary.png`, если доступен `matplotlib`.

Пример строки `steps.jsonl`:

```json
{"step":1,"parent_state_id":"state_0000","candidate_id":"step_001_state_0001","action":{"action_type":"gain","channel_id":2,"instrument":"lead_vocal","parameters":{"delta_db":1.0}},"muq_score_before":0.51,"muq_score_after":0.54,"delta_muq":0.03,"loudness_after":-14.0,"peak_after":-3.0,"phase_score":0.96,"final_score":0.61,"accepted":true,"rejection_reason":"","render_path":"~/Desktop/Ai MIXING/muq_director/test_001/candidate_renders/step_001_state_0001.wav","audio_window_scores":[0.53,0.55]}
```

Почему нельзя включать в live: director делает много candidate renders, может быть медленным,
исследует рискованные гипотезы и использует MuQ как основной оптимизатор. Это полезно для
оценки способности метрики вести процесс сведения, но недостаточно надежно для FOH без
человека и existing safety pipeline.

### Автообучение агентов через интернет

Включается через секцию `training` в `config/automixer.yaml` (по умолчанию выключено).

Пример ключей в YAML:

```yaml
training:
  enabled: false
  safe_autostart: false
  interval_minutes: 360
  manifest_url: ""
  request_timeout_sec: 30
  max_retries: 2
  max_dataset_bytes: 21474836480
  study:
    enabled: false
    max_resources_per_run: 0
    max_resources_per_type: 0
    max_resource_bytes: 0
    timeout_sec: 30
    source_types: [books, articles, videos]
    allowed_domains: [arxiv.org, openlibrary.org, archive.org, wikipedia.org, ieee.org, acm.org, nature.com, mdpi.com]
    queries:
      books:
        - "audio engineering handbook"
      articles:
        - "audio mixing article"
      videos:
        - "audio mixing tutorial"
    resource_urls: []
    feeds: []
  cleanup_downloads: true
  discovery:
    enabled: false
    max_candidates_per_target: 2
    hf_search_limit: 8
    candidate_urls: []
    queries:
      channel_classifier:
        - "audio instrument classification dataset jsonl"
        - "music spectral features instrument"
      gain_pan_predictor:
        - "multichannel audio gains pans dataset npz"
        - "music mix parameters gain pan npz"
      mix_console:
        - "multitrack music stems dataset"
        - "audio multitrack reference mix dataset"
  targets:
    channel_classifier:
      enabled: false
```

Переключение через переменные окружения:

- `AUTOMIXER_TRAINING_ENABLED=true|false`
- `AUTOMIXER_TRAINING_INTERVAL_MINUTES=360`
- `AUTOMIXER_TRAINING_MANIFEST_URL=https://.../manifest.json`
- `AUTOMIXER_TRAINING_MAX_DATASET_BYTES=21474836480`
- `AUTOMIXER_TRAINING_CLEANUP_DOWNLOADS=true|false`
- `AUTOMIXER_TRAINING_SAFE_AUTOSTART=true|false`
- `AUTOMIXER_TRAINING_DISCOVERY_ENABLED=true|false`
- `AUTOMIXER_TRAINING_HF_SEARCH_LIMIT=8`
- `AUTOMIXER_TRAINING_MAX_CANDIDATES_PER_TARGET=2`
- `AUTOMIXER_TRAINING_DISCOVERY_CANDIDATE_URLS='["org/dataset", "https://..."]'` (опционально)
- `AUTOMIXER_TRAINING_STUDY_ENABLED=true|false`
- `AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES=0` (0 = без лимита на все выбранные ресурсы за один прогон)
- `AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES_PER_TYPE=0` (0 = без лимита на тип источника)
- `AUTOMIXER_TRAINING_STUDY_MAX_BYTES=0` (`0` = без лимита по размеру ресурса)
- `AUTOMIXER_TRAINING_STUDY_TIMEOUT=30`
- `AUTOMIXER_TRAINING_STUDY_ALLOWED_DOMAINS='["arxiv.org","openlibrary.org","archive.org","wikipedia.org","ieee.org","acm.org","nature.com","mdpi.com"]'`

`safe_autostart: true` выполняет один запуск в режиме dry-run после первого старта: сервис только выбирает подходящие кандидаты датасетов и `study`-материалов, применяя лимиты (`max_dataset_bytes`, `max_candidates_per_target`, `hf_search_limit`, `max_resources_per_run`, `max_resources_per_type`, `max_resource_bytes`), пишет их в лог и в `last_run` (`dataset`/`study`), но не выполняет download/training. На следующем запуске обучение идет уже реально.

`training.study` использует allowlist-домены (`allowed_domains`) и поддерживает как автоматически найденные источники, так и ручной список `learning_resources` в манифесте.

По-умолчанию, если `manifest_url` не задан, сервис:
- берет список источников из `training.discovery`,
- подбирает ближайшие public-источники в интернете (Hugging Face),
- пропускает файлы, если размер выше `max_dataset_bytes`,
- после завершения запуска очищает загруженные и промежуточные временные файлы.

`discovery.candidate_urls` поддерживает как прямые URL файлов, так и `dataset_id` формата Hugging Face (`org/dataset_name`): в последнем случае будет выбран подходящий файл из репозитория.

Ключи для манифеста описаны в [config/training_manifest.example.json](/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/config/training_manifest.example.json).

#### WebSocket API для управления обучением

Все команды шлются через WS `type` сообщения.

- `start_training`
```json
{
  "type": "start_training",
  "force": false,
  "reason": "manual",
  "manifest_url": "https://.../manifest.json"
}
```
- `stop_training`
```json
{
  "type": "stop_training"
}```
- `get_training_status`
```json
{
  "type": "get_training_status"
}```

Возвраты:
- `training_started` с полем `result` (`started`, `run_id`, `reason`)
- `training_status` с деталями по `running/last_success/targets/...`
- `training_stopped` и `training_start_failed` по завершению/ошибкам

Пример манифеста (`channel_classifier`, `gain_pan_predictor`, `mix_console`):

```json
{
  "dataset_id": "global-v1",
  "targets": {
    "channel_classifier": {
      "dataset_url": "https://example.com/data/channel_classifier_train.jsonl",
      "dataset_id": "channel-classifier-v1",
      "n_epochs": 40
    },
    "gain_pan_predictor": {
      "dataset_url": "https://example.com/data/gain_pan_train.npz",
      "dataset_id": "gain-pan-v1",
      "n_samples": 800
    },
  "mix_console": {
      "dataset_url": "https://example.com/data/mix_console_train.npz",
      "dataset_id": "mix-console-v1",
      "n_epochs": 35
    }
  },
  "learning_resources": [
    {
      "type": "articles",
      "url": "https://www.nature.com/articles/audio"
    },
    {
      "type": "books",
      "url": "https://openlibrary.org/works/OL123456W/example"
    },
    {
      "type": "articles",
      "url": "https://ieeexplore.ieee.org/document/example"
    }
  }
}
```

Отредактируйте `config/default_config.json` для изменения:
- Параметров подключения по умолчанию
- Настроек модулей автоматизации
- Safety Limits (защита от перегрузок)

## Тестирование OSC-связи

Используйте тестовый скрипт для проверки подключения:

```bash
cd backend
python test_wing_connection.py
```

Скрипт выведет все входящие OSC-сообщения. Покрутите ручки на пульте и убедитесь, что изменения отображаются.

## Следующие шаги

### Этап 2: Модули автоматизации
- Auto-Gain: автоматическая регулировка входного уровня
- Auto-EQ: FFT-анализ и подавление резонансов
- Auto-Mix / Ducker: балансировка каналов

### Этап 3: Система пресетов и финализация
- Сохранение/загрузка пресетов микса
- Safety Limits
- Оптимизация GUI

## Ссылки на документацию

- [Behringer Wing OSC Implementation](https://wiki.munichmakerlab.de/wiki/Behringer_Wing)
- [Mixing Secrets for the Small Studio - Mike Senior](https://www.cambridge-mt.com/ms/mix-book/)
- [Sound Systems: Design and Optimization - Bob McCarthy](https://bobmccarthy.com/)

## Поддержка

При возникновении проблем:
1. Проверьте сетевое подключение Wing и компьютера
2. Убедитесь, что порты 2222/2223 не заняты
3. Проверьте версию прошивки Wing (должна быть 3.0.5)
4. Проверьте логи backend-сервера

---

**Firmware:** Wing Rack fw 3.0.5  
**Protocol:** OSC (Open Sound Control)  
**License:** MIT
