# Auto Mixer Tubeslave

Приложение для автоматического микширования с использованием микшерного пульта **Behringer Wing Rack** (fw 3.0.5) через протокол OSC.

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
