# План доработки и исправлений AUTO-MIXER-Tubeslave

**Дата анализа:** 17 марта 2025  
**Версия документа:** 1.0

---

## 1. Резюме анализа

Проект — AI-управляемый автоматический микшер для live sound с поддержкой Behringer WING (OSC), Allen & Heath dLive (MIDI/TCP) и Mixing Station. Backend на Python, frontend на React + Electron. Архитектура в целом зрелая: handlers, automation-модули, тесты. Однако есть критические пробелы в безопасности, дублирование конфигурации, неиспользуемый код и расхождения между документацией и реализацией.

---

## 2. Критические исправления (приоритет 1)

### 2.1 Безопасность live sound

**Проблема:** `.cursorrules` и `CLAUDE.md` требуют fader ≤ 0 dBFS и проверку true peak < -1.0 dBTP перед повышением gain. В коде эти ограничения **не применяются**.

| Место | Текущее поведение | Требуемое |
|-------|-------------------|-----------|
| `mixer_handlers.py` | Передаёт `value` в mixer без проверок | Валидация channel, value; применение `safety.max_fader`, `safety.max_gain` |
| `wing_client.set_channel_fader()` | Clamp -144..+10 dB (WING hardware limit) | Дополнительно ограничить по `config.safety.max_fader` (по умолчанию 0 dBFS) |
| `wing_client.set_channel_gain()` | Clamp -18..+18 dB | Ограничить по `config.safety.max_gain` |
| `default_config.json` | `safety.enable_limits: true` | **Не используется** ни в одном handler |

**Действия:**

1. В `mixer_handlers.py`:
   - Проверять `channel` (1..40 для WING) и тип `value`
   - Перед вызовом `set_channel_fader`/`set_channel_gain` применять `server.config.get("safety", {})`:
     - Если `enable_limits: true` — clamp fader до `min(value, max_fader)` (по умолчанию 0), gain до `min(value, max_gain)`
   - Логировать отклонённые значения

2. В `WingClient` (и других mixer clients):
   - Добавить опциональный параметр `safety_limits: dict` в конструктор
   - В `set_channel_fader`/`set_channel_gain` применять эти лимиты **до** hardware clamp

3. Добавить проверку true peak перед повышением gain в:
   - `lufs_gain_staging.py` — уже есть `true_peak_limit`, проверить применение
   - `auto_fader.py` — при изменении fader/gain
   - `mixer_handlers` — при ручном `set_gain` (если есть доступ к аудио-буферу; иначе — только clamp по config)

### 2.2 CI/CD — сломанный путь

**Проблема:** В `.github/workflows/test.yml` строка 57:

```yaml
cd /home/user/workspace/AUTO-MIXER-Tubeslave-affabb77
```

Жёстко заданный путь, которого нет в GitHub Actions. Тесты не выполняются в CI.

**Действие:** Удалить `cd` или заменить на `$GITHUB_WORKSPACE`. Репозиторий уже в рабочей директории после `actions/checkout@v4`. Запускать:

```yaml
- name: Run tests
  run: PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

### 2.3 Валидация входных данных в handlers

**Проблема:** `mixer_handlers` не проверяет `channel` и `value` перед вызовом mixer client. Возможны:
- `channel=None` → crash
- `channel=999` → невалидный OSC-адрес
- `value="abc"` → `float(value)` → ValueError

**Действие:** Добавить в каждый handler:
- Проверку наличия обязательных полей
- Валидацию диапазонов (channel 1..40, value в допустимых пределах)
- `try/except` с логированием и ответом клиенту об ошибке

---

## 3. Конфигурация и архитектура (приоритет 2)

### 3.1 Два источника конфигурации

**Проблема:**
- `config/default_config.json` — используется `server.py`
- `config/automixer.yaml` — используется только `ConfigManager` в тестах
- `ConfigManager` с hot-reload **не подключён** к server

**Действия (выбрать один вариант):**

**Вариант A (минимальный):** Оставить только `default_config.json`. Удалить или пометить `automixer.yaml` и `ConfigManager` как deprecated/experimental. Обновить документацию.

**Вариант B (рекомендуемый):** Унифицировать на YAML:
- Перенести структуру `default_config.json` в `automixer.yaml` (или объединить)
- Подключить `ConfigManager` в `server.py` вместо прямой загрузки JSON
- Добавить fallback на JSON при отсутствии YAML
- Удалить дублирование

### 3.2 Интеграция FeedbackDetector — ВЫПОЛНЕНО

Реализовано: handlers/feedback_handlers, server.start_feedback_detection (device_id, channels, channel_mapping), AudioCapture + FeedbackDetector, broadcast feedback_detected, config safety.feedback_detection

---

## 4. Рефакторинг и качество кода (приоритет 3)

### 4.1 server.py — монолит

**Проблема:** `server.py` ~4500 строк. Много ответственностей: WebSocket, подключение, voice control, gain staging, auto EQ, phase, fader, soundcheck, compressor, snapshot, bleed, emergency stop и т.д.

**Действие:** Декомпозиция уже частично сделана (handlers вынесены). Дополнительно:
- Вынести инициализацию контроллеров (gain_staging, auto_eq, phase, fader, compressor, voice) в отдельный модуль `backend/controllers.py` или `backend/initialization.py`
- Оставить в `server.py` только: создание сервера, загрузка config, регистрация handlers, цикл обработки сообщений
- Цель: `server.py` < 1500 строк

### 4.2 Дублирование docs/ и Docs/

**Проблема:** Есть и `docs/` (ARCHITECTURE, CONVENTIONS, TOOLS), и `Docs/` (TECHNICAL, CODE_REVIEW, AUTO_EQ_INTEGRATION, STAGE1_COMPLETE, PDF). Часть файлов дублируется или пересекается.

**Действие:**
- Оставить `Docs/` для пользовательской/технической документации (PDF, детальные гайды)
- Использовать `docs/` только для инженерных правил агента (ARCHITECTURE, CONVENTIONS, TOOLS, IMPROVEMENT_PLAN)
- Удалить дубликаты; в `docs/ARCHITECTURE.md` добавить ссылки на `Docs/` где нужно

### 4.3 Зависимости

**Проблема:**
- Корневой `requirements.txt` и `backend/requirements.txt` различаются (prometheus-client 0.16 vs 0.17, torch, chromadb, essentia, faster-whisper только в backend)
- `essentia` — Linux-only; на macOS установка может падать
- `faster-whisper` — тяжёлая зависимость; не во всех сценариях нужна

**Действие:**
- Унифицировать в один `backend/requirements.txt` (или корневой с путём)
- Закрепить версии (pin) для воспроизводимости
- `essentia`: оставить `; sys_platform == "linux"` в requirements
- Вынести `faster-whisper` в optional/extra, если используется только в voice_control_sherpa

---

## 5. Документация (приоритет 4)

### 5.1 README.md — устаревший

**Проблема:**
- "48 channel strips" — WING имеет 40 каналов
- "EQ/Comp в разработке" — реализованы
- Нет упоминания dLive, Mixing Station, Voice Control, Auto Soundcheck, Auto Compressor, Bleed Service

**Действие:** Обновить README:
- Актуальное описание возможностей (EQ, Compressor, Gain Staging, Phase, Auto Fader, Soundcheck, Voice Control)
- Правильное число каналов (40 для WING)
- Ссылки на `docs/` и `Docs/`
- Краткий раздел по безопасности (safety limits, emergency stop)

### 5.2 .cursorrules и CLAUDE.md — синхронизация с кодом

**Проблема:** Упоминаются agents, ai/, ConfigManager, feedback_detector — часть не интегрирована.

**Действие:**
- В `.cursorrules` добавить примечание: "agents/ и ai/ — модули есть, интеграция в server — в планах"
- Явно указать: "FeedbackDetector — планируется интеграция (см. IMPROVEMENT_PLAN)"
- После интеграции FeedbackDetector и ConfigManager — обновить правила

---

## 6. Неиспользуемый и экспериментальный код (приоритет 5)

### 6.1 agents/ и ai/

**Проблема:** `agents/` (GainAgent, EQAgent, FaderAgent, Coordinator) и `ai/` (KnowledgeBase, RuleEngine, LLMClient, MixingAgent) не импортируются в `server.py`. Автоматика реализована через отдельные контроллеры (auto_eq, auto_fader, lufs_gain_staging и т.д.).

**Действия (на выбор):**
- **A:** Оставить как экспериментальные. Добавить в docs пометку "Experimental / Future integration"
- **B:** Интегрировать Coordinator как оркестратор поверх существующих контроллеров (большой объём работы)
- **C:** Если не планируется использование — вынести в отдельную ветку или архив

### 6.2 MixingStationClient, AbletonClient

**Проблема:** Есть клиенты для Mixing Station и Ableton. Mixing Station используется. Ableton — неясно, используется ли в production.

**Действие:** Проверить использование Ableton в коде. Если не используется — пометить в docs как optional/experimental.

---

## 7. Frontend (приоритет 6)

### 7.1 State management

**Проблема:** Всё состояние в `App.js` через `useState`. При росте числа вкладок и параметров — сложно поддерживать.

**Действие (опционально):** Ввести лёгкий state manager (Zustand, Jotai) или Context API для глобального состояния (connection, selectedChannels, automation status). Не критично для текущего масштаба.

### 7.2 console.log в production

**Проблема:** В `websocket.js` и других сервисах могут оставаться `console.log` для отладки.

**Действие:** Заменить на условный лог (например, `if (process.env.NODE_ENV === 'development')`) или использовать единый logger.

---

## 8. Тестирование (приоритет 7)

### 8.1 Покрытие safety

**Проблема:** Нет тестов, проверяющих, что `safety.enable_limits` и `max_fader`/`max_gain` применяются в handlers.

**Действие:** Добавить в `tests/test_infrastructure/` или `tests/`:
- `test_mixer_handlers_safety.py`: mock server с `config.safety.enable_limits=True`, `max_fader=0`; отправить `set_fader` с value=5 → убедиться, что в mixer уходит ≤0
- Аналогично для `set_gain`

### 8.2 Интеграционные тесты

**Проблема:** Много unit-тестов, мало интеграционных (server + handlers + mock mixer).

**Действие:** Добавить `tests/integration/test_server_handlers.py`: поднять server, подключить WebSocket-клиент, отправить `set_fader`/`get_state` и проверить ответы. Использовать `virtual_mixer` или mock.

---

## 9. План выполнения (рекомендуемый порядок)

| Этап | Задачи | Оценка |
|------|--------|--------|
| 1 | Safety: mixer_handlers + wing_client + config | 1–2 дня |
| 2 | CI: исправить workflow | 0.5 дня |
| 3 | Валидация входных данных в handlers | 0.5 дня |
| 4 | Конфиг: унификация (вариант A или B) | 1 день |
| 5 | Интеграция FeedbackDetector | 1–2 дня |
| 6 | Рефакторинг server.py (вынос контроллеров) | 1–2 дня |
| 7 | Документация: README, docs/, .cursorrules | 0.5 дня |
| 8 | Зависимости: унификация requirements | 0.5 дня |
| 9 | Тесты: safety, интеграционные | 1 день |
| 10 | Frontend: убрать console.log, опционально state | 0.5 дня |

**Итого:** ~7–10 рабочих дней для приоритетов 1–4.

---

## 10. Чек-лист перед релизом

- [ ] `safety.enable_limits` применяется во всех путях изменения fader/gain
- [ ] CI проходит на push/PR
- [ ] Handlers валидируют входные данные
- [ ] Один источник конфигурации (JSON или YAML)
- [ ] FeedbackDetector интегрирован или явно помечен как experimental
- [ ] README актуален
- [ ] Нет дублирования docs/Docs
- [ ] requirements.txt унифицирован и закреплён
- [ ] Тесты на safety добавлены
- [ ] `PYTHONPATH=backend python -m pytest tests/ -x` проходит

---

*Документ создан на основе анализа кодовой базы и может обновляться по мере выполнения работ.*
