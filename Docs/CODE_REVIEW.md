# Deep Code Review: AUTO-MIXER-Tubeslave

**Дата:** 2026-03-15
**Проект:** AUTO-MIXER-Tubeslave (Python backend + React frontend, Behringer Wing Rack, OSC)
**Общий объём:** ~24 600 строк Python, 29 .py файлов + 2 конфиг-файла
**Стадия зрелости:** Поздний прототип / Ранняя бета

---

## 1. Статус модулей

### 1.1 DSP-модули (обработка сигнала)

| Модуль | Строк | Статус | Ключевые проблемы |
|--------|-------|--------|-------------------|
| `lufs_gain_staging.py` | 1504 | **WORKING** | K-weighting захардкожен под 48kHz; PyAudio-поток конфликтует с другими модулями; правильная double-gating по ITU-R BS.1770-4; true peak с 4x oversampling |
| `auto_eq.py` | 3076 | **WORKING** | 15+ инструментных профилей; bleed-aware boost reduction; Essentia/scipy спектральный анализ; частотный conflict resolver; большой, но хорошо структурированный |
| `auto_compressor.py` | 747 | **WORKING** | Soundcheck последовательность disable→record→re-enable; live коррекция с порогами; bleed-aware (пропускает при ratio > 0.5); детекция pumping и потери транзиентов |
| `auto_fader.py` | 2244 | **WORKING** | Integrated LUFS с double gating; fuzzy logic контроллер; pre-fader компенсация; жанровые профили баланса; static auto-balance с 90-й перцентиль нормализацией |
| `auto_fader_hybrid.py` | 752 | **WORKING** | 6-уровневая архитектура; L_hybrid = 0.45·LUFS + 0.35·RMS + 0.20·Peak; PI-контроллер с anti-windup; адаптивный rate limiting; ScenarioDetector инстанцируется каждый кадр (антипаттерн) |
| `auto_phase_gcc_phat.py` | 429 | **WORKING** ⚠️ | Корректная GCC-PHAT математика с параболической интерполяцией; **BUG: ref/tgt аргументы перепутаны в `process_audio()` — инвертирует полярность задержки** |
| `auto_panner_adaptive.py` | 698 | **WORKING** | 3-band crossover (LR4); 3-tier feature extraction; жанровые panning шаблоны; sine/cosine panning law; **LR4 коэффициенты — заглушка, не рассчитаны для реальных частот** |
| `cross_adaptive_eq.py` | 163 | **WORKING** | Mirror EQ по IMP 7.3; cut masker (Q=4), boost masked (Q=2, 50%); нет темпоральной сглаживания между кадрами |
| `signal_analysis.py` | 249 | **WORKING** ⚠️ | 7-band energy; spectral flux (IMP [48,53]); transient detection; **BUG: `ratio_float_to_wing()` ссылается на несуществующие `WING_RATIO_VALUES` / `WING_RATIO_STRINGS`** |
| `system_measurement.py` | 784 | **WORKING** | Sine sweep (Farina); RT60 через Schroeder integration; EQ correction; **coherence захардкожена на 0.95 (заглушка)** |

### 1.2 Агентная система

| Модуль | Строк | Статус | Ключевые проблемы |
|--------|-------|--------|-------------------|
| `base_agent.py` | 418 | **WORKING** ⚠️ | ODA-паттерн (Observe-Decide-Act) ABC; dual execution (async + sync fallback); **BUG: проверка PAUSED вызывает выход из цикла вместо паузы (PAUSED != RUNNING ломает while-условие)** |
| `gain_agent.py` | 506 | **WORKING** | LUFS error + peak limiting + smoothing; линейный dB mapping (может не совпадать с нелинейным Wing trim law); **race condition на `_channel_states`** |
| `eq_agent.py` | 1132 | **WORKING** | 4 режима (corrective/creative/surgical/profile); 9 инструментных профилей; cross-channel conflict resolution; **медленная сходимость: alpha=0.15 × max_change=1.0 = 0.15 dB/цикл** |
| `fader_agent.py` | 778 | **PARTIAL** | Cross-adaptive gain sharing работает; **Adaptive mode — ЗАГЛУШКА (делегирует в auto mode)**; `_on_gain_agent_message` — пустой метод; нелинейный Wing fader law корректно моделирован |
| `coordinator.py` | 736 | **WORKING** | Singleton; топологическая сортировка (Kahn's); 5 стратегий конфликтов (**VOTING не реализована**); параллельная оркестрация; **история конфликтов растёт без ограничений** |
| `agents_config.yaml` | 198 | **WORKING** | 5 пресетов (live_performance, broadcast, recording, live_band, worship); **отсутствуют профили toms, overhead, room — на них ссылаются пресеты** |

### 1.3 Инфраструктура и коммуникация

| Модуль | Строк | Статус | Ключевые проблемы |
|--------|-------|--------|-------------------|
| `server.py` | 5031 | **WORKING** | Монолитный WebSocket-сервер; вся автоматизация — методы одного класса; **нет аутентификации, нет валидации входных данных, нет TLS**; mixed async/sync с проблемами потокобезопасности |
| `wing_client.py` | 1565 | **WORKING** | OSC-клиент с Wing handshake; /xremote подписка; throttling сообщений; **нет авто-реконнекта**; **`find_snap_by_name()` деструктивна — загружает снапшоты чтобы прочитать имена** |
| `enhanced_osc_client.py` | 560 | **WORKING** ⚠️ | State machine для соединения; авто-реконнект; логирование; статистика; **НЕ ИНТЕГРИРОВАН — server.py импортирует WingClient, а не EnhancedOSCClient** |
| `controller.py` | 1651 | **WORKING** | Главный оркестратор; интеграция C++ DSP bridge; acoustic analysis, classification, bleed detection; **ML data collection — заглушка** |
| `default_config.json` | 234 | **WORKING** | Per-instrument калибровочные таргеты; много фич отключено по умолчанию; **пустые channel_priorities и vocal_channels** |

### 1.4 V2 модули (вспомогательные)

| Модуль | Строк | Статус | Ключевые проблемы |
|--------|-------|--------|-------------------|
| `activity_detector.py` | 15 | **SKELETON** | Один порог (-50 LUFS), одна функция `is_active()`. Нет гистерезиса, нет временного сглаживания |
| `channel_classifier.py` | 24 | **SKELETON** | Только 4 типа ударных по spectral centroid. Возвращает 'unknown' для всего остального — вокал, гитара, клавиши не распознаются |
| `bleed_compensator.py` | 128 | **WORKING** | Band-level bleed с K-weighting весами; порог доминирования 0.7; linear fallback |
| `hierarchical_mixer.py` | 65 | **WORKING** | Priority-based attenuation при перегрузке; max step cut 3dB; progressive factor (1.0 - index × 0.12) |
| `pid_controller.py` | 125 | **WORKING** | GainSharingController (не PID!); IMP Section 4.3 cross-adaptive; EMA smoothing; dead zone 0.5dB; gate -50 LUFS |
| `spectral_masking.py` | 189 | **WORKING** | Vocal dominance в 1-3kHz; пропорциональные EQ cuts (max -6dB); поддержка coarse bands (umf/lmf) |
| `vocal_activity_detector.py` | 166 | **WORKING** | Attack/release smoothing; hold time; per-channel state; activity level 0-1 для плавного ducking |
| `dynamic_mixer.py` | 467 | **WORKING** | 3-phase mixing (CALIBRATION→STABILIZATION→MAINTENANCE); dead-zone hysteresis; **debugging артефакты в коде** |

---

## 2. Архитектурные проблемы

### 2.1 КРИТИЧЕСКИЕ

**P1: Монолитный server.py (5031 строк)**
Весь WebSocket-сервер — один файл с одним классом. Автоматизация, UI-логика, конфигурация, OSC-коммуникация — всё смешано. Невозможно тестировать изолированно, невозможно масштабировать.

**Рекомендация:** Декомпозировать на:
- `ws_server.py` — WebSocket endpoint + маршрутизация сообщений
- `automation_engine.py` — логика автоматизации
- `config_manager.py` — управление конфигурацией
- `session_manager.py` — управление сессиями клиентов

**P2: Отсутствие аутентификации и авторизации**
WebSocket-сервер принимает любые соединения. Нет TLS. Нет валидации входных данных. В концертной среде это означает, что любой в локальной сети может отправлять OSC-команды на пульт.

**P3: Enhanced OSC Client не интегрирован**
`enhanced_osc_client.py` реализует всё, чего не хватает `wing_client.py`: state machine, авто-реконнект, логирование, статистику. Но server.py использует старый WingClient. На концерте потеря OSC-соединения без авто-реконнекта = тишина в зале.

**P4: Единственный PyAudio поток**
`lufs_gain_staging.py`, `auto_fader.py`, `auto_fader_hybrid.py` и `auto_phase_gcc_phat.py` все создают собственные PyAudio потоки. На одной машине они будут конфликтовать за аудио-устройство.

**Рекомендация:** Единый `AudioCapture` сервис, раздающий PCM-буферы подписчикам.

### 2.2 СЕРЬЁЗНЫЕ

**P5: Смешанный async/sync код**
`base_agent.py` имеет dual execution (async + sync fallback), но `server.py` вызывает агентов из asyncio event loop. Sync fallback блокирует event loop, замедляя все WebSocket-соединения.

**P6: Thread safety агентов**
`gain_agent.py` — race condition на `_channel_states` (dict мутируется из callback-потока OSC и из основного цикла агента). Нет locks, нет thread-safe контейнеров.

**P7: Дублирование LUFS-измерений**
LUFS вычисляется в: `lufs_gain_staging.py`, `auto_fader.py`, `auto_fader_hybrid.py`, `controller.py` (через bridge). Четыре раздельные реализации одного и того же — рассинхронизация показаний гарантирована.

**P8: Неограниченный рост данных**
- `coordinator.py`: история конфликтов растёт без ограничения
- `dynamic_mixer.py`: калибровочные данные накапливаются
- `auto_eq.py`: история спектральных коррекций

На многочасовом концерте это приведёт к OOM.

### 2.3 УМЕРЕННЫЕ

**P9: Именование не соответствует реализации**
`pid_controller.py` содержит `GainSharingController` — это не PID-контроллер. `PIDLoudnessController` — alias для обратной совместимости. Вводит в заблуждение при чтении кода.

**P10: Конфигурация с пустыми обязательными полями**
`default_config.json` имеет пустые `channel_priorities: {}` и `vocal_channels: []`. Без них hierarchical_mixer и vocal_activity_detector не работают, но ошибки нет — просто тихо ничего не делают.

**P11: Debugging артефакты в production-коде**
`dynamic_mixer.py` содержит обширные блоки отладочного логирования с захардкоженными путями файлов. `auto_fader_hybrid.py` инстанцирует ScenarioDetector каждый кадр (100Hz = 100 объектов/сек).

---

## 3. Алгоритмические пробелы

### 3.1 Подтверждённые баги

| # | Модуль | Серьёзность | Описание |
|---|--------|-------------|----------|
| B1 | `auto_phase_gcc_phat.py` | **ВЫСОКАЯ** | `process_audio(reference, target)` вызывает `gcc_phat(target, reference)` — аргументы перепутаны, задержка инвертируется. Фазовая коррекция работает в противоположную сторону |
| B2 | `signal_analysis.py` | **СРЕДНЯЯ** | `ratio_float_to_wing()` ссылается на `WING_RATIO_VALUES` и `WING_RATIO_STRINGS` — они нигде не определены. Функция упадёт с NameError при вызове |
| B3 | `base_agent.py` | **ВЫСОКАЯ** | Цикл `while self.state == AgentState.RUNNING` — при переходе в PAUSED цикл завершается вместо ожидания. Агент не может быть возобновлён после паузы |
| B4 | `wing_client.py` | **СРЕДНЯЯ** | `find_snap_by_name()` последовательно загружает снапшоты чтобы прочитать их имена. На концерте это перезагрузит настройки пульта |

### 3.2 Алгоритмические упрощения

| # | Модуль | Описание | Влияние |
|---|--------|----------|---------|
| A1 | `lufs_gain_staging.py` | K-weighting фильтры захардкожены для 48kHz | Некорректные измерения при 44.1kHz/96kHz |
| A2 | `system_measurement.py` | Coherence = 0.95 (константа-заглушка) | Нельзя оценить качество измерения |
| A3 | `auto_panner_adaptive.py` | Linkwitz-Riley коэффициенты — заглушка | Crossover не работает на заявленных частотах (250Hz/4kHz) |
| A4 | `channel_classifier.py` | Только 4 типа ударных по centroid | Вокал, гитара, клавиши — всё 'unknown', не классифицируется |
| A5 | `activity_detector.py` | Один порог, без гистерезиса | На границе порога — непрерывное переключение (chattering) |
| A6 | `cross_adaptive_eq.py` | Нет темпорального сглаживания | Резкие скачки EQ между кадрами; слышимые артефакты |
| A7 | `eq_agent.py` | alpha=0.15 × max_change=1.0 = 0.15 dB/цикл | При отклонении 10dB сходимость займёт ~67 циклов (>1 мин при 1Hz цикле) |
| A8 | `gain_agent.py` | Линейный dB→fader mapping | Wing использует нелинейный trim law; -10dB отклонение в крайних положениях |
| A9 | `pid_controller.py` + `controller.py` | Двойное сглаживание: EMA alpha=0.3 + дополнительный фактор 0.30 в controller | Эффективный отклик ~4x медленнее задуманного (~1.5с вместо ~350мс для сходимости) |

### 3.3 Пробелы в DSP

- **Нет латентности компенсации:** Phase correction (`auto_phase_gcc_phat.py`) определяет задержку, но нет механизма отправки delay compensation на Wing
- **Нет частотно-зависимого gating:** LUFS gate в `auto_fader.py` работает по broadband; низкочастотный шум (кондиционер, сцена) проходит через gate
- **Нет спектрального баланса по жанрам:** `auto_eq.py` имеет инструментные профили, но нет overall mix target curve (например, pink noise -3dB/oct для рока)
- **Нет детекции feedback:** Критично для живого звука — нет ни одного модуля для определения акустической обратной связи

---

## 4. Что отсутствует для Production

### 4.1 Безопасность и надёжность

| # | Компонент | Статус | Критичность |
|---|-----------|--------|-------------|
| M1 | Аутентификация WebSocket | Отсутствует | **КРИТИЧЕСКАЯ** |
| M2 | TLS/WSS | Отсутствует | **КРИТИЧЕСКАЯ** |
| M3 | Валидация входных данных | Отсутствует | **КРИТИЧЕСКАЯ** |
| M4 | OSC авто-реконнект | Реализован, не интегрирован | **ВЫСОКАЯ** |
| M5 | Graceful shutdown | Частичный | **ВЫСОКАЯ** |
| M6 | Health checks / watchdog | Отсутствует | **ВЫСОКАЯ** |
| M7 | Rate limiting (WebSocket) | Отсутствует | СРЕДНЯЯ |

### 4.2 Тестирование

| # | Тип тестов | Статус |
|---|-----------|--------|
| T1 | Unit tests | **Полностью отсутствуют** |
| T2 | Integration tests | **Полностью отсутствуют** |
| T3 | DSP algorithm tests | **Полностью отсутствуют** |
| T4 | OSC protocol tests | **Полностью отсутствуют** |
| T5 | Load/stress tests | **Полностью отсутствуют** |
| T6 | CI/CD pipeline | **Полностью отсутствует** |

**Ни одного тестового файла во всём проекте.** Для аудио-системы реального времени это неприемлемо.

### 4.3 Операционная готовность

- **Нет логирования в structured формате** (JSON logs для ELK/Grafana)
- **Нет метрик** (Prometheus/StatsD) — невозможно мониторить latency, CPU, память в реальном времени
- **Нет конфигурации через ENV** — все параметры в JSON/YAML файлах, перезагрузка = перезапуск
- **Нет hot-reload конфигурации** — смена жанра/preset на концерте требует перезапуска
- **Нет backup/restore состояния** — если сервер упадёт, все калибровки потеряны
- **Нет документации API** — WebSocket-сообщения не документированы
- **Нет ограничения concurrent connections** — один злонамеренный клиент может создать тысячи соединений

### 4.4 Недостающие межмодульные связи

| Источник | Назначение | Описание разрыва |
|----------|-----------|------------------|
| `enhanced_osc_client.py` | `server.py` | Улучшенный клиент не импортирован; server использует старый wing_client |
| `vocal_activity_detector.py` | `auto_fader.py` | VAD существует, но ducking в auto_fader не использует его (собственная реализация) |
| `spectral_masking.py` | `cross_adaptive_eq.py` | Оба работают с masking, но не связаны; дублирование логики |
| `hierarchical_mixer.py` | `coordinator.py` | Coordinator не вызывает hierarchical mixer при перегрузке |
| `channel_classifier.py` | `eq_agent.py` | eq_agent имеет собственные 9 профилей; classifier возвращает только drum types |
| `bleed_compensator.py` | `lufs_gain_staging.py` | Bleed compensation не применяется к LUFS-измерениям; показания загрязнены bleed |
| `dynamic_mixer.py` | `fader_agent.py` | dynamic_mixer существует отдельно от fader_agent; неясно кто главный |
| `auto_phase_gcc_phat.py` | `wing_client.py` | Phase detection не отправляет delay compensation на Wing (нет OSC-команды) |

---

## 5. Приоритетные рекомендации

Ранжированы по критичности для перехода к production-ready концертному применению.

### Приоритет 1: Блокирующие проблемы (исправить до любого живого использования)

**R1. Исправить bug B1 (фазовая инверсия в GCC-PHAT)**
Поменять аргументы в `process_audio()` строка ~180. Одна строка кода, но без неё phase alignment работает в обратную сторону.

**R2. Исправить bug B3 (паузе агента = остановка)**
Заменить `while self.state == RUNNING` на `while self.state in (RUNNING, PAUSED)` с `await asyncio.sleep()` внутри PAUSED-ветки.

**R3. Интегрировать enhanced_osc_client.py**
Заменить `WingClient` на `EnhancedOSCClient` в server.py. Авто-реконнект уже реализован — нужно только подключить.

**R4. Добавить аутентификацию WebSocket**
Минимально: token-based auth в handshake. Без этого любой в сети может управлять пультом.

**R5. Единый AudioCapture сервис**
Создать один PyAudio поток, раздающий буферы подписчикам. Устранит конфликты B4-уровня.

### Приоритет 2: Важные улучшения (до первого реального концерта)

**R6. Декомпозировать server.py**
Разбить 5031 строк на 4-5 модулей. Это разблокирует тестирование и позволит работать нескольким разработчикам.

**R7. Написать DSP unit tests**
Начать с: LUFS measurement (сравнение с эталонными WAV), GCC-PHAT (известная задержка), EQ biquad коэффициенты. Цель: 80% покрытие DSP-модулей.

**R8. Добавить thread safety в агенты**
`threading.Lock` или `asyncio.Lock` на все shared state в `gain_agent.py`, `eq_agent.py`, `fader_agent.py`.

**R9. Ограничить рост данных**
Добавить `collections.deque(maxlen=N)` или periodic cleanup для конфликтной истории в coordinator, калибровочных данных в dynamic_mixer, spectral history в auto_eq.

**R10. Feedback detection**
Добавить модуль определения акустической обратной связи (narrow-band peak detection + быстрый notch). Критично для живого звука.

### Приоритет 3: Улучшения качества (после стабилизации)

**R11. Динамические K-weighting фильтры**
Рассчитывать коэффициенты по ITU-R BS.1770-4 для произвольной sample rate, не только 48kHz.

**R12. Реализовать channel_classifier**
Расширить за пределы 4 drum types. Минимум: vocal, guitar, bass, keys, drums. Использовать spectral+temporal features.

**R13. Связать spectral_masking ↔ cross_adaptive_eq**
Объединить в единый anti-masking pipeline. Сейчас два модуля делают похожую работу, но не знают друг о друге.

**R14. Добавить structured logging и метрики**
JSON logging + Prometheus endpoints. Без этого невозможно диагностировать проблемы на реальном концерте.

**R15. Hot-reload конфигурации**
File watcher или WebSocket-команда для обновления параметров без перезапуска сервера.

---

## 6. Сводная оценка

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| **Доменная экспертиза** | 9/10 | Отличное знание audio DSP, ITU стандартов, IMP, live sound практик |
| **Алгоритмическая корректность** | 7/10 | Основные алгоритмы правильны; 2 критических бага; несколько заглушек |
| **Архитектура** | 4/10 | Монолитный server.py; отсутствие DI; дублирование; несвязанные модули |
| **Thread Safety** | 3/10 | Race conditions; блокирующие вызовы в async коде; нет locks |
| **Тестируемость** | 2/10 | 0 тестов; монолит не позволяет unit testing; нет моков для OSC |
| **Production Readiness** | 2/10 | Нет auth, нет TLS, нет health checks, нет graceful shutdown |
| **Безопасность** | 2/10 | Нет аутентификации; нет валидации; find_snap_by_name деструктивна |
| **Документация кода** | 7/10 | Хорошие docstrings; ссылки на IMP/ITU; но нет API docs |
| **Общая зрелость** | 4/10 | Поздний прототип. Алгоритмы сильные, инфраструктура слабая |

### Вердикт

Проект демонстрирует **глубокое понимание audio DSP и автоматического микширования**. Алгоритмическая база — одна из самых полных open-source реализаций IMP-подхода. Однако инфраструктурная зрелость значительно отстаёт от алгоритмической. Для безопасного использования на реальном концерте необходимо:

1. Исправить 4 подтверждённых бага (особенно B1 и B3)
2. Интегрировать уже написанный enhanced_osc_client
3. Добавить минимальную безопасность (auth + TLS)
4. Написать тесты для DSP-ядра
5. Декомпозировать server.py

**Оценочный объём работы для production-ready:** 3-4 человеко-месяца при условии сохранения текущей архитектуры агентов.
