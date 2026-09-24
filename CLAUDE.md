# AUTO-MIXER-Tubeslave

AI-управляемый автоматический микшер для саундчеков и концертов.
Behringer WING Rack (OSC) + Allen & Heath dLive (MIDI/TCP) + Dante/SoundGrid.

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt && python server.py

# Frontend
cd frontend && npm install && npm start

# Tests
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Структура (TOC)

```
backend/
  server.py              — WebSocket координатор, точка входа
  wing_client.py         — Behringer WING OSC client (MixerClientBase)
  dlive_client.py        — Allen & Heath dLive MIDI/TCP client (MixerClientBase)
  mixer_client_base.py   — ABC для всех mixer clients
  audio_capture.py       — Единый аудио-сервис (ring buffer, subscribe)
  feedback_detector.py   — Обнаружение обратной связи, notch EQ, fader fallback
  thread_safety.py       — ThreadSafeMixerState (asyncio.Lock, copy-on-read)
  config_manager.py      — YAML config с hot-reload (watchdog)
  handlers/              — 18 WebSocket handler-модулей (см. docs/ARCHITECTURE.md)
  agents/                — Multi-agent система (coordinator → eq/fader/gain agents)
  ai/                    — LLM orchestrator + ChromaDB/RAG knowledge base
  ml/                    — PyTorch ML модели (classifier, predictor, style transfer)
  osc/                   — Enhanced OSC client с retry и батчингом
frontend/
  src/components/        — React компоненты (табы: EQ, Compressor, Fader, Gate, etc.)
config/
  automixer.yaml         — Главный конфиг (mixer, audio, websocket, agent, ai, safety)
tests/                   — pytest (test_ml/, test_ai/, test_infrastructure/)
virtual_mixer/           — Эмулятор микшера для тестов без оборудования
Docs/                    — PDF документация WING, технические документы
```

Подробная архитектура: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
Соглашения по коду и DSP: [docs/CONVENTIONS.md](docs/CONVENTIONS.md)
Протоколы и инструменты: [docs/TOOLS.md](docs/TOOLS.md)

## Критические ограничения (SAFETY)

Это система управления живым звуком. Ошибки = физический вред слуху.

1. **Fader ceiling**: никогда не устанавливать fader > 0 dBFS без явного запроса оператора
2. **True peak**: всегда проверять < -1.0 dBTP перед повышением gain
3. **Feedback first**: feedback_detector.py имеет абсолютный приоритет — может снизить фейдер без одобрения других агентов. **Статус 2026-07-29: механизм реализован (EmergencyChannelFaderMove/notch-транслятор, бюджеты −3 дБ/шаг ≤ −9/канал, DEC-0005), но по умолчанию ВЫКЛЮЧЕН — включается операторским флагом `feedback_emergency_live_apply: true` в action_limits. До включения и proof-прогона на железе система только детектирует и алертит**
4. **Snap safety**: find_snap_by_name() читает только кэш имён. Загрузка снапшота перезаписывает настройки пульта
5. **При сомнениях — снижай, не повышай**

## Зафиксированные правила (обновляй при обнаружении нового)

### Баги, которые повторялись
- `gcc_phat(target, reference)` — НЕ `gcc_phat(reference, target)`. Инверсия = фаза корректируется в обратную сторону
- Agent `while self.state == RUNNING` завершается при PAUSED. Использовать `asyncio.Event.wait()`
- `find_snap_by_name()` — КЭШИРОВАТЬ имена. Загрузка = деструкция mixer state на концерте
- `ratio_float_to_wing()` — константы WING_RATIO_VALUES/WING_RATIO_STRINGS должны быть на уровне модуля
- NumPy типы (np.float64) не сериализуются в JSON → `convert_numpy_types()` перед `json.dumps()`
- WING USB-поток снимается ДО цифрового трима (pre-trim tap): изменение `/ch/N/in/set/trim` НЕ отражается в USB-захвате. Замкнутый контур gain staging по USB невозможен — трим считать открытым контуром от pre-trim замера. Подтверждено на реальном пульте 2026-07-03
- WING FX: имена моделей — только официальные (FW 3.1): стерео-дилей `ST-DL` (не STEREO_DELAY), список в `Docs/Behringer_WING_Official/WING_RACK_OSC_FX_GUIDE.md`. Адреса `/fx/N/on` НЕ существует (слот активен с моделью). FX-возвраты работают только если шина имеет insert слота: `/bus/N/preins/ins = FXn`, `/bus/N/preins/on = 1`
- WING FX слоты: премиум-модели (HALL, ST-DL, ROOM, ...) грузятся ТОЛЬКО в слоты 1–8; запись премиум-mdl в слоты 9–16 молча отвергается — readback остаётся `NONE` (тихий отказ, как у OSC-троттла). Стандартные модели (SPKMAN) принимаются в 9–16. Параметрические токены (`pdel`, `time`, ...) существуют только при загруженной модели — без mdl запрос таймаутит. Загрузка модели ~0.6 с. SPKMAN-токены: `hpf lpf dist lim phase`. Подтверждено на реальном пульте 2026-07-05
- WING канальная задержка — `/ch/N/in/set/dly` (+ `dlymode`); X32-стиля `/ch/N/dly/*` НЕ существует. Квант 0.1 мс (запись 1.23 → readback 1.2). Pan — `/ch/N/pan`, не `/ch/N/mix/pan`. Выходные секции `/main|mtx/n/{eq,dly,dyn,fdr,in/set/inv}` включая кроссоверные типы `leq/heq` (LR24/LR48) — подтверждены на реальном пульте 2026-07-05
- WING USB loopback — USB-выходы 14/15 замыкаются на входные каналы 14/15: цифровая петля с RT ~258.6 мс, плоская TF 0.00 дБ, когерентность 1.000 (проверено ESS+H1 2026-07-05). Годится для self-test измерительного движка pa_v3 без микрофона; маппинг зависит от текущего роутинга — перед использованием проверять корреляционной пробой на −42 dBFS
- Оценка сведения — ТОЛЬКО по мастер-петле Main LR → USB-отправки 47/48 (`/io/out/USB/47|48`: grp=MAIN, in=L/R; записи `in` 1-базные, readback 0-базный!) — единственный замкнутый контур: pre-trim tap каналов не слышит НИ ОДНОЙ применённой коррекции. Источник MAIN снимается пост-мьют: замьюченный Main LR = честный NO_SIGNAL, а не «тихий микс». Tap-каналы исключать из коррекций (master-reference). Модуль backend/master_mix_eval.py, хук в _apply_corrections. Внедрено по директиве оператора 2026-07-06
- Непрерывные (цикличные) коррекции — только идемпотентные АБСОЛЮТНЫЕ цели, никогда относительные приращения по замеру, который не отражает применённое изменение (pre-trim USB tap!): относительный трим/фейдер-якорь от текущих значений = интеграторный храповик (за 92 цикла тримы ушли в +18, фейдеры в пол −40; подтверждено live-тестом 2026-07-03). Якорь баланса — к константному потолку, не к текущим фейдерам
- Feedback-детекция — устойчивый тон ≠ заводка: у ноты инструмента есть гармоники 2f/3f (в пределах ~25 dB от пика), у заводки — чистый растущий синус. Проверять гармоники и монотонный рост, иначе ложные срабатывания на аккордеоне/органе. ВАЖНО: проверки гармоник ВВЕРХ недостаточно — 2-я гармоника ноты сама проходит PHPR (у неё нет 2f/3f выше); обязательна проверка СУБ-гармоник f/2, f/3 (compute_subharmonic_ratio_db в feedback_detector). Критерии PHPR/PNPR/IMSD — van Waterschoot & Moonen 2010, пороги калибруются под зал
- GCC-PHAT в live с bleed — замер по всей записи ненадёжен; использовать onset-gated окна + полосовую префильтрацию по телу инструмента. Полярность/задержки применять только при MAD ≤ 0.5 мс и достаточном числе ударов; при противоречивых прогонах — нужен соло-проход инструмента. Пары одного источника (snare T/B, kick in/out) — микрофоны в паре сантиметров: замер задержки > 2 мс физически неправдоподобен — это периодическая неоднозначность GCC-PHAT (резонанс снейра ~100 Гц = период 10 мс; live 2026-07-06: «9.90 мс» при coherence 1.0), пару пропускать целиком, включая polarity (_same_source_delay_plausible)
- WING compressor threshold — адрес зависит от модели динамики: COMP → `/dyn/thr`, ряд эмуляций → `/dyn/thr26`. Запись только в `thr26` при модели COMP молча теряется (подтверждено на реальном пульте). `set_compressor_threshold` пишет в оба адреса; readback проверять через оба
- WING OSC throttle — `WingClient.send()` молча дропает повторную запись в тот же адрес чаще ~10 Hz и возвращает `False`. Из-за этого apply + мгновенный rollback в тот же адрес терял rollback (`rollback_not_applied` в CH3 fader proof, подтверждено на реальном пульте). Manual-write путь сервера ретраит сброшенную запись с паузой больше интервала троттлера; в proof-раннерах ставить rollback delay больше интервала троттлера
- Write-методы mixer client ОБЯЗАНЫ возвращать результат `send()`. `set_channel_mute`/`set_channel_pan` возвращали `None`, а серверный manual-write gate проверяет `write_result is False` — провал записи рапортовался как applied. Для mute на концерте это ложное «снял mute»
- Read-методы mixer client — pan/EQ/trim НЕ стримятся `/xremote`, поэтому пассивное `self.state.get(addr)` возвращает `None`/устаревшее и молча ломает read-back верификацию коррекций. `get_channel_pan/get_eq_on/get_eq_band_gain/get_eq_band_frequency/get_channel_gain` обязаны делать активный GET, как `get_gate_threshold/get_compressor_threshold`. Реализовано `_active_readback(addr, ttl, timeout)`: активный запрос при устаревании кэша, ранний выход по обновлению timestamp (горячие вызовы бьют по кэшу). Fader/mute трогать НЕ надо — их `/xremote` стримит живьём. Верифицировано на реальном пульте 2026-07-09 (pan 16/16 против 0/16 до фикса)
- Live soundcheck trim — по умолчанию НЕ поднимать input trim по короткому сигналу: bleed может выглядеть как тихий источник. Авто-трим в live режиме только снижает перегруз, boost требует явного opt-in.
- Bleed-доминантный близкий микрофон (kick/snare/tom без СОБСТВЕННЫХ ударов) — никаких EQ/comp/pan/trim/fader коррекций и никакого участия в кросс-адаптивных прогонах (маскирование, w_p/w_f); только гейт. Признак собственного удара — proximity: в 30-мс окне онсета канал энергичнее остальных близких микрофонов; микрофоны ОДНОГО источника (snare T/B, kick in/out) между собой не конкурируют — нижний снейр горячее верхнего на том же ударе, это не bleed. Порог: ≥3 собственных удара в окне анализа. Замечание оператора на живом прогоне 2026-07-06: TOM/F TOM получили EQ/pan по чужому сигналу
- Полярность многомикрофонных пар — детекция по знаку корреляции ПОСЛЕ выравнивания на замеренный лаг (r на нулевом лаге при сдвиге 0.58 мс ≈ 0 у широкополосного снейра — признак инверсии не срабатывает). USB-tap стоит ДО /in/set/inv: проверить сумму после переворота замером нельзя, судья — оператор на месте
- WING EQ bands — существующая полоса может быть feedback-notch/ручной полосой, а не “band 2 = low-mid”. При EQ-коррекции проверять частотное совпадение или перестраивать полосу, и включать `eq/on`, чтобы оператор слышал результат.
- EQ не «давит громкое»: выпуклый пик спектра — ещё не проблема (форманты аккордеона, грудной регистр вокала — это ГОЛОС источника). Глубокий срез (>3 дБ) требует доказательства: проблемный резонанс ЗВЕНИТ — его полоса затухает медленнее широкополосной огибающей в паузах (ring-тест resonance_eq, ratio<0.6); без подтверждения — гигиенический кэп −3. Маскинг-срезы не трогают тембровое ядро маскера (полосы в 6 дБ от его максимума) глубже −3 — конфликт решается в не-ядерных полосах или балансом. Вопрос оператора на живом прогоне 2026-07-06
- Master-reference Dante channels — каналы, заведённые как master/reference feed, не являются обычными source channels и должны исключаться из source gain/EQ/FX correction.
- Compressor makeup — при сжатии компенсировать потерянный уровень через bounded `dyn/gain`, ориентируясь на GR и true-peak headroom; не использовать makeup как обход headroom, feedback или bleed safety.
- OSC throttle — НЕ дропать молча троттлированные записи: очередь trailing-edge (последнее значение побеждает) + flush по таймеру. Сеттеры клиентов ОБЯЗАНЫ возвращать bool из send()/_send_raw — `sent=True` на потерянном пакете даёт ложные rollback_not_applied.
- Autopilot truth-цикл — валидацию прогонять ПОСЛЕ финальной записи `ops/autopilot/*` и синхронизации всех operator-поверхностей (9 файлов со строкой «Latest ledger handoff row»); прогон до дозаписи + деселекция CP05-теста скрыли 9 красных рельс (inspector iteration 001). Строка `RUN_LEDGER.jsonl` обязана нести полную handoff-схему (`authority`, `decision_ledger`, `current_blocker`, `next_operator_action`, `external_validation_requirements`, `hardware_contacted`, `live_writes`). Голый формат `N passed, M skipped` в `results` зарезервирован за M115 baseline — новые полные прогоны только с текстовым префиксом. Checkpoint = source-коммит → ровно один marker-only CP05 rebind (`backend/m110_cp05_hardware_gate.py`). Список deliverables сверять с committed tree (001 оставил `tools/automixer.py` untracked)
- AudioCapture silence fallback — если InputStream не открылся, capture молча деградирует в генератор тишины. Проверять `capture.silence_fallback` после start(); саундчек на тишине обязан завершаться ERROR, а не успехом.
- Input click repair — короткая вершина выше 0.95 не доказывает цифровой клик: перед интерполяцией проверять длительность всего выброса и резкие изолированные границы. Иначе гладкие атаки барабанов удаляются, а простое изменение входного gain меняет микс. Регрессии: `tests/test_input_health.py`; на реальном snare старый ремонт удалял 11.45% энергии.
- Offline drum phase — не выбирать только отрицательные GCC-замеры и не применять `max(old_delay, measured_delay)`: это даёт случайные задержки и не исправляет overshoot. Проверять отдельные атаки, MAD/IQR, полярность и сумму до изменения плана; цель абсолютная, задержка с нулевым заполнением. PHAT confidence не является gain-инвариантным допуском. Регрессии: `tests/test_offline_agent_mix_phase.py`.
- AutoFOH per-step клампы (±1 dB/шаг) + однократный вызов `_apply_fader`/`_apply_eq` физически не достигают вычисленной цели. В SNAPSHOT_LOCK использовать `_execute_action_stepped` (bounded-шаги до цели, rate-limit exempt), потолки и total-cap сохраняются.
- device_id из фронтенда — это PyAudio-индекс; движок матчит устройства по имени (sounddevice). Резолвить индекс→имя через `audio_devices.get_device_by_id` до передачи в engine. `channel_mapping` (console→capture) обязателен при WING USB/Dante-тапах — см. ChannelMappedCapture.
- Ayaic gain staging/balance — при offline сведении, саундчеках и live-режимах
  использовать Ayaic Mix Monolith уровни как дефолтные плоскости для отдельных
  источников и для стемов/групп одновременно. Не балансировать только
  изолированные дорожки: сначала выровнять слои внутри инструмента/стема,
  затем проверить сумму шины. Базовые ориентиры: lead/main element `-22 LUFS`,
  strong sources и subgroup buses вроде Drums/Bass/Guitars/BG Vocals `-25 LUFS`,
  tucked backing vocals `-27 LUFS`, keys/synth/ambient support `-30 LUFS`, FX
  returns/deep effects `-35 LUFS`; doubles/layers держать примерно на 3 LU ниже
  целевой плоскости группы. В live это только цель для bounded fader/bus moves:
  true-peak/headroom/feedback safety и запрет fader > 0 dBFS важнее попадания в
  Ayaic-уровень.
- Основной offline pipeline — стандартный no-reference pipeline: Ayaic баланс
  источников/стемов/шин, Gate по необходимости, bounded source/drum-bus
  compression до финального gain для контроля crest factor, FX15-style
  section/mod space и `spectral_ceiling_eq` как главный спектральный потолок.
  Не использовать `AIR_MINUS_0_80`, финальный limiter, soft clipper или
  reference matching по умолчанию; это только явные варианты по запросу
  оператора.

- TruePeakMeter: cutoff `firwin` задаётся в долях Найквиста АПСЕМПЛИРОВАННОГО потока — `0.5/OS` режет полосу на fs/8 (12 кГц) и занижает ВЧ-пики до −20 дБ; правильно `1.0/OS` + страховка `max(interpolated, sample_peak)`. Регресс: tests/test_transport_safety_invariants.py (ВЧ-синусы + межсэмпловый пик)
- auto_eq: `max_cut` в профилях ОТРИЦАТЕЛЕН — масштабирование среза только через `abs(max_cut)`, иначе срез меняет знак и становится бустом. Позиционная конвертация крайних полос в shelf запрещена (узкий рез 250 Гц как low-shelf выпиливает фундамент)

### Архитектурные решения
- AudioCapture — единый сервис. НЕ создавать PyAudio потоки в отдельных модулях
- ScenarioDetector — кэшировать экземпляр, НЕ создавать на каждый аудиоблок
- server.py — координатор. Логика в handlers/. НЕ добавлять новые маршруты в server.py напрямую
- ML classifier — использовать полноценный из backend/ml/channel_classifier.py (25 классов), НЕ заглушку
- Fader ceiling — потолок enforced на транспортной границе: `WingClient` клампит записи фейдера выше `fader_ceiling_db` (default 0.0 dBFS) с warning; опт-ин оператора через `AUTOMIXER_FADER_CEILING_DB`. Upper layers больше не единственная защита (defence-in-depth, ADR p1-fader-ceiling-defence-in-depth). С 2026-07-29 кламп на ВСЕХ уровневых поверхностях (`_clamp_fader_db`): ch/bus/dca/main/fx-return/sends WING + `DLiveClient.set_fader` — DCA/шина выше unity обесценивали канальный потолок
- Feedback live monitor — always-on детекция feedback вне саундчека: `backend/feedback_live_monitor.py`, конфиг `feedback_live:` в automixer.yaml, только детекция + алерты оператору, без записей в пульт (ADR c3-always-on-feedback-detection)

### DSP стандарты
- LUFS: ITU-R BS.1770-4, double gating (absolute -70 LUFS + relative -10 LU)
- True Peak: 4x oversampling
- K-Weighting: high-shelf +4dB @ 1681 Hz + HPF 38 Hz (два biquad каскада)
- EQ biquads: по Audio EQ Cookbook (Robert Bristow-Johnson)
- Компрессия ratio: `1 + (max_ratio - 1) * factor` — НЕ инвертировать
- Gate: объявленный hysteresis ОБЯЗАН применяться в GateProcessor
- Gain staging/balance: Ayaic Mix Monolith planes применяются к channels и
  stem/group buses; bus sum проверяется после channel trims.

## Тестирование

```bash
# Все тесты
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q

# Только ML
PYTHONPATH=backend python -m pytest tests/test_ml/ -v

# Только AI
PYTHONPATH=backend python -m pytest tests/test_ai/ -v

# Только инфраструктура
PYTHONPATH=backend python -m pytest tests/test_infrastructure/ -v

# Virtual mixer (без оборудования)
python virtual_mixer/virtual_mixer.py
```

CI: GitHub Actions — Python 3.10, 3.11, 3.12. Все PR должны пройти `pytest`.

## Workflow для агента

1. Перед началом работы — прочитать этот файл и `.cursorrules`
2. При изменении DSP — написать тест с синтетическими сигналами
3. При изменении handlers — обновить регистрацию в `handlers/__init__.py`
4. При добавлении OSC/MIDI команд — сверяться с `Docs/WING Remote Protocols v3.0.5.pdf`
5. При обнаружении бага или паттерна — добавить в раздел «Зафиксированные правила» выше
6. Перед коммитом — `PYTHONPATH=backend python -m pytest tests/ -x`
7. Не добавлять зависимости без явной необходимости

## Мульти-агентная разработка (Claude ⇄ Codex ⇄ Kimi)

Над проектом совместно работают три AI-ассистента: Claude, Codex (ChatGPT,
ведёт `ops/autopilot/`) и Kimi Code v3. Канал связи и зоны владения —
`ops/collab/PROTOCOL.md`; в начале каждой сессии проверь свой
`ops/collab/inbox/<имя>.md`, совместные решения — `ops/collab/DECISIONS.md`.
Мастер-план до рабочей версии — `docs/MASTER_PLAN_TO_V1.md`.
