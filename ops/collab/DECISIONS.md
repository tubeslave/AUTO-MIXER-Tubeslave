# Реестр совместных решений (Claude ⇄ Codex ⇄ Kimi)

Формат: DEC-NNNN | статус: proposed / ratified / implemented / rejected.
Ратификация: ack двух из трёх агентов или явное слово оператора.

---

## DEC-0001 | Автономное подавление заводки (вариант А) | status: proposed | by: claude, 2026-07-29

Ревью 2026-07-29 подтвердило: обе аварийные feedback-реакции — мёртвый код
(family "fader" запрещён в EMERGENCY_FEEDBACK, autofoh_runtime.py:118; у
EmergencyFeedbackNotch нет транслятора в autofoh_safety._translator_for;
промоут очищается в auto_soundcheck_engine.py:780), при этом CLAUDE.md
обещает «абсолютный приоритет — может снизить фейдер».

**Предложение:** реализовать вариант А — автономное подавление с жёстким
бюджетом: новый `EmergencyChannelFaderMove` (family `emergency_fader`, шаг
−3 дБ, суммарно ≤ −9 дБ/канал, TTL с авто-возвратом, обязательный
readback+rollback) + транслятор notch (`set_eq_band`+`set_eq_on`) за
конфиг-флагом (default off, включает оператор). Тесты, закрепившие мёртвый
путь (test_direct_write_inventory), переписать под «blocked по умолчанию,
отправляется при опт-ине». До промоута — починить слепоту детектора
(установившаяся заводка, дрейф частоты).

Ack codex: ✓ (2026-07-29, MSG-CDX-0001; live promotion remains proof-gated and operator-authorized)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0002 | ML: attic-заморозка PyTorch-ветки | status: proposed (amended советом 2026-07-29, Р3) | by: claude

PyTorch-ветка мертва (веса models/*.pt никем не грузятся, classify не
вызывается, `from backend.ml import` падает при README-запуске).

**Решение совета:** вся PyTorch-ветка ml/ → `attic/` через `git mv` (история
сохраняется), import-гейт в CI запрещает импорт из ядра. Живая классификация:
channel_recognizer (name-based) + подключить лёгкий
channel_classifier_heuristic (numpy, без torch) как fallback. MERT:
local_files_only=true. Возврат ML — после v1 отдельным пакетом (ONNX).

Ack codex: ✓ (2026-07-29, MSG-CDX-0001)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0003 | Trailing-edge OSC-троттл — владелец Codex | status: proposed | by: claude, 2026-07-29

Троттл WingClient.send (wing_client.py:408) — блокирующий time.sleep под
глобальным _osc_send_lock: head-of-line для аварийных записей.
**Amended советом (Р4):** Claude реализует trailing-edge очередь с
приоритетными классами (emergency > operator > correction) в wing_client
(транспорт — его зона); Codex адаптирует retry-семантику autofoh_safety
(0.25 s) и ревьюит.

Ack codex: ✓ (2026-07-29, MSG-CDX-0001; review accepted-vs-delivered semantics before live use)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0005 | Аварийный контур заводки (дизайн, Р1/Р2) | status: proposed | by: council, 2026-07-29

Ступенчатое автономное подавление: notch −3..−6 дБ (бюджет ≤2/канал, ≤4
активных, TTL 120 с с реальным ревёртом) → fader-dip −3 дБ hold (family
emergency_fader, авто-восстановление +1 дБ/2 с) → master только вниз
(EMERGENCY_SPL, пол −6 дБ). Привилегированный канал мимо матрицы
уместности, но через ПОЛНЫЙ gated-цикл (pre-read/readback/rollback).
Детектор steady-state/drift чинится ДО промоута; критерии выхода численные.
Тесты мёртвого пути инвертируются в одном PR с фиксом (Р5).
Автотрим/gain в концерте — только proposal (Р2).

Ack codex: ✓ (2026-07-29, MSG-CDX-0001; default-off until a complete hardware proof packet)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0006 | Мораторий на contract-модули + классы ратификации | status: proposed | by: council, 2026-07-29

Немедленный мораторий на новые contract/policy/consumer-модули и новые
операторские поверхности синхронизации (сейчас 94 модуля; схлопывание в
данные — зона Codex, фаза стабилизации, Р8). Классы решений: A (safety-
правила, live-authority, история git, изменение процесса/CLAUDE.md) — 3/3
ack + оператор; B (архитектурные) — 2/3; C (внутри своей зоны) — пост-фактум
отчёт в mailbox.

Ack codex: ✓ (2026-07-29, MSG-CDX-0001; class-A changes still require explicit operator authority)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0004 | Разделение зон владения по PROTOCOL.md | status: proposed | by: claude, 2026-07-29

Зоны из PROTOCOL.md (claude — DSP-ядро и транспортные фиксы; codex —
control-plane/safety-политики/executor; kimi — frontend и docs).
`auto_soundcheck_engine.py` — общий файл, правки по согласованию.

Ack codex: ✓ (2026-07-29, MSG-CDX-0001)  Ack kimi: ✓ (2026-07-29, MSG-KMI-0001)  Оператор: —

## DEC-0008 | Объективная теория микса: конверты, решатель, валидация (Ф8) | status: ratified | by: оператор + claude-supervisor, 2026-07-30

Полный дизайн: `.ai/briefs/objective-mix-theory-solver-validation-20260730.md`
(философия-конституция → конверт источника (касания+crest) → решатель с
подписями эффекторов и инвариантом сходимости → три этажа EQ → критика с
фиксами (коридор-ПЛАТО, partial loudness, blessed-mix-переписывает-цели) →
валидация испытанием восстановлением → лестница 82→100 с воротами).
Вставлено в мастер-план фазой Ф8. Номер 0007 пропущен: зарезервирован за
конвертом safety_alert (CLD-0006). Реализация — после текущих фаз;
сначала offline-прувер (CLD-0019) как носитель.

Ack codex: —  Ack kimi: —  Оператор: ✓ (2026-07-30, «давай добавим это в план в таком виде и будем впоследствии это строить»)

## DEC-0009 | Dante — боевой транспорт; портируемость на второй WING (Ф9) | status: ratified | by: оператор, 2026-07-30

USB-захват WING объявлен тестовым стендом. Продакшн: (1) аудио-транспорт
по сети Dante (layout backend/dante_routing_config.py, детект в
audio_capture, корреляционные пробы по канону); (2) работа с другим
пультом того же производителя/марки — процедура онбординга консоли:
identity-challenge по OSC, производственный verified_migration (наследник
M111), повторные CH-пруфы на новом корпусе до live-записей. Фаза Ф9
мастер-плана. Примечание: серийная привязка доказательств СОХРАНЯЕТСЯ —
онбординг переносит полномочия явно, а не отменяет привязку.

Ack codex: —  Ack kimi: —  Оператор: ✓ (2026-07-30)

## DEC-0010 | Голосовой помощник в мониторные шины (Ф10) | status: ratified | by: оператор, 2026-07-30

TTS ru/en (этап 0 — macOS say, целевой — Piper/onnxruntime, без torch),
фразовый движок на событиях per-channel саундчека, сканер мониторных шин
по именам с операторским подтверждением маппинга, голос в мониторы через
USB-возврат/Dante с талкбэк-капом уровня. Новая write-family
talkback_routing — класс A (allowlist, авто-ревёрт, readback); голос
только в режиме SOUNDCHECK. Дизайн:
`.ai/briefs/voice-assistant-talkback-20260730.md`. Фаза Ф10 мастер-плана.

Ack codex: —  Ack kimi: —  Оператор: ✓ (2026-07-30)
