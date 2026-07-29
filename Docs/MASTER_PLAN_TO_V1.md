# Мастер-план до рабочей версии (v1) — AUTO-MIXER

Утверждён советом 2026-07-29 (`COUNCIL_2026-07-29.md`); алгоритм —
`WORKING_ALGORITHM.md`; acceptance-критерии v1 — протокол §7.
Исполнители: **Claude** (DSP-ядро, транспорт, safety-DSP), **Codex/ChatGPT**
(control-plane, оркестровка стадий, леджер/промоуты, CI-гейты),
**Kimi Code v3** (frontend, механические рефакторинги, docs/веб-исследования).
Канал: `ops/collab/` (PROTOCOL.md, DECISIONS.md). Правки вне своей зоны — по
DEC-решению; тест, закрепляющий признанный баг, меняется в одном PR с фиксом.

## Ф0 — Немедленно (день 0)

- [x] Канал связи ops/collab/, протокол, реестр решений, стартовые сообщения.
- [x] Протокол совещания, алгоритм, мастер-план (этот файл).
- [ ] Мораторий на новые contract-модули и новые операторские поверхности
      синхронизации (DEC-0006). — Codex ack
- [ ] CLAUDE.md: честная строка «feedback сейчас = detect + alert only» до
      закрытия Ф2 (документация не обещает несуществующую защиту). — Claude

## Ф1 — Метрология и транспортные инварианты (Claude, ~3 дня)

Пока судьи лгут, любая перестройка недоказуема. Каждый пункт = код + числовой тест.

1. TruePeakMeter: cutoff 24 кГц (0.9/OS), страховка max(sample_peak);
   тест-векторы 12/16/20 кГц ±0.5 дБ. [lufs_gain_staging.py:4637]
2. auto_eq: знак среза (abs(max_cut)); убрать позиционную shelf-конвертацию;
   тест на синтетическом спектре. [auto_eq.py:2115, :2171]
3. Fader ceiling на ВСЕХ поверхностях: общий _clamp_fader_db → set_main/
   dca/bus_fader, fx_return, sends (WING) + dlive set_fader; property-тест.
4. dLive: bool-возвраты set_phantom/pad/name/colour/eq_band.
5. Троттл: trailing-edge очередь + приоритетные классы (emergency > operator
   > correction), lock только на sendto; Codex адаптирует retry autofoh (DEC-0003).
6. _apply_bounds: current=None → blocked "current_unknown" (зона Codex —
   согласованная правка, Claude готовит патч+тест, Codex ревьюит).
7. master_mix_eval: stereo BS.1770 (−0.691 + 10·log10(z_L+z_R)); Welch-спектр.
8. Персистентный K-фильтр в signal_metrics; когерентность Welch-усреднением
   + гейт |r|≥0.25 в ветке задержки compare_channels.
9. CAIG: frame_samples=len(audio); тест длительностей фаз гейта.
10. ratio_float_to_wing на транспортной границе set_compressor.

## Ф2 — Аварийный контур заводки (Claude + Codex + Kimi, ~4 дня)

- Детектор: ветка steady-state (без is_growing), фикс дрейф-трекинга,
  tolerance от ширины бина, dBFS-нормировка; численные критерии выхода
  (recall на стационарных тонах, 0 FP на корпусе аккордеон/орган). — Claude
- Привилегированный emergency-канал: EmergencyChannelFaderMove (family
  emergency_fader, −3 дБ/шаг, ≤−9 дБ/канал, TTL, авто-восстановление),
  транслятор EmergencyFeedbackNotch (set_eq_band + set_eq_on), TTL-менеджер
  ревёрта notch, бюджеты; per-target демоут с cooldown. — Claude (DSP/действия)
  + Codex (политика/промоут/леджер-гейт).
- Детекция вне _monitor_iteration_lock (выделенный поток, бюджет <50 мс). — Claude
- Инверсия тестов мёртвого пути (в одном PR с фиксом). — Claude
- UI: алерт-центр (feedback/TP/silence, severity, звук, ≤200 мс), кнопки
  STOP AI / REVERT. — Kimi
- Монитор: channel_mapping, исключение master-tap, silence_fallback check. — Claude

## Ф3 — Один владелец на контрол + порядок стадий (Codex + Claude, ~3 дня)

- Слить _apply_gain_correction/_apply_fader; расписание фейдера channel →
  bus → DCA, стерео-идентичность как constraint решателя. — Codex (оркестровка)
- Фазовая стадия ДО тональных; drum_image объединить (один писатель delay);
  phase_v3 onset-gated как default-измеритель; same-source ДО polarity. — Claude
- Gate-порог follows trim (симметрично comp). — Claude
- Тихий fallback v3→legacy заменить на SKIPPED + residual. — Codex
- Синхронные буферы (get_synchronized_buffers) во всех межканальных замерах. — Claude

## Ф4 — Разминирование и чистка (Kimi под ревью Claude/Codex, ~3 дня)

- Удалить заряженные ружья (не флагами): legacy realtime-трим контур
  lufs_gain_staging, _reset_faders_to_zero, относительный fader-fallback
  (заменить абсолютным якорем — Claude). 
- attic/: PyTorch-ветка ml/ целиком, agents/*, bleed_leakage, instability_revert
  (или подключить — DEC), level_riding, dугan-дубликат, root-дубликаты,
  худший phase_alignment; import-гейт в CI. (Р3)
- 5 модулей с PyAudio-потоками → подписка AudioCapture.
- Git: artifacts/ из индекса, .gitignore, манифесты SHA-256; вложенную копию
  репо удалить. (Р7)
- tests/ → dsp_numeric / contracts / hardware_proof / ops_rails / integration.
- README-запуск без torch: ленивые ML-импорты, smoke-тест холодного старта.

## Ф5 — Замкнутые циклы и конвергенция (Claude, ~2 дня)

- Итеративный мастер-цикл K5: абсолютные цели, лимит проходов,
  anti-oscillation, журнал дельт.
- K3-конвергенция трима (медиана 3 проходов, коридор ±1.5 LU) как штатный
  критерий стадии linecheck.
- Venue-калибровочный проход в Prep (шум зала, запас до заводки, per-venue
  конфиг порогов PHPR/PNPR/IMSD и TARGET_MIX_LUFS).

## Ф6 — Программа изучения пульта + новая поверхность (Claude + Codex, ~4 дня + железо)

Proof-эксперименты на реальном WING (методика по образцу gain-proof; rollback
delay > троттл-интервала; фантом vph НЕ трогать):
1. Headamp `/io/in/<grp>/<n>/g`: квант 0.5 дБ, троттл, RO-статус `$g`.
2. `/snap/store` — существует ли (node-дамп `/$ctl/lib`); иначе SAVE вычеркнуть.
3. `$solo` писабельность (SIP не включать).
4. `/mgrp/N/mute` запись/чтение/стрим.
5. Бинарные метры 0xa2/0xa3/0xa6/0xaa (RTA!) + V2 0xab–0xaf.
6. Talkback `$on`, PUSH/LATCH.
7. W-Live цикл REC → session → `/io/altsw` c interlock-семантикой.
8. FX-карты параметров (node-дампы моделей в FX1) → таблица в Docs.
9. Систематический тест /xremote-покрытия по категориям.
10. `/ch/N/send/MX1..8` запись/readback.

Реализация по итогам: mute-группы + DCA-сцены + user-кнопка/GPIO как
аппаратная паника (приоритет 1); interlock «GO/altsw только вне live»;
headamp-gain стадия linecheck (supervised, шаги ≤3 дБ); RTA-метры в
feedback-детекцию; monitor sends — bounded-вниз (v1.5); W-Live виртуальный
саундчек (v1.5). Handlers/manual-write gate — Codex; клиент/метры — Claude;
FX-карты по community-источникам — Kimi.

## Ф7 — Полевая валидация (все + оператор)

Acceptance-чек-лист §7 протокола: стенд (virtual_mixer + USB loopback) →
реальный WING supervised → 3 концерта под присмотром. CI-гейты Codex: fast
(dsp_numeric+contracts <5 мин) → full → ops_rails (только control-plane) →
hardware_proof (вручную).

## Порядок и зависимости

Ф1 — пререквизит Ф5 (метрология раньше циклов). Ф2 не ждёт Ф1 (кроме
троттл-приоритетов из Ф1.5). Ф3 после Ф1.7-8. Ф4 параллельно всему (кроме
п.1 — сразу). Ф6-эксперименты — при первом доступе к железу. Оценка: ~3–4
недели календарно при трёх исполнителях.
