# Фреймворк детекции блиддинга (Drums + Vocals)

**Дата:** 18 марта 2025  
**Контекст:** Логи Safe Gain показывают ложные срабатывания и пропуски. Нужна точная детекция bleed для барабанов и вокалов.

---

## 1. Анализ логов (что не так)

### 1.1 Проблемы из терминальных логов

| Канал | Проблема | bleed_rej | own% |
|-------|----------|-----------|------|
| Ch3 (tom_mid) | Почти всё отвергнуто как bleed | 435 | 0.4% |
| Ch4 (tom_lo) | То же | 273 | 1.0% |
| Ch5 (hi_hat) | Агрессивный срез gain | 10 | 8.1% |
| Ch6 (ride) | Много rejection | 71 | 3.5% |
| Ch17–19 (vocal) | 47–87 reject, при этом gain +5..+18 dB | 47–87 | 2–3.4% |

**Вывод:** Bleed detector считает bleed там, где идёт основной сигнал (томы, vocals). И наоборот — может не распознавать bleed в тихих фазах.

### 1.2 Критическая ошибка: Safe Gain не передаёт band metrics

В `gain_staging_service.py` строка 366:

```python
self.bleed_service.update(levels, centroids, {})  # all_channel_metrics = {}
```

**Спектральная детекция** в BleedDetector требует `all_channel_metrics` (band_energy). При пустом — **spectral полностью отключена**. Работают только:
- temporal (envelope history + delayed repetition)
- level-based (ambient < -25 LUFS → ratio=0.95; high > -18 LUFS → ratio *= 0.5)

---

## 2. Ограничения текущего BleedDetector

| Метод | Ограничение |
|-------|-------------|
| **Spectral** | Не используется в Safe Gain (нет band metrics). Для drums: snare/tom/kick имеют похожие формы — легко false positive. |
| **Temporal** | Требует «target peaks после source»; при одновременных ударах (beat) — неоднозначность. Источник: envelope_history. |
| **Level-based** | Жёсткие пороги (-25 LUFS, -18 LUFS). Тихий вокал = ambient, всё считается bleed. |

### 2.1 Drums: специфика

- **Близкие формы**: kick, snare, tom — все percussive, transient.
- **Одновременность**: удары часто совпадают по времени.
- **Bleed-пути**: kick → snare (через общую подставку), snare → tom, hi-hat → всё.
- **Характеристики**: kick — sub/bass; snare — low_mid..high; tom — bass..mid; hi-hat — mid..air.

### 2.2 Vocals: специфика

- **Bleed**: вокал подхватывает drums (kick, snare), bass, guitar.
- **Форма**: vocal — mid, high_mid, high.
- **Тихий вокал**: baseline -27..-42 dBFS → level-based даёт ratio=0.95 (всё bleed).
- **Громкий вокал**: при правильном спектре — own signal, но spectral не работает без band metrics.

---

## 3. Архитектура фреймворка

### 3.1 Принципы

1. **Инструмент-специфичные детекторы** — drums и vocals имеют свои стратегии.
2. **Обязательные band metrics** — Safe Gain должен получать/вычислять band_energy для каждой полосы.
3. **Фазовая логика** — в BLEED_LEARN и WAIT использовать разные пороги.
4. **Конфигурируемые пороги** — per-instrument overrides.

### 3.2 Компоненты

```
BleedDetectionFramework
├── BandMetricsProvider (вычисление band_energy из samples)
├── DrumBleedDetector
│   ├── spectral: band dominance in source's characteristic bands
│   ├── transient: attack coincidence (onset detection)
│   └── level: source must be N dB louder in overlapping bands
├── VocalBleedDetector
│   ├── spectral: drum shape ≠ vocal shape
│   ├── centroid: vocal 120–8k Hz; kick/snare outside
│   └── level: ambient vs. own (с учётом формы)
└── BleedAggregator (combine per-instrument results)
```

### 3.3 BandMetricsProvider

Safe Gain должен получать band energies. Варианты:

- **A)** LUFSGainStagingController уже имеет `_audio_buffers`; добавить FFT/band analysis для каждого блока.
- **B)** В gain_staging_service вызывать анализатор (например, из auto_fader_v2) для получения band_energy.
- **C)** Простой band-pass RMS: 7 полос (sub, bass, low_mid, mid, high_mid, high, air) по RBJ biquads или FFT.

**Рекомендация:** Вариант C — простой band RMS в `lufs_gain_staging.py` или отдельном `band_analyzer.py`, передавать в `bleed_service.update(levels, centroids, band_metrics)`.

### 3.4 DrumBleedDetector — стратегии

| Сигнал | Собственный | Bleed |
|--------|-------------|-------|
| **Tom** | Peak в bass/low_mid/mid, форма соответствует tom curve | Peak в sub (kick) или high (snare/hihat) при совпадении по времени |
| **Snare** | Peak в low_mid..high, характерный crack | Sub/bass доминирует (kick) |
| **Hi-hat** | mid..air, форма hi-hat | Bass/low_mid при тихом hi-hat |
| **Kick** | sub/bass | mid/high при тихом kick |

**Ключевые правила:**
- Source в своих characteristic bands > target в тех же bands на 4+ dB.
- Target форма не соответствует target curve (target ≈ source curve) → bleed.
- Transient: если target peak на 0–20 ms после source peak и source громче → вероятный bleed.

### 3.5 VocalBleedDetector — стратегии

| Сигнал | Собственный | Bleed |
|--------|-------------|-------|
| **Vocal** | Peak в mid/high_mid, centroid 120–8k Hz | Sub/bass dominant (kick) или low_mid (snare) при тихом vocal |
| **Ambient** | — | LUFS < -25 → ratio=0.95 (текущий) |
| **Громкий вокал** | LUFS > -18, форма vocal | ratio *= 0.5 (scale down) |

**Ключевые правила:**
- Vocal curve: mid, high_mid, high.
- Kick/snare: sub, bass, low_mid, mid, high.
- Если в vocal канале sub/bass > mid на 6+ dB → скорее bleed от kick.
- Centroid: vocal 120–8k; если centroid < 80 Hz → kick bleed.

---

## 4. План реализации

### Этап 1: Band metrics в Safe Gain (критично) — **ВЫПОЛНЕНО**

1. Добавлен `backend/band_analyzer.py`: `compute_band_energy()`, `BandMetrics`, `samples_to_band_metrics()`.
2. В `gain_staging_service` передаётся `all_channel_metrics` с band_energy_* в `bleed_service.update()`.
3. Spectral detection в BleedDetector теперь получает band metrics и работает.

### Этап 2: Инструмент-специфичные детекторы — **ЧАСТИЧНО ВЫПОЛНЕНО**

1. Реализовано внутри `backend/auto_fader_v2/core/bleed_detector.py`:
   - Drum-specific refinement (`DRUM_INSTRUMENTS`, source-band dominance, own-curve protection)
   - Vocal-specific refinement (low-band excess detection, vocal own-curve protection)
   - Инструмент-специфичные пороги confidence/ratio из `bleed_protection`.
2. Следующий шаг: вынести логику в отдельные pluggable-модули (`drum_bleed.py`, `vocal_bleed.py`)
   без изменения алгоритма.

### Этап 3: Настройка порогов — **ЧАСТИЧНО ВЫПОЛНЕНО**

Добавлены новые параметры в `automation.bleed_protection`:
- `ambient_bleed_ratio`, `ambient_bleed_ratio_vocal`
- `instrument_specific_enabled`
- `drums_bleed_min_ratio`, `drums_bleed_min_confidence`
- `vocal_bleed_min_ratio`, `vocal_bleed_min_confidence`
- `vocal_low_band_excess_db`, `vocal_own_curve_similarity`

Добавлены новые параметры в `automation.safe_gain_calibration`:
- `exclude_bleed_from_own_capture` (по умолчанию `true`)
- `capture_bleed_guard_ratio`
- `capture_bleed_guard_confidence`

Эти параметры включают исключение bleed-событий из `WAIT_FOR_OWN_SIGNAL` и `CAPTURE_OWN_LEVEL`.

### Этап 4: Тесты

1. Синтетические сигналы: tom + kick bleed, vocal + snare bleed.
2. Unit-тесты для DrumBleedDetector, VocalBleedDetector.
3. Интеграционный тест с Safe Gain.

---

## 5. Ссылки

- `backend/auto_fader_v2/core/bleed_detector.py` — текущий BleedDetector
- `backend/bleed_service.py` — BleedService
- `backend/services/gain_staging_service.py` — Safe Gain (строка 366: `update(levels, centroids, {})`)
- `docs/CONVENTIONS.md` — DSP стандарты (RBJ biquads, LUFS)
