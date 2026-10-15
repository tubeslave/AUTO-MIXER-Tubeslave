# ADR: сначала корректировать источник bleed в параллельной ветви

- **Дата:** 2026-10-15 09:00 MSK
- **Статус:** accepted as experiment candidate
- **Update:** `ML-2026-10-15-0900`

## Контекст

Открытый разбор Nail The Mix описывает случай, где parallel drum compression усиливала cymbal bleed из snare-top. Инженер уменьшил соответствующий send, а не добавил новый post-compressor processor. Produce Like A Pro отдельно показывает функционально разделённые drum-FX returns, но не предоставляет контролируемого сравнения или универсальных параметров.

## Решение

1. Перед новым EQ/gate/debleed processor локализовать, какая ветвь и какой send усиливают bleed.
2. Создать `EXP-ML-20261015-0900-01`: текущий snare-top send против −3 dB, остальные параметры заморожены.
3. −3 dB считать только проектной проверочной точкой, не числом из источника.
4. Сохранить raw A/B и отдельно создать BS.1770-matched listening copies.
5. Не принимать уменьшение bleed, если одновременно ухудшились snare punch/body, ghost notes, groove или room depth.
6. Не считать одинаковое число plugins доказательством временного совпадения параллельных ветвей; использовать DAW PDC audit и null/impulse test.
7. Правило оставить `auto_apply:false`, эксперимент — `not_run`.

## Последствия

- Не добавляется универсальная drum-FX topology или compressor preset.
- Свежий reference-разбор *Sea Change* сохраняется как стилистический catalogue reference без DSP-метрик и raw audio.
- Следующий исполнимый приоритет остаётся `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Аудио, DSP, runtime и модели проекта не изменяются.
