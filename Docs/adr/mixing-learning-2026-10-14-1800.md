# ADR: MPASCNet остаётся заблокирован до provenance, parity, stereo и listening gates

- **Дата:** 2026-10-14 18:00 MSK
- **Статус:** accepted
- **Update:** `ML-2026-10-14-1800`
- **Источник поиска:** `ML-2026-10-14-1500`

## Контекст

MPASCNet сообщает улучшение SDR, SI-SNR и ViSQOL на коротких синтетически сжатых MP3-фрагментах, но не содержит слушательского теста. Углублённый аудит публичного кода показал, что тестовые файлы не несут source/time/bitrate provenance, стерео обрабатывается независимо, ViSQOL считает mono downmix, а разные семейства метрик используют выход до и после clamp. Точный checkpoint manifest не получен.

В уже существующей независимой карточке AEROMamba/PAQM объективный выигрыш ViSQOL/LSD для двух 32-kbps MP3-вариантов не дал значимого субъективного выигрыша относительно low-bitrate входа; лучший слушательский результат дал другой loss-вариант. Поэтому objective-only promotion противоречит накопленной базе.

## Решение

1. Не запускать и не интегрировать MPASCNet, пока не закреплены checkpoint filename/revision/SHA-256, code SHA, окружение и права.
2. Новый локальный evaluation manifest обязан сохранять source ID, start/duration, bitrate, encoder/decoder versions, common gain, delay estimate и hashes; фрагменты не должны пересекаться.
3. Все метрики считать и отчётно разделять для raw model output и сохранённого/clamped output; ViSQOL считать только mono diagnostic, не stereo gate.
4. Обязательны L/R, M/S, mono и correlation проверки, поскольку каналы обрабатываются независимо.
5. Принятие возможно только при повторяемом loudness-matched blind-listening выигрыше с hidden lossless reference; objective-only win отменяет кандидат.
6. `RULE-ML-20261014-1500-01` остаётся `auto_apply:false`; `EXP-ML-20261014-1500-01` остаётся `not_run_blocked`.

## Последствия

- Не переносим битрейты статьи в mixing preset и не называем restoration научно доказанным улучшением мастера.
- Не создаём дубликат `SRC-ML-20260930-1500-01`; используем его как независимый контрпример metric-only acceptance.
- Следующий исполнимый эксперимент не меняется: `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Аудио, DSP, runtime и модели проекта не изменены.
