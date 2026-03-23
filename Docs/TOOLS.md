# Инструменты и протоколы

## Behringer WING Rack (OSC)

- **Send port**: 2222
- **Receive port**: 2223
- **Keepalive/subscribe**: `/xremote` (периодически)
- **Адреса каналов**: `/ch/{N}/...` (см. `backend/wing_addresses.py`)

### Примеры

- Fader канала: `/ch/01/mix/fader`
- Gain/trim: `/ch/01/preamp/trim`
- EQ band1 freq: `/ch/01/eq/1/f`

### Нормировка fader (WING)

- 0.0 = -∞
- 0.7498 ≈ 0 dB
- 1.0 ≈ +10 dB

## Ableton Live (AbletonOSC)

- **Send port**: 11000 (Ableton слушает)
- **Receive port**: 11001 (ответы Ableton)
- **Требуется**: AbletonOSC Remote Script (Live 11+)

### Задержка и инверсия фазы (Phase Alignment)

AutoMixer применяет в Ableton (Live 11/12):

1. **Utility** (индекс 0) — инверсия фазы: параметры 1 (Left Inv), 2 (Right Inv) — по OSC
2. **Track Delay** — UI automation отключён (не работает должным образом). Настраивайте вручную в Ableton: View → Arrangement Track Controls → Track Options → Track Delay
3. **Gain/Trim** — в Ableton нет pre-fader gain; `set_channel_gain` маппится на track volume (fader) для Gain Staging и Safe Gain Calibration
4. **Volume curve** — dB↔linear по кривой marcobn (AbletonOSC #44): -18..+6 dB → (db+34)/40

**Настройка сета в Ableton:**

- На каждый трек, участвующий в Phase Alignment, добавьте **Utility** первым (индекс 0)
- Track Delay настраивайте вручную в поле Track Delay (mixer control / track option)

Если порядок Utility другой, укажите `utility_device_index` в конфиге Ableton.

**Подробнее (Utility Gain, EQ Eight, Compressor, примеры OSC):** [ABLETON_OSC_SETUP.md](ABLETON_OSC_SETUP.md)

## Документация

- `Docs/WING Remote Protocols v3.0.5.pdf` — первоисточник по OSC протоколу WING

## Локальные инструменты проекта

- Скрипты запуска: `start_backend.sh`, `start_frontend.sh`
- Сборка: `build.sh`
