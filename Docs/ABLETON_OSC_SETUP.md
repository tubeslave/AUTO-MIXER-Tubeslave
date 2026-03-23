# Ableton Live + AbletonOSC: настройка и команды

Практическое руководство: открыть Live, включить [AbletonOSC](https://github.com/ideoforms/AbletonOSC), управлять **громкостью трека**, **Gain в Utility**, **EQ Eight**, **Compressor** через OSC.

Порты и интеграция с AutoMixer см. также [TOOLS.md](TOOLS.md).

## 1. Установка AbletonOSC

1. Скачайте репозиторий [ideoforms/AbletonOSC](https://github.com/ideoforms/AbletonOSC) (ZIP).
2. Папку распакуйте и **переименуйте** в ровно `AbletonOSC` (не `AbletonOSC-main`).
3. Положите в каталог Remote Scripts:
   - **macOS**: `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC`
   - Официально: [Installing third-party remote scripts](https://help.ableton.com/hc/en-us/articles/209072009-Installing-third-party-remote-scripts)
4. Полностью закройте и снова откройте **Ableton Live** (11+).
5. **Preferences → Link / Tempo / MIDI**:
   - **Control Surface**: выберите **AbletonOSC**.
   - Внизу/в статусе должно появиться сообщение вроде «Listening for OSC on port **11000**».
6. Исходящие ответы Live идёт на **11001**. Клиент (AutoMixer или тестовый скрипт) должен **слушать UDP на 11001** на своей машине, а команды слать на **IP Live:11000**.

Проверка: отправьте OSC ` /live/test ` на `127.0.0.1:11000` — при корректной связке придёт ответ на recv-порт.

## 2. Индексы треков и устройств

- **track_id** — **с нуля**: первый аудиотрек = `0`, второй = `1`, …
- **device_id** — позиция устройства на цепочке трека, **с нуля**. Порядок **не угадывать**: всегда сначала запросить имена параметров (см. ниже).
- Значения параметров в OSC у AbletonOSC — **нормализованные float 0.0…1.0** (как в Live API), не «сырые» dB/Гц, если не используете отдельные human-readable эндпоинты скрипта.

## 3. Громкость (fader трека и мастера)

| Действие | OSC |
|----------|-----|
| Установить громкость трека | `/live/track/set/volume` `<track_id>` `<0..1>` |
| Запросить | `/live/track/get/volume` `<track_id>` |
| Мастер | `/live/master/set/volume` `<0..1>` / `/live/master/get/volume` |

В **AutoMixer** уровень в dB переводится в 0…1 по кривой marcobn (см. [TOOLS.md](TOOLS.md)); прямой OSC — только нормализованное значение.

## 4. Параметры устройств (Utility, EQ, Compressor)

### Узнать индексы параметров

Запрос (пример: трек 0, устройство 1):

```text
/live/device/get/parameters/name 0 1
```

В ответе придёт список имён; **номер в списке = param_index** для команд ниже.

### Установить параметр

```text
/live/device/set/parameter/value <track_id> <device_id> <param_index> <0..1>
```

### Прочитать параметр (ответ в лог)

```text
/live/device/get/parameter/value <track_id> <device_id> <param_index>
```

Ответ AbletonOSC: тот же адрес и аргументы `<track_id> <device_id> <param_index> <0..1>`. В **`AbletonClient`** значения кладутся в кэш; **`log_eq_eight_readback()`** / **`fetch_eq_eight_physical_bands()`** запрашивают Gain/Freq/Q по картам полос и пишут в лог расшифровку в dB / Hz / Q. После **одноканального** Auto-EQ apply и после **reset** readback вызывается автоматически; при **multi-channel apply всем** опрос не выполняется (задержка и нагрузка на OSC) — при необходимости вызовите **`log_eq_eight_readback(channel, tag="...")`** из кода или добавьте отдельную команду позже.

### Utility: Gain

1. Найдите устройство **Utility** на нужном треке: перебором `device_id` (0, 1, 2, …) или по памятке цепочки (см. §5).
2. В ответе на `get/parameters/name` найдите параметр с именем вроде **Gain** (точная строка зависит от языка/версии Live).
3. Подберите `<0..1>` экспериментально или через UI, параллельно подписавшись на изменения (см. AbletonOSC: `start_listen/parameter/value`).

**Важно:** параметр с индексом **0** у устройства — обычно **Device On**; его отключение = bypass всего устройства. Не путать с «гейном».

### EQ Eight

Те же команды: сначала **`get/parameters/name`**, затем **`set/parameter/value`**. Индексы полос зависят от версии плагина — надёжный путь только через список имён.

В **`AbletonClient`** по умолчанию: **Utility = устройство 0**, **EQ Eight = устройство 1** (`utility_device_index`, `eq_eight_device_index` в `config/default_config.json` → `ableton`, либо поля `utility_device_index` / `eq_eight_device_index` в сообщении `connect_ableton`). Индексы параметров Gain/Frequency/Resonance подтягиваются из ответа **`/live/device/get/parameters/name`** при подключении (английские имена вроде `1 Gain A`, `1 Frequency A`) для **физических** полос EQ Eight 1–8; при сбое — запасной маппинг.

**Переразметка Wing → EQ Eight (Auto-EQ):** у WING отдельные low shelf, 4 PEQ и high shelf; у EQ Eight одна тройка Gain/Freq/Q на полосу. Чтобы не затирать друг друга: **low shelf → физ. полоса 1**, **Wing PEQ 1…4 → физ. 2…5**, **high shelf → физ. 6** (полосы 7–8 не используются авто-EQ). Режимы shelf/bell в плагине по-прежнему задаются вручную в Live при необходимости.

**Сброс EQ (`reset_eq` / «сбросить всем»):** вызывается **`AbletonClient.reset_channel_eq_gains_zero()`**. Сначала снова запрашиваются имена параметров EQ Eight на **этом** треке. Если пришёл список (английский UI), для **каждой** строки вида **`N Gain`** (`1 Gain A`, …) отправляется **`/live/device/set/parameter/value`** с **точным индексом строки** и значением **0 dB** (норм. 0.5) — это надёжнее, чем только карта (gain/freq/q). Если имён нет (нет ответа на **recv**), используется запасной путь по **phys 1–6** и числовому fallback **(7,6,8)+10k**. Индексы в OSC приводятся к **int/float** как ожидает AbletonOSC.

### Compressor (стоковый Compressor)

Аналогично: **`get/parameters/name`** для нужного `device_id`, затем маппинг **Threshold, Ratio, Attack, Release, Makeup** (имена в списке — как в Live). В текущем `AbletonClient` методы `set_compressor*` для Ableton **не реализованы** (заглушки с предупреждением); управление — сырым OSC или будущим расширением клиента.

### Включить/выключить устройство целиком

```text
/live/device/set/enabled <track_id> <device_id> <0|1>
```

## 5. Рекомендуемый порядок устройств на треке

Чтобы совпасть с **дефолтами AutoMixer** (фаза + Auto-EQ):

- **Utility** — `utility_device_index` (по умолчанию **0**).
- **EQ Eight** сразу после Utility — `eq_eight_device_index` (по умолчанию **1**).

Если первым идёт встроенный **mixer device** (индекс 0), а Utility — вторым, задайте в конфиге `ableton.utility_device_index` / `ableton.eq_eight_device_index` по результатам `ableton_param_discovery.py`.

Примеры (проверяйте у себя через discovery):

- Шаблон по умолчанию в клиенте: `[Utility 0] → [EQ Eight 1] → …`
- С встроенным микшером: `[Mixer 0] → [Utility 1] → [EQ Eight 2] → …`

## 6. Скрипт в репозитории

```bash
cd backend && PYTHONPATH=. python ../scripts/ableton_param_discovery.py
```

Скрипт запрашивает имена параметров для треков `0..2` и устройств `0..2`. Для длинной цепочки временно увеличьте диапазоны в циклах в `scripts/ableton_param_discovery.py`.

## 7. Ссылки

- [ideoforms/AbletonOSC](https://github.com/ideoforms/AbletonOSC) — исходники и описание OSC-адресов
- [TOOLS.md](TOOLS.md) — порты, кривая громкости, Phase Alignment
