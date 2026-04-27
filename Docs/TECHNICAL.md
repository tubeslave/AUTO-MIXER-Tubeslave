# Техническая документация - Этап 1

## Архитектура приложения

### Компоненты системы

```
┌─────────────────┐         WebSocket (8765)        ┌─────────────────┐
│                 │◄────────────────────────────────►│                 │
│  React Frontend │                                  │  Python Backend │
│  (Browser GUI)  │                                  │  (WS Server)    │
│                 │                                  │                 │
└─────────────────┘                                  └────────┬────────┘
                                                              │
                                                              │ OSC
                                                              │ UDP 2222/2223
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  Behringer Wing │
                                                     │  Rack fw 3.0.5  │
                                                     └─────────────────┘
```

### Backend (Python)

**Файлы:**
- `wing_client.py` - Класс WingClient для OSC-коммуникации
- `wing_addresses.py` - Справочник OSC-адресов Wing
- `server.py` - WebSocket-сервер (asyncio)
- `test_wing_connection.py` - Тестовый скрипт

**Основные возможности WingClient:**
1. Подключение к Wing через OSC (UDP)
2. Двусторонний обмен данными
3. Автоматическое сканирование состояния пульта при подключении
4. Callback-система для обработки входящих OSC-сообщений
5. API для управления:
   - Фейдерами каналов
   - Gain/Trim
   - 4-полосным EQ
   - Компрессорами/гейтами
   - Роутингом

**Протокол коммуникации:**

Backend ↔ Frontend (WebSocket JSON):
```json
{
  "type": "connect_wing",
  "ip": "192.168.1.100",
  "send_port": 2222,
  "receive_port": 2223
}

{
  "type": "set_fader",
  "channel": 1,
  "value": 0.75
}

{
  "type": "wing_update",
  "address": "/ch/01/mix/fader",
  "values": [0.75]
}
```

Backend ↔ Wing (OSC):
```
/xremote                          # Подписка на обновления
/ch/01/mix/fader <float>          # Фейдер канала 1
/ch/01/preamp/trim <float>        # Gain канала 1
/ch/01/eq/1/f <float>             # EQ band 1 частота
/ch/01/eq/1/g <float>             # EQ band 1 gain
/ch/01/eq/1/q <float>             # EQ band 1 Q-фактор
/ch/01/dyn/thr26 <float>          # Компрессор threshold
```

### Frontend (React)

**Компоненты:**
1. `App.js` - Главный компонент, управление состоянием
2. `ConnectionPanel.js` - Панель подключения к Wing
3. `MixerView.js` - Основной view микшера с табами
4. `ChannelStrip.js` - Канальная полоса (фейдер + gain + meter)

**Сервисы:**
- `websocket.js` - Singleton для WS-коммуникации с бэкендом

**Состояние приложения:**
- `isConnected` - Статус WS-соединения с бэкендом
- `wingConnected` - Статус подключения к Wing
- `mixerState` - Объект с текущим состоянием всех параметров

**Структура mixerState:**
```javascript
{
  "/ch/01/mix/fader": 0.75,
  "/ch/01/preamp/trim": 6.0,
  "/ch/01/eq/1/f": 1000.0,
  "/ch/01/eq/1/g": -3.0,
  "/ch/01/eq/1/q": 2.5,
  ...
}
```

## OSC-адреса Wing Rack (WING Remote Protocols v3.0.5)

**Важно:** Все адреса используют формат `/ch/1`, `/ch/2` и т.д. (без нулей), как указано в официальной документации.

### Каналы (1-40)

#### Входные настройки (Input Settings)
- `/ch/X/in/set/trim` - Trim (-18 до +18 dB) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
- `/ch/X/in/set/bal` - Balance (-9 до +9 dB)
- `/ch/X/in/set/inv` - Инверсия фазы (0/1)
- `/ch/X/in/set/dly` - Задержка (зависит от режима)
- `/ch/X/in/set/dlyon` - Включение задержки (0/1)
- `/ch/X/in/set/dlymode` - Режим задержки: M, FT, MS, SMP

#### Фильтры (Filters)
- `/ch/X/flt/lc` - Low cut включение (0/1)
- `/ch/X/flt/lcf` - Low cut частота (20-2000 Hz)
- `/ch/X/flt/lcs` - Low cut slope: 6, 12, 18, 24
- `/ch/X/flt/hc` - High cut включение (0/1) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
- `/ch/X/flt/hcf` - High cut частота (50-20000 Hz) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
- `/ch/X/flt/hcs` - High cut slope: 6, 12 ✅ **ПРАВИЛЬНЫЙ АДРЕС**

#### Gate
- `/ch/X/gate/on` - Включение (0/1)
- `/ch/X/gate/mdl` - Модель gate: GATE, DUCK, E88, 9000G, D241, DS902, WAVE, DEQ, WARM, 76LA, LA, RIDE, PSE, CMB
- `/ch/X/gate/thr` - Threshold (-80 до 0 dB)
- `/ch/X/gate/range` - Range (3-60 dB)
- `/ch/X/gate/att` - Attack (0-120 ms)
- `/ch/X/gate/hld` - Hold (0-200 ms)
- `/ch/X/gate/rel` - Release (4-4000 ms)
- `/ch/X/gate/acc` - Accent (0-100)
- `/ch/X/gate/ratio` - Ratio: 1:1.5, 1:2, 1:3, 1:4, GATE

#### Динамика (Компрессор)
- `/ch/X/dyn/on` - Включение (0/1)
- `/ch/X/dyn/mdl` - Модель: COMP, EXP, B160, B560, D241, ECL33, 9000C, SBUS, RED3, 76LA, LA, F670, BLISS, NSTR, WAVE, RIDE, 2250, L100, CMB
- `/ch/X/dyn/thr26` - Threshold (-60 до 0 dB)
- `/ch/X/dyn/ratio` - Ratio: 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10, 20, 50, 100
- `/ch/X/dyn/knee` - Knee (0-5)
- `/ch/X/dyn/att` - Attack (0-120 ms)
- `/ch/X/dyn/hld` - Hold (1-200 ms)
- `/ch/X/dyn/rel` - Release (4-4000 ms)
- `/ch/X/dyn/gain` - Make-up gain (-6 до +12 dB)
- `/ch/X/dyn/mix` - Mix (0-100 %)
- `/ch/X/dyn/det` - Detection: PEAK, RMS
- `/ch/X/dyn/env` - Envelope: LIN, LOG
- `/ch/X/dyn/auto` - Auto switch (0/1)

#### EQ (6 полос: Low shelf + 4 bands + High shelf)
- `/ch/X/eq/on` - Включение всего EQ (0/1) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
- `/ch/X/eq/mdl` - Модель EQ: STD, SOUL, E88, E84, F110, PULSAR, MACH4
- `/ch/X/eq/mix` - Mix (0-125 %)
- **Low shelf:**
  - `/ch/X/eq/lg` - Low gain (-15 до +15 dB)
  - `/ch/X/eq/lf` - Low frequency (20-2000 Hz)
  - `/ch/X/eq/lq` - Low Q (0.44-10)
  - `/ch/X/eq/leq` - Low type: PEQ, SHV
- **Bands 1-4:**
  - `/ch/X/eq/1f` - Частота полосы 1 (20-20000 Hz) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
  - `/ch/X/eq/1g` - Gain полосы 1 (-15 до +15 dB) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
  - `/ch/X/eq/1q` - Q полосы 1 (0.44-10) ✅ **ПРАВИЛЬНЫЙ АДРЕС**
  - `/ch/X/eq/2f`, `/ch/X/eq/2g`, `/ch/X/eq/2q` - Полоса 2
  - `/ch/X/eq/3f`, `/ch/X/eq/3g`, `/ch/X/eq/3q` - Полоса 3
  - `/ch/X/eq/4f`, `/ch/X/eq/4g`, `/ch/X/eq/4q` - Полоса 4
- **High shelf:**
  - `/ch/X/eq/hg` - High gain (-15 до +15 dB)
  - `/ch/X/eq/hf` - High frequency (50-20000 Hz)
  - `/ch/X/eq/hq` - High Q (0.44-10)
  - `/ch/X/eq/heq` - High type: SHV, PEQ

#### Основные параметры канала
- `/ch/X/mute` - Mute (0/1)
- `/ch/X/fdr` - Фейдер (dB, -144 до +10)
- `/ch/X/pan` - Панорама (-100 до +100)
- `/ch/X/wid` - Ширина (%) -150 до +150
- `/ch/X/name` - Имя канала (до 16 символов)
- `/ch/X/col` - Цвет (1-12)
- `/ch/X/icon` - Иконка (0-999)

#### Sends (посылы на шины)
- `/ch/X/send/1/on` - Включение send 1 (0/1)
- `/ch/X/send/1/lvl` - Уровень send 1 (dB, -144 до +10)
- `/ch/X/send/1/mode` - Режим: PRE, POST, GRP
- `/ch/X/send/1/pan` - Pan send 1 (-100 до +100)
- `/ch/X/send/1/pon` - Pre always on (0/1)
- `/ch/X/send/MX1/on` - Matrix send 1 (0/1)
- `/ch/X/send/MX1/lvl` - Matrix send 1 уровень

#### Main sends
- `/ch/X/main/1/on` - Включение send на Main 1 (0/1)
- `/ch/X/main/1/lvl` - Уровень send на Main 1 (dB)
- `/ch/X/main/pre` - Pre fader на Main

### Bus (1-16)
- `/bus/X/fdr` - Фейдер bus
- `/bus/X/mute` - Mute bus
- `/bus/X/pan` - Pan bus
- `/bus/X/eq/on` - Включение EQ (8 полос)
- `/bus/X/eq/1f` - EQ полоса 1 частота
- `/bus/X/eq/1g` - EQ полоса 1 gain
- `/bus/X/eq/1q` - EQ полоса 1 Q
- (полосы 1-6 + low/high shelf)

### Main (1-4)
- `/main/X/fdr` - Фейдер main
- `/main/X/mute` - Mute main
- `/main/X/pan` - Pan main
- `/main/X/eq/on` - Включение EQ (8 полос)

### Matrix (1-8)
- `/mtx/X/fdr` - Фейдер matrix
- `/mtx/X/mute` - Mute matrix

### DCA (1-16)
- `/dca/X/fdr` - Фейдер DCA
- `/dca/X/mute` - Mute DCA

### Подписка на обновления
- `/xremote` - Отправляется периодически для получения обновлений от Wing

## Формат данных OSC

Wing использует следующие типы данных:
- **Float** - Большинство параметров (0.0-1.0, -1.0 до +1.0, и т.д.)
- **Integer** - Переключатели (0/1), индексы
- **String** - Имена каналов, метки

### Примеры значений:

**Фейдер (0.0 - 1.0):**
- 0.0 = -∞ dB
- 0.75 = 0 dB (unity)
- 1.0 = +10 dB

**Gain (-12 до +60 dB):**
- Передается как float, значения в dB

**Pan (-1.0 до +1.0):**
- -1.0 = крайний левый
- 0.0 = центр
- +1.0 = крайний правый

## Особенности реализации

### Двусторонняя синхронизация

1. **При подключении:**
   - Backend отправляет `/xremote` для подписки
   - Backend сканирует начальное состояние (запрашивает все параметры)
   - Wing отправляет текущие значения
   - Backend обновляет `state` и транслирует во Frontend

2. **Изменения на Wing:**
   - Wing отправляет OSC-сообщение на Backend
   - Backend обновляет `state`
   - Backend транслирует изменение во все подключенные Frontend-клиенты

3. **Изменения во Frontend:**
   - Frontend отправляет команду через WebSocket
   - Backend преобразует в OSC и отправляет на Wing
   - Wing применяет изменение
   - Wing отправляет подтверждение (новое значение)
   - Цикл завершается синхронизацией

### Обработка ошибок

- **Таймаут подключения:** Backend пытается подключиться 5 секунд
- **Потеря соединения:** Frontend автоматически переподключается к WS каждые 3 секунды
- **Неверные значения:** Backend проверяет диапазоны перед отправкой на Wing

### Safety Limits (планируется)

Защита от критических значений:
- Max Fader: +10 dB
- Max Gain: +18 dB
- Предупреждения при критических уровнях

## Следующие этапы разработки

### Этап 2: Модули автоматизации
1. **Auto-Gain** - Автоматическая регулировка входного уровня
2. **Auto-EQ** - FFT-анализ и подавление резонансов
3. **Auto-Mix / Ducker** - Dugan-style автомикс

### Этап 3: Финализация
1. Система пресетов (JSON)
2. Real-time meters через OSC `/meters/`
3. Графическая визуализация EQ
4. Safety Limits
5. Логирование сессий

## Ссылки

- [Wing OSC Protocol Documentation](https://wiki.music-tribe.com/wing)
- [Python-OSC Library](https://github.com/attwad/python-osc)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

---

**Версия:** 1.0 (Этап 1 завершен)  
**Дата:** 2025  
**Статус:** Готов к тестированию с реальным Wing Rack
