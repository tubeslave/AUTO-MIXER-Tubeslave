# API для маршрутизации каналов

Добавлена возможность проводить любой роутинг каналов по запросу через OSC команды.

## Методы WingClient

### `route_output(output_group, output_number, source_group, source_channel)`

Маршрутизация одного выхода на источник.

**Параметры:**
- `output_group` (str): OUTPUT GROUP (куда посылать)
  - `"MOD"` - Module (DANTE модуль)
  - `"CRD"` - Card (карты, включая WLIVE PLAY)
  - `"LCL"` - Local Outputs
  - `"AUX"` - Aux Outputs
  - `"A"`, `"B"`, `"C"` - AES50 A/B/C Outputs
  - `"SC"` - StageConnect Outputs
  - `"USB"` - USB Outputs
  - `"AES"` - AES/EBU Outputs
  - `"REC"` - USB Record Outputs

- `output_number` (int): Номер выхода в OUTPUT GROUP (1..64)

- `source_group` (str): SOURCE GROUP (откуда брать сигнал)
  - `"CRD"` - Card (WLIVE PLAY карта)
  - `"PLAY"` - USB Player
  - `"CH"` - Channels (каналы пульта) - может не работать, проверьте документацию
  - `"AUX"` - Aux
  - `"BUS"` - Bus
  - `"MAIN"` - Main
  - `"MTX"` - Matrix
  - `"SEND"` - FX Send
  - `"MON"` - Monitor
  - `"USR"` - User Signal
  - `"OSC"` - Oscillator
  - И другие из документации

- `source_channel` (int): Номер канала источника из SOURCE GROUP (1..64)

**Пример:**
```python
from wing_client import WingClient

client = WingClient("192.168.1.102", 2223)
client.connect()

# Маршрутизация DANTE выхода 1 на WLIVE PLAY канал 1
client.route_output("MOD", 1, "CRD", 1)

# Маршрутизация локального выхода 5 на канал пульта 10
client.route_output("LCL", 5, "CH", 10)
```

### `route_multiple_outputs(output_group, start_output, num_outputs, source_group, start_source_channel)`

Маршрутизация нескольких выходов последовательно.

**Параметры:**
- `output_group` (str): OUTPUT GROUP
- `start_output` (int): Начальный номер выхода
- `num_outputs` (int): Количество выходов
- `source_group` (str): SOURCE GROUP
- `start_source_channel` (int): Начальный номер канала источника

**Возвращает:** Количество успешно маршрутизированных выходов

**Пример:**
```python
# Маршрутизация 24 DANTE выходов (MOD 1-24) на WLIVE PLAY каналы (CRD 1-24)
success_count = client.route_multiple_outputs("MOD", 1, 24, "CRD", 1)
print(f"Маршрутизировано {success_count} из 24 выходов")
```

### `get_output_routing(output_group, output_number)`

Получить текущую маршрутизацию выхода.

**Параметры:**
- `output_group` (str): OUTPUT GROUP
- `output_number` (int): Номер выхода

**Возвращает:** Dict с ключами:
- `output_group` - OUTPUT GROUP
- `output_number` - номер выхода
- `source_group` - SOURCE GROUP
- `source_channel` - номер канала источника

**Пример:**
```python
routing = client.get_output_routing("MOD", 1)
if routing:
    print(f"MOD выход 1: SOURCE GROUP={routing['source_group']}, канал={routing['source_channel']}")
```

## WebSocket API

Через WebSocket можно отправлять команды маршрутизации:

### `route_output`
```json
{
  "type": "route_output",
  "output_group": "MOD",
  "output_number": 1,
  "source_group": "CRD",
  "source_channel": 1
}
```

**Ответ:**
```json
{
  "type": "route_output_result",
  "success": true,
  "output_group": "MOD",
  "output_number": 1,
  "source_group": "CRD",
  "source_channel": 1
}
```

### `route_multiple_outputs`
```json
{
  "type": "route_multiple_outputs",
  "output_group": "MOD",
  "start_output": 1,
  "num_outputs": 24,
  "source_group": "CRD",
  "start_source_channel": 1
}
```

**Ответ:**
```json
{
  "type": "route_multiple_outputs_result",
  "success_count": 24,
  "total": 24,
  "output_group": "MOD",
  "start_output": 1,
  "source_group": "CRD",
  "start_source_channel": 1
}
```

### `get_output_routing`
```json
{
  "type": "get_output_routing",
  "output_group": "MOD",
  "output_number": 1
}
```

**Ответ:**
```json
{
  "type": "output_routing",
  "routing": {
    "output_group": "MOD",
    "output_number": 1,
    "source_group": "CRD",
    "source_channel": 1
  }
}
```

## Командная строка

Универсальный скрипт `route_channels.py`:

```bash
python3 route_channels.py <IP> <OUTPUT_GROUP> <START_OUTPUT> <NUM_OUTPUTS> <SOURCE_GROUP> <START_SOURCE_CHANNEL>
```

**Примеры:**

```bash
# DANTE выходы (MOD 1-24) <- WLIVE PLAY каналы (CRD 1-24)
python3 route_channels.py 192.168.1.102 MOD 1 24 CRD 1

# Локальные выходы (LCL 1-8) <- каналы пульта (CH 1-8)
python3 route_channels.py 192.168.1.102 LCL 1 8 CH 1

# USB выходы (USB 1-4) <- BUS шины (BUS 1-4)
python3 route_channels.py 192.168.1.102 USB 1 4 BUS 1
```

## Специальный скрипт для DANTE

Скрипт `route_dante_outputs.py` теперь использует универсальную функцию:

```bash
python3 route_dante_outputs.py [IP] [START_CHANNEL] [NUM_CHANNELS] [DANTE_START] [CARD_NAME]
```

По умолчанию маршрутизирует 24 канала на DANTE выходы из WLIVE PLAY карты.

## Маршрутизация входов каналов (Channel Input Routing)

### Методы WingClient

#### `set_channel_input(channel, source_group, source_channel)`

Назначить входной источник для канала (Channel Main).

**Параметры:**
- `channel` (int): Номер канала пульта (1-40)
- `source_group` (str): SOURCE GROUP (откуда брать сигнал) - "MOD", "CRD", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
- `source_channel` (int): Номер канала источника из SOURCE GROUP (1-64)

**Пример:**
```python
# Канал 1 получает сигнал с MOD канала 1
client.set_channel_input(1, "MOD", 1)

# Канал 10 получает сигнал с CRD канала 5
client.set_channel_input(10, "CRD", 5)
```

#### `set_channel_alt_input(channel, source_group, source_channel)`

Назначить альтернативный входной источник для канала (Channel ALT).

**Параметры:** Те же, что и для `set_channel_input`

**Пример:**
```python
# Канал 1 ALT получает сигнал с CRD канала 1
client.set_channel_alt_input(1, "CRD", 1)
```

#### `get_channel_input_routing(channel)`

Получить текущую маршрутизацию входов канала.

**Параметры:**
- `channel` (int): Номер канала пульта (1-40)

**Возвращает:** Dict с ключами:
- `channel` - номер канала
- `main_group` - SOURCE GROUP для Main
- `main_channel` - номер канала источника для Main
- `alt_group` - SOURCE GROUP для ALT
- `alt_channel` - номер канала источника для ALT

**Пример:**
```python
routing = client.get_channel_input_routing(1)
if routing:
    print(f"Channel 1 Main: {routing['main_group']}/{routing['main_channel']}")
    print(f"Channel 1 ALT: {routing['alt_group']}/{routing['alt_channel']}")
```

### WebSocket API

#### `set_channel_input`
```json
{
  "type": "set_channel_input",
  "channel": 1,
  "source_group": "MOD",
  "source_channel": 1
}
```

**Ответ:**
```json
{
  "type": "set_channel_input_result",
  "success": true,
  "channel": 1,
  "source_group": "MOD",
  "source_channel": 1
}
```

#### `set_channel_alt_input`
```json
{
  "type": "set_channel_alt_input",
  "channel": 1,
  "source_group": "CRD",
  "source_channel": 1
}
```

**Ответ:**
```json
{
  "type": "set_channel_alt_input_result",
  "success": true,
  "channel": 1,
  "source_group": "CRD",
  "source_channel": 1
}
```

#### `get_channel_input_routing`
```json
{
  "type": "get_channel_input_routing",
  "channel": 1
}
```

**Ответ:**
```json
{
  "type": "channel_input_routing",
  "routing": {
    "channel": 1,
    "main_group": "MOD",
    "main_channel": 1,
    "alt_group": "CRD",
    "alt_channel": 1
  }
}
```

### Командная строка

Скрипт `route_channel_inputs.py`:

```bash
python3 route_channel_inputs.py <IP> [START_CHANNEL] [NUM_CHANNELS] [MAIN_SOURCE] [ALT_SOURCE]
```

**Примеры:**

```bash
# Каналы 1-40: Main <- MOD 1-40, ALT <- CRD 1-40
python3 route_channel_inputs.py 192.168.1.102

# Каналы 1-20: Main <- MOD 1-20, ALT <- CRD 1-20
python3 route_channel_inputs.py 192.168.1.102 1 20

# Каналы 1-40: Main <- CRD 1-40, ALT <- MOD 1-40
python3 route_channel_inputs.py 192.168.1.102 1 40 CRD MOD
```
