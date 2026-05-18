# Backup and Restore Guide

## Создание бэкапа

Создать бэкап всех настроек всех 40 каналов:

```bash
cd backend
python3 backup_channels.py [IP] [PORT] [OUTPUT_FILE]
```

**Примеры:**
```bash
# Использовать значения по умолчанию (192.168.1.102:2223)
python3 backup_channels.py

# Указать IP и порт
python3 backup_channels.py 192.168.1.102 2223

# Указать имя файла
python3 backup_channels.py 192.168.1.102 2223 ../presets/my_backup.json
```

**Что сохраняется:**
- Все параметры входов (trim, balance, phase, delay)
- Все параметры каналов (fader, pan, width, mute)
- Все настройки фильтров (low cut, high cut, модель, tool filter)
- Все настройки EQ (включая все полосы)
- Все настройки компрессора
- Все настройки gate
- Все инсерты (pre и post) с полными настройками FX модулей
- Все параметры FX модулей

Файл сохраняется в `presets/channel_backup_YYYYMMDD_HHMMSS.json`

## Восстановление из бэкапа

Восстановить все настройки из бэкапа:

```bash
cd backend
python3 restore_channels.py <backup_file> [IP] [PORT]
```

**Примеры:**
```bash
# Восстановить из последнего бэкапа
python3 restore_channels.py ../presets/channel_backup_20260129_120246.json

# Указать IP и порт
python3 restore_channels.py ../presets/channel_backup_20260129_120246.json 192.168.1.102 2223
```

**Важно:** Восстановление требует подтверждения (введите 'YES')

## Структура файла бэкапа

```json
{
  "timestamp": "2026-01-29T12:02:54.919997",
  "wing_ip": "192.168.1.102",
  "wing_port": 2223,
  "channels": {
    "1": {
      "channel": 1,
      "input": { "trim": 0.0, "balance": 0.0, ... },
      "controls": { "fader": 0.0, "pan": 0.0, ... },
      "filters": { "model": "TILT", ... },
      "eq": { "on": 0, "bands": {...}, ... },
      "compressor": { "on": 0, ... },
      "gate": { "on": 0, ... },
      "inserts": {
        "pre_insert": {
          "slot": "FX13",
          "fx_module": {
            "model": "P-BASS",
            "parameters": { 1: -2.0, 2: -16.5, ... }
          }
        }
      }
    },
    ...
  }
}
```

## Использование в коде

```python
from backup_channels import backup_all_channels
from restore_channels import restore_from_backup

# Создать бэкап
backup_file = backup_all_channels("192.168.1.102", 2223)
print(f"Backup saved to: {backup_file}")

# Восстановить из бэкапа
restore_from_backup(backup_file, "192.168.1.102", 2223)
```

## Примечания

- Бэкап включает все 40 каналов
- Все инсерты и FX модули сохраняются с полными параметрами
- Файлы бэкапа сохраняются в директории `presets/`
- Размер файла обычно 80-100 KB для всех 40 каналов
- Восстановление может занять 10-15 секунд для всех каналов
