# Инструкция по тестированию функции Scan Mixer

## Шаг 1: Запуск backend сервера

```bash
cd backend
python3 server.py
```

Сервер должен запуститься и показать:
```
INFO:__main__:AutoMixer Server initialized on localhost:8765
INFO:__main__:Starting WebSocket server on ws://localhost:8765
```

## Шаг 2: Запуск теста через WebSocket

В другом терминале:

```bash
cd backend
python3 test_scan_mixer_websocket.py 192.168.1.102
```

Тест должен:
1. Подключиться к WebSocket серверу
2. Подключиться к микшеру
3. Отправить запрос scan_mixer_channel_names
4. Получить ответ с именами каналов

## Шаг 3: Проверка логов

После выполнения теста проверьте логи:

```bash
tail -100 .cursor/debug.log | grep -E "scan_mixer|Function called|Sending response"
```

## Ожидаемый результат

Тест должен показать:
- ✅ Подключено к микшеру
- ✅ ОТВЕТ ПОЛУЧЕН
- ✅ Количество каналов в ответе: 40
- ✅ Примеры имен каналов
- ✅ ТЕСТ ПРОЙДЕН УСПЕШНО
