# Инструкция по запуску голосового управления

## Проблема была решена

Старый процесс сервера занимал порт 8765. Процесс остановлен.

## Запуск сервера

### 1. Запустите backend сервер:

```bash
cd backend
python server.py
```

Сервер должен запуститься без ошибок. Вы увидите:
```
INFO:__main__:AutoMixer Server initialized on localhost:8765
INFO:__main__:Starting WebSocket server on localhost:8765
```

### 2. Запустите frontend:

В другом терминале:
```bash
cd frontend
npm start
```

### 3. Откройте приложение в браузере

Обычно открывается автоматически на `http://localhost:3000`

## Использование голосового управления

1. **Откройте вкладку "VOICE CONTROL"** в приложении
2. **Выберите аудиоустройство** из списка
3. **Выберите канал** для записи голосовых команд
4. **Нажмите "Start Voice Control"**

### Что должно произойти:

1. В консоли браузера (F12 → Console) появятся логи:
   - `Starting voice control with params: {...}`
   - `startVoiceControl sending: {...}`

2. В терминале сервера появятся логи:
   ```
   ============================================================
   RECEIVED start_voice_control MESSAGE
   ============================================================
   START_VOICE_CONTROL CALLED
   Step 1: Initializing VoiceControl...
   Step 2: Starting voice control listening...
   Step 3: Broadcasting success message...
   ```

3. В интерфейсе:
   - Статус изменится на "Voice Control Active"
   - Кнопка изменится на "Stop Voice Control"

### Если ничего не происходит:

1. **Проверьте логи сервера** - должны быть логи обработки запроса
2. **Проверьте консоль браузера** - должны быть логи отправки
3. **Убедитесь, что сервер запущен** - проверьте порт 8765
4. **Попробуйте перезапустить сервер** - остановите (Ctrl+C) и запустите заново

## Тестирование через терминал

Можно протестировать напрямую:

```bash
cd backend
python test_voice_button.py
```

Этот скрипт проверит работу голосового управления через WebSocket.

## Поддерживаемые команды

### Русский язык:
- `канал 1` - установить фейдер канала 1
- `гейн 3` - установить гейн канала 3
- `загрузить концерт` - загрузить снапшот "концерт"
- `мут 2` - заглушить канал 2
- `громче 4` - увеличить громкость канала 4
- `тише 6` - уменьшить громкость канала 6

### English:
- `channel 10` - set fader for channel 10
- `gain 5` - set gain for channel 5
- `load test` - load snapshot "test"
- `mute 1` - mute channel 1
- `louder 2` - increase volume for channel 2
- `quieter 3` - decrease volume for channel 3
