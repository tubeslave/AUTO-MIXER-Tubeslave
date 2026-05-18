# Отладка голосового управления

## Шаги для диагностики проблемы

### 1. Проверка консоли браузера

Откройте консоль браузера (F12 → Console) и проверьте:

1. **При нажатии кнопки "Start Voice Control"** должны появиться логи:
   ```
   Starting voice control with params: {modelSize: "small", language: "ru", deviceId: "0", channel: 0}
   startVoiceControl sending: {type: "start_voice_control", ...}
   Sending WebSocket message: {type: "start_voice_control", ...}
   ```

2. **При получении ответа от сервера**:
   ```
   WebSocket message received: voice_control_status {active: true, message: "Voice control started successfully"}
   Received voice_control_status: {active: true, message: "..."}
   ```

### 2. Проверка логов backend

В терминале, где запущен `server.py`, должны появиться логи:

```
INFO:server:Received start_voice_control request: {'type': 'start_voice_control', ...}
INFO:server:Starting voice control: model=small, language=ru, device_id=0, channel=0
INFO:server:Using audio device index: 0
INFO:server:Initializing VoiceControl...
INFO:voice_control:VoiceControl initialized (model: small, language: ru)
INFO:voice_control:Loading Whisper model: small on cpu
INFO:voice_control:Whisper model loaded successfully
INFO:server:Starting voice control listening...
INFO:voice_control:Voice control started listening
INFO:server:Voice control started successfully
```

### 3. Возможные проблемы и решения

#### Проблема: Кнопка не реагирует
- **Проверка**: Убедитесь, что выбран аудиоустройство
- **Решение**: Выберите устройство из списка перед нажатием кнопки

#### Проблема: WebSocket не подключен
- **Проверка**: В консоли браузера должно быть "WebSocket connected"
- **Решение**: Убедитесь, что backend сервер запущен на порту 8765

#### Проблема: Ошибка загрузки модели Whisper
- **Проверка**: В логах backend должна быть ошибка загрузки модели
- **Решение**: 
  ```bash
  pip install faster-whisper
  ```
  При первой загрузке модель скачается автоматически (~500MB)

#### Проблема: Ошибка открытия аудио устройства
- **Проверка**: В логах backend должна быть ошибка PyAudio
- **Решение**: 
  - Проверьте, что устройство не используется другим приложением
  - Попробуйте выбрать другое устройство
  - Проверьте системные разрешения для микрофона

#### Проблема: Сообщение отправляется, но нет ответа
- **Проверка**: В консоли браузера видно отправку, но нет получения
- **Решение**: 
  - Проверьте логи backend на наличие ошибок
  - Убедитесь, что обработчик сообщений работает правильно

### 4. Тестовая команда через консоль браузера

Можно протестировать отправку команды напрямую через консоль:

```javascript
// В консоли браузера
websocketService.startVoiceControl('small', 'ru', '0', 0);
```

### 5. Проверка состояния WebSocket

```javascript
// В консоли браузера
console.log('WebSocket state:', websocketService.ws?.readyState);
// 0 = CONNECTING
// 1 = OPEN (нормально)
// 2 = CLOSING
// 3 = CLOSED
```

### 6. Ручная проверка backend

Можно протестировать backend напрямую:

```python
# В Python консоли
from voice_control import VoiceControl

vc = VoiceControl(model_size="small", language="ru", input_device_index=0)
vc.load_model()

def test_callback(cmd):
    print(f"Command received: {cmd}")

vc.start_listening(test_callback)
# Говорите команды в микрофон
# Через 30 секунд:
vc.stop_listening()
```

## Типичные ошибки

1. **"No module named 'faster_whisper'"**
   - Решение: `pip install faster-whisper`

2. **"No audio devices found"**
   - Решение: Проверьте подключение микрофона и системные настройки

3. **"Invalid device_id"**
   - Решение: Убедитесь, что device_id - это число (индекс устройства)

4. **"Voice control already active"**
   - Решение: Сначала остановите текущее голосовое управление

5. **WebSocket connection refused**
   - Решение: Убедитесь, что backend сервер запущен
