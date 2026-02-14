# Auto Mixer Tubeslave

Приложение для автоматического микширования с использованием микшерного пульта **Behringer Wing Rack** (fw 3.0.5) через протокол OSC.

## Структура проекта

```
AUTO MIXER Tubeslave/
├── backend/              # Python-бэкенд с OSC-клиентом
│   ├── wing_client.py    # Класс для управления Wing через OSC
│   ├── wing_addresses.py # Справочник OSC-адресов Wing
│   ├── server.py         # WebSocket-сервер для связи с фронтендом
│   └── requirements.txt  # Зависимости Python
├── frontend/             # React-фронтенд
│   ├── public/
│   ├── src/
│   │   ├── components/   # Компоненты GUI
│   │   ├── services/     # WebSocket-сервис
│   │   └── App.js
│   └── package.json
├── config/               # Конфигурационные файлы
│   └── default_config.json
├── presets/              # Сохраненные пресеты микса
└── Docs/                 # Документация
```

## Возможности (Этап 1)

### Реализовано:
- ✅ OSC-клиент с двусторонней синхронизацией с Wing Rack
- ✅ WebSocket-сервер для real-time коммуникации
- ✅ GUI с полным отображением параметров пульта:
  - Настройки подключения (IP, порты, модель)
  - 48 канальных полос с фейдерами и gain
  - Индикаторы уровней
  - 4-полосный параметрический EQ на канал
  - Компрессор с настройками threshold, ratio, attack, release
  - Интерфейс для Gate и Routing (в разработке)

### В разработке:
- ⏳ Модули автоматизации (Auto-Gain, Auto-EQ, Auto-Mix)
- ⏳ Система пресетов
- ⏳ Real-time анализ аудио с FFT

## Требования

### Backend:
- Python 3.9+
- python-osc
- websockets
- numpy, scipy (для модулей автоматизации)

### Frontend:
- Node.js 16+
- React 18

## Установка

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

## Запуск

### 1. Запуск Backend-сервера

```bash
cd backend
python server.py
```

Сервер запустится на `ws://localhost:8765`

### 2. Запуск Frontend

```bash
cd frontend
npm start
```

Приложение откроется в браузере на `http://localhost:3000`

## Подключение к Wing Rack

1. Подключите Wing Rack к той же сети, что и компьютер
2. Найдите IP-адрес пульта в настройках Wing (Setup → Network)
3. В интерфейсе приложения:
   - Выберите модель: **Wing Rack**
   - Введите IP-адрес (по умолчанию: `192.168.1.100`)
   - Send Port: `2222` (стандартный для Wing)
   - Receive Port: `2223`
   - Нажмите **Connect to Wing**

4. При успешном подключении:
   - Статус изменится на "Wing: Connected"
   - GUI синхронизируется с текущим состоянием пульта
   - Все изменения на пульте отобразятся в реальном времени

## OSC-адреса Wing

Примеры адресов согласно спецификации Wing OSC API:

- Фейдер канала: `/ch/01/mix/fader`
- Gain канала: `/ch/01/preamp/trim`
- EQ полоса 1 частота: `/ch/01/eq/1/f`
- Компрессор threshold: `/ch/01/dyn/thr`
- Подписка на обновления: `/xremote`

Полный справочник в файле `backend/wing_addresses.py`

## Настройка конфигурации

Отредактируйте `config/default_config.json` для изменения:
- Параметров подключения по умолчанию
- Настроек модулей автоматизации
- Safety Limits (защита от перегрузок)

## Тестирование OSC-связи

Используйте тестовый скрипт для проверки подключения:

```bash
cd backend
python test_wing_connection.py
```

Скрипт выведет все входящие OSC-сообщения. Покрутите ручки на пульте и убедитесь, что изменения отображаются.

## Следующие шаги

### Этап 2: Модули автоматизации
- Auto-Gain: автоматическая регулировка входного уровня
- Auto-EQ: FFT-анализ и подавление резонансов
- Auto-Mix / Ducker: балансировка каналов

### Этап 3: Система пресетов и финализация
- Сохранение/загрузка пресетов микса
- Safety Limits
- Оптимизация GUI

## Ссылки на документацию

- [Behringer Wing OSC Implementation](https://wiki.munichmakerlab.de/wiki/Behringer_Wing)
- [Mixing Secrets for the Small Studio - Mike Senior](https://www.cambridge-mt.com/ms/mix-book/)
- [Sound Systems: Design and Optimization - Bob McCarthy](https://bobmccarthy.com/)

## Поддержка

При возникновении проблем:
1. Проверьте сетевое подключение Wing и компьютера
2. Убедитесь, что порты 2222/2223 не заняты
3. Проверьте версию прошивки Wing (должна быть 3.0.5)
4. Проверьте логи backend-сервера

---

**Firmware:** Wing Rack fw 3.0.5  
**Protocol:** OSC (Open Sound Control)  
**License:** MIT
