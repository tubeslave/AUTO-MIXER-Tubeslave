# OpenClaw AppleScript Automation

Интеграция OpenClaw с macOS через AppleScript для управления приложениями и терминалом.

## Скрипты

### 1. OpenClaw Launcher.scpt
Главный лаунчер для запуска и управления OpenClaw Gateway.

**Функции:**
- Запуск/проверка OpenClaw Gateway
- Отправка команд агенту
- Управление приложениями (открыть/закрыть/скрыть/показать)
- Открытие терминала
- Проверка статуса

**Установка:**
```bash
# Копировать в Script Menu
mkdir -p ~/Library/Scripts
cp "OpenClaw Launcher.scpt" ~/Library/Scripts/

# Или создать приложение через Script Editor
```

### 2. OpenClaw Terminal Helper.scpt
Вспомогательный скрипт для работы с Terminal.app.

**Функции:**
- Запуск openclaw команд в новом окне терминала
- Выполнение и отображение результатов
- Поддержка интерактивных команд (dashboard, tui)

### 3. OpenClaw Application Bridge.scpt
Мост между OpenClaw и macOS приложениями.

**Функции:**
- Получение списка запущенных приложений
- Активация/закрытие приложений
- Выполнение произвольного AppleScript кода
- Создание быстрых команд на рабочем столе

## Требования

- macOS с AppleScript поддержкой
- OpenClaw установлен через npm/nvm
- Node.js v24+ (путь: `~/.nvm/versions/node/v24.11.1/bin/openclaw`)

## Настройка путей

Если путь к openclaw отличается, измените переменную в скриптах:
```applescript
property openclawPath : "/Users/YOUR_USERNAME/.nvm/versions/node/v24.11.1/bin/openclaw"
```

## Использование

### Вариант 1: Script Menu
1. Включите Script Menu: `Script Editor > Settings > Show Script menu in menu bar`
2. Скопируйте скрипты в `~/Library/Scripts/`
3. Доступ через иконку скриптов в меню-баре

### Вариант 2: Automator Quick Action
1. Откройте Automator
2. Создайте Quick Action
3. Добавьте "Run AppleScript"
4. Вставьте код из скрипта
5. Сохраните и назначьте горячую клавишу в `System Settings > Keyboard > Keyboard Shortcuts > Services`

### Вариант 3: Alfred/raycast
- Импортируйте скрипты как workflows

## Безопасность

- Gateway token хранится в открытом виде в скриптах
- Для продакшена рекомендуется использовать Keychain
- Проверяйте команды перед выполнением

## Интеграция с AUTO-MIXER

### OpenClaw Ableton Bridge.scpt
Специальный скрипт для интеграции с AUTO-MIXER Tubeslave:

**Функции:**
- 🔥 Полный сетап (Audio MIDI + AUTO-MIXER + OpenClaw)
- 🚀 Запуск AUTO-MIXER backend
- 🎵 Открытие Ableton Live проектов
- 🔊 Настройка Audio MIDI Setup
- 📡 Отправка OSC команд на WING mixer
- 🔍 Проверка статуса всех компонентов

**Быстрый запуск:**
```applescript
-- Полный сетап для live sound
fullLiveSetup()

-- Запуск AUTO-MIXER
startAutomixer()

-- Проверка всех систем
checkAllStatus()
```

### Пример использования для управления звуковыми приложениями:
```applescript
-- Открыть Audio MIDI Setup
controlApplication("Audio MIDI Setup", "activate")

-- Открыть Ableton Live
controlApplication("Ableton Live", "activate")
```
