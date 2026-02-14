# Сборка .app для macOS

## Требования

- macOS 10.13 или новее
- Python 3.9+
- Node.js 16+
- npm или yarn

## Быстрая сборка

```bash
./build.sh
```

Этот скрипт автоматически:
1. Собирает Python backend с PyInstaller
2. Устанавливает npm зависимости (если нужно)
3. Собирает React frontend
4. Создает .app bundle с electron-builder

## Пошаговая сборка

### Шаг 1: Сборка Backend

```bash
./build_backend.sh
```

Или вручную:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pyinstaller --clean server.spec
deactivate
cd ..
```

Результат: `backend/dist/server` (исполняемый файл)

### Шаг 2: Установка Frontend зависимостей

```bash
cd frontend
npm install
```

### Шаг 3: Сборка .app

```bash
cd frontend
npm run dist
```

Результат: `frontend/dist/AUTO MIXER Tubeslave.app`

## Установка

```bash
cp -r "frontend/dist/AUTO MIXER Tubeslave.app" /Applications/
```

Или перетащите .app в папку Applications через Finder.

## Структура .app bundle

```
AUTO MIXER Tubeslave.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── AUTO MIXER Tubeslave (Electron executable)
│   ├── Resources/
│   │   ├── app.asar (React frontend)
│   │   ├── backend/
│   │   │   └── server (Python backend executable)
│   │   └── icon.icns
│   └── Frameworks/ (Electron frameworks)
```

## Как работает приложение

1. При запуске .app, Electron main process (`electron.js`) запускает:
   - Python backend сервер (`backend/server`) в фоновом режиме
   - WebSocket сервер на порту 8765
2. Через 2 секунды открывается Electron окно с React frontend
3. Frontend подключается к backend через WebSocket
4. Backend управляет OSC-коммуникацией с Wing Rack

## Отладка

### Режим разработки

```bash
cd frontend
npm run electron:dev
```

Это запустит:
- React dev server на `http://localhost:3000`
- Electron окно с DevTools
- Python backend из исходников (не из bundle)

### Логи

В режиме разработки логи backend выводятся в консоль Electron.

В production логи можно увидеть через:

```bash
open /Applications/AUTO\ MIXER\ Tubeslave.app
# Затем: View → Toggle Developer Tools
```

## Известные проблемы

### PyInstaller и PyAudio

Если PyInstaller не может найти PyAudio:

```bash
cd backend
source venv/bin/activate
pip install --force-reinstall pyaudio
pyinstaller --clean server.spec
```

### Electron builder fails

Если electron-builder не может создать .app:

1. Очистите кэш:
   ```bash
   cd frontend
   rm -rf dist build node_modules/electron
   npm install
   ```

2. Проверьте, что backend собран:
   ```bash
   ls -la backend/dist/server
   ```

### Иконка не отображается

Замените `frontend/public/icon.png` на вашу иконку (512x512 PNG).

## Подпись и нотаризация (для распространения)

Для распространения вне App Store нужна подпись Apple Developer ID.

### 1. Получите Developer ID Certificate

- Войдите в https://developer.apple.com
- Создайте Developer ID Application certificate

### 2. Настройте подпись

В `frontend/package.json`, в секции `build.mac`:

```json
"identity": "Developer ID Application: Your Name (TEAM_ID)",
"hardenedRuntime": true
```

### 3. Соберите и подпишите

```bash
cd frontend
npm run dist
```

### 4. Нотаризация (для macOS 10.15+)

```bash
xcrun notarytool submit "frontend/dist/AUTO MIXER Tubeslave.dmg" \
  --apple-id "your@email.com" \
  --password "app-specific-password" \
  --team-id "TEAM_ID" \
  --wait
```

## Альтернативные методы распространения

### DMG (рекомендуется)

DMG создается автоматически при `npm run dist`.

Результат: `frontend/dist/AUTO MIXER Tubeslave-1.0.0.dmg`

### ZIP

ZIP также создается автоматически.

Результат: `frontend/dist/AUTO MIXER Tubeslave-1.0.0-mac.zip`

### Homebrew Cask (для опытных пользователей)

Создайте формулу Homebrew Cask для автоматической установки.

## Обновление версии

1. Обновите версию в `frontend/package.json`
2. Соберите заново: `./build.sh`

## CI/CD

Пример GitHub Actions для автоматической сборки:

```yaml
name: Build macOS App

on: [push, pull_request]

jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Build
        run: ./build.sh
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: AUTO-MIXER-Tubeslave-macOS
          path: frontend/dist/*.dmg
```

## Помощь

Если возникли проблемы:

1. Проверьте логи: `npm run electron:dev`
2. Убедитесь, что backend собирается: `./build_backend.sh`
3. Очистите и пересоберите: `rm -rf backend/dist backend/build frontend/dist frontend/build && ./build.sh`

---

**Версия:** 1.0  
**Дата:** 2025
