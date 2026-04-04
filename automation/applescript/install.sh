#!/bin/bash
# Установка OpenClaw AppleScript автоматизации

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SCRIPTS_DIR="$HOME/Library/Scripts"
OPENCLAW_BIN="$HOME/.nvm/versions/node/v24.11.1/bin/openclaw"

echo "🦞 OpenClaw AppleScript Automation Installer"
echo "============================================"

# Проверка OpenClaw
if [ ! -f "$OPENCLAW_BIN" ]; then
    echo "❌ OpenClaw не найден в $OPENCLAW_BIN"
    echo "Проверьте установку: npm install -g openclaw"
    exit 1
fi

echo "✅ OpenClaw найден"

# Создание директории скриптов
mkdir -p "$USER_SCRIPTS_DIR"

# Копирование скриптов
echo "📁 Копирование скриптов в $USER_SCRIPTS_DIR..."

# Конвертация .scpt файлов в текстовые для редактирования
for script in "$SCRIPT_DIR"/*.scpt; do
    if [ -f "$script" ]; then
        script_name=$(basename "$script")
        cp "$script" "$USER_SCRIPTS_DIR/$script_name"
        echo "  ✅ $script_name"
    fi
done

# Создание .app версии для лаунчера
echo ""
echo "🚀 Создание приложения OpenClaw Launcher..."

APP_DIR="$HOME/Applications/OpenClaw Launcher.app"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Info.plist
cat > "$APP_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>OpenClaw Launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.openclaw.launcher</string>
    <key>CFBundleName</key>
    <string>OpenClaw Launcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>OpenClaw Launcher needs to control other applications</string>
</dict>
</plist>
EOF

# Wrapper script
cat > "$APP_DIR/Contents/MacOS/OpenClaw Launcher" << EOF
#!/bin/bash
export PATH="\$HOME/.nvm/versions/node/v24.11.1/bin:\$PATH"
osascript "$USER_SCRIPTS_DIR/OpenClaw Launcher.scpt" &
EOF
chmod +x "$APP_DIR/Contents/MacOS/OpenClaw Launcher"

# Иконка (используем системную)
cp /System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/ExecutableBinaryIcon.icns "$APP_DIR/Contents/Resources/Applet.icns" 2>/dev/null || true

echo "✅ Приложение создано: $APP_DIR"

# Создание Automator Workflow
echo ""
echo "🤖 Создание Automator Quick Action..."

AUTOMATOR_DIR="$HOME/Library/Services"
mkdir -p "$AUTOMATOR_DIR"

# Quick Action для запуска OpenClaw
cat > "$AUTOMATOR_DIR/OpenClaw Quick Action.workflow" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSServices</key>
    <array>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>Run OpenClaw Command</string>
            </dict>
            <key>NSMessage</key>
            <string>runWorkflowAsService</string>
            <key>NSRequiredContext</key>
            <dict>
                <key>NSApplicationIdentifier</key>
                <string>com.apple.finder</string>
            </dict>
        </dict>
    </array>
</dict>
</plist>
EOF

echo "✅ Quick Action создана"

# Инструкции
echo ""
echo "============================================"
echo "✅ Установка завершена!"
echo ""
echo "📍 Локации:"
echo "  - Скрипты: $USER_SCRIPTS_DIR"
echo "  - Приложение: $APP_DIR"
echo ""
echo "🚀 Использование:"
echo "  1. Включите Script Menu:"
echo "     Script Editor → Settings → Show Script menu in menu bar"
echo ""
echo "  2. Или запустите приложение из Dock/Applications"
echo ""
echo "  3. Для горячих клавиш:"
echo "     System Settings → Keyboard → Keyboard Shortcuts → Services"
echo ""
echo "🔧 Настройка путей:"
echo "  Если путь к openclaw отличается, отредактируйте:"
echo "  $USER_SCRIPTS_DIR/OpenClaw Launcher.scpt"
echo ""
