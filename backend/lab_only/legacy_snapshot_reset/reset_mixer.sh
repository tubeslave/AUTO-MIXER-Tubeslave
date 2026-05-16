#!/bin/bash
# Legacy lab-only wrapper for full mixer reset scripts.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Загрузить IP из конфига по умолчанию
DEFAULT_IP="192.168.1.100"
DEFAULT_PORT="2223"

if [ -f "$REPO_ROOT/config/default_config.json" ]; then
    DEFAULT_IP=$(python3 -c "import json; f=open('$REPO_ROOT/config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('default_ip', '$DEFAULT_IP'))" 2>/dev/null || echo "$DEFAULT_IP")
    DEFAULT_PORT=$(python3 -c "import json; f=open('$REPO_ROOT/config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('receive_port', '$DEFAULT_PORT'))" 2>/dev/null || echo "$DEFAULT_PORT")
fi

# Использовать переданные аргументы или значения по умолчанию
IP="${1:-$DEFAULT_IP}"
PORT="${2:-$DEFAULT_PORT}"

echo "=========================================="
echo "Сброс настроек пульта (микшера)"
echo "=========================================="
echo "IP адрес: $IP"
echo "Порт: $PORT"
echo ""
echo "ВНИМАНИЕ: Это сбросит настройки на ВСЕХ 40 каналах!"
echo "  - Все модули ВЫКЛЮЧЕНЫ"
echo "  - Все параметры сброшены к значениям по умолчанию"
echo "  - Trim = 0дБ, Fader = 0дБ"
echo ""
read -p "Введите 'ДА' для продолжения: " confirm

if [ "$confirm" != "ДА" ]; then
    echo "Операция отменена"
    exit 1
fi

echo ""
echo "Запуск скрипта сброса..."
cd "$REPO_ROOT" || exit 1
python3 -m backend.lab_only.legacy_snapshot_reset.reset_modules_trim_faders "$IP" "$PORT"
