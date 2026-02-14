#!/bin/bash
# Скрипт для выключения всех модулей и установки фейдеров на 0дБ

cd "$(dirname "$0")"

# Загрузить IP из конфига по умолчанию
DEFAULT_IP="192.168.1.102"
DEFAULT_PORT="2223"

if [ -f "../config/default_config.json" ]; then
    DEFAULT_IP=$(python3 -c "import json; f=open('../config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('default_ip', '$DEFAULT_IP'))" 2>/dev/null || echo "$DEFAULT_IP")
    DEFAULT_PORT=$(python3 -c "import json; f=open('../config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('receive_port', '$DEFAULT_PORT'))" 2>/dev/null || echo "$DEFAULT_PORT")
fi

# Использовать переданные аргументы или значения по умолчанию
IP="${1:-$DEFAULT_IP}"
PORT="${2:-$DEFAULT_PORT}"

echo "=========================================="
echo "Выключение модулей и установка фейдеров"
echo "=========================================="
echo "IP адрес: $IP"
echo "Порт: $PORT"
echo ""
echo "ВНИМАНИЕ: Это выключит все модули и установит фейдеры на 0дБ на ВСЕХ 40 каналах!"
echo "  - Все модули ВЫКЛЮЧЕНЫ (EQ, PreEQ, Gate, Dynamics, Filters, Inserts)"
echo "  - Все фейдеры = 0дБ"
echo ""
read -p "Введите 'ДА' для продолжения: " confirm

if [ "$confirm" != "ДА" ]; then
    echo "Операция отменена"
    exit 1
fi

echo ""
echo "Запуск скрипта..."
python3 disable_modules_set_faders.py "$IP" "$PORT"
