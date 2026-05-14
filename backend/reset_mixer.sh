#!/bin/bash
# Скрипт для сброса настроек пульта (микшера)

cd "$(dirname "$0")"

DEFAULT_IP="192.168.1.100"
DEFAULT_PORT="2223"

if [ -f "../config/default_config.json" ]; then
    DEFAULT_IP=$(python3 -c "import json; f=open('../config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('default_ip', '$DEFAULT_IP'))" 2>/dev/null || echo "$DEFAULT_IP")
    DEFAULT_PORT=$(python3 -c "import json; f=open('../config/default_config.json'); c=json.load(f); print(c.get('wing', {}).get('receive_port', '$DEFAULT_PORT'))" 2>/dev/null || echo "$DEFAULT_PORT")
fi

IP="${1:-$DEFAULT_IP}"
PORT="${2:-$DEFAULT_PORT}"
CONFIRM=0
ROLLBACK=""
DRY_RUN=0

if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --confirm-maintenance)
            CONFIRM=1
            shift
            ;;
        --rollback-snapshot)
            if [ -z "$2" ]; then
                echo "Ошибка: --rollback-snapshot требует указания PATH"
                exit 1
            fi
            ROLLBACK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            echo "Использование: ./reset_mixer.sh [IP] [PORT] --confirm-maintenance --rollback-snapshot PATH [--dry-run]"
            echo "  IP    - IP пульта (по умолчанию из config/default_config.json или 192.168.1.100)"
            echo "  PORT  - OSC порт (по умолчанию из config/default_config.json или 2223)"
            echo ""
            echo "  --confirm-maintenance   Подтверждение maintenance"
            echo "  --rollback-snapshot PATH  Обязательный путь к rollback snapshot"
            echo "  --dry-run               Не выполнять команды, только отчёт"
            exit 0
            ;;
        *)
            echo "Неизвестный параметр: $1"
            echo "  --confirm-maintenance --rollback-snapshot PATH [--dry-run]"
            exit 1
            ;;
    esac
done

if [ "$CONFIRM" -ne 1 ]; then
    echo "Операция заблокирована: нужен флаг --confirm-maintenance"
    echo "  Пример: ./reset_mixer.sh 192.168.1.102 2223 --confirm-maintenance --rollback-snapshot backup.snap"
    exit 1
fi

if [ -z "$ROLLBACK" ]; then
    echo "Операция заблокирована: нужен --rollback-snapshot PATH"
    exit 1
fi

echo "=========================================="
echo "Сброс настроек пульта (микшера)"
echo "=========================================="
echo "IP адрес: $IP"
echo "Порт: $PORT"
echo "Rollback snapshot: $ROLLBACK"
echo ""

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run: реальные команды не выполняются"
    exit 0
fi

python3 reset_modules_trim_faders.py "$IP" "$PORT" --confirm-maintenance --rollback-snapshot "$ROLLBACK"
