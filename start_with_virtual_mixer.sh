#!/bin/bash
# AUTO-MIXER Tubeslave — запуск с виртуальным микшером для тестов
# Запускает: Virtual Mixer (OSC 2222) → Backend (WS 8765) → Frontend (3000)

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   AUTO-MIXER Tubeslave — тест с виртуальным микшером           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Use backend venv for Python (has python-osc, websockets)
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROOT}/backend/venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}Ошибка: venv не найден. Запустите: cd backend && pip install -r requirements.txt${NC}"
    exit 1
fi

# 1. Virtual Mixer (OSC 2222, WING? handshake)
echo -e "${GREEN}1. Запуск виртуального микшера (OSC port 2222)...${NC}"
cd virtual_mixer
$PYTHON test_integration.py &
VM_PID=$!
cd ..
sleep 2
if ! kill -0 $VM_PID 2>/dev/null; then
    echo -e "${RED}Ошибка: виртуальный микшер не запустился${NC}"
    exit 1
fi
echo -e "${GREEN}   ✓ Virtual Mixer запущен (PID $VM_PID)${NC}"
echo ""

# 2. Backend
echo -e "${GREEN}2. Запуск backend (WebSocket 8765)...${NC}"
cd backend
$PYTHON server.py &
BACKEND_PID=$!
cd ..
sleep 2
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Ошибка: backend не запустился${NC}"
    kill $VM_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}   ✓ Backend запущен (PID $BACKEND_PID)${NC}"
echo ""

# 3. Frontend
echo -e "${GREEN}3. Запуск frontend (React 3000)...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo "   Установка npm зависимостей..."
    npm install
fi
npm start &
FRONTEND_PID=$!
cd ..
sleep 3
echo -e "${GREEN}   ✓ Frontend запускается (PID $FRONTEND_PID)${NC}"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Всё запущено!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  🌐 Frontend:     http://localhost:3000"
echo "  🔌 WebSocket:    ws://localhost:8765"
echo "  🎛  Virtual Mixer: localhost:2222 (OSC)"
echo ""
echo -e "${YELLOW}Подключение к микшеру во Frontend:${NC}"
echo "  • Wing IP:  127.0.0.1 (или localhost)"
echo "  • OSC Port: 2222"
echo ""
echo "  Нажмите Ctrl+C для остановки всех процессов"
echo ""

cleanup() {
    echo ""
    echo -e "${YELLOW}Остановка...${NC}"
    kill $VM_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Готово.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for any process to exit
wait
