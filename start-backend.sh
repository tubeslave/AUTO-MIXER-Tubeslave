#!/bin/bash
# Запуск backend (WebSocket сервер на порту 8765)
cd "$(dirname "$0")/backend" && python server.py
