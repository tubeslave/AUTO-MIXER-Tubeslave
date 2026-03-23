#!/usr/bin/env bash
# Просмотр логов применения EQ после Auto-EQ (консоль + файл logs/automixer-backend.log).
# Запуск: из корня репозитория: ./scripts/tail_automixer_eq_logs.sh
# Или: bash scripts/tail_automixer_eq_logs.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/automixer-backend.log"
mkdir -p "$ROOT/logs"
touch "$LOG"

echo "Файл: $LOG"
echo "Фильтр: Auto-EQ apply | Ableton EQ OSC | WING EQ OSC | Applied EQ | Error applying EQ"
echo "Полный лог без фильтра: tail -f \"$LOG\""
echo "---"

if command -v rg >/dev/null 2>&1; then
  tail -f "$LOG" | rg --line-buffered \
    'Auto-EQ apply|Ableton EQ OSC|WING EQ OSC|Applied EQ|Error applying EQ'
else
  tail -f "$LOG" | grep --line-buffered -E \
    'Auto-EQ apply|Ableton EQ OSC|WING EQ OSC|Applied EQ|Error applying EQ'
fi
