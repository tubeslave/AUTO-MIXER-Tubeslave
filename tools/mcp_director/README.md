# Automixer MCP Director

This MCP server exposes tools for:

- `create_task` — создать задачу в `.paperclip/tasks/`
- `run_agent` — подготовить задачу для агента (DSP / ML / Refactor / Analyzer)
- `run_full_review` — запустить полный multi-agent обзор
- `git_report` — сгенерировать отчёт по git-статусу
- `telegram_notify` — отправить уведомление в Telegram

## Install

```bash
cd tools/mcp_director
npm install
npm start
```

## Environment

```bash
export AUTOMIXER_PROJECT_DIR="/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## Codex MCP config

Add a local MCP server command:

```json
{
  "mcpServers": {
    "automixer-paperclip-director": {
      "command": "node",
      "args": ["/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/tools/mcp_director/server.js"],
      "env": {
        "AUTOMIXER_PROJECT_DIR": "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main"
      }
    }
  }
}
```

## Architecture

```
iPhone GPT Chat / Codex
        ↓
       MCP
        ↓
  Paperclip Director  ←  этот сервер
        ↓
 ┌───────────────┐
 │ Codex Agents  │
 │ DSP Agent     │
 │ ML Agent      │
 │ Refactor      │
 │ Analyzer      │
 └───────────────┘
        ↓
 Git Branches / Reports
        ↓
  Telegram / GPT Chat
```
