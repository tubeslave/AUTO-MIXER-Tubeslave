---
name: telegram
description: Send Telegram bot messages and inspect recent updates. Use when the user wants a Telegram notification, a quick bot test, or help finding a chat ID for the Telegram Bot API.
---

# Telegram

Use this skill when you need to send a quick Telegram message from the local workspace or inspect recent bot updates.

## Prerequisites

- `TELEGRAM_BOT_TOKEN` must be set in the environment.
- `TELEGRAM_CHAT_ID` is optional, but recommended for repeated sends.

## Quick start

Send a message to the default chat:

```bash
python3 ../../scripts/send_message.py --message "Build finished successfully"
```

Inspect recent updates to confirm the correct `chat_id`:

```bash
python3 ../../scripts/get_updates.py --limit 10
```

## When to use which script

- Use `send_message.py` for alerts, summaries, and manual bot tests.
- Use `get_updates.py` when the user does not know the chat ID yet or wants to inspect incoming updates before wiring automation.

## Expected behavior

- Prefer `TELEGRAM_CHAT_ID` from the environment unless the user specifies another chat explicitly.
- Keep messages short and clear by default.
- If Telegram returns an API error, surface the response text instead of hiding it.
