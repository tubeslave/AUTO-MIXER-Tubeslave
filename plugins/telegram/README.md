# Telegram Plugin

This repo-local Codex plugin gives you a lightweight Telegram Bot API workflow:

- send a message to a chat from the terminal
- inspect recent updates to discover the right `chat_id`
- expose those helpers through a plugin skill

## What is included

- `.codex-plugin/plugin.json` with Telegram-facing plugin metadata
- `skills/telegram/SKILL.md` with usage guidance for Codex
- `scripts/send_message.py` to send Telegram messages
- `scripts/get_updates.py` to inspect recent updates and find chat IDs
- `.agents/plugins/marketplace.json` entry so the plugin can be discovered locally

## Setup

1. Create a bot in Telegram with `@BotFather`.
2. Copy the bot token.
3. Export the token in your shell:

```bash
export TELEGRAM_BOT_TOKEN="123456:example-token"
```

4. Send your bot a message in Telegram, or add it to a group and post a message there.
5. Inspect updates to find the target chat:

```bash
python3 ./scripts/get_updates.py --limit 5
```

6. Set a default chat for future sends:

```bash
export TELEGRAM_CHAT_ID="123456789"
```

## Send a test message

```bash
python3 ./scripts/send_message.py --message "AUTO MIXER test message"
```

You can also override the default chat:

```bash
python3 ./scripts/send_message.py --chat-id "-1001234567890" --message "Mix is ready"
```

## Notes

- `getUpdates` will not work while an outgoing webhook is active for the bot.
- The helper scripts use the official Telegram Bot API over HTTPS.
- The plugin manifest still contains a few `[TODO: ...]` fields for author, license, and repository metadata.
