#!/usr/bin/env python3
"""Send a message using the Telegram Bot API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


API_BASE = "https://api.telegram.org"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a Telegram message with a bot token."
    )
    parser.add_argument("--token", help="Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN.")
    parser.add_argument(
        "--chat-id",
        help="Target chat ID. Defaults to TELEGRAM_CHAT_ID.",
    )
    parser.add_argument(
        "--message",
        help="Text to send. Use --stdin to read the message body from standard input.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the message body from standard input.",
    )
    parser.add_argument(
        "--parse-mode",
        choices=("HTML", "MarkdownV2"),
        help="Optional Telegram parse mode.",
    )
    parser.add_argument(
        "--thread-id",
        type=int,
        help="Optional forum topic thread id (message_thread_id).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Send without sound notifications.",
    )
    parser.add_argument(
        "--protect-content",
        action="store_true",
        help="Ask Telegram to protect the message from forwarding and saving.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full Telegram API response as JSON.",
    )
    return parser


def fail(message: str, *, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


def load_message(args: argparse.Namespace) -> str:
    if args.stdin:
        message = sys.stdin.read()
    else:
        message = args.message or ""
    return message.strip()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return fail("Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN or pass --token.")

    chat_id = args.chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not chat_id:
        return fail("Missing chat id. Set TELEGRAM_CHAT_ID or pass --chat-id.")

    message = load_message(args)
    if not message:
        return fail("Message body is empty. Pass --message or use --stdin.")

    payload: dict[str, object] = {
        "chat_id": chat_id,
        "text": message,
    }
    if args.parse_mode:
        payload["parse_mode"] = args.parse_mode
    if args.thread_id is not None:
        payload["message_thread_id"] = args.thread_id
    if args.silent:
        payload["disable_notification"] = True
    if args.protect_content:
        payload["protect_content"] = True

    request = urllib.request.Request(
        f"{API_BASE}/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return fail(f"Telegram API request failed ({exc.code}): {error_body}")
    except urllib.error.URLError as exc:
        return fail(f"Telegram API request failed: {exc.reason}")

    if not body.get("ok"):
        return fail(f"Telegram API error: {json.dumps(body, ensure_ascii=False)}")

    if args.json:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    result = body.get("result", {})
    message_id = result.get("message_id")
    result_chat_id = (result.get("chat") or {}).get("id", chat_id)
    print(f"Sent message {message_id} to chat {result_chat_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
