#!/usr/bin/env python3
"""Inspect recent Telegram bot updates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


API_BASE = "https://api.telegram.org"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch recent updates for a Telegram bot."
    )
    parser.add_argument("--token", help="Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN.")
    parser.add_argument(
        "--offset",
        type=int,
        help="First update id to return.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of updates to request, between 1 and 100. Default: 10.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Long-poll timeout in seconds. Default: 0.",
    )
    parser.add_argument(
        "--allowed-updates",
        nargs="*",
        help="Optional list of update types, for example: message callback_query.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of a condensed summary.",
    )
    return parser


def fail(message: str, *, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


def summarize_update(update: dict[str, object]) -> str:
    for field in (
        "message",
        "edited_message",
        "channel_post",
        "edited_channel_post",
        "business_message",
    ):
        payload = update.get(field)
        if isinstance(payload, dict):
            chat = payload.get("chat") or {}
            from_user = payload.get("from") or {}
            text = payload.get("text") or payload.get("caption") or ""
            username = from_user.get("username")
            sender = f"@{username}" if username else from_user.get("first_name") or "unknown"
            title = chat.get("title") or chat.get("username") or chat.get("first_name") or "unknown"
            return (
                f"update_id={update.get('update_id')} type={field} "
                f"chat_id={chat.get('id')} chat={title!r} sender={sender!r} text={text!r}"
            )

    callback_query = update.get("callback_query")
    if isinstance(callback_query, dict):
        from_user = callback_query.get("from") or {}
        data = callback_query.get("data") or ""
        username = from_user.get("username")
        sender = f"@{username}" if username else from_user.get("first_name") or "unknown"
        return (
            f"update_id={update.get('update_id')} type=callback_query "
            f"sender={sender!r} data={data!r}"
        )

    return f"update_id={update.get('update_id')} keys={sorted(update.keys())}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return fail("Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN or pass --token.")

    limit = max(1, min(args.limit, 100))
    query: dict[str, object] = {"limit": limit, "timeout": args.timeout}
    if args.offset is not None:
        query["offset"] = args.offset
    if args.allowed_updates is not None:
        query["allowed_updates"] = json.dumps(args.allowed_updates)

    url = f"{API_BASE}/bot{token}/getUpdates?{urllib.parse.urlencode(query)}"

    try:
        with urllib.request.urlopen(url, timeout=max(args.timeout + 5, 10)) as response:
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

    updates = body.get("result", [])
    if not updates:
        print("No updates returned.")
        return 0

    for update in updates:
        if isinstance(update, dict):
            print(summarize_update(update))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
