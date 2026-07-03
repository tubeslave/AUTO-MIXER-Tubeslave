#!/usr/bin/env python3
"""Fetch a sanitized ChatGPT Pulse snapshot for Automixer review automation.

The script uses the local ChatGPT browser session through the existing
`chatgpt-list-chats` helper. It deliberately writes only card text and short
conversation excerpts. It never prints cookies, access tokens, session payloads,
signed image URLs, or account identity fields.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHATGPT_BASE = "https://chatgpt.com"
DEFAULT_HELPER_DIR = (
    Path.home() / ".codex" / "skills" / "chatgpt-list-chats" / "scripts"
)
MAX_TEXT_CHARS = 5000
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{16,}\b"),
)
GENERIC_SECRET_KEY_PATTERN = (
    r"(?:[A-Z0-9_]*(?:API[_-]?KEY|PRIVATE[_-]?KEY)[A-Z0-9_]*|"
    r"(?:[A-Z0-9_]+[_-])?(?:PASSWORD|SECRET|TOKEN)(?:[_-][A-Z0-9_]+)*|"
    r"[A-Z0-9_]*(?:ACCESS|REFRESH)[_-]?TOKEN[A-Z0-9_]*)"
)
GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(^|[^\w.-])((?:export\s+)?[\"']?({GENERIC_SECRET_KEY_PATTERN})[\"']?\s*[:=]\s*)"
    r"(?:([\"'])([^\"'\r\n]*)([\"'])|([^\s#,}]+))",
    re.IGNORECASE | re.MULTILINE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_chatgpt_helper(helper_dir: Path) -> Any:
    helper_dir = helper_dir.expanduser().resolve()
    if not helper_dir.exists():
        raise SystemExit(f"ChatGPT helper is missing: {helper_dir}")
    sys.path.insert(0, str(helper_dir))
    try:
        import list_chatgpt_chats as helper  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(f"Failed to import ChatGPT helper from {helper_dir}: {exc}") from exc
    return helper


def get_json(session: Any, url: str, headers: dict[str, str] | None = None, tries: int = 4) -> dict[str, Any]:
    last_status = "not_started"
    last_head = ""
    for attempt in range(tries):
        response = session.get(url, timeout=45, headers=headers or {})
        last_status = f"{response.status_code} {response.headers.get('content-type')}"
        text = response.text or ""
        last_head = text[:120].replace("\n", " ")
        if response.status_code == 200 and text.strip().startswith("{"):
            return response.json()
        if response.status_code == 404:
            raise RuntimeError(f"Not found: {url}")
        time.sleep(min(2.0, 0.5 + attempt * 0.5))
    raise RuntimeError(f"Could not fetch JSON from {url}: {last_status} {last_head!r}")


def get_first_json(
    session: Any,
    urls: list[str],
    headers: dict[str, str] | None = None,
    tries: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    for url in urls:
        try:
            return get_json(session, url, headers, tries=tries)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_access_token(session: Any) -> str:
    payload = get_json(
        session,
        f"{CHATGPT_BASE}/api/auth/session",
        {"Accept": "application/json", "Referer": f"{CHATGPT_BASE}/pulse"},
    )
    token = payload.get("accessToken")
    if not isinstance(token, str) or not token:
        raise RuntimeError("ChatGPT access token is unavailable from the local session")
    return token


def text_from_message_parts(parts: list[Any]) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, str):
            values.append(part)
        elif isinstance(part, dict):
            for key in ("text", "name", "summary"):
                value = part.get(key)
                if isinstance(value, str):
                    values.append(value)
    return "\n".join(value.strip() for value in values if value and value.strip()).strip()


def _redact_secret_assignment(match: re.Match[str]) -> str:
    leading = match.group(1)
    prefix = match.group(2)
    quote = match.group(4)
    if quote:
        return f"{leading}{prefix}{quote}<redacted>{match.group(6)}"
    return f"{leading}{prefix}<redacted>"


def redact_secret_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return GENERIC_SECRET_ASSIGNMENT_RE.sub(_redact_secret_assignment, redacted)


def sanitize_text(value: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = value.replace("\r", "")
    text = re.sub(r"https://sdmntpr[^)\s]+", "[redacted-image-url]", text)
    text = re.sub(r"```.*?```", "[code block omitted]", text, flags=re.S)
    text = redact_secret_text(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def extract_conversation_text(payload: dict[str, Any]) -> dict[str, str]:
    mapping = payload.get("mapping") or {}
    messages: list[tuple[str, str]] = []
    if isinstance(mapping, dict):
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            role = ((message.get("author") or {}).get("role") or "unknown").strip()
            content = message.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            text = text_from_message_parts(parts)
            if text:
                messages.append((role, text))

    first_user = next((text for role, text in messages if role == "user"), "")
    assistant_messages = [text for role, text in messages if role == "assistant"]
    final_assistant = assistant_messages[-1] if assistant_messages else ""
    return {
        "user_excerpt": sanitize_text(first_user, max_chars=1200),
        "assistant_excerpt": sanitize_text(final_assistant, max_chars=MAX_TEXT_CHARS),
    }


def fetch_pulse(helper: Any, browser: str) -> dict[str, Any]:
    bundles = helper.load_cookie_bundles(browser)
    if not bundles:
        raise RuntimeError(f"No ChatGPT cookies found for browser={browser}")

    session = helper.build_http_session(bundles[0])
    token = fetch_access_token(session)
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {token}",
        "Origin": CHATGPT_BASE,
        "Referer": f"{CHATGPT_BASE}/pulse",
    }

    entrypoint = get_json(session, f"{CHATGPT_BASE}/backend-api/feed/entrypoint", headers)
    feed_id = entrypoint.get("id")
    mixed_urls = [f"{CHATGPT_BASE}/backend-api/feed/mixed"]
    if isinstance(feed_id, str) and feed_id:
        mixed_urls.insert(0, f"{CHATGPT_BASE}/backend-api/feed/mixed/{feed_id}")
    try:
        feed = get_first_json(session, mixed_urls, headers, tries=3)
    except Exception as exc:
        preview_items = entrypoint.get("preview_items")
        if not isinstance(preview_items, list):
            raise
        feed = {
            "id": feed_id,
            "items": [
                {
                    "content_type": "card",
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "preview_markdown_text": item.get("preview"),
                }
                for item in preview_items
                if isinstance(item, dict)
            ],
            "fallback_reason": str(exc),
        }
    raw_items = feed.get("items") if isinstance(feed.get("items"), list) else []
    cards: list[dict[str, Any]] = []

    for item in raw_items:
        if not isinstance(item, dict) or item.get("content_type") != "card":
            continue
        card: dict[str, Any] = {
            "id": item.get("id"),
            "title": item.get("title"),
            "conversation_id": item.get("conversation_id"),
            "preview": sanitize_text(str(item.get("preview_markdown_text") or ""), max_chars=1200),
        }
        conversation_id = card.get("conversation_id")
        if isinstance(conversation_id, str) and conversation_id:
            try:
                conversation = get_json(
                    session,
                    f"{CHATGPT_BASE}/backend-api/conversation/{conversation_id}",
                    headers,
                    tries=2,
                )
                card.update(extract_conversation_text(conversation))
            except Exception as exc:
                card["conversation_fetch_error"] = str(exc)
        cards.append(card)

    return {
        "schema_version": "0.1",
        "generated_at": utc_now(),
        "source": "chatgpt_pulse",
        "source_url": f"{CHATGPT_BASE}/pulse",
        "feed_id": feed.get("id") or entrypoint.get("id"),
        "title": entrypoint.get("title"),
        "caption": entrypoint.get("caption"),
        "card_count": len(cards),
        "cards": cards,
    }


def write_markdown(snapshot: dict[str, Any], path: Path) -> None:
    lines = [
        f"# ChatGPT Pulse Snapshot {snapshot.get('generated_at')}",
        "",
        f"- Source: {snapshot.get('source_url')}",
        f"- Title: {snapshot.get('title') or 'n/a'}",
        f"- Caption: {snapshot.get('caption') or 'n/a'}",
        f"- Cards: {snapshot.get('card_count')}",
        "",
    ]
    for index, card in enumerate(snapshot.get("cards") or [], start=1):
        lines.extend(
            [
                f"## {index}. {card.get('title') or 'Untitled'}",
                "",
                f"Preview: {card.get('preview') or 'n/a'}",
                "",
            ]
        )
        excerpt = card.get("assistant_excerpt")
        if excerpt:
            lines.extend(["Assistant excerpt:", "", str(excerpt), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="chrome", choices=("chrome", "brave", "edge", "chromium"))
    parser.add_argument("--helper-dir", type=Path, default=DEFAULT_HELPER_DIR)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    helper = load_chatgpt_helper(args.helper_dir)
    snapshot = fetch_pulse(helper, args.browser)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        write_markdown(snapshot, args.output_md)

    print(
        json.dumps(
            {
                "generated_at": snapshot["generated_at"],
                "card_count": snapshot["card_count"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md) if args.output_md else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
