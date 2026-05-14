#!/usr/bin/env python3
"""Small Telegram bridge for local Codex/project commands.

The bridge intentionally keeps the command surface narrow. It polls Telegram
updates, accepts messages from a single chat id, and replies with safe project
state or branch-management results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
MAX_TELEGRAM_TEXT = 3900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Telegram and handle local project commands.")
    parser.add_argument("--repo", default=os.getcwd(), help="Repository directory to manage.")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Telegram bot token.")
    parser.add_argument(
        "--chat-id",
        default=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
        help="Allowed Telegram chat id.",
    )
    parser.add_argument("--poll-timeout", type=int, default=30, help="Telegram long-poll timeout.")
    parser.add_argument("--skip-existing", action="store_true", help="Ignore messages that arrived before startup.")
    parser.add_argument("--startup-message", action="store_true", help="Send a startup message to the allowed chat.")
    return parser.parse_args()


def run(args: list[str], *, cwd: Path, timeout: float = 20.0) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return 124, f"Command timed out.\n{output}".strip()
    return completed.returncode, completed.stdout.strip()


def telegram_request(token: str, method: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    context = ssl_context()
    try:
        with urllib.request.urlopen(
            request,
            timeout=max(int(payload.get("timeout", 30)) + 10, 20),
            context=context,
        ) as response_file:
            response = json.loads(response_file.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram request failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Telegram request failed: {exc.reason}") from exc
    if not response.get("ok"):
        raise RuntimeError(json.dumps(response, ensure_ascii=False))
    return response


def ssl_context():
    try:
        import certifi
    except Exception:
        return None

    import ssl

    return ssl.create_default_context(cafile=certifi.where())


def send_message(token: str, chat_id: str, text: str) -> None:
    for part in split_message(text):
        telegram_request(token, "sendMessage", {"chat_id": chat_id, "text": part})


def split_message(text: str) -> list[str]:
    text = text.strip() or "(empty response)"
    if len(text) <= MAX_TELEGRAM_TEXT:
        return [text]
    return [text[i : i + MAX_TELEGRAM_TEXT] for i in range(0, len(text), MAX_TELEGRAM_TEXT)]


def git_output(repo: Path, args: list[str], timeout: float = 20.0) -> str:
    code, output = run(["git", *args], cwd=repo, timeout=timeout)
    if code != 0:
        return output or f"git {' '.join(args)} failed with exit code {code}"
    return output


def current_branch(repo: Path) -> str:
    branch = git_output(repo, ["branch", "--show-current"])
    return branch or "(detached HEAD)"


def is_dirty(repo: Path) -> bool:
    return bool(git_output(repo, ["status", "--porcelain"]))


def validate_branch(raw: str) -> tuple[bool, str]:
    branch = raw.strip()
    if not branch:
        return False, "Укажи имя ветки."
    if branch.startswith("-") or ".." in branch or not BRANCH_RE.fullmatch(branch):
        return False, "Небезопасное имя ветки. Используй буквы, цифры, `-`, `_`, `.`, `/`."
    return True, branch


def command_help() -> str:
    return "\n".join(
        [
            "Команды:",
            "/ping - проверить связь",
            "/branch - текущая ветка",
            "/status - git status",
            "/switch <branch> - переключиться на существующую ветку",
            "/switch <branch> --allow-dirty - переключиться даже с незакоммиченными изменениями",
            "/newbranch <name> - создать ветку; без `/` добавится префикс `codex/`",
            "/ask <вопрос> - спросить Codex в read-only режиме",
            "/help - показать команды",
        ]
    )


def handle_message(repo: Path, text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return "Пустое сообщение."
    if lowered in {"/help", "help", "помощь"}:
        return command_help()
    if lowered == "/ping":
        return "pong"
    if lowered == "/branch" or "какой ветк" in lowered or "какая ветк" in lowered:
        return f"Текущая ветка: {current_branch(repo)}"
    if lowered == "/status" or "git status" in lowered or "статус" in lowered:
        status = git_output(repo, ["status", "--short", "--branch"])
        return status or "Рабочее дерево чистое."

    if stripped.startswith("/switch "):
        return handle_switch(repo, stripped.removeprefix("/switch ").strip())
    if stripped.startswith("/newbranch "):
        return handle_new_branch(repo, stripped.removeprefix("/newbranch ").strip())
    if stripped.startswith("/ask "):
        return handle_codex_ask(repo, stripped.removeprefix("/ask ").strip())

    return "Сообщение получил. Для управления проектом используй /help."


def handle_switch(repo: Path, raw: str) -> str:
    parts = shlex.split(raw)
    allow_dirty = "--allow-dirty" in parts
    branch_parts = [part for part in parts if part != "--allow-dirty"]
    if len(branch_parts) != 1:
        return "Формат: /switch <branch> [--allow-dirty]"
    ok, branch = validate_branch(branch_parts[0])
    if not ok:
        return branch
    if is_dirty(repo) and not allow_dirty:
        return (
            "Есть незакоммиченные изменения. Переключение остановлено.\n"
            "Посмотри /status или повтори: /switch "
            f"{branch} --allow-dirty"
        )
    code, output = run(["git", "switch", branch], cwd=repo, timeout=30.0)
    if code != 0:
        return output or f"Не удалось переключиться на {branch}."
    return f"Переключился на ветку: {current_branch(repo)}"


def handle_new_branch(repo: Path, raw: str) -> str:
    parts = shlex.split(raw)
    if len(parts) != 1:
        return "Формат: /newbranch <name>"
    name = parts[0]
    branch = name if "/" in name else f"codex/{name}"
    ok, branch_or_error = validate_branch(branch)
    if not ok:
        return branch_or_error
    code, output = run(["git", "switch", "-c", branch_or_error], cwd=repo, timeout=30.0)
    if code != 0:
        return output or f"Не удалось создать ветку {branch_or_error}."
    return f"Создал и переключился на ветку: {current_branch(repo)}"


def handle_codex_ask(repo: Path, prompt: str) -> str:
    if not prompt:
        return "Формат: /ask <вопрос>"
    instruction = (
        "Ответь кратко на русском. Работай только в read-only режиме. "
        "Не изменяй файлы и не запускай долгие команды.\n\n"
        f"Вопрос пользователя: {prompt}"
    )
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="telegram-codex-", suffix=".txt", delete=False) as file:
            output_path = Path(file.name)
        code, output = run(
            [
                "codex",
                "exec",
                "-C",
                str(repo),
                "-s",
                "read-only",
                "--output-last-message",
                str(output_path),
                instruction,
            ],
            cwd=repo,
            timeout=180.0,
        )
        if code != 0:
            return output or "Codex exec завершился с ошибкой."
        if output_path.exists():
            final = output_path.read_text(encoding="utf-8").strip()
            if final:
                return final
        return output or "Codex не вернул текстовый ответ."
    finally:
        if output_path and output_path.exists():
            output_path.unlink()


def extract_message(update: dict[str, object]) -> tuple[str, str] | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_id = chat.get("id")
    text = message.get("text")
    if chat_id is None or not isinstance(text, str):
        return None
    return str(chat_id), text


def initial_offset(token: str) -> int | None:
    response = telegram_request(token, "getUpdates", {"limit": 100, "timeout": 1})
    updates = response.get("result", [])
    if not isinstance(updates, list) or not updates:
        return None
    ids = [update.get("update_id") for update in updates if isinstance(update, dict)]
    numeric_ids = [int(update_id) for update_id in ids if isinstance(update_id, int)]
    if not numeric_ids:
        return None
    return max(numeric_ids) + 1


def poll_loop(repo: Path, token: str, chat_id: str, poll_timeout: int, offset: int | None) -> None:
    while True:
        payload: dict[str, object] = {
            "limit": 20,
            "timeout": poll_timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        try:
            response = telegram_request(token, "getUpdates", payload)
            updates = response.get("result", [])
            if not isinstance(updates, list):
                continue
            for update in updates:
                if not isinstance(update, dict):
                    continue
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                extracted = extract_message(update)
                if not extracted:
                    continue
                incoming_chat_id, text = extracted
                if incoming_chat_id != chat_id:
                    send_message(token, incoming_chat_id, "Этот бот принимает команды только из разрешённого чата.")
                    continue
                reply = handle_message(repo, text)
                send_message(token, chat_id, reply)
        except Exception as exc:  # noqa: BLE001 - keep the bridge alive and report the failure.
            print(f"bridge error: {exc}", file=sys.stderr, flush=True)
            time.sleep(5)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if not args.token:
        print("Missing TELEGRAM_BOT_TOKEN or --token.", file=sys.stderr)
        return 2
    if not args.chat_id:
        print("Missing TELEGRAM_CHAT_ID/TELEGRAM_ALLOWED_CHAT_ID or --chat-id.", file=sys.stderr)
        return 2
    if not (repo / ".git").exists():
        print(f"{repo} is not a git repository.", file=sys.stderr)
        return 2

    offset = initial_offset(args.token) if args.skip_existing else None
    if args.startup_message:
        send_message(
            args.token,
            str(args.chat_id),
            f"Telegram-Codex bridge запущен.\nТекущая ветка: {current_branch(repo)}\n/help - команды",
        )
    poll_loop(repo, args.token, str(args.chat_id), args.poll_timeout, offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
