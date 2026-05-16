#!/usr/bin/env python3
"""Schedule conservative recurring Paperclip -> ChromaDB sync runs."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.project_vector_index import automixer_chromadb_index as indexer  # noqa: E402

WATCHER_DIR = Path(__file__).resolve().parent
DEFAULT_INTERVAL_SECONDS = 900
DEFAULT_LOG_PATH = "logs/automixer_vector_paperclip_sync.log"
DEFAULT_STATE_PATH = ".paperclip/vector_paperclip_sync_state.json"
DEFAULT_LOCK_PATH = ".paperclip/vector_paperclip_sync.lock"


@dataclass
class Settings:
    project_root: Path
    persist_dir: str
    collection: str
    batch_size: int
    paperclip_url: str
    company_id: str | None
    token: str | None
    statuses: str
    limit: int
    timeout_seconds: float
    interval_seconds: int
    dry_run: bool
    log_path: Path
    state_path: Path
    lock_path: Path

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if environ is None else environ)
        load_env_file(env)
        project_root = Path(env.get("AUTOMIXER_VECTOR_SYNC_PROJECT_ROOT", REPO_ROOT)).expanduser().resolve()
        return cls(
            project_root=project_root,
            persist_dir=env.get("AUTOMIXER_VECTOR_SYNC_PERSIST_DIR", indexer.DEFAULT_PERSIST_DIR),
            collection=env.get("AUTOMIXER_VECTOR_SYNC_COLLECTION", indexer.DEFAULT_COLLECTION),
            batch_size=parse_int(
                env.get("AUTOMIXER_VECTOR_SYNC_BATCH_SIZE"),
                indexer.DEFAULT_BATCH_SIZE,
            ),
            paperclip_url=env.get("PAPERCLIP_API_URL", indexer.DEFAULT_PAPERCLIP_API_URL).rstrip("/"),
            company_id=(env.get("PAPERCLIP_COMPANY_ID") or "").strip() or None,
            token=(env.get("PAPERCLIP_TOKEN") or "").strip() or None,
            statuses=env.get("AUTOMIXER_VECTOR_SYNC_STATUSES", indexer.DEFAULT_PAPERCLIP_STATUSES),
            limit=parse_int(
                env.get("AUTOMIXER_VECTOR_SYNC_LIMIT"),
                indexer.DEFAULT_PAPERCLIP_LIMIT,
            ),
            timeout_seconds=parse_float(
                env.get("AUTOMIXER_VECTOR_SYNC_TIMEOUT_SECONDS"),
                indexer.DEFAULT_HTTP_TIMEOUT_SECONDS,
            ),
            interval_seconds=parse_int(
                env.get("AUTOMIXER_VECTOR_SYNC_INTERVAL_SECONDS"),
                DEFAULT_INTERVAL_SECONDS,
            ),
            dry_run=parse_bool(env.get("AUTOMIXER_VECTOR_SYNC_DRY_RUN"), default=False),
            log_path=repo_path(project_root, env.get("AUTOMIXER_VECTOR_SYNC_LOG", DEFAULT_LOG_PATH)),
            state_path=repo_path(project_root, env.get("AUTOMIXER_VECTOR_SYNC_STATE", DEFAULT_STATE_PATH)),
            lock_path=repo_path(project_root, env.get("AUTOMIXER_VECTOR_SYNC_LOCK", DEFAULT_LOCK_PATH)),
        )


@dataclass
class SyncResult:
    result: str
    summary: dict[str, Any]
    state: dict[str, Any]
    skipped: bool = False


def repo_path(project_root: Path, raw: str) -> Path:
    return (project_root / raw).expanduser().resolve()


def parse_int(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def parse_float(raw: str | None, default: float) -> float:
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_env_file(env: dict[str, str]) -> None:
    env_path = WATCHER_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env.setdefault(key, value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(timestamp: datetime | None = None) -> str:
    current = timestamp or utc_now()
    return current.astimezone(timezone.utc).isoformat(timespec="seconds")


def sanitize_text(value: str, *, secrets: list[str] | None = None) -> str:
    sanitized = indexer.redact_secrets(value)
    for secret in secrets or []:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized


def sanitize_payload(payload: Any, *, secrets: list[str] | None = None) -> Any:
    if isinstance(payload, dict):
        return {str(key): sanitize_payload(value, secrets=secrets) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_payload(value, secrets=secrets) for value in payload]
    if isinstance(payload, str):
        return sanitize_text(payload, secrets=secrets)
    return payload


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "state_error": "invalid_json"}
    if not isinstance(data, dict):
        return {"version": 1, "state_error": "invalid_type"}
    data.setdefault("version", 1)
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_logging(settings: Settings) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)


@contextlib.contextmanager
def advisory_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "locked_at": iso_utc()}) + "\n")
        handle.flush()
        try:
            yield True
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_sync_once(
    settings: Settings,
    *,
    now: datetime | None = None,
) -> SyncResult:
    current_time = now or utc_now()
    configure_logging(settings)
    state = load_state(settings.state_path)
    started_at = iso_utc(current_time)
    secrets = [settings.token] if settings.token else []
    with advisory_lock(settings.lock_path) as locked:
        if not locked:
            summary = {
                "result": "skipped",
                "reason": "lock_busy",
                "started_at": started_at,
            }
            state.update(
                {
                    "last_attempt_at": started_at,
                    "last_result": "skipped",
                    "last_reason": "lock_busy",
                }
            )
            save_state(settings.state_path, state)
            logging.info("vector paperclip sync skipped: lock busy")
            return SyncResult("skipped", summary, state, skipped=True)

        state.update(
            {
                "last_attempt_at": started_at,
                "last_result": "running",
                "last_reason": None,
            }
        )
        save_state(settings.state_path, state)

        try:
            records, summary = indexer.build_paperclip_chunks(
                base_url=settings.paperclip_url,
                company_id=settings.company_id,
                token=settings.token,
                statuses=settings.statuses,
                limit=settings.limit,
                timeout_seconds=settings.timeout_seconds,
                indexed_at=started_at,
            )
            if not settings.dry_run and records:
                persist_dir = indexer.parse_path(settings.persist_dir, settings.project_root)
                collection = indexer.chroma_collection(persist_dir, settings.collection, reset=False)
                indexer.upsert_records(collection, records, batch_size=settings.batch_size)

            finished_at = iso_utc()
            summary_payload = sanitize_payload(
                {
                    **summary,
                    "result": "ok" if records else "empty",
                    "dry_run": settings.dry_run,
                    "collection": settings.collection,
                    "record_count": len(records),
                    "finished_at": finished_at,
                },
                secrets=secrets,
            )
            state.update(
                {
                    "last_finished_at": finished_at,
                    "last_result": summary_payload["result"],
                    "last_reason": None,
                    "last_summary": summary_payload,
                }
            )
            if records:
                state["last_success_at"] = finished_at
            save_state(settings.state_path, state)
            logging.info("vector paperclip sync finished: %s", json.dumps(summary_payload, ensure_ascii=False))
            return SyncResult(summary_payload["result"], summary_payload, state)
        except Exception as exc:
            finished_at = iso_utc()
            error_text = sanitize_text(str(exc), secrets=secrets)
            state.update(
                {
                    "last_finished_at": finished_at,
                    "last_result": "error",
                    "last_reason": error_text,
                }
            )
            save_state(settings.state_path, state)
            logging.error("vector paperclip sync failed: %s", error_text)
            summary = {
                "result": "error",
                "error": error_text,
                "finished_at": finished_at,
            }
            return SyncResult("error", summary, state)


def command_once(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    result = run_sync_once(settings)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return 0 if result.result in {"ok", "empty", "skipped"} else 1


def command_loop(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    while True:
        result = run_sync_once(settings)
        print(json.dumps(result.summary, indent=2, ensure_ascii=False))
        if args.max_runs and args.max_runs > 0:
            args.max_runs -= 1
            if args.max_runs <= 0:
                return 0 if result.result != "error" else 1
        time.sleep(max(settings.interval_seconds, 1))


def command_status(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    secrets = [settings.token] if settings.token else []
    payload = sanitize_payload(
        {
            "settings": {
                **asdict(settings),
                "project_root": str(settings.project_root),
                "log_path": str(settings.log_path),
                "state_path": str(settings.state_path),
                "lock_path": str(settings.lock_path),
                "token": "<redacted>" if settings.token else None,
            },
            "state": load_state(settings.state_path),
        },
        secrets=secrets,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate recurring read-only Paperclip sync into the local vector index",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    once_parser = subparsers.add_parser("once", help="run one sync pass")
    once_parser.set_defaults(func=command_once)

    loop_parser = subparsers.add_parser("loop", help="run sync in a local loop")
    loop_parser.add_argument("--max-runs", type=int, default=0, help="stop after N passes; 0 means infinite")
    loop_parser.set_defaults(func=command_loop)

    status_parser = subparsers.add_parser("status", help="show effective config and last sync state")
    status_parser.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
