"""Session lifecycle for concert/soundcheck learning telemetry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .event_writer import TelemetryEventWriter, write_json_atomic
    from .schemas import (
        SCHEMA_VERSION,
        SceneMetadata,
        SessionRecord,
        TelemetryManifest,
        generate_session_id,
        utc_now,
    )
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.event_writer import TelemetryEventWriter, write_json_atomic
    from telemetry.schemas import (
        SCHEMA_VERSION,
        SceneMetadata,
        SessionRecord,
        TelemetryManifest,
        generate_session_id,
        utc_now,
    )


DEFAULT_STORAGE_DIR = "data/telemetry/sessions"


def default_telemetry_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "storage_dir": DEFAULT_STORAGE_DIR,
        "schema_version": SCHEMA_VERSION,
        "write_jsonl": True,
        "write_parquet": False,
        "collect_interval_ms": 1000,
        "collect_channel_meters": True,
        "collect_bus_meters": True,
        "collect_processing_snapshots": True,
        "collect_engineer_actions": True,
        "spl": {
            "enabled": False,
            "provider": "mock",
            "file_path": None,
        },
        "safety": {
            "allow_live_writes": False,
            "telemetry_read_only": True,
        },
    }


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_git_commit(repo_root: str | Path | None = None) -> Optional[str]:
    root = Path(repo_root or Path.cwd())
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


class TelemetrySessionManager:
    def __init__(
        self,
        *,
        storage_dir: str | Path = DEFAULT_STORAGE_DIR,
        config: Optional[Dict[str, Any]] = None,
        repo_root: str | Path | None = None,
    ):
        self.config = _deep_merge(default_telemetry_config(), config)
        self.storage_dir = Path(self.config.get("storage_dir") or storage_dir)
        if storage_dir != DEFAULT_STORAGE_DIR and not (config or {}).get("storage_dir"):
            self.storage_dir = Path(storage_dir)
        self.repo_root = Path(repo_root or Path.cwd())
        self.current_session: Optional[SessionRecord] = None
        self.writer: Optional[TelemetryEventWriter] = None

    @property
    def current_session_path(self) -> Path:
        return self.storage_dir / "current_session.json"

    def session_dir(self, session_id: str) -> Path:
        return self.storage_dir / session_id

    def start_session(
        self,
        *,
        mode: str = "soundcheck",
        venue_name: str = "",
        artist_name: str = "",
        scene_metadata: Optional[SceneMetadata | Dict[str, Any]] = None,
        project_version: str = "",
        console_type: str = "",
        console_ip: str = "",
        safety_mode: str = "dry-run",
        notes: str = "",
        session_id: Optional[str] = None,
        force: bool = False,
    ) -> Optional[SessionRecord]:
        if not force and not bool(self.config.get("enabled", False)):
            return None
        if not bool(self.config.get("safety", {}).get("telemetry_read_only", True)):
            raise ValueError("telemetry.safety.telemetry_read_only must remain true")
        if bool(self.config.get("safety", {}).get("allow_live_writes", False)):
            raise ValueError("telemetry.safety.allow_live_writes must remain false")

        session_id = session_id or generate_session_id(mode)
        session = SessionRecord(
            session_id=session_id,
            started_at=utc_now(),
            mode=mode,
            venue_name=venue_name,
            artist_name=artist_name,
            project_version=project_version,
            console_type=console_type,
            console_ip=console_ip,
            safety_mode=safety_mode,
            notes=notes,
        )
        metadata = scene_metadata
        if metadata is None:
            metadata = SceneMetadata(venue_name=venue_name)
        elif isinstance(metadata, dict):
            metadata = SceneMetadata(**metadata)

        session_dir = self.session_dir(session.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self.writer = TelemetryEventWriter(session_dir, session_id=session.session_id)
        self.current_session = session
        write_json_atomic(session_dir / "session.json", session.to_dict())
        write_json_atomic(session_dir / "scene_metadata.json", metadata.to_dict())
        write_json_atomic(self.current_session_path, {"session_id": session.session_id, "session_dir": str(session_dir)})
        self.write_manifest()
        return session

    def get_current_session(self) -> Optional[SessionRecord]:
        if self.current_session is not None:
            return self.current_session
        if not self.current_session_path.exists():
            return None
        try:
            pointer = json.loads(self.current_session_path.read_text(encoding="utf-8"))
            session_path = Path(pointer["session_dir"]) / "session.json"
            return self.load_session_record(str(pointer["session_id"]), session_path=session_path)
        except Exception:
            return None

    def load_session_record(self, session_id: str, *, session_path: Optional[str | Path] = None) -> SessionRecord:
        path = Path(session_path or self.session_dir(session_id) / "session.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionRecord(**payload)

    def stop_session(self, session_id: Optional[str] = None, *, reason: str = "") -> Optional[SessionRecord]:
        session = self.current_session
        if session_id:
            session = self.load_session_record(session_id)
        elif session is None:
            session = self.get_current_session()
        if session is None:
            return None
        session.ended_at = utc_now()
        if reason:
            session.notes = f"{session.notes}\nStop reason: {reason}".strip()
        session_dir = self.session_dir(session.session_id)
        write_json_atomic(session_dir / "session.json", session.to_dict())
        if self.writer is None or self.writer.session_id != session.session_id:
            self.writer = TelemetryEventWriter(session_dir, session_id=session.session_id)
            self._load_existing_counters(session_dir)
        self.write_manifest(session=session)
        if self.current_session_path.exists():
            self.current_session_path.unlink()
        self.current_session = None
        if self.writer:
            self.writer.close()
        return session

    def _load_existing_counters(self, session_dir: Path) -> None:
        if not self.writer:
            return
        for event_type, path_name in self.writer.event_path_map().items() if hasattr(self.writer, "event_path_map") else []:
            path = session_dir / path_name
            if path.exists():
                self.writer.counters[event_type] = len(path.read_text(encoding="utf-8").splitlines())

    def append_event(self, event_type: str, payload: Any) -> Dict[str, Any]:
        session = self.get_current_session()
        if session is None:
            raise RuntimeError("No active telemetry session")
        if self.writer is None:
            self.writer = TelemetryEventWriter(self.session_dir(session.session_id), session_id=session.session_id)
        row = self.writer.append_jsonl(event_type, payload)
        self.write_manifest(session=session)
        return row

    def write_manifest(self, *, session: Optional[SessionRecord] = None) -> Dict[str, Any]:
        session = session or self.current_session or self.get_current_session()
        if session is None:
            raise RuntimeError("No telemetry session available for manifest")
        if self.writer is None:
            self.writer = TelemetryEventWriter(self.session_dir(session.session_id), session_id=session.session_id)
        manifest = TelemetryManifest(
            session_id=session.session_id,
            created_at=session.started_at,
            schema_version=str(self.config.get("schema_version") or SCHEMA_VERSION),
            project_git_commit=project_git_commit(self.repo_root),
            safety_mode=session.safety_mode,
        )
        return self.writer.write_manifest(manifest)


def _manager_from_args(args: argparse.Namespace) -> TelemetrySessionManager:
    return TelemetrySessionManager(storage_dir=args.storage_dir, repo_root=Path.cwd())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automixer learning telemetry session manager")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start")
    start.add_argument("--mode", default="soundcheck", choices=sorted({"soundcheck", "concert", "rehearsal", "offline_test"}))
    start.add_argument("--venue", default="")
    start.add_argument("--artist", default="")
    start.add_argument("--console-type", default="wing")
    start.add_argument("--console-ip", default="")
    start.add_argument("--safety-mode", default="dry-run")
    start.add_argument("--notes", default="")
    start.add_argument("--session-id", default=None)

    stop = sub.add_parser("stop")
    stop.add_argument("--session-id", default=None)
    stop.add_argument("--reason", default="")

    args = parser.parse_args(argv)
    manager = _manager_from_args(args)
    if args.command == "start":
        session = manager.start_session(
            mode=args.mode,
            venue_name=args.venue,
            artist_name=args.artist,
            console_type=args.console_type,
            console_ip=args.console_ip,
            safety_mode=args.safety_mode,
            notes=args.notes,
            session_id=args.session_id,
            force=True,
        )
        print(json.dumps({"started": bool(session), "session": session.to_dict() if session else None}, ensure_ascii=False))
        return 0
    if args.command == "stop":
        session = manager.stop_session(args.session_id, reason=args.reason)
        print(json.dumps({"stopped": bool(session), "session": session.to_dict() if session else None}, ensure_ascii=False))
        return 0 if session else 1
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
