"""
Session Lifecycle and Configuration Management

Manages mixing sessions for the AUTO MIXER system with:
- Session creation and destruction with unique IDs
- Config loading from YAML and JSON files with dot-notation access
- Session state persistence to disk (JSON)
- Multiple concurrent sessions with active-session selection
- Session metadata tracking (duration, channel count, event name)
- Auto-save on configurable interval
"""

import copy
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.debug("PyYAML not installed — YAML config loading disabled")


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Represents a single mixing session."""
    session_id: str
    name: str
    created_at: float
    config: Dict[str, Any] = field(default_factory=dict)
    state: str = "active"  # active, paused, destroyed
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime tracking (not persisted by default)
    _dirty: bool = field(default=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to a plain dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "config": self.config,
            "state": self.state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        """Deserialize session from a dictionary."""
        return cls(
            session_id=d["session_id"],
            name=d.get("name", ""),
            created_at=d.get("created_at", time.time()),
            config=d.get("config", {}),
            state=d.get("state", "active"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_nested(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Retrieve a value from a nested dict using dot notation.

    Example:
        _get_nested({"a": {"b": 1}}, "a.b") -> 1
    """
    parts = dotted_key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set_nested(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Set a value in a nested dict using dot notation, creating
    intermediate dicts as needed.

    Example:
        d = {}
        _set_nested(d, "a.b.c", 42)
        # d == {"a": {"b": {"c": 42}}}
    """
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _delete_nested(data: Dict[str, Any], dotted_key: str) -> bool:
    """
    Delete a key from a nested dict using dot notation.
    Returns True if the key existed and was removed.
    """
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    if isinstance(current, dict) and parts[-1] in current:
        del current[parts[-1]]
        return True
    return False


def _load_yaml_file(path: str) -> Dict[str, Any]:
    """Load a YAML file, returning an empty dict on failure."""
    if not HAS_YAML:
        logger.warning("PyYAML not installed — cannot load %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to load YAML %s: %s", path, exc)
        return {}


def _load_json_file(path: str) -> Dict[str, Any]:
    """Load a JSON file, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to load JSON %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages mixing session lifecycle and configuration.

    Supports:
    - Multiple concurrent sessions (each with its own config namespace)
    - Global config that serves as the default for new sessions
    - YAML and JSON config file loading
    - Dot-notation config get/set
    - Periodic auto-save of the active session
    - Session state persistence to disk
    - Listeners for session lifecycle events
    """

    def __init__(
        self,
        config_dir: str = "config",
        sessions_dir: str = "sessions",
        auto_save_interval_sec: float = 0.0,
    ):
        """
        Args:
            config_dir: Directory to search for config files.
            sessions_dir: Directory to persist session JSON files.
            auto_save_interval_sec: If > 0, start a background thread that
                                    saves the active session at this interval.
        """
        self._sessions: Dict[str, Session] = {}
        self._active_session_id: Optional[str] = None
        self._config_dir = config_dir
        self._sessions_dir = sessions_dir
        self._global_config: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Lifecycle event listeners
        self._listeners: List[Callable[[str, Session], None]] = []

        # Auto-save
        self._auto_save_interval = auto_save_interval_sec
        self._auto_save_stop = threading.Event()
        self._auto_save_thread: Optional[threading.Thread] = None

        if auto_save_interval_sec > 0:
            self._start_auto_save()

        logger.info(
            "SessionManager initialized (config_dir=%s, sessions_dir=%s, "
            "auto_save=%.1fs)",
            config_dir, sessions_dir, auto_save_interval_sec,
        )

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new mixing session.

        Args:
            name: Human-readable session name.  Auto-generated if empty.
            config: Initial config dict. Defaults to a copy of the global config.
            metadata: Optional metadata (event name, venue, etc.).

        Returns:
            The newly created Session.
        """
        session_id = str(uuid.uuid4())[:8]

        with self._lock:
            initial_config = config if config is not None else copy.deepcopy(self._global_config)
            session = Session(
                session_id=session_id,
                name=name or f"Session-{session_id}",
                created_at=time.time(),
                config=initial_config,
                metadata=metadata or {},
            )
            self._sessions[session_id] = session

            # If no active session, make this one active
            if self._active_session_id is None:
                self._active_session_id = session_id

        self._emit("created", session)
        logger.info("Session created: %s (%s)", session.name, session_id)
        return session

    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session and clean up. If it was the active session,
        the next available session becomes active.

        Returns True if the session existed.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            session.state = "destroyed"

            if self._active_session_id == session_id:
                remaining = list(self._sessions.keys())
                self._active_session_id = remaining[0] if remaining else None

        self._emit("destroyed", session)
        logger.info("Session destroyed: %s (%s)", session.name, session_id)
        return True

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_active_session(self) -> Optional[Session]:
        """Get the currently active session."""
        with self._lock:
            if self._active_session_id:
                return self._sessions.get(self._active_session_id)
        return None

    def set_active_session(self, session_id: str) -> bool:
        """
        Set the active session by ID.
        Returns True if the session exists and was activated.
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            prev_id = self._active_session_id
            self._active_session_id = session_id

        logger.info("Active session changed: %s -> %s", prev_id, session_id)
        session = self.get_session(session_id)
        if session:
            self._emit("activated", session)
        return True

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summary info."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "state": s.state,
                    "created_at": s.created_at,
                    "active": s.session_id == self._active_session_id,
                    "config_keys": len(s.config),
                    "metadata": s.metadata,
                }
                for s in self._sessions.values()
            ]

    def pause_session(self, session_id: str) -> bool:
        """Pause a session (mark as paused). Returns True on success."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.state != "active":
                return False
            session.state = "paused"
        self._emit("paused", session)
        return True

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session. Returns True on success."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.state != "paused":
                return False
            session.state = "active"
        self._emit("resumed", session)
        return True

    # ------------------------------------------------------------------
    # Config access
    # ------------------------------------------------------------------

    def load_config(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file into the global config.

        The loaded keys are merged into the existing global config.

        Args:
            path: Absolute or relative path. Relative paths are resolved
                  against config_dir.

        Returns:
            The loaded config dict (may be empty on error).
        """
        # Resolve relative paths against config_dir
        if not os.path.isabs(path):
            path = os.path.join(self._config_dir, path)

        if not os.path.isfile(path):
            logger.warning("Config file not found: %s", path)
            return {}

        if path.endswith((".yaml", ".yml")):
            config = _load_yaml_file(path)
        else:
            config = _load_json_file(path)

        with self._lock:
            self._global_config.update(config)

        logger.info("Config loaded from %s: %d top-level keys", path, len(config))
        return config

    def load_config_into_session(
        self, session_id: str, path: str
    ) -> Dict[str, Any]:
        """Load a config file directly into a specific session's config."""
        if not os.path.isabs(path):
            path = os.path.join(self._config_dir, path)

        if not os.path.isfile(path):
            logger.warning("Config file not found: %s", path)
            return {}

        if path.endswith((".yaml", ".yml")):
            config = _load_yaml_file(path)
        else:
            config = _load_json_file(path)

        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.config.update(config)
                session._dirty = True

        return config

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.

        Looks in the active session's config first, then falls back to
        the global config.

        Args:
            key: Dot-separated key, e.g. 'mixer.osc.ip'.
            default: Default value if key is not found.
        """
        with self._lock:
            session = self.get_active_session()
            if session:
                val = _get_nested(session.config, key)
                if val is not None:
                    return val
            return _get_nested(self._global_config, key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        Set a config value using dot notation.

        Writes to the active session's config if one exists, otherwise
        to the global config.
        """
        with self._lock:
            session = self.get_active_session()
            if session:
                _set_nested(session.config, key, value)
                session._dirty = True
            else:
                _set_nested(self._global_config, key, value)

    def delete_config(self, key: str) -> bool:
        """Delete a config key. Returns True if it existed."""
        with self._lock:
            session = self.get_active_session()
            if session:
                removed = _delete_nested(session.config, key)
                if removed:
                    session._dirty = True
                return removed
            return _delete_nested(self._global_config, key)

    def get_global_config(self) -> Dict[str, Any]:
        """Return a deep copy of the global config."""
        with self._lock:
            return copy.deepcopy(self._global_config)

    def set_global_config(self, config: Dict[str, Any]) -> None:
        """Replace the entire global config."""
        with self._lock:
            self._global_config = copy.deepcopy(config)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_session(
        self, session_id: str, path: Optional[str] = None
    ) -> bool:
        """
        Save a session to disk as JSON.

        Args:
            session_id: ID of the session to save.
            path: File path. Defaults to sessions_dir/{session_id}.json.

        Returns:
            True on success.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning("Cannot save: session %s not found", session_id)
                return False
            data = session.to_dict()

        save_path = path or os.path.join(
            self._sessions_dir, f"{session_id}.json"
        )
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]._dirty = False
            logger.info("Session saved: %s -> %s", session_id, save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save session %s: %s", session_id, exc)
            return False

    def save_active_session(self) -> bool:
        """Save the active session to its default path."""
        with self._lock:
            sid = self._active_session_id
        if sid:
            return self.save_session(sid)
        return False

    def save_all_sessions(self) -> int:
        """Save all sessions. Returns the count of successfully saved sessions."""
        saved = 0
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            if self.save_session(sid):
                saved += 1
        return saved

    def load_session(self, path: str) -> Optional[Session]:
        """
        Load a session from a JSON file on disk.

        Args:
            path: Path to the session JSON file.

        Returns:
            The loaded Session, or None on failure.
        """
        if not os.path.isfile(path):
            logger.warning("Session file not found: %s", path)
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Failed to load session from %s: %s", path, exc)
            return None

        session = Session.from_dict(data)

        with self._lock:
            self._sessions[session.session_id] = session

        self._emit("loaded", session)
        logger.info("Session loaded: %s (%s) from %s",
                     session.name, session.session_id, path)
        return session

    def load_all_sessions(self) -> int:
        """
        Load all session JSON files from the sessions directory.
        Returns the number of sessions loaded.
        """
        sessions_path = Path(self._sessions_dir)
        if not sessions_path.is_dir():
            return 0

        loaded = 0
        for json_file in sessions_path.glob("*.json"):
            if self.load_session(str(json_file)):
                loaded += 1

        logger.info("Loaded %d sessions from %s", loaded, self._sessions_dir)
        return loaded

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    def add_listener(
        self, callback: Callable[[str, Session], None]
    ) -> None:
        """
        Register a session lifecycle listener.

        The callback receives (event_name, session) where event_name is
        one of: 'created', 'destroyed', 'activated', 'paused', 'resumed',
        'loaded'.
        """
        self._listeners.append(callback)

    def remove_listener(
        self, callback: Callable[[str, Session], None]
    ) -> bool:
        """Remove a listener. Returns True if found and removed."""
        try:
            self._listeners.remove(callback)
            return True
        except ValueError:
            return False

    def _emit(self, event: str, session: Session) -> None:
        """Emit a lifecycle event to all listeners."""
        for listener in self._listeners:
            try:
                listener(event, session)
            except Exception as exc:
                logger.error("Session listener error (%s): %s", event, exc)

    # ------------------------------------------------------------------
    # Auto-save
    # ------------------------------------------------------------------

    def _start_auto_save(self) -> None:
        """Start the auto-save background thread."""
        self._auto_save_stop.clear()
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop, name="session-autosave", daemon=True
        )
        self._auto_save_thread.start()
        logger.info("Auto-save started (interval=%.1fs)", self._auto_save_interval)

    def _auto_save_loop(self) -> None:
        """Periodically save dirty sessions."""
        while not self._auto_save_stop.is_set():
            self._auto_save_stop.wait(timeout=self._auto_save_interval)
            if self._auto_save_stop.is_set():
                break

            with self._lock:
                dirty_ids = [
                    sid for sid, s in self._sessions.items()
                    if s._dirty and s.state != "destroyed"
                ]

            for sid in dirty_ids:
                self.save_session(sid)

    def stop_auto_save(self) -> None:
        """Stop the auto-save background thread."""
        self._auto_save_stop.set()
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=3.0)
        self._auto_save_thread = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Graceful shutdown: save all sessions, stop auto-save,
        and clean up resources.
        """
        logger.info("SessionManager shutting down...")
        self.stop_auto_save()
        self.save_all_sessions()
        logger.info("SessionManager shutdown complete")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<SessionManager sessions={len(self._sessions)} "
            f"active={self._active_session_id}>"
        )

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions
