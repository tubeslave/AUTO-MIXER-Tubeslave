"""
Session lifecycle and configuration management.

Manages mixing sessions, user configurations, and session persistence.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a nested value using dot notation."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set_nested(data: Dict[str, Any], path: str, value: Any):
    """Write a nested value using dot notation."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


@dataclass
class SessionConfig:
    """Configuration for a mixing session."""
    name: str = "default"
    venue: str = ""
    genre: str = "pop_rock"
    num_channels: int = 40
    sample_rate: int = 48000
    target_lufs: float = -18.0
    safety_limits: Dict[str, float] = field(default_factory=lambda: {
        "max_fader_db": 10.0,
        "max_gain_change_db": 6.0,
        "max_eq_cut_db": 15.0,
        "max_eq_boost_db": 12.0,
    })
    automation_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "gain_staging": True,
        "auto_eq": False,
        "auto_fader": False,
        "auto_compressor": False,
        "phase_alignment": False,
        "feedback_detection": True,
    })
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "venue": self.venue,
            "genre": self.genre,
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "target_lufs": self.target_lufs,
            "safety_limits": self.safety_limits,
            "automation_enabled": self.automation_enabled,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(init=False)
class Session:
    """A mixing session with state and history."""
    id: str
    config: Any
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active: bool = True
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        id: Optional[str] = None,
        config: Any = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        active: bool = True,
        events: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        state: str = "active",
    ):
        resolved_id = id or session_id or f"session_{int(time.time())}"
        if config is None:
            config = SessionConfig(name=name or "default")
        elif isinstance(config, dict):
            config = dict(config)

        self.id = resolved_id
        self.config = config
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.active = active if state is None else (state == "active")
        self.events = list(events or [])
        self.metadata = dict(metadata or {})
        if name is not None:
            self.metadata.setdefault("name", name)

    def add_event(self, event_type: str, data: Dict[str, Any] = None):
        self.events.append({
            "type": event_type,
            "data": data or {},
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    @property
    def session_id(self) -> str:
        return self.id

    @property
    def name(self) -> str:
        if "name" in self.metadata:
            return str(self.metadata["name"])
        if isinstance(self.config, SessionConfig):
            return self.config.name
        return "default"

    @property
    def state(self) -> str:
        return "active" if self.active else "ended"

    def to_dict(self) -> Dict[str, Any]:
        config = self.config.to_dict() if isinstance(self.config, SessionConfig) else dict(self.config or {})
        data = {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "config": config,
            "metadata": dict(self.metadata),
            "events": list(self.events),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        config = data.get("config", {})
        if isinstance(config, dict) and {"venue", "genre", "num_channels", "sample_rate", "target_lufs"} & set(config.keys()):
            config = SessionConfig.from_dict(config)
        return cls(
            session_id=data.get("session_id") or data.get("id"),
            name=data.get("name"),
            config=config,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            state=data.get("state", "active"),
            metadata=data.get("metadata", {}),
            events=data.get("events", []),
            active=data.get("active", data.get("state", "active") == "active"),
        )


class SessionManager:
    """
    Manages mixing session lifecycle and persistence.

    Provides:
    - Session creation and destruction
    - Configuration save/load
    - Session history tracking
    - Auto-save on changes
    """

    def __init__(self, sessions_dir: str = "sessions"):
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[Session] = None
        self._sessions: Dict[str, Session] = {}

    def create_session(self, name: str = "default", **config_kwargs) -> Session:
        """Create a new mixing session."""
        config = SessionConfig(name=name, **config_kwargs)
        session_id = f"{name}_{int(time.time())}"
        session = Session(id=session_id, config=config)
        self._sessions[session_id] = session
        self._current_session = session
        session.add_event("created")
        logger.info(f"Created session '{session_id}'")
        self._save_session(session)
        return session

    def get_current_session(self) -> Optional[Session]:
        """Get the current active session."""
        return self._current_session

    def set_current_session(self, session_id: str) -> Optional[Session]:
        """Set the active session by ID."""
        session = self._sessions.get(session_id)
        if session:
            if self._current_session:
                self._current_session.active = False
            self._current_session = session
            session.active = True
            session.add_event("activated")
            logger.info(f"Activated session '{session_id}'")
        return session

    def update_config(self, **kwargs) -> Optional[SessionConfig]:
        """Update current session configuration."""
        if not self._current_session:
            return None
        config = self._current_session.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self._current_session.add_event("config_updated", kwargs)
        self._save_session(self._current_session)
        return config

    def end_session(self, session_id: str = None):
        """End a session."""
        session = self._sessions.get(
            session_id or (self._current_session.id if self._current_session else "")
        )
        if session:
            session.active = False
            session.add_event("ended")
            self._save_session(session)
            if session == self._current_session:
                self._current_session = None
            logger.info(f"Ended session '{session.id}'")

    def destroy_session(self, session_id: str) -> bool:
        """Compatibility alias that removes a session from memory."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if self._current_session and self._current_session.id == session_id:
            self._current_session = None
        path = self._sessions_dir / f"{session_id}.json"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return True

    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        result = []
        for sid, session in self._sessions.items():
            result.append({
                "id": sid,
                "name": session.config.name,
                "active": session.active,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "events_count": len(session.events),
            })
        return result

    def _save_session(self, session: Session):
        """Save session to disk."""
        filepath = self._sessions_dir / f"{session.id}.json"
        try:
            config = session.config.to_dict() if isinstance(session.config, SessionConfig) else dict(session.config or {})
            data = {
                "id": session.id,
                "config": config,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "active": session.active,
                "events": session.events[-100:],  # Keep last 100 events
                "metadata": session.metadata,
            }
            filepath.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load_sessions(self):
        """Load all sessions from disk."""
        for filepath in self._sessions_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                config_data = data.get("config", {})
                config = SessionConfig.from_dict(config_data) if isinstance(config_data, dict) else config_data
                session = Session(
                    id=data["id"],
                    config=config,
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                    active=data.get("active", False),
                    events=data.get("events", []),
                    metadata=data.get("metadata", {}),
                )
                self._sessions[session.id] = session
            except Exception as e:
                logger.warning(f"Failed to load session {filepath}: {e}")
        logger.info(f"Loaded {len(self._sessions)} sessions")

    def save_session(self, session_id: str, path: str) -> bool:
        """Save a single session to a specific path."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        data = session.to_dict()
        Path(path).write_text(json.dumps(data, indent=2))
        return True

    def load_session(self, path: str) -> Optional[Session]:
        """Load one session from disk and register it."""
        try:
            data = json.loads(Path(path).read_text())
        except OSError:
            return None
        session = Session.from_dict(data)
        self._sessions[session.session_id] = session
        return session

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read dot-notated config from the active session."""
        if not self._current_session:
            return default
        config = self._current_session.config
        if isinstance(config, SessionConfig):
            config = config.to_dict()
        return _get_nested(config, key, default)

    def set_config(self, key: str, value: Any) -> bool:
        """Write dot-notated config into the active session."""
        if not self._current_session:
            return False
        config = self._current_session.config
        if isinstance(config, SessionConfig):
            config = config.to_dict()
            self._current_session.config = config
        _set_nested(config, key, value)
        self._current_session.updated_at = time.time()
        return True
