"""
Tests for backend/session_manager.py — Session dataclass, SessionManager lifecycle,
config access, persistence with temp files.
"""

import json
import os
import tempfile
import time
import pytest

try:
    from session_manager import Session, SessionManager, _get_nested, _set_nested
except ImportError:
    pytest.skip("session_manager module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session_mgr():
    """Fresh SessionManager instance."""
    return SessionManager()


@pytest.fixture
def sample_session():
    """A pre-built Session instance for tests."""
    return Session(
        session_id="test-id-123",
        name="Test Session",
        created_at=time.time(),
        config={"mixer": {"channels": 40}},
        metadata={"event": "Concert"},
    )


# ---------------------------------------------------------------------------
# Session dataclass tests
# ---------------------------------------------------------------------------

class TestSessionDataclass:

    def test_to_dict(self, sample_session):
        d = sample_session.to_dict()
        assert d["session_id"] == "test-id-123"
        assert d["name"] == "Test Session"
        assert d["config"]["mixer"]["channels"] == 40
        assert d["metadata"]["event"] == "Concert"

    def test_from_dict_roundtrip(self, sample_session):
        d = sample_session.to_dict()
        restored = Session.from_dict(d)
        assert restored.session_id == sample_session.session_id
        assert restored.name == sample_session.name
        assert restored.config == sample_session.config

    def test_from_dict_defaults(self):
        """from_dict should handle missing optional fields."""
        d = {"session_id": "minimal"}
        s = Session.from_dict(d)
        assert s.session_id == "minimal"
        assert s.state == "active"
        assert s.config == {}

    def test_state_field(self, sample_session):
        assert sample_session.state == "active"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestNestedHelpers:

    def test_get_nested_simple(self):
        data = {"a": {"b": {"c": 42}}}
        assert _get_nested(data, "a.b.c") == 42

    def test_get_nested_missing_returns_default(self):
        data = {"a": 1}
        assert _get_nested(data, "a.b.c", "default") == "default"

    def test_set_nested_creates_path(self):
        data = {}
        _set_nested(data, "x.y.z", 99)
        assert data["x"]["y"]["z"] == 99

    def test_set_nested_overwrites(self):
        data = {"x": {"y": 10}}
        _set_nested(data, "x.y", 20)
        assert data["x"]["y"] == 20


# ---------------------------------------------------------------------------
# SessionManager lifecycle tests
# ---------------------------------------------------------------------------

class TestSessionManagerLifecycle:

    def test_create_session(self, session_mgr):
        session = session_mgr.create_session(name="My Session")
        assert session is not None
        assert session.name == "My Session"
        assert session.state == "active"
        assert len(session.session_id) > 0

    def test_create_multiple_sessions(self, session_mgr):
        s1 = session_mgr.create_session(name="S1")
        s2 = session_mgr.create_session(name="S2")
        assert s1.session_id != s2.session_id

    def test_destroy_session(self, session_mgr):
        session = session_mgr.create_session(name="Doomed")
        sid = session.session_id
        result = session_mgr.destroy_session(sid)
        assert result is True

    def test_destroy_nonexistent_session(self, session_mgr):
        result = session_mgr.destroy_session("nonexistent-id")
        assert result is False


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestSessionPersistence:

    def test_save_and_load(self, session_mgr, tmp_dir):
        session = session_mgr.create_session(name="Persist Test")
        sid = session.session_id
        path = os.path.join(tmp_dir, f"{sid}.json")
        session_mgr.save_session(sid, path)
        assert os.path.isfile(path)

        # Verify JSON content
        with open(path) as f:
            data = json.load(f)
        assert data["session_id"] == sid
        assert data["name"] == "Persist Test"

    def test_load_session(self, session_mgr, tmp_dir):
        # Create and save
        session = session_mgr.create_session(name="Load Test")
        sid = session.session_id
        path = os.path.join(tmp_dir, f"{sid}.json")
        session_mgr.save_session(sid, path)

        # Create a new manager and load
        mgr2 = SessionManager()
        loaded = mgr2.load_session(path)
        assert loaded is not None
        assert loaded.session_id == sid
        assert loaded.name == "Load Test"

    def test_save_nonexistent_session_fails(self, session_mgr, tmp_dir):
        """Saving a session that doesn't exist should fail gracefully."""
        path = os.path.join(tmp_dir, "nope.json")
        result = session_mgr.save_session("fake-id", path)
        assert result is False


# ---------------------------------------------------------------------------
# Config access tests
# SessionManager.get_config(key, default) and set_config(key, value)
# operate on the ACTIVE session's config (no session_id parameter).
# ---------------------------------------------------------------------------

class TestSessionConfig:

    def test_get_config(self, session_mgr):
        session = session_mgr.create_session(name="Config Test")
        # Set initial config on the session directly
        session.config = {"mixer": {"channels": 40}}
        # get_config uses the active session
        val = session_mgr.get_config("mixer.channels")
        assert val == 40

    def test_set_config(self, session_mgr):
        session_mgr.create_session(name="Config Set")
        session_mgr.set_config("osc.port", 2223)
        val = session_mgr.get_config("osc.port")
        assert val == 2223

    def test_get_config_default(self, session_mgr):
        session_mgr.create_session(name="Default Config")
        val = session_mgr.get_config("missing.key", "default_val")
        assert val == "default_val"
