"""Minimal lifecycle handle for the canonical LIVE/SOUNDCHECK session.

This object deliberately contains no mixer, DSP or decision policy.  It exists
so the migrated live runtime can expose start/stop/status semantics without
constructing or running the legacy ``AutoSoundcheckEngine``.  The owner remains
``LiveSoundcheckService``; ``stop()`` only delegates back to that owner.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class LiveSessionPhase(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class LiveSessionLifecycle:
    """Policy-free lifecycle/event primitive owned by ``live_runtime``."""

    canonical_live_session = True

    def __init__(
        self,
        *,
        on_state_change: Callable[[str, str], None] | None = None,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> None:
        self.state = LiveSessionPhase.IDLE
        self.last_error: str | None = None
        self._on_state_change = on_state_change
        self._audit_sink = audit_sink
        self._stop_callback = stop_callback

    @property
    def active(self) -> bool:
        return self.state in {
            LiveSessionPhase.STARTING,
            LiveSessionPhase.RUNNING,
            LiveSessionPhase.STOPPING,
        }

    def _transition(self, phase: LiveSessionPhase, message: str) -> None:
        previous = self.state
        self.state = phase
        if self._audit_sink is not None:
            self._audit_sink(
                {
                    "event": "live_session_lifecycle",
                    "state_before": previous.value,
                    "state_after": phase.value,
                    "message": message,
                }
            )
        if self._on_state_change is not None:
            self._on_state_change(phase.value, message)

    def begin_start(self) -> None:
        if self.active:
            raise RuntimeError("Canonical live session is already active")
        self.last_error = None
        self._transition(LiveSessionPhase.STARTING, "Canonical live runtime starting")

    def mark_running(self) -> None:
        if self.state is not LiveSessionPhase.STARTING:
            raise RuntimeError(
                f"Cannot mark canonical live session running from {self.state.value}"
            )
        self._transition(LiveSessionPhase.RUNNING, "Canonical live runtime ready")

    def begin_stop(self) -> None:
        if self.state in (LiveSessionPhase.STOPPED, LiveSessionPhase.IDLE):
            return
        if self.state is LiveSessionPhase.STOPPING:
            return
        self._transition(LiveSessionPhase.STOPPING, "Canonical live runtime stopping")

    def mark_stopped(self) -> None:
        if self.state is LiveSessionPhase.STOPPED:
            return
        self._transition(LiveSessionPhase.STOPPED, "Canonical live runtime stopped")

    def mark_error(self, reason: BaseException | str) -> None:
        self.last_error = str(reason)
        self._transition(LiveSessionPhase.ERROR, self.last_error)

    def stop(self) -> bool:
        """Compatibility-safe stop handle delegating to the owning service."""
        if self._stop_callback is None:
            return False
        return bool(self._stop_callback())

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "audio_running": self.active,
            "mixer_connected": False,
            "lifecycle_error": self.last_error,
            "legacy_engine_attached": False,
        }
