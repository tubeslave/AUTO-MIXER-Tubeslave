"""Authoritative live soundcheck construction and lifecycle seam.

This module is the only new-live layer allowed to construct the legacy
``AutoSoundcheckEngine`` while its useful hardware/audio plumbing is migrated
behind ``live_runtime`` interfaces. Callers depend on this service rather than
constructing, starting, stopping or inspecting the legacy decision engine
independently.

The compatibility bridge is temporary. It removes parallel decision authority
from the UI/transport layer without forcing a flag-day rewrite of WING/audio
capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import LiveMode


@dataclass(frozen=True)
class LiveStartRequest:
    mixer_type: str
    mixer_ip: str
    mixer_port: int
    audio_device_name: str
    num_channels: int
    selected_channels: list[int]
    mode: LiveMode = LiveMode.OBSERVE


class LiveSoundcheckService:
    """Authoritative lifecycle seam for the live pipeline.

    ``AutoSoundcheckEngine`` remains a MIGRATE dependency because it still owns
    useful mixer discovery, audio capture, readback and action logging. New
    callers do not own that engine directly: they start/stop/query this service.
    """

    legacy_bridge = True

    def __init__(self, engine_factory: Callable[..., Any] | None = None):
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._request: LiveStartRequest | None = None

    @property
    def active_engine(self) -> Any | None:
        """Temporary compatibility view for legacy server cleanup/readback."""
        return self._engine

    @property
    def active_mode(self) -> LiveMode | None:
        return self._request.mode if self._request else None

    @staticmethod
    def _legacy_flags(mode: LiveMode) -> tuple[bool, bool]:
        """Map new runtime modes onto the temporary legacy engine flags.

        OBSERVE/PROPOSE must not write. BENCH_TEST and production write modes
        may write; detailed authorization belongs to live_runtime policy, not UI
        handlers. The bridge never infers a mode from mixer presence.
        """
        if mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE):
            return True, False
        return False, True

    def _factory(self) -> Callable[..., Any]:
        factory = self._engine_factory
        if factory is None:
            # Deliberately lazy and isolated: this is the sole compatibility
            # import to delete when the new live runtime fully owns the loop.
            from auto_soundcheck_engine import AutoSoundcheckEngine

            factory = AutoSoundcheckEngine
        return factory

    def create_engine(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        """Construct an engine without starting it.

        Kept for focused bridge tests and staged migration. Runtime callers
        should use :meth:`start` so lifecycle ownership stays in live_runtime.
        """
        observe_only, auto_apply = self._legacy_flags(request.mode)
        engine = self._factory()(
            mixer_type=request.mixer_type,
            mixer_ip=request.mixer_ip,
            mixer_port=request.mixer_port,
            audio_device_name=request.audio_device_name,
            num_channels=request.num_channels,
            selected_channels=request.selected_channels,
            observe_only=observe_only,
            auto_apply=auto_apply,
            on_state_change=on_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        setattr(engine, "live_runtime_mode", request.mode.value)
        return engine

    def is_active(self) -> bool:
        engine = self._engine
        if engine is None:
            return False
        state = getattr(getattr(engine, "state", None), "value", None)
        return state not in ("stopped", "error")

    def start(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        """Construct and start exactly one live engine instance."""
        if self.is_active():
            raise RuntimeError("Live soundcheck engine already running")

        engine = self.create_engine(
            request,
            on_state_change=on_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        self._engine = engine
        self._request = request
        try:
            engine.start_async()
        except Exception:
            self._engine = None
            self._request = None
            raise
        return engine

    def stop(self) -> bool:
        """Stop the active engine and release lifecycle ownership."""
        engine = self._engine
        self._engine = None
        self._request = None
        if engine is None:
            return False
        engine.stop()
        return True

    def get_status(self) -> dict[str, Any]:
        """Return a transport-safe live status without exposing engine ownership."""
        engine = self._engine
        if engine is None:
            return {
                "state": "idle",
                "mixer_connected": False,
                "audio_running": False,
            }

        raw = engine.get_status() if hasattr(engine, "get_status") else {}
        status = dict(raw or {})
        mode = self.active_mode
        if mode is not None:
            status["mode"] = mode.value
        return status
