"""Authoritative live soundcheck construction seam.

This module is the only new-live layer allowed to construct the legacy
``AutoSoundcheckEngine`` while its useful hardware/audio plumbing is migrated
behind ``live_runtime`` interfaces.  Callers must depend on this service rather
than importing the legacy decision engine directly.

The compatibility bridge is temporary.  It exists to remove parallel decision
authorities from the composition root without forcing a flag-day rewrite of
WING/audio capture.
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
    """Factory/lifecycle seam for the live pipeline.

    ``AutoSoundcheckEngine`` remains a MIGRATE dependency for now because it
    contains proven mixer discovery, audio capture, readback and action logging.
    No UI/server module should import it directly after this migration.
    """

    legacy_bridge = True

    def __init__(self, engine_factory: Callable[..., Any] | None = None):
        self._engine_factory = engine_factory

    @staticmethod
    def _legacy_flags(mode: LiveMode) -> tuple[bool, bool]:
        """Map new runtime modes onto the temporary legacy engine flags.

        OBSERVE/PROPOSE must not write.  BENCH_TEST and production write modes
        may write; their detailed authorization is owned by live_runtime policy,
        not by UI handlers.  The legacy bridge must not invent a mode from mixer
        presence.
        """
        if mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE):
            return True, False
        return False, True

    def create_engine(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        factory = self._engine_factory
        if factory is None:
            # Deliberately lazy and isolated: this is the sole compatibility
            # import to delete when the new live runtime fully owns the loop.
            from auto_soundcheck_engine import AutoSoundcheckEngine

            factory = AutoSoundcheckEngine

        observe_only, auto_apply = self._legacy_flags(request.mode)
        engine = factory(
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
        # Surface the authoritative mode for logging/UI even while the legacy
        # engine remains underneath the adapter.
        setattr(engine, "live_runtime_mode", request.mode.value)
        return engine
