"""Frozen compatibility adapter for the legacy soundcheck engine.

This module is the only live_runtime source file allowed to import the legacy
``AutoSoundcheckEngine``.  The import is deliberately lazy so importing or
constructing the canonical ``LiveSoundcheckService`` cannot load the legacy
decision stack.

The adapter exists only for unconfigured compatibility sessions while runtime
references are severed.  It must not gain new musical decision behaviour.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import LiveMode


def _legacy_flags(mode: LiveMode) -> tuple[bool, bool]:
    """Map canonical modes onto the frozen legacy engine flags."""
    if mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE):
        return True, False
    return False, True


def _default_engine_factory() -> Callable[..., Any]:
    # Sole legacy decision import allowed by the live_runtime source boundary.
    # Keep it inside this function so canonical imports stay legacy-free.
    from auto_soundcheck_engine import AutoSoundcheckEngine

    return AutoSoundcheckEngine


def create_legacy_soundcheck_engine(
    request: Any,
    *,
    engine_factory: Callable[..., Any] | None = None,
    on_state_change: Callable[..., Any] | None = None,
    on_channel_update: Callable[..., Any] | None = None,
    on_observation: Callable[..., Any] | None = None,
) -> Any:
    """Construct the frozen compatibility engine without starting it."""
    observe_only, auto_apply = _legacy_flags(request.mode)
    factory = engine_factory or _default_engine_factory()
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
    setattr(engine, "live_runtime_mode", request.mode.value)
    return engine
