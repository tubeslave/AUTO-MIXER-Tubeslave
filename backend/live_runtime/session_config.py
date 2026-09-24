"""Explicit configuration resolver for the canonical live soundcheck runtime.

The composition root is the authority for channel roles and the post-console
Main tap contract.  This module deliberately does not infer either from legacy
``auto_*`` classifiers, mixer state caches, channel names, or WING defaults.

For WING, an enabled capture bridge config has this shape::

    live_soundcheck:
      capture_bridge:
        enabled: true
        roles: {"1": "lead_vocal", "2": "guitar"}
        channel_names: {"1": "Lead Vocal", "2": "Guitar"}
        window_frames: 2048
        analysis_interval_s: 0.1
        main_tap:
          left_channel: 47
          right_channel: 48
          routes:
            - {usb_slot: 47, source_group: MAIN, source_channel: 1}
            - {usb_slot: 48, source_group: MAIN, source_channel: 1}

The source channel values above are only an example.  No source channel, tap
slot, or musical role is supplied by code defaults.  The exact route declared
here is later proved by PATCH_VERIFY using fresh WING readback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .feature_stream import USB_CHANNEL_COUNT
from .main_evidence import PostConsoleMainTap
from .patch_verify import (
    MainTapPatchContract,
    MainTapRouteExpectation,
)
from .service import LiveCaptureBridgeConfig


class LiveSessionConfigError(ValueError):
    """Raised when an explicitly enabled live configuration is unsafe/invalid."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveSessionConfigError(f"{label} must be a mapping")
    return value


def _channel_number(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise LiveSessionConfigError(f"{label} must be an integer channel number")
    try:
        channel = int(value)
    except (TypeError, ValueError) as exc:
        raise LiveSessionConfigError(f"{label} must be an integer channel number") from exc
    if not 1 <= channel <= USB_CHANNEL_COUNT:
        raise LiveSessionConfigError(
            f"{label} must be inside 1..{USB_CHANNEL_COUNT}; got {channel}"
        )
    return channel


def _channel_text_map(value: Any, *, label: str, require_nonempty: bool) -> dict[int, str]:
    raw = _mapping(value, label=label)
    result: dict[int, str] = {}
    for key, text_value in raw.items():
        channel = _channel_number(key, label=f"{label} key")
        text = str(text_value).strip()
        if not text:
            raise LiveSessionConfigError(f"{label}[{channel}] must not be empty")
        result[channel] = text
    if require_nonempty and not result:
        raise LiveSessionConfigError(f"{label} must declare at least one channel")
    return result


def _selected_channels(values: Sequence[int] | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(
        sorted(
            {
                _channel_number(value, label="selected channel")
                for value in values
            }
        )
    )


def resolve_live_capture_bridge_config(
    config: Mapping[str, Any],
    *,
    mixer_type: str,
    selected_channels: Sequence[int] | None = None,
) -> LiveCaptureBridgeConfig | None:
    """Resolve an explicit service-owned capture bridge configuration.

    Missing/disabled configuration returns ``None`` and therefore cannot open
    the autonomous feature path.  Once ``enabled: true`` is present, validation
    is fail-closed: roles, Main tap slots and every route source are explicit.

    The bridge is currently WING-specific because its PATCH_VERIFY contract and
    native physical Main meter evidence are WING-specific.  Other mixer types
    keep their compatibility lifecycle but do not receive this bridge.
    """

    if not isinstance(config, Mapping):
        raise LiveSessionConfigError("server config must be a mapping")
    if str(mixer_type).strip().lower() != "wing":
        return None

    live_raw = config.get("live_soundcheck")
    if live_raw is None:
        return None
    live = _mapping(live_raw, label="live_soundcheck")

    bridge_raw = live.get("capture_bridge")
    if bridge_raw is None:
        return None
    bridge = _mapping(bridge_raw, label="live_soundcheck.capture_bridge")

    enabled = bridge.get("enabled", False)
    if not isinstance(enabled, bool):
        raise LiveSessionConfigError("live_soundcheck.capture_bridge.enabled must be bool")
    if not enabled:
        return None

    try:
        roles = _channel_text_map(
            bridge.get("roles"),
            label="live_soundcheck.capture_bridge.roles",
            require_nonempty=True,
        )
        channel_names_raw = bridge.get("channel_names", {})
        channel_names = _channel_text_map(
            channel_names_raw,
            label="live_soundcheck.capture_bridge.channel_names",
            require_nonempty=False,
        )

        main_tap_raw = _mapping(
            bridge.get("main_tap"),
            label="live_soundcheck.capture_bridge.main_tap",
        )
        if "left_channel" not in main_tap_raw:
            raise LiveSessionConfigError(
                "live_soundcheck.capture_bridge.main_tap.left_channel is required"
            )
        left_channel = _channel_number(
            main_tap_raw["left_channel"],
            label="main_tap.left_channel",
        )
        right_value = main_tap_raw.get("right_channel")
        right_channel = (
            None
            if right_value is None
            else _channel_number(right_value, label="main_tap.right_channel")
        )
        tap = PostConsoleMainTap(left_channel, right_channel)

        routes_raw = main_tap_raw.get("routes")
        if not isinstance(routes_raw, (list, tuple)) or not routes_raw:
            raise LiveSessionConfigError("main_tap.routes must be a non-empty list")
        routes: list[MainTapRouteExpectation] = []
        for index, item in enumerate(routes_raw):
            route = _mapping(item, label=f"main_tap.routes[{index}]")
            if "usb_slot" not in route or "source_channel" not in route:
                raise LiveSessionConfigError(
                    f"main_tap.routes[{index}] requires usb_slot and source_channel"
                )
            source_group = str(route.get("source_group", "MAIN")).strip().upper()
            routes.append(
                MainTapRouteExpectation(
                    usb_slot=_channel_number(
                        route["usb_slot"],
                        label=f"main_tap.routes[{index}].usb_slot",
                    ),
                    source_group=source_group,
                    source_channel=int(route["source_channel"]),
                )
            )

        patch_contract = MainTapPatchContract(tap=tap, routes=tuple(routes))
        reserved = set(tap.channels)

        role_overlap = reserved.intersection(roles)
        if role_overlap:
            raise LiveSessionConfigError(
                "reserved Main tap channels cannot have musical roles: "
                f"{sorted(role_overlap)}"
            )
        name_overlap = reserved.intersection(channel_names)
        if name_overlap:
            raise LiveSessionConfigError(
                "reserved Main tap channels cannot have input channel names: "
                f"{sorted(name_overlap)}"
            )

        selected = _selected_channels(selected_channels)
        selected_overlap = reserved.intersection(selected)
        if selected_overlap:
            raise LiveSessionConfigError(
                "selected input channels overlap reserved Main tap channels: "
                f"{sorted(selected_overlap)}"
            )
        missing_roles = [channel for channel in selected if channel not in roles]
        if missing_roles:
            raise LiveSessionConfigError(
                "explicit roles are required for every selected input channel: "
                f"missing {missing_roles}"
            )

        return LiveCaptureBridgeConfig(
            patch_contract=patch_contract,
            roles=roles,
            channel_names=channel_names,
            window_frames=bridge.get("window_frames", 2048),
            analysis_interval_s=bridge.get("analysis_interval_s", 0.100),
        )
    except LiveSessionConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise LiveSessionConfigError(str(exc)) from exc
