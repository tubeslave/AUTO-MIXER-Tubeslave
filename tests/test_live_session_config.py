"""Tests for the explicit live soundcheck composition-root configuration."""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, BACKEND)

from live_runtime.session_config import (  # noqa: E402
    LiveSessionConfigError,
    resolve_live_capture_bridge_config,
)


def _enabled_config():
    return {
        "live_soundcheck": {
            "capture_bridge": {
                "enabled": True,
                "roles": {"1": "lead_vocal", "2": "guitar"},
                "channel_names": {"1": "Lead Vocal", "2": "Guitar"},
                "window_frames": 4096,
                "analysis_interval_s": 0.2,
                "main_tap": {
                    "left_channel": 47,
                    "right_channel": 48,
                    "routes": [
                        {"usb_slot": 47, "source_group": "MAIN", "source_channel": 1},
                        {"usb_slot": 48, "source_group": "MAIN", "source_channel": 1},
                    ],
                },
            }
        }
    }


def test_missing_or_disabled_config_does_not_invent_capture_bridge():
    assert resolve_live_capture_bridge_config({}, mixer_type="wing") is None

    config = _enabled_config()
    config["live_soundcheck"]["capture_bridge"]["enabled"] = False
    assert resolve_live_capture_bridge_config(config, mixer_type="wing") is None


def test_non_wing_does_not_receive_wing_patch_contract():
    config = _enabled_config()
    assert resolve_live_capture_bridge_config(config, mixer_type="dlive") is None


def test_explicit_config_resolves_roles_names_and_exact_main_patch_contract():
    resolved = resolve_live_capture_bridge_config(
        _enabled_config(),
        mixer_type="WING",
        selected_channels=[1, 2],
    )

    assert resolved is not None
    assert resolved.roles == {1: "lead_vocal", 2: "guitar"}
    assert resolved.channel_names == {1: "Lead Vocal", 2: "Guitar"}
    assert resolved.window_frames == 4096
    assert resolved.analysis_interval_s == pytest.approx(0.2)
    assert resolved.patch_contract.tap.channels == (47, 48)
    assert [(route.usb_slot, route.source_group, route.source_channel) for route in resolved.patch_contract.routes] == [
        (47, "MAIN", 1),
        (48, "MAIN", 1),
    ]


def test_selected_input_requires_explicit_role():
    config = _enabled_config()
    with pytest.raises(LiveSessionConfigError, match="missing \[3\]"):
        resolve_live_capture_bridge_config(
            config,
            mixer_type="wing",
            selected_channels=[1, 3],
        )


def test_reserved_main_tap_channel_cannot_receive_musical_role():
    config = _enabled_config()
    config["live_soundcheck"]["capture_bridge"]["roles"]["47"] = "playback"

    with pytest.raises(LiveSessionConfigError, match="reserved Main tap channels"):
        resolve_live_capture_bridge_config(config, mixer_type="wing")


def test_selected_input_cannot_overlap_reserved_main_tap():
    with pytest.raises(LiveSessionConfigError, match="overlap reserved Main tap"):
        resolve_live_capture_bridge_config(
            _enabled_config(),
            mixer_type="wing",
            selected_channels=[1, 47],
        )


def test_patch_routes_must_cover_declared_tap_exactly():
    config = _enabled_config()
    config["live_soundcheck"]["capture_bridge"]["main_tap"]["routes"] = [
        {"usb_slot": 47, "source_group": "MAIN", "source_channel": 1},
    ]

    with pytest.raises(LiveSessionConfigError, match="routes must cover exactly"):
        resolve_live_capture_bridge_config(config, mixer_type="wing")


def test_non_main_route_is_rejected_fail_closed():
    config = _enabled_config()
    config["live_soundcheck"]["capture_bridge"]["main_tap"]["routes"][0]["source_group"] = "BUS"

    with pytest.raises(LiveSessionConfigError, match="source_group"):
        resolve_live_capture_bridge_config(config, mixer_type="wing")
