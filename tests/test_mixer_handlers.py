"""Tests for backend/handlers/mixer_handlers.py."""

import os
import sys
from typing import Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from handlers import mixer_handlers

try:
    from operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS


class FakeServer:
    def _parse_manual_write_address(self, address: object) -> Optional[tuple[int, str]]:
        normalized = str(address or "").strip()
        if not normalized:
            return None
        if normalized.startswith("/ch/") and normalized.endswith("/fdr"):
            try:
                return int(normalized.split("/")[2]), "fader"
            except (IndexError, TypeError, ValueError):
                return None
        if normalized.startswith("/ch/") and normalized.endswith("/mute"):
            try:
                return int(normalized.split("/")[2]), "mute"
            except (IndexError, TypeError, ValueError):
                return None
        if normalized.startswith("/ch/") and normalized.endswith("/pan"):
            try:
                return int(normalized.split("/")[2]), "pan"
            except (IndexError, TypeError, ValueError):
                return None
        parts = normalized.split("/")
        if (
            len(parts) == 6
            and parts[1] == "ch"
            and parts[3:] == ["in", "set", "trim"]
        ):
            try:
                return int(parts[2]), "gain"
            except (TypeError, ValueError, IndexError):
                return None
        return None

    def _normalize_manual_write_kind(self, raw_value: object) -> str:
        if raw_value is None:
            return ""
        value = str(raw_value).strip().lower()
        if value == "preamp":
            return "gain"
        if value == "pan_percent":
            return "pan"
        return value

    def _manual_write_address(self, channel: int, value_type: str) -> str:
        if value_type == "fader":
            return f"/ch/{channel}/fdr"
        if value_type == "gain":
            return f"/ch/{channel}/in/set/trim"
        if value_type == "mute":
            return f"/ch/{channel}/mute"
        if value_type == "pan":
            return f"/ch/{channel}/pan"
        return ""


@pytest.mark.parametrize("value_type", sorted(SUPPORTED_MANUAL_LIVE_WRITE_KINDS))
def test_parse_readonly_parameter_request_supports_supported_manual_kinds(value_type: str):
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {
            "channel": 7,
            "value_type": value_type,
        },
    )

    assert error is None
    assert channel == 7
    assert parsed_kind == value_type
    assert address == server._manual_write_address(7, value_type)


def test_parse_readonly_parameter_request_aliases_gain_like_input():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"channel": 3, "value_type": "preamp"},
    )

    assert error is None
    assert channel == 3
    assert parsed_kind == "gain"
    assert address == server._manual_write_address(3, "gain")


def test_parse_readonly_parameter_request_supports_pan_value_type():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"channel": 5, "value_type": "pan"},
    )

    assert error is None
    assert channel == 5
    assert parsed_kind == "pan"
    assert address == server._manual_write_address(5, "pan")


def test_parse_readonly_parameter_request_supports_pan_percent_alias():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"channel": 11, "value_type": "pan_percent"},
    )

    assert error is None
    assert channel == 11
    assert parsed_kind == "pan"
    assert address == server._manual_write_address(11, "pan")


def test_parse_readonly_parameter_request_by_address():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"address": "/ch/12/fdr"},
    )

    assert error is None
    assert channel == 12
    assert parsed_kind == "fader"
    assert address == "/ch/12/fdr"


def test_parse_readonly_parameter_request_by_pan_address():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"address": "/ch/9/pan"},
    )

    assert error is None
    assert channel == 9
    assert parsed_kind == "pan"
    assert address == "/ch/9/pan"


def test_parse_readonly_parameter_request_rejects_unsupported_kind():
    server = FakeServer()
    channel, parsed_kind, address, error = mixer_handlers._parse_readonly_parameter_request(
        server,
        {"channel": 2, "value_type": "eq"},
    )

    assert error == "missing_read_target"
    assert channel is None
    assert parsed_kind is None
    assert address is None
