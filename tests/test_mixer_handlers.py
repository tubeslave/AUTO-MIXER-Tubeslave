"""Tests for backend/handlers/mixer_handlers.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from handlers import mixer_handlers

try:
    from operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS


class FakeServer:
    def _normalize_manual_write_kind(self, raw_value: object) -> str:
        if raw_value is None:
            return ""
        value = str(raw_value).strip().lower()
        if value == "preamp":
            return "gain"
        return value

    def _manual_write_address(self, channel: int, value_type: str) -> str:
        if value_type == "fader":
            return f"/ch/{channel}/fdr"
        if value_type == "gain":
            return f"/ch/{channel}/in/set/trim"
        if value_type == "mute":
            return f"/ch/{channel}/mute"
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
