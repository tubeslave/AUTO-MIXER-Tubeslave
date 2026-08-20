from __future__ import annotations

import json
from pathlib import Path

import pytest

from wing_lab_autowrite import (
    ALLOWED_FAMILIES,
    WingLabAutowriteSession,
    compile_wing_lab_decision,
)


class FakeWingTransport:
    def __init__(self, host: str, port: int, *, mismatch_address: str | None = None):
        self.host = host
        self.port = port
        self.is_connected = False
        self.handshake_info = "WING Rack 3.0.5"
        self.mismatch_address = mismatch_address
        self.values = {
            "/ch/1/fdr": -12.0,
            "/ch/2/eq/1g": 0.0,
            "/ch/3/gate/mdl": "DS902",
            "/ch/3/gate/f": 6500.0,
        }
        self.sends: list[tuple[str, object]] = []

    def connect(self, timeout: float = 5.0) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        return True

    def send(self, address: str, *values) -> bool:
        assert self.is_connected
        if values:
            self.sends.append((address, values[0]))
            if address != self.mismatch_address:
                self.values[address] = values[0]
        return True

    def query_parameter(self, address: str, timeout: float = 0.5):
        assert self.is_connected
        return {
            "status": "received",
            "confirmed_round_trips": 2,
            "peer_ip": self.host,
            "peer_port": self.port,
            "value": self.values.get(address),
        }


def test_lab_is_disabled_by_default_and_production_policy_is_unchanged(tmp_path):
    session = WingLabAutowriteSession(
        lambda host, port: FakeWingTransport(host, port),
        runtime_root=tmp_path,
        enabled=False,
    )

    assert session.status()["production_policy_changed"] is False
    with pytest.raises(PermissionError, match="wing_lab_disabled"):
        session.arm(host="192.168.1.10", console_model="WING Rack")


def test_arm_requires_private_ipv4_and_observed_wing_rack_identity(tmp_path):
    session = WingLabAutowriteSession(
        lambda host, port: FakeWingTransport(host, port),
        runtime_root=tmp_path,
        enabled=True,
    )
    with pytest.raises(ValueError, match="private_ipv4"):
        session.arm(host="wing.example.com", console_model="WING Rack")

    transport = FakeWingTransport("192.168.1.10", 2223)
    transport.handshake_info = "different console"
    session = WingLabAutowriteSession(
        lambda _host, _port: transport,
        runtime_root=tmp_path,
        enabled=True,
    )
    with pytest.raises(RuntimeError, match="identity_not_observed"):
        session.arm(host="192.168.1.10", console_model="WING Rack")
    assert transport.is_connected is False


def test_one_arm_allows_multiple_decisions_without_per_decision_confirmation(tmp_path):
    transport = FakeWingTransport("192.168.1.10", 2223)
    session = WingLabAutowriteSession(
        lambda _host, _port: transport,
        runtime_root=tmp_path,
        enabled=True,
    )
    armed = session.arm(
        host="192.168.1.10",
        console_model="WING Rack",
        duration_sec=60,
        max_corrections=8,
    )
    assert armed["per_decision_confirmation_required"] is False

    result = session.apply_decisions(
        [
            {
                "id": "level-1", "family": "channel_fader", "channel": 1,
                "accepted": True, "reason": "measured level above target",
                "confidence": 0.9, "value": -14.0,
            },
            {
                "id": "eq-1",
                "family": "channel_eq",
                "accepted": True,
                "reason": "measured narrow excess",
                "confidence": 0.8,
                "channel": 2,
                "parameter": "gain",
                "band": 1,
                "value": -1.5,
            },
            {
                "id": "deess-1",
                "family": "channel_deesser",
                "accepted": True,
                "reason": "measured sibilance peak",
                "confidence": 0.85,
                "channel": 3,
                "parameter": "frequency",
                "value": 7200.0,
            },
        ]
    )

    assert result["corrections_written"] == 3
    assert result["per_decision_confirmation_requested"] is False
    assert transport.values["/ch/1/fdr"] == -14.0
    stopped = session.stop()
    assert stopped["rollback"]["failures"] == []
    assert transport.values["/ch/1/fdr"] == -12.0
    assert transport.values["/ch/2/eq/1g"] == 0.0
    assert transport.values["/ch/3/gate/f"] == 6500.0

    wal_path = next((tmp_path / "wing_lab_autowrite").glob("*.jsonl"))
    rows = [json.loads(line) for line in wal_path.read_text().splitlines()]
    assert {row["phase"] for row in rows} >= {
        "armed",
        "prepared",
        "write_attempted",
        "verified",
        "rollback_verified",
        "stopped",
    }
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_event_sha256"] == previous["event_sha256"]


def test_readback_mismatch_rolls_back_prior_decisions_and_emergency_stops(tmp_path):
    transport = FakeWingTransport(
        "192.168.1.10", 2223, mismatch_address="/ch/2/eq/1g"
    )
    session = WingLabAutowriteSession(
        lambda _host, _port: transport,
        runtime_root=tmp_path,
        enabled=True,
    )
    session.arm(host="192.168.1.10", console_model="WING Rack", duration_sec=60)

    with pytest.raises(RuntimeError, match="wing_lab_readback_mismatch"):
        session.apply_decisions(
            [
                {
                    "id": "level-1", "family": "channel_fader", "channel": 1,
                    "accepted": True, "reason": "measured level above target",
                    "confidence": 0.9, "value": -14.0,
                },
                {
                    "id": "eq-1",
                    "family": "channel_eq",
                    "accepted": True,
                    "reason": "measured narrow excess",
                    "confidence": 0.8,
                    "channel": 2,
                    "parameter": "gain",
                    "band": 1,
                    "value": -1.5,
                },
            ]
        )

    assert transport.values["/ch/1/fdr"] == -12.0
    assert session.status()["emergency_stopped"] is True


@pytest.mark.parametrize("family", ["mute", "unmute", "routing", "scene", "snapshot"])
def test_lab_rejects_excluded_console_families(family):
    with pytest.raises(ValueError, match="family_not_allowed"):
        compile_wing_lab_decision(
            {
                "id": "blocked", "family": family, "accepted": True,
                "reason": "test", "confidence": 1.0, "value": 1,
            }
        )


def test_declared_lab_scope_covers_requested_non_routing_correction_families():
    assert ALLOWED_FAMILIES == {
        "channel_fader",
        "bus_fader",
        "master_fader",
        "input_gain",
        "channel_eq",
        "channel_gate",
        "channel_compressor",
        "channel_deesser",
        "channel_pan",
        "channel_phase",
        "channel_delay",
        "fx",
        "channel_send",
    }
