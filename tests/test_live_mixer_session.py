"""Tests for canonical live mixer discovery and connection ownership."""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

from mixer_discovery import DiscoveredMixer  # noqa: E402
from live_runtime.mixer_session import (  # noqa: E402
    LiveMixerConfig,
    LiveMixerSession,
    LiveMixerSessionError,
)


class FakeClient:
    def __init__(self, *, connect_result=True, connected_after=True):
        self.connect_result = connect_result
        self.is_connected = False
        self.connected_after = connected_after
        self.connect_calls = []
        self.disconnect_calls = 0

    def connect(self, timeout=0):
        self.connect_calls.append(timeout)
        self.is_connected = bool(self.connected_after)
        return self.connect_result

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False


def test_explicit_wing_target_skips_discovery_and_owns_connection_once():
    audits = []
    discovery_calls = []
    created = []

    def discover(**kwargs):
        discovery_calls.append(kwargs)
        raise AssertionError("explicit target must not invoke discovery")

    def factory(target, midi_base_channel):
        created.append((target, midi_base_channel))
        return FakeClient()

    session = LiveMixerSession(
        LiveMixerConfig(
            mixer_type="wing_rack",
            mixer_ip="192.168.1.50",
            auto_discover=True,
        ),
        discover=discover,
        client_factory=factory,
        audit_sink=audits.append,
    )

    client = session.start()
    assert discovery_calls == []
    assert session.start() is client
    assert len(created) == 1
    target, midi_base_channel = created[0]
    assert target.mixer_type == "wing"
    assert target.ip == "192.168.1.50"
    assert target.port == 2223
    assert target.discovery_method == "explicit"
    assert midi_base_channel == 0
    assert client.connect_calls == [10.0]
    assert session.status().connected is True

    assert session.stop() is True
    assert session.stop() is False
    assert client.disconnect_calls == 1
    assert [event["event"] for event in audits] == [
        "live_mixer_connected",
        "live_mixer_disconnected",
    ]


def test_discovery_resolves_missing_target_without_legacy_engine_policy():
    discovered = DiscoveredMixer(
        mixer_type="dlive",
        ip="10.0.0.70",
        port=51329,
        name="dLive S5000",
        tls=True,
        discovery_method="tcp_probe",
        response_time_ms=11.0,
    )
    discovery_calls = []
    targets = []

    def discover(**kwargs):
        discovery_calls.append(kwargs)
        return discovered

    def factory(target, midi_base_channel):
        targets.append((target, midi_base_channel))
        return FakeClient()

    session = LiveMixerSession(
        LiveMixerConfig(
            mixer_type="dlive",
            mixer_ip=None,
            midi_base_channel=3,
            scan_subnet=True,
            discovery_timeout_s=1.25,
        ),
        discover=discover,
        client_factory=factory,
    )

    session.start()
    assert discovery_calls == [
        {
            "preferred_type": "dlive",
            "preferred_ip": None,
            "scan_subnet": True,
            "timeout": 1.25,
        }
    ]
    target, midi_base_channel = targets[0]
    assert target.mixer_type == "dlive"
    assert target.ip == "10.0.0.70"
    assert target.port == 51329
    assert target.tls is True
    assert target.discovery_method == "tcp_probe"
    assert midi_base_channel == 3
    session.stop()


def test_discovery_type_conflict_fails_closed_before_client_construction():
    factory_calls = []

    def discover(**kwargs):
        return DiscoveredMixer(
            mixer_type="dlive",
            ip="10.0.0.70",
            port=51328,
        )

    def factory(target, midi_base_channel):
        factory_calls.append((target, midi_base_channel))
        return FakeClient()

    session = LiveMixerSession(
        LiveMixerConfig(mixer_type="wing", mixer_ip=None),
        discover=discover,
        client_factory=factory,
    )

    with pytest.raises(LiveMixerSessionError, match="contradicts explicit mixer_type"):
        session.start()

    assert factory_calls == []
    assert session.client is None
    assert session.connected is False


def test_incomplete_target_without_discovery_fails_closed():
    session = LiveMixerSession(
        LiveMixerConfig(
            mixer_type="wing",
            mixer_ip=None,
            auto_discover=False,
        ),
        client_factory=lambda target, midi: FakeClient(),
    )

    with pytest.raises(LiveMixerSessionError, match="incomplete"):
        session.start()

    assert session.client is None
    assert session.connected is False


def test_failed_connect_releases_partial_client_and_audits_failure():
    audits = []
    client = FakeClient(connect_result=False, connected_after=False)

    session = LiveMixerSession(
        LiveMixerConfig(
            mixer_type="wing",
            mixer_ip="192.168.1.50",
            connect_timeout_s=4.5,
        ),
        client_factory=lambda target, midi: client,
        audit_sink=audits.append,
    )

    with pytest.raises(LiveMixerSessionError, match="connect returned false"):
        session.start()

    assert client.connect_calls == [4.5]
    assert client.disconnect_calls == 1
    assert session.client is None
    assert session.connected is False
    assert audits[-1]["event"] == "live_mixer_connect_failed"


def test_config_rejects_unknown_mixer_type_and_bad_port():
    with pytest.raises(ValueError, match="Unsupported mixer type"):
        LiveMixerConfig(mixer_type="mystery-console")

    with pytest.raises(ValueError, match="1..65535"):
        LiveMixerConfig(mixer_type="wing", mixer_ip="192.168.1.50", mixer_port=70000)
