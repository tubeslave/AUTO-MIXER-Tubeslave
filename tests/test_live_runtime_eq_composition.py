import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import ChannelFeatures, LiveMode, MixFeatures
from live_runtime.eq_locator import EqTargetEvidence
from live_runtime.service import LiveSoundcheckService, LiveStartRequest


def _request(mode):
    return LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[2],
        mode=mode,
    )


class FakeWingClient:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.callbacks = {}
        self.sent = []

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            self.values[address] = values[0] if len(values) == 1 else values
            return True
        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


class ConnectedWingEngine:
    transport = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self.started = False
        self.stopped = False
        self._real_mixer_client = type(self).transport
        self.mixer_client = self._real_mixer_client

    def start_async(self):
        self.started = True
        self.state = type("State", (), {"value": "running"})()

    def stop(self):
        self.stopped = True
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {
            "state": self.state.value,
            "mixer_connected": True,
            "audio_running": self.started and not self.stopped,
        }


def _wing_eq_state(channel=2):
    return {
        f"/ch/{channel}/eq/1f": 180.0,
        f"/ch/{channel}/eq/1q": 0.8,
        f"/ch/{channel}/eq/1g": 0.0,
        f"/ch/{channel}/eq/2f": 950.0,
        f"/ch/{channel}/eq/2q": 1.0,
        f"/ch/{channel}/eq/2g": 0.0,
        f"/ch/{channel}/eq/3f": 3150.0,
        f"/ch/{channel}/eq/3q": 1.4,
        f"/ch/{channel}/eq/3g": 2.0,
        f"/ch/{channel}/eq/4f": 7600.0,
        f"/ch/{channel}/eq/4q": 1.2,
        f"/ch/{channel}/eq/4g": 0.0,
    }


def _harsh_guitar_features():
    guitar = ChannelFeatures(
        2,
        "Guitar",
        -18.0,
        -5.0,
        13.0,
        activity=.9,
        harshness=.9,
    )
    return MixFeatures([guitar], -14.0, -4.0, 10.0)


def test_service_composes_evidence_selector_director_and_verified_eq_write():
    client = FakeWingClient(_wing_eq_state())
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(engine_factory=ConnectedWingEngine)
    service.start(_request(LiveMode.BENCH_TEST))

    hypothesis = service.propose_hypothesis(
        _harsh_guitar_features(),
        {2: "guitar"},
        eq_evidence={
            (2, "harshness"): EqTargetEvidence(
                center_frequency_hz=3300.0,
                confidence=.92,
                max_octave_distance=.35,
                preferred_q=1.4,
                source="harshness_peak_tracker",
            )
        },
    )

    assert hypothesis is not None
    assert hypothesis.name == "source_harshness"
    assert hypothesis.action.eq_locator is not None
    assert hypothesis.action.eq_locator.band == 3
    assert hypothesis.action.parameter == "eq_gain_delta_db"

    result = service.execute_action(hypothesis.action)

    assert result.wrote is True
    assert result.verified.before == 2.0
    assert result.verified.after == pytest.approx(1.4)
    assert result.verified.readback == pytest.approx(1.4)
    assert result.verified.accepted is True
    assert result.verified.rollback_value == 2.0
    assert client.values["/ch/2/eq/3g"] == pytest.approx(1.4)
    assert service.control_audit_events[0]["event"] == "live_eq_locator_selected"
    assert service.control_audit_events[-1]["event"] == "live_write_verified"


def test_low_confidence_eq_evidence_fails_closed_without_console_queries():
    client = FakeWingClient(_wing_eq_state())
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(engine_factory=ConnectedWingEngine)
    service.start(_request(LiveMode.PROPOSE))

    hypothesis = service.propose_hypothesis(
        _harsh_guitar_features(),
        {2: "guitar"},
        eq_evidence={
            (2, "harshness"): EqTargetEvidence(
                center_frequency_hz=3300.0,
                confidence=.5,
                source="harshness_peak_tracker",
            )
        },
    )

    assert hypothesis is None
    assert client.sent == []
    assert service.control_audit_events == [
        {
            "event": "live_eq_locator_unresolved",
            "channel": 2,
            "evidence": {
                "center_frequency_hz": 3300.0,
                "confidence": .5,
                "max_octave_distance": .5,
                "preferred_q": None,
                "min_q": None,
                "max_q": None,
                "source": "harshness_peak_tracker",
            },
            "locator": None,
        }
    ]
