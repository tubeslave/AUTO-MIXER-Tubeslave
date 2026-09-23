import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import LiveMode, ProposedAction
from live_runtime.control_plane import LiveControlPlane


class FakeMixerAdapter:
    def __init__(self, initial=None, *, readback_transform=None):
        self.values = dict(initial or {})
        self.writes = []
        self.readback_transform = readback_transform

    @staticmethod
    def _key(action):
        parameter = "fader_db" if action.parameter == "fader_delta_db" else action.parameter
        return action.target, parameter

    def read_value(self, action):
        value = self.values.get(self._key(action))
        if self.readback_transform is not None:
            return self.readback_transform(action, value)
        return value

    def write_value(self, action):
        self.writes.append(action)
        self.values[self._key(action)] = action.value


def _action(**overrides):
    payload = dict(
        target="ch:1",
        parameter="fader_db",
        value=-4.0,
        reason="test move",
        confidence=.9,
        max_step=1.0,
        reversible=True,
        risk="low",
    )
    payload.update(overrides)
    return ProposedAction(**payload)


def test_observe_never_writes_but_keeps_before_value_for_audit():
    adapter = FakeMixerAdapter({("ch:1", "fader_db"): -5.0})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)

    result = plane.execute(_action(), LiveMode.OBSERVE)

    assert result.wrote is False
    assert result.authorization_reason == "writes_disabled"
    assert result.verified.before == -5.0
    assert result.verified.accepted is False
    assert adapter.writes == []
    assert events[-1]["event"] == "live_write_blocked"


def test_bench_test_bypasses_production_routing_risk_and_confidence_gates():
    adapter = FakeMixerAdapter({("output:1", "routing"): "USB/1"})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    dangerous_for_production = _action(
        target="output:1",
        parameter="routing",
        value="MOD/1",
        confidence=.1,
        reversible=False,
        risk="high",
        max_step=None,
    )

    result = plane.execute(dangerous_for_production, LiveMode.BENCH_TEST)

    assert result.wrote is True
    assert result.authorization_reason == "bench_test_unrestricted"
    assert result.verified.accepted is True
    assert result.verified.readback == "MOD/1"
    assert len(adapter.writes) == 1
    assert events[-1]["event"] == "live_write_verified"
    assert events[-1]["mode"] == "bench_test"


def test_auto_safe_blocks_same_routing_action_before_transport_write():
    adapter = FakeMixerAdapter({("output:1", "routing"): "USB/1"})
    plane = LiveControlPlane(adapter)
    action = _action(
        target="output:1",
        parameter="routing",
        value="MOD/1",
        confidence=.99,
        reversible=True,
    )

    result = plane.execute(action, LiveMode.AUTO_SAFE)

    assert result.wrote is False
    assert result.authorization_reason == "parameter_not_allowlisted"
    assert adapter.writes == []


def test_manual_freeze_still_wins_in_bench_test():
    adapter = FakeMixerAdapter({("ch:1", "fader_db"): -5.0})
    plane = LiveControlPlane(adapter)

    result = plane.execute(_action(), LiveMode.BENCH_TEST, manual_freeze=True)

    assert result.wrote is False
    assert result.authorization_reason == "frozen"
    assert adapter.writes == []


def test_relative_fader_move_resolves_against_fresh_current_value():
    adapter = FakeMixerAdapter({("main:1", "fader_db"): -6.0})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    action = _action(
        target="main:1",
        parameter="fader_delta_db",
        value=-0.5,
        max_step=0.5,
        reason="restore main headroom",
    )

    result = plane.execute(action, LiveMode.BENCH_TEST)

    assert result.wrote is True
    assert result.verified.before == -6.0
    assert result.verified.after == -6.5
    assert result.verified.readback == -6.5
    assert result.verified.proposal.parameter == "fader_delta_db"
    assert result.verified.proposal.value == -0.5
    assert adapter.writes[0].parameter == "fader_db"
    assert adapter.writes[0].value == -6.5
    assert events[-1]["resolved_action"]["parameter"] == "fader_db"
    assert events[-1]["resolved_action"]["value"] == -6.5


def test_relative_fader_move_cannot_exceed_its_declared_max_step_even_in_bench_test():
    adapter = FakeMixerAdapter({("main:1", "fader_db"): -6.0})
    plane = LiveControlPlane(adapter)
    action = _action(
        target="main:1",
        parameter="fader_delta_db",
        value=-1.0,
        max_step=0.5,
    )

    result = plane.execute(action, LiveMode.BENCH_TEST)

    assert result.wrote is False
    assert result.authorization_reason == "max_step_exceeded"
    assert result.verified.before == -6.0
    assert adapter.writes == []


def test_transport_success_is_rejected_when_readback_does_not_match():
    state = {"reads": 0}

    def stale_after_write(action, value):
        state["reads"] += 1
        if state["reads"] == 1:
            return -5.0
        return -4.5

    adapter = FakeMixerAdapter(
        {("ch:1", "fader_db"): -5.0},
        readback_transform=stale_after_write,
    )
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append, readback_tolerance=.02)

    result = plane.execute(_action(value=-4.0), LiveMode.BENCH_TEST)

    assert result.wrote is True
    assert result.verified.accepted is False
    assert result.verified.rollback_value == -5.0
    assert result.verified.readback == -4.5
    assert events[-1]["event"] == "live_write_mismatch"


def test_small_numeric_readback_quantization_is_accepted():
    def quantized(action, value):
        if value is None:
            return value
        return float(value) + .01

    adapter = FakeMixerAdapter(
        {("ch:1", "fader_db"): -5.0},
        readback_transform=quantized,
    )
    plane = LiveControlPlane(adapter, readback_tolerance=.02)

    result = plane.execute(_action(value=-4.0), LiveMode.BENCH_TEST)

    assert result.verified.accepted is True
