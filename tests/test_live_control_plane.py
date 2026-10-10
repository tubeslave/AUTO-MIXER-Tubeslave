import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import EqBandLocator, LiveMode, ProposedAction
from live_runtime.control_plane import LiveControlPlane


class FakeMixerAdapter:
    def __init__(self, initial=None, *, readback_transform=None):
        self.values = dict(initial or {})
        self.writes = []
        self.readback_transform = readback_transform

    @staticmethod
    def _key(action):
        relative_to_absolute = {
            "fader_delta_db": "fader_db",
            "eq_gain_delta_db": "eq_gain_db",
        }
        parameter = relative_to_absolute.get(action.parameter, action.parameter)
        locator_key = None
        if action.eq_locator is not None:
            locator_key = (
                action.eq_locator.band,
                action.eq_locator.frequency_hz,
                action.eq_locator.q,
            )
        return action.target, parameter, locator_key

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
    adapter = FakeMixerAdapter({("ch:1", "fader_db", None): -5.0})
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
    adapter = FakeMixerAdapter({("output:1", "routing", None): "USB/1"})
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
    adapter = FakeMixerAdapter({("output:1", "routing", None): "USB/1"})
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
    adapter = FakeMixerAdapter({("ch:1", "fader_db", None): -5.0})
    plane = LiveControlPlane(adapter)

    result = plane.execute(_action(), LiveMode.BENCH_TEST, manual_freeze=True)

    assert result.wrote is False
    assert result.authorization_reason == "frozen"
    assert adapter.writes == []


def test_relative_fader_move_resolves_against_fresh_current_value():
    adapter = FakeMixerAdapter({("main:1", "fader_db", None): -6.0})
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
    adapter = FakeMixerAdapter({("main:1", "fader_db", None): -6.0})
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


def test_relative_eq_gain_resolves_against_fresh_located_band_gain():
    locator = EqBandLocator(3, 3200.0, 1.4)
    key = ("ch:2", "eq_gain_db", (3, 3200.0, 1.4))
    adapter = FakeMixerAdapter({key: -1.0})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    action = _action(
        target="ch:2",
        parameter="eq_gain_delta_db",
        value=-0.7,
        max_step=1.0,
        eq_locator=locator,
        reason="release vocal masking",
    )

    result = plane.execute(action, LiveMode.BENCH_TEST)

    assert result.wrote is True
    assert result.verified.before == -1.0
    assert result.verified.after == -1.7
    assert result.verified.readback == -1.7
    assert adapter.writes[0].parameter == "eq_gain_db"
    assert adapter.writes[0].eq_locator == locator
    assert events[-1]["resolved_action"]["eq_locator"] == {
        "band": 3,
        "frequency_hz": 3200.0,
        "q": 1.4,
    }


def test_relative_eq_gain_cannot_exceed_its_declared_max_step_even_in_bench_test():
    locator = EqBandLocator(3, 3200.0, 1.4)
    key = ("ch:2", "eq_gain_db", (3, 3200.0, 1.4))
    adapter = FakeMixerAdapter({key: -1.0})
    plane = LiveControlPlane(adapter)
    action = _action(
        target="ch:2",
        parameter="eq_gain_delta_db",
        value=-1.2,
        max_step=0.8,
        eq_locator=locator,
    )

    result = plane.execute(action, LiveMode.BENCH_TEST)

    assert result.wrote is False
    assert result.authorization_reason == "max_step_exceeded"
    assert adapter.writes == []


def test_transport_success_is_rejected_when_readback_does_not_match():
    state = {"reads": 0}

    def stale_after_write(action, value):
        state["reads"] += 1
        if state["reads"] == 1:
            return -5.0
        return -4.5

    adapter = FakeMixerAdapter(
        {("ch:1", "fader_db", None): -5.0},
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
        {("ch:1", "fader_db", None): -5.0},
        readback_transform=quantized,
    )
    plane = LiveControlPlane(adapter, readback_tolerance=.02)

    result = plane.execute(_action(value=-4.0), LiveMode.BENCH_TEST)

    assert result.verified.accepted is True


def test_relative_fader_rollback_restores_captured_absolute_value_and_verifies():
    adapter = FakeMixerAdapter({("main:1", "fader_db", None): -6.0})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    action = _action(
        target="main:1",
        parameter="fader_delta_db",
        value=-0.5,
        max_step=0.5,
        reason="temporary headroom test",
    )

    write = plane.execute(action, LiveMode.BENCH_TEST)
    rollback = plane.rollback(write.verified, LiveMode.BENCH_TEST)

    assert rollback.wrote is True
    assert rollback.restored is True
    assert rollback.before_rollback == -6.5
    assert rollback.readback == -6.0
    assert adapter.writes[-1].parameter == "fader_db"
    assert adapter.writes[-1].value == -6.0
    assert events[-1]["event"] == "live_rollback_verified"
    assert events[-1]["target"] == -6.0


def test_rollback_restoration_bypasses_auto_safe_new_action_allowlist():
    adapter = FakeMixerAdapter({("output:1", "routing", None): "USB/1"})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    bench_action = _action(
        target="output:1",
        parameter="routing",
        value="MOD/1",
        confidence=.2,
        reversible=True,
        risk="high",
        max_step=None,
    )

    write = plane.execute(bench_action, LiveMode.BENCH_TEST)
    rollback = plane.rollback(write.verified, LiveMode.AUTO_SAFE)

    assert rollback.wrote is True
    assert rollback.authorization_reason == "rollback_authorized"
    assert rollback.restored is True
    assert rollback.readback == "USB/1"
    assert adapter.values[("output:1", "routing", None)] == "USB/1"
    assert events[-1]["event"] == "live_rollback_verified"


def test_manual_freeze_blocks_rollback_even_in_bench_test():
    adapter = FakeMixerAdapter({("ch:1", "fader_db", None): -5.0})
    plane = LiveControlPlane(adapter)
    write = plane.execute(_action(value=-4.0), LiveMode.BENCH_TEST)

    rollback = plane.rollback(write.verified, LiveMode.BENCH_TEST, manual_freeze=True)

    assert rollback.wrote is False
    assert rollback.restored is False
    assert rollback.authorization_reason == "frozen"
    assert adapter.values[("ch:1", "fader_db", None)] == -4.0
    assert len(adapter.writes) == 1


def test_non_reversible_action_has_no_rollback_write():
    adapter = FakeMixerAdapter({("ch:1", "fader_db", None): -5.0})
    events = []
    plane = LiveControlPlane(adapter, audit_sink=events.append)
    write = plane.execute(_action(value=-4.0, reversible=False), LiveMode.BENCH_TEST)

    rollback = plane.rollback(write.verified, LiveMode.BENCH_TEST)

    assert rollback.wrote is False
    assert rollback.restored is False
    assert rollback.authorization_reason == "not_reversible"
    assert len(adapter.writes) == 1
    assert events[-1]["event"] == "live_rollback_unavailable"
