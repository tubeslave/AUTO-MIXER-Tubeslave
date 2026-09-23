import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import LiveMode, ProposedAction
from live_runtime.control_plane import LiveControlPlane
from live_runtime.wing_adapter import WingWriteAdapter


class FakeWingClient:
    """Minimal callback-driven WING transport double.

    Queries emit an inbound callback using WING's common
    [display_string, normalized_value, actual_value] response shape.
    """

    def __init__(self, values=None, *, fail_queries=False, fail_writes=False, drop_queries=False):
        self.values = dict(values or {})
        self.callbacks = {}
        self.sent = []
        self.fail_queries = fail_queries
        self.fail_writes = fail_writes
        self.drop_queries = drop_queries

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            if self.fail_writes:
                return False
            self.values[address] = values[0] if len(values) == 1 else values
            return True

        if self.fail_queries:
            return False
        if self.drop_queries:
            return True

        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


def _fader_action(value=-4.0, **overrides):
    payload = dict(
        target="ch:1",
        parameter="fader_db",
        value=value,
        reason="fader migration test",
        confidence=.9,
        max_step=1.0,
        reversible=True,
        risk="low",
    )
    payload.update(overrides)
    return ProposedAction(**payload)


def test_reads_fresh_wing_fader_from_inbound_callback():
    client = FakeWingClient({"/ch/1/fdr": -5.25})
    adapter = WingWriteAdapter(client)

    value = adapter.read_value(_fader_action())

    assert value == -5.25
    assert client.sent == [("/ch/1/fdr", ())]


def test_control_plane_fader_roundtrip_uses_real_query_after_write():
    client = FakeWingClient({"/ch/1/fdr": -5.0})
    adapter = WingWriteAdapter(client)
    plane = LiveControlPlane(adapter)

    result = plane.execute(_fader_action(-4.0), LiveMode.BENCH_TEST)

    assert result.wrote is True
    assert result.verified.before == -5.0
    assert result.verified.readback == -4.0
    assert result.verified.accepted is True
    assert result.verified.rollback_value == -5.0
    assert client.sent == [
        ("/ch/1/fdr", ()),
        ("/ch/1/fdr", (-4.0,)),
        ("/ch/1/fdr", ()),
    ]


def test_missing_fresh_readback_times_out_instead_of_trusting_cache():
    client = FakeWingClient({"/ch/1/fdr": -5.0}, drop_queries=True)
    adapter = WingWriteAdapter(client, readback_timeout=.01)

    with pytest.raises(TimeoutError, match="fresh WING readback"):
        adapter.read_value(_fader_action())


def test_transport_write_failure_is_not_silently_accepted():
    client = FakeWingClient({"/ch/1/fdr": -5.0}, fail_writes=True)
    adapter = WingWriteAdapter(client)

    with pytest.raises(RuntimeError, match="write transport failed"):
        adapter.write_value(_fader_action(-4.0))


def test_only_migrated_channel_fader_surface_is_available():
    client = FakeWingClient({"/ch/1/fdr": -5.0})
    adapter = WingWriteAdapter(client)

    with pytest.raises(NotImplementedError):
        adapter.write_value(_fader_action(parameter="eq_gain_db"))
    with pytest.raises(ValueError, match="channel out of range"):
        adapter.write_value(_fader_action(target="ch:41"))
    with pytest.raises(ValueError, match="outside WING range"):
        adapter.write_value(_fader_action(value=12.0))
