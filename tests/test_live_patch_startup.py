import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import SoundcheckState
from live_runtime.feature_stream import MainFeatureEvidence
from live_runtime.main_evidence import PostConsoleMainTap
from live_runtime.patch_startup import MainTapPatchStartupCoordinator
from live_runtime.patch_verify import (
    MainTapPatchContract,
    MainTapPatchVerifier,
    MainTapRouteExpectation,
    PhysicalMainMeterEvidence,
)
from live_runtime.wing_adapter import WingWriteAdapter


class FakeWingClient:
    def __init__(self, values, *, order=None):
        self.values = dict(values)
        self.callbacks = {}
        self.sent = []
        self.order = order

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            raise AssertionError('PATCH_VERIFY startup must never write to WING')
        if self.order is not None:
            self.order.append(('route', address))
        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


class FakeMeterProvider:
    def __init__(self, evidence=None, *, error=None, order=None):
        self.evidence = evidence
        self.error = error
        self.calls = 0
        self.order = order

    def read(self):
        self.calls += 1
        if self.order is not None:
            self.order.append(('meter', self.calls))
        if self.error is not None:
            raise self.error
        return self.evidence


def _contract():
    return MainTapPatchContract(
        tap=PostConsoleMainTap(left_channel=47, right_channel=48),
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
    )


def _tap(*, peak=-6.0, rms=-18.0, timestamp=10.0):
    return MainFeatureEvidence(
        rms_dbfs=rms,
        peak_dbfs=peak,
        crest_db=max(0.0, peak - rms),
        timestamp_s=timestamp,
    )


def _physical(*, peak=-6.4, timestamp=10.05):
    return PhysicalMainMeterEvidence(
        peak_dbfs=peak,
        rms_dbfs=None,
        timestamp_s=timestamp,
        source='wing-native-main-meter:MAIN.1',
    )


def _routes(*, left_group='MAIN'):
    return {
        '/io/out/USB/46/grp': left_group,
        '/io/out/USB/46/in': 1,
        '/io/out/USB/47/grp': 'MAIN',
        '/io/out/USB/47/in': 2,
    }


def _coordinator(client, meter, *, audit_sink=None):
    return MainTapPatchStartupCoordinator(
        MainTapPatchVerifier(WingWriteAdapter(client)),
        meter,
        audit_sink=audit_sink,
    )


def test_patch_verify_success_advances_to_listen_after_route_then_meter():
    order = []
    client = FakeWingClient(_routes(), order=order)
    meter = FakeMeterProvider(_physical(), order=order)
    audit = []
    coordinator = _coordinator(client, meter, audit_sink=audit.append)

    result = coordinator.run(SoundcheckState.PATCH_VERIFY, _contract(), _tap())

    assert result.verified is True
    assert result.reason == 'verified'
    assert result.state is SoundcheckState.LISTEN
    assert meter.calls == 1
    assert [kind for kind, _ in order] == ['route', 'route', 'route', 'route', 'meter']
    assert all(values == () for _, values in client.sent)
    assert audit[-1]['state_before'] == 'patch_verify'
    assert audit[-1]['state_after'] == 'listen'
    assert audit[-1]['physical_source'] == 'wing-native-main-meter:MAIN.1'


def test_route_failure_enters_hold_without_reading_physical_meter():
    client = FakeWingClient(_routes(left_group='BUS'))
    meter = FakeMeterProvider(_physical())
    coordinator = _coordinator(client, meter)

    result = coordinator.run(SoundcheckState.PATCH_VERIFY, _contract(), _tap())

    assert result.verified is False
    assert result.state is SoundcheckState.HOLD
    assert result.reason.startswith('route_failed: route_mismatch:')
    assert meter.calls == 0
    assert result.physical_evidence is None
    assert client.sent == [
        ('/io/out/USB/46/grp', ()),
        ('/io/out/USB/46/in', ()),
    ]


def test_meter_failure_after_route_proof_enters_hold():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider(error=RuntimeError('native meter unavailable'))
    coordinator = _coordinator(client, meter)

    result = coordinator.run(SoundcheckState.PATCH_VERIFY, _contract(), _tap())

    assert result.verified is False
    assert result.state is SoundcheckState.HOLD
    assert result.reason == 'meter_failed: RuntimeError: native meter unavailable'
    assert result.verification.route.verified is True
    assert result.verification.level is None
    assert meter.calls == 1
    assert all(values == () for _, values in client.sent)


def test_invalid_meter_provider_result_fails_closed():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider(evidence={'peak_dbfs': -6.0})
    coordinator = _coordinator(client, meter)

    result = coordinator.run(SoundcheckState.PATCH_VERIFY, _contract(), _tap())

    assert result.verified is False
    assert result.state is SoundcheckState.HOLD
    assert result.reason.startswith('meter_failed: TypeError: physical Main meter provider returned dict')


def test_level_mismatch_enters_hold_with_physical_evidence_preserved():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider(_physical(peak=-10.0))
    coordinator = _coordinator(client, meter)

    result = coordinator.run(SoundcheckState.PATCH_VERIFY, _contract(), _tap(peak=-6.0))

    assert result.verified is False
    assert result.state is SoundcheckState.HOLD
    assert result.reason.startswith('level_failed: peak_mismatch:')
    assert result.verification.level is not None
    assert result.physical_evidence is meter.evidence


def test_wrong_fsm_state_is_rejected_before_any_transport_read():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider(_physical())
    coordinator = _coordinator(client, meter)

    with pytest.raises(RuntimeError, match='only in PATCH_VERIFY'):
        coordinator.run(SoundcheckState.LISTEN, _contract(), _tap())

    assert client.sent == []
    assert meter.calls == 0
