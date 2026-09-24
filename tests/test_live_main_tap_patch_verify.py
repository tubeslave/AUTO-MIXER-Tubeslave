import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.feature_stream import MainFeatureEvidence
from live_runtime.main_evidence import PostConsoleMainTap
from live_runtime.patch_verify import (
    MainTapLevelCoherencePolicy,
    MainTapLevelCoherenceVerifier,
    MainTapPatchContract,
    MainTapPatchGateVerifier,
    MainTapPatchVerifier,
    MainTapRouteExpectation,
    PhysicalMainMeterEvidence,
)
from live_runtime.wing_adapter import WingWriteAdapter


class FakeWingClient:
    def __init__(self, values=None, *, drop_queries=False, fail_queries=False):
        self.values = dict(values or {})
        self.callbacks = {}
        self.sent = []
        self.drop_queries = drop_queries
        self.fail_queries = fail_queries

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            raise AssertionError('PATCH_VERIFY must never write to WING')
        if self.fail_queries:
            return False
        if self.drop_queries:
            return True
        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


def _stereo_contract():
    tap = PostConsoleMainTap(left_channel=47, right_channel=48)
    return MainTapPatchContract(
        tap=tap,
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
    )


def _tap_evidence(*, peak=-6.0, rms=-18.0, timestamp=10.0):
    return MainFeatureEvidence(
        rms_dbfs=rms,
        peak_dbfs=peak,
        crest_db=max(0.0, peak - rms),
        timestamp_s=timestamp,
    )


def _physical_evidence(*, peak=-6.4, rms=None, timestamp=10.05):
    return PhysicalMainMeterEvidence(
        peak_dbfs=peak,
        rms_dbfs=rms,
        timestamp_s=timestamp,
        source='wing-main-meter',
    )


def test_fresh_stereo_main_routes_verify_without_any_write():
    client = FakeWingClient({
        '/io/out/USB/46/grp': 'MAIN',
        '/io/out/USB/46/in': 1,
        '/io/out/USB/47/grp': 'MAIN',
        '/io/out/USB/47/in': 2,
    })
    verifier = MainTapPatchVerifier(WingWriteAdapter(client))

    result = verifier.verify(_stereo_contract())

    assert result.verified is True
    assert result.reason == 'verified'
    assert [(r.output_number, r.source_group, r.source_channel) for r in result.observed] == [
        (47, 'MAIN', 1),
        (48, 'MAIN', 2),
    ]
    assert client.sent == [
        ('/io/out/USB/46/grp', ()),
        ('/io/out/USB/46/in', ()),
        ('/io/out/USB/47/grp', ()),
        ('/io/out/USB/47/in', ()),
    ]


def test_route_mismatch_fails_closed_and_does_not_query_later_slot():
    client = FakeWingClient({
        '/io/out/USB/46/grp': 'BUS',
        '/io/out/USB/46/in': 1,
        '/io/out/USB/47/grp': 'MAIN',
        '/io/out/USB/47/in': 2,
    })
    verifier = MainTapPatchVerifier(WingWriteAdapter(client))

    result = verifier.verify(_stereo_contract())

    assert result.verified is False
    assert result.reason == 'route_mismatch: USB 47 expected MAIN 1, read BUS 1'
    assert client.sent == [
        ('/io/out/USB/46/grp', ()),
        ('/io/out/USB/46/in', ()),
    ]


def test_missing_fresh_route_readback_fails_closed_instead_of_using_cache():
    client = FakeWingClient(
        {
            '/io/out/USB/46/grp': 'MAIN',
            '/io/out/USB/46/in': 1,
        },
        drop_queries=True,
    )
    verifier = MainTapPatchVerifier(WingWriteAdapter(client, readback_timeout=0.01))

    result = verifier.verify(
        MainTapPatchContract(
            tap=PostConsoleMainTap(left_channel=47),
            routes=(MainTapRouteExpectation(usb_slot=47, source_channel=1),),
        )
    )

    assert result.verified is False
    assert result.reason.startswith('readback_failed: USB 47: TimeoutError:')
    assert result.observed == ()
    assert client.sent == [('/io/out/USB/46/grp', ())]


def test_transport_query_failure_is_reported_as_failed_patch_proof():
    client = FakeWingClient(fail_queries=True)
    verifier = MainTapPatchVerifier(WingWriteAdapter(client))

    result = verifier.verify(
        MainTapPatchContract(
            tap=PostConsoleMainTap(left_channel=47),
            routes=(MainTapRouteExpectation(usb_slot=47, source_channel=1),),
        )
    )

    assert result.verified is False
    assert 'RuntimeError' in result.reason
    assert client.sent == [('/io/out/USB/46/grp', ())]


def test_contract_must_cover_exact_reserved_tap_slots():
    tap = PostConsoleMainTap(left_channel=47, right_channel=48)

    with pytest.raises(ValueError, match='exactly cover reserved tap slots'):
        MainTapPatchContract(
            tap=tap,
            routes=(MainTapRouteExpectation(usb_slot=47, source_channel=1),),
        )


def test_main_tap_expectation_rejects_non_main_source_group():
    with pytest.raises(ValueError, match='source_group MAIN'):
        MainTapRouteExpectation(usb_slot=47, source_group='BUS', source_channel=1)


def test_level_coherence_accepts_close_peak_evidence_without_requiring_rms():
    result = MainTapLevelCoherenceVerifier().verify(
        _tap_evidence(peak=-6.0, rms=-18.0, timestamp=10.0),
        _physical_evidence(peak=-6.4, rms=None, timestamp=10.05),
    )

    assert result.verified is True
    assert result.reason == 'verified'
    assert result.physical_source == 'wing-main-meter'
    assert result.timestamp_skew_s == pytest.approx(0.05)
    assert result.peak_delta_db == pytest.approx(0.4)
    assert result.rms_delta_db is None


def test_level_coherence_rejects_stale_meter_sample():
    result = MainTapLevelCoherenceVerifier().verify(
        _tap_evidence(timestamp=10.0),
        _physical_evidence(timestamp=10.2),
    )

    assert result.verified is False
    assert result.reason.startswith('timestamp_skew:')
    assert result.timestamp_skew_s == pytest.approx(0.2)


def test_level_coherence_cannot_be_proved_in_near_silence():
    result = MainTapLevelCoherenceVerifier().verify(
        _tap_evidence(peak=-80.0, rms=-90.0),
        _physical_evidence(peak=-80.2),
    )

    assert result.verified is False
    assert result.reason.startswith('signal_too_low:')


def test_level_coherence_rejects_peak_mismatch():
    result = MainTapLevelCoherenceVerifier().verify(
        _tap_evidence(peak=-6.0),
        _physical_evidence(peak=-9.0),
    )

    assert result.verified is False
    assert result.reason.startswith('peak_mismatch:')
    assert result.peak_delta_db == pytest.approx(3.0)


def test_level_coherence_optional_rms_is_checked_when_present():
    result = MainTapLevelCoherenceVerifier().verify(
        _tap_evidence(peak=-6.0, rms=-18.0),
        _physical_evidence(peak=-6.4, rms=-21.0),
    )

    assert result.verified is False
    assert result.reason.startswith('rms_mismatch:')
    assert result.rms_delta_db == pytest.approx(3.0)


def test_level_coherence_can_require_independent_rms_evidence():
    verifier = MainTapLevelCoherenceVerifier(
        MainTapLevelCoherencePolicy(require_rms=True)
    )

    result = verifier.verify(_tap_evidence(), _physical_evidence(rms=None))

    assert result.verified is False
    assert result.reason == 'rms_missing: policy requires independent RMS evidence'


def test_physical_meter_evidence_rejects_nonfinite_values():
    with pytest.raises(ValueError, match='peak_dbfs must be finite'):
        PhysicalMainMeterEvidence(
            peak_dbfs=float('nan'),
            timestamp_s=10.0,
            source='wing-main-meter',
        )


def test_complete_patch_gate_requires_route_and_level_proof_without_writes():
    client = FakeWingClient({
        '/io/out/USB/46/grp': 'MAIN',
        '/io/out/USB/46/in': 1,
        '/io/out/USB/47/grp': 'MAIN',
        '/io/out/USB/47/in': 2,
    })
    gate = MainTapPatchGateVerifier(
        MainTapPatchVerifier(WingWriteAdapter(client))
    )

    result = gate.verify(
        _stereo_contract(),
        _tap_evidence(peak=-6.0, rms=-18.0, timestamp=10.0),
        _physical_evidence(peak=-6.4, rms=-18.6, timestamp=10.05),
    )

    assert result.verified is True
    assert result.reason == 'verified'
    assert result.route.verified is True
    assert result.level is not None and result.level.verified is True
    assert all(values == () for _, values in client.sent)


def test_complete_patch_gate_short_circuits_level_proof_when_route_fails():
    client = FakeWingClient({
        '/io/out/USB/46/grp': 'BUS',
        '/io/out/USB/46/in': 1,
    })
    gate = MainTapPatchGateVerifier(
        MainTapPatchVerifier(WingWriteAdapter(client))
    )

    result = gate.verify(
        _stereo_contract(),
        _tap_evidence(),
        _physical_evidence(peak=-20.0),
    )

    assert result.verified is False
    assert result.reason.startswith('route_failed: route_mismatch:')
    assert result.level is None
    assert client.sent == [
        ('/io/out/USB/46/grp', ()),
        ('/io/out/USB/46/in', ()),
    ]
