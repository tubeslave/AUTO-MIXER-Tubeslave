import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.main_evidence import PostConsoleMainTap
from live_runtime.patch_verify import (
    MainTapPatchContract,
    MainTapPatchVerifier,
    MainTapRouteExpectation,
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
    tap = PostConsoleMainTap(left_slot=47, right_slot=48)
    return MainTapPatchContract(
        tap=tap,
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
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
            tap=PostConsoleMainTap(left_slot=47),
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
            tap=PostConsoleMainTap(left_slot=47),
            routes=(MainTapRouteExpectation(usb_slot=47, source_channel=1),),
        )
    )

    assert result.verified is False
    assert 'RuntimeError' in result.reason
    assert client.sent == [('/io/out/USB/46/grp', ())]


def test_contract_must_cover_exact_reserved_tap_slots():
    tap = PostConsoleMainTap(left_slot=47, right_slot=48)

    with pytest.raises(ValueError, match='exactly cover reserved tap slots'):
        MainTapPatchContract(
            tap=tap,
            routes=(MainTapRouteExpectation(usb_slot=47, source_channel=1),),
        )


def test_main_tap_expectation_rejects_non_main_source_group():
    with pytest.raises(ValueError, match='source_group MAIN'):
        MainTapRouteExpectation(usb_slot=47, source_group='BUS', source_channel=1)
