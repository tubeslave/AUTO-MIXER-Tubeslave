import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import ChannelFeatures, EqBandLocator, MixFeatures
from live_runtime.decision_engine import propose_one
from live_runtime.eq_locator import (
    EqTargetEvidence,
    RealtimeEqLocatorSelector,
    select_existing_eq_band,
)
from live_runtime.wing_adapter import WingWriteAdapter


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


def _wing_eq_state(channel=2):
    return {
        f"/ch/{channel}/eq/1f": 180.0,
        f"/ch/{channel}/eq/1q": 0.8,
        f"/ch/{channel}/eq/2f": 950.0,
        f"/ch/{channel}/eq/2q": 1.0,
        f"/ch/{channel}/eq/3f": 3150.0,
        f"/ch/{channel}/eq/3q": 1.4,
        f"/ch/{channel}/eq/4f": 7600.0,
        f"/ch/{channel}/eq/4q": 1.2,
    }


def test_wing_adapter_reads_all_eq_band_fingerprints_from_fresh_callbacks():
    client = FakeWingClient(_wing_eq_state())
    adapter = WingWriteAdapter(client)

    locators = adapter.read_eq_locators(2)

    assert locators == [
        EqBandLocator(1, 180.0, 0.8),
        EqBandLocator(2, 950.0, 1.0),
        EqBandLocator(3, 3150.0, 1.4),
        EqBandLocator(4, 7600.0, 1.2),
    ]
    assert client.sent == [
        ("/ch/2/eq/1f", ()), ("/ch/2/eq/1q", ()),
        ("/ch/2/eq/2f", ()), ("/ch/2/eq/2q", ()),
        ("/ch/2/eq/3f", ()), ("/ch/2/eq/3q", ()),
        ("/ch/2/eq/4f", ()), ("/ch/2/eq/4q", ()),
    ]


def test_selector_chooses_existing_band_nearest_explicit_spectral_evidence():
    bands = [
        EqBandLocator(1, 180.0, 0.8),
        EqBandLocator(2, 950.0, 1.0),
        EqBandLocator(3, 3150.0, 1.4),
        EqBandLocator(4, 7600.0, 1.2),
    ]
    evidence = EqTargetEvidence(
        center_frequency_hz=3300.0,
        confidence=.91,
        max_octave_distance=.35,
        preferred_q=1.5,
        source="harshness_peak_tracker",
    )

    assert select_existing_eq_band(bands, evidence) == EqBandLocator(3, 3150.0, 1.4)


def test_selector_fails_closed_when_no_existing_band_is_close_enough():
    bands = [
        EqBandLocator(1, 180.0, 0.8),
        EqBandLocator(2, 950.0, 1.0),
        EqBandLocator(3, 1600.0, 1.4),
        EqBandLocator(4, 7600.0, 1.2),
    ]
    evidence = EqTargetEvidence(
        center_frequency_hz=3200.0,
        confidence=.95,
        max_octave_distance=.3,
    )

    assert select_existing_eq_band(bands, evidence) is None


def test_q_constraints_can_reject_closest_frequency_without_moving_q():
    bands = [
        EqBandLocator(2, 2600.0, 1.5),
        EqBandLocator(3, 3150.0, 0.5),
        EqBandLocator(4, 3600.0, 2.0),
    ]
    evidence = EqTargetEvidence(
        center_frequency_hz=3200.0,
        confidence=.9,
        max_octave_distance=.5,
        min_q=1.0,
        max_q=3.0,
        preferred_q=2.0,
    )

    assert select_existing_eq_band(bands, evidence) == EqBandLocator(4, 3600.0, 2.0)


def test_low_confidence_evidence_does_not_query_console():
    client = FakeWingClient(_wing_eq_state())
    selector = RealtimeEqLocatorSelector(WingWriteAdapter(client), min_confidence=.75)
    evidence = EqTargetEvidence(center_frequency_hz=3200.0, confidence=.6)

    assert selector.select(2, evidence) is None
    assert client.sent == []


def test_fresh_selected_locator_can_make_harshness_hypothesis_actionable():
    client = FakeWingClient(_wing_eq_state())
    selector = RealtimeEqLocatorSelector(WingWriteAdapter(client))
    evidence = EqTargetEvidence(
        center_frequency_hz=3300.0,
        confidence=.92,
        max_octave_distance=.35,
        preferred_q=1.4,
        source="harshness_peak_tracker",
    )
    locator = selector.select(2, evidence)

    guitar = ChannelFeatures(2, "Guitar", -18, -5, 13, activity=.9, harshness=.9)
    features = MixFeatures([guitar], -14, -4, 10)
    hypothesis = propose_one(
        features,
        {2: "guitar"},
        eq_locators={(2, "harshness"): locator} if locator else {},
    )

    assert locator == EqBandLocator(3, 3150.0, 1.4)
    assert hypothesis is not None
    assert hypothesis.name == "source_harshness"
    assert hypothesis.action.eq_locator == locator
    assert hypothesis.action.parameter == "eq_gain_delta_db"


def test_invalid_evidence_and_channel_fail_before_selection():
    with pytest.raises(ValueError, match="center_frequency_hz"):
        EqTargetEvidence(center_frequency_hz=0.0, confidence=.9)
    with pytest.raises(ValueError, match="confidence"):
        EqTargetEvidence(center_frequency_hz=3200.0, confidence=1.1)

    client = FakeWingClient(_wing_eq_state())
    adapter = WingWriteAdapter(client)
    with pytest.raises(ValueError, match="channel out of range"):
        adapter.read_eq_locators(41)
