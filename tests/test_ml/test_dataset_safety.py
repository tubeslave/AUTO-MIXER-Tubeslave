"""Tests for backend/ml/dataset_safety.py."""
import pytest

from ml.dataset_safety import (
    SafetyLimits,
    filter_events_for_training,
    redact_training_event,
    validate_training_event_v1,
)


def _minimal_event(**kwargs):
    base = {
        "schema_version": "1.0",
        "event_id": "e1",
        "recorded_at": "2026-04-04T12:00:00Z",
        "source": "synthetic",
    }
    base.update(kwargs)
    return base


def test_validate_ok_empty_automation():
    ok, errs = validate_training_event_v1(_minimal_event())
    assert ok
    assert errs == []


def test_reject_wrong_schema_version():
    ok, errs = validate_training_event_v1(_minimal_event(schema_version="0.9"))
    assert not ok
    assert any("schema_version" in e for e in errs)


def test_reject_fader_above_ceiling():
    ev = _minimal_event(
        operator={
            "parameters_final": {"fader_dbfs": 3.0},
        },
    )
    ok, errs = validate_training_event_v1(ev)
    assert not ok
    assert any("fader" in e.lower() for e in errs)


def test_allow_fader_above_ceiling_with_override():
    ev = _minimal_event(
        operator={"parameters_final": {"fader_dbfs": 3.0}},
        safety={"explicit_override_positive_fader": True},
    )
    ok, errs = validate_training_event_v1(ev)
    assert ok


def test_reject_positive_delta_with_high_true_peak():
    ev = _minimal_event(
        observation={"true_peak_dbtp": 0.5},
        automation={
            "recommended_action": "boost",
            "parameters": {"delta_db": 2.0},
        },
    )
    ok, errs = validate_training_event_v1(ev)
    assert not ok
    assert any("true_peak" in e.lower() for e in errs)


def test_allow_positive_delta_when_true_peak_safe():
    ev = _minimal_event(
        observation={"true_peak_dbtp": -2.0},
        automation={"parameters": {"delta_db": 1.0}},
    )
    ok, errs = validate_training_event_v1(ev)
    assert ok


def test_reject_trim_out_of_range():
    ev = _minimal_event(
        automation={"parameters": {"trim_db": 20.0}},
    )
    ok, errs = validate_training_event_v1(
        ev, limits=SafetyLimits(max_gain_trim_db=12.0)
    )
    assert not ok


def test_filter_events_for_training():
    good = _minimal_event(event_id="g1")
    bad = _minimal_event(
        event_id="b1",
        operator={"parameters_final": {"fader_dbfs": 6.0}},
    )
    acc, rej = filter_events_for_training([good, bad])
    assert len(acc) == 1
    assert acc[0]["event_id"] == "g1"
    assert len(rej) == 1
    assert rej[0][0] == "b1"


def test_redact_training_event():
    ev = _minimal_event()
    ev["artist_name"] = "X"
    ev["venue_name"] = "Y"
    redact_training_event(ev)
    assert "artist_name" not in ev
    assert "venue_name" not in ev
