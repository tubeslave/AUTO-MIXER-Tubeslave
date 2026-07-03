"""Replay-oriented room compensation validation scenarios."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pytest

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo, ChannelSnapshot
from backend.system_measurement import (
    EQCorrection,
    FrequencyResponse,
    MeasurementPosition,
    MeasurementResult,
    RT60Data,
    RoomAnalysisResult,
    SystemMeasurementController,
)
from signal_metrics import ChannelMetrics

from tests.replay_support import ReplayMixer, install_transport_blockers


def _measurement(
    *,
    overall_quality: float = 0.87,
    rt60_sec: float | Iterable[float] = 0.9,
    magnitude_db: list[float] | None = None,
    coherence: list[float] | None = None,
    positions: int = 6,
) -> MeasurementResult:
    frequencies = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=float)
    magnitude = np.array(
        magnitude_db
        if magnitude_db is not None
        else [2.0, 3.5, 2.0, 0.5, 0.0, -0.5, -1.0, -2.0],
        dtype=float,
    )
    coh = np.array(coherence or [0.94] * len(frequencies), dtype=float)
    if isinstance(rt60_sec, (list, tuple)):
        rt60_values = list(rt60_sec)
    else:
        rt60_values = [float(rt60_sec)] * 8
    position_list = [
        MeasurementPosition(position_id=idx, x=float(idx), y=0.0, height=1.4, quality_score=overall_quality)
        for idx in range(positions)
    ]
    return MeasurementResult(
        magnitude_response=FrequencyResponse(
            frequencies=frequencies,
            magnitude_db=magnitude,
            phase_deg=np.zeros_like(frequencies),
            coherence=coh,
        ),
        phase_response=FrequencyResponse(
            frequencies=frequencies,
            magnitude_db=magnitude,
            phase_deg=np.zeros_like(frequencies),
            coherence=coh,
        ),
        rt60=RT60Data(
            bands=[63, 125, 250, 500, 1000, 2000, 4000, 8000],
            rt60=rt60_values,
        ),
        room_analysis=RoomAnalysisResult(),
        positions=position_list,
        overall_quality=overall_quality,
    )


def _catalog_cases() -> list[dict[str, Any]]:
    return [
        {
            "scenario": "reflective_room",
            "measurement": _measurement(rt60_sec=2.8, magnitude_db=[0.0, 0.5, 0.0, 0.5, 0.0, -0.5, -1.0, -1.5], positions=8),
            "expected_classification": "reverberant",
            "expected_recommended_target": "matrix",
            "expected_controller_target": "Use Group EQ + Matrix delay for large room",
        },
        {
            "scenario": "conference_hall",
            "measurement": _measurement(rt60_sec=0.9, magnitude_db=[1.2, 1.5, 1.0, 0.4, 0.0, -0.2, -0.4, -0.8], positions=8),
            "expected_classification": "balanced",
            "expected_recommended_target": "master",
            "expected_controller_target": "Use Master EQ for small room",
        },
        {
            "scenario": "stage_monitor",
            "measurement": _measurement(rt60_sec=0.6, magnitude_db=[9.0, 8.5, 6.5, 1.0, 0.0, -0.5, -1.0, -2.0], positions=7),
            "expected_classification": "boomy",
            "expected_recommended_target": "group",
            "expected_controller_target": "Use Master EQ for small room",
        },
        {
            "scenario": "noisy_venue",
            "measurement": _measurement(
                overall_quality=0.5,
                rt60_sec=(1.1, 1.5, 1.4, 1.0, 0.8, 1.3, 1.6, 1.2),
                coherence=[0.2] * 8,
                magnitude_db=[4.0, 3.0, 2.5, -1.5, 0.0, -1.0, -2.5, -2.0],
                positions=1,
            ),
            "expected_classification": "boomy",
            "expected_recommended_target": "group",
            "expected_controller_target": "Use Master or Group EQ",
        },
        {
            "scenario": "mixed_speech_music",
            "measurement": _measurement(
                rt60_sec=(1.2, 1.1, 1.4, 1.3, 1.2, 1.1, 1.4, 1.3),
                magnitude_db=[7, -7, 6, -6, 5, -5, 4, -4],
                positions=8,
            ),
            "expected_classification": "uneven",
            "expected_recommended_target": "group",
            "expected_controller_target": "Use Master or Group EQ",
        },
    ]


@pytest.mark.parametrize("case", _catalog_cases())
def test_room_compensation_catalog_replay_paths(case: dict[str, Any]):
    controller = SystemMeasurementController()
    measurement = case["measurement"]
    controller.last_result = measurement

    corrections = controller.correction_calculator.calculate_corrections(measurement)
    plan = controller.room_compensation_planner.plan(measurement, corrections)

    assert controller.get_recommended_target() == case["expected_controller_target"]
    assert plan.summary.classification == case["expected_classification"]
    assert plan.recommended_target == case["expected_recommended_target"]
    assert plan.filter_cap >= len(plan.filters)
    if case["scenario"] == "noisy_venue":
        assert plan.blocked is True
        assert plan.blocked_reason == "measurement_confidence_too_low"
        assert plan.filters == []
    else:
        assert plan.blocked is False
        assert plan.filters
    assert len(plan.filters) <= plan.filter_cap
    expected_max_q = {"master": 2.2, "group": 2.6, "matrix": 2.0}[plan.recommended_target]
    for item in plan.filters:
        assert 0.7 <= float(item.q) <= expected_max_q
        assert 1.0 <= float(item.frequency) <= 20000.0
        assert abs(float(item.gain_db)) >= 0.75


@pytest.mark.parametrize("case", _catalog_cases())
def test_room_compensation_replay_plans_are_deterministic(case: dict[str, Any]):
    controller = SystemMeasurementController()
    measurement = case["measurement"]
    corrections = [
        EQCorrection(frequency=125.0, gain_db=-6.5, q=2.6, type="peak"),
        EQCorrection(frequency=500.0, gain_db=4.0, q=2.0, type="peak"),
    ]

    controller.last_result = measurement
    first = controller.room_compensation_planner.plan(measurement, corrections).to_dict()
    second = controller.room_compensation_planner.plan(measurement, corrections).to_dict()

    assert first == second


def test_room_compensation_recommendations_preserve_read_only_safety_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
):
    install_transport_blockers(monkeypatch)
    mixer = ReplayMixer(scenario="room_compensation_read_only")
    engine = AutoSoundcheckEngine(auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    confidence_by_case = {
        True: 0.8,
        False: 0.65,
    }
    for recognized, true_peak in [(True, -16.0), (False, -27.0)]:
        metrics = ChannelMetrics(channel=1)
        metrics.level.true_peak_dbtp = true_peak
        metrics.level.rms_db = -33.0
        metrics.level.crest_factor_db = 14.0

        engine.channels[1] = ChannelInfo(
            channel=1,
            name="Room Main",
            preset="room",
            recognized=recognized,
            has_signal=True,
            peak_db=true_peak,
            rms_db=-33.0,
            metrics=metrics,
            original_snapshot=ChannelSnapshot(channel=1, gain_db=0.0),
        )

        recs = engine.get_gain_recommendations()
        rec = recs[1]

        assert rec["live_apply_safe"] is False
        assert rec["dry_run_only"] is True
        assert rec["metadata"]["trim_bounds_db"]["min"] == -12.0
        assert rec["metadata"]["trim_bounds_db"]["max"] == 12.0
        assert rec["metadata"]["auto_apply_enabled"] is False
        assert rec["confidence"] == confidence_by_case[recognized]

    assert mixer.incident_records() == []
