"""Dry-run room compensation planner tests."""

from __future__ import annotations

import numpy as np

from room_compensation_planner import (
    RoomCompensationPlanner,
    build_room_compensation_operator_payload,
)
from system_measurement import (
    EQCorrection,
    FrequencyResponse,
    MeasurementPosition,
    MeasurementResult,
    RT60Data,
    RoomAnalysisResult,
    SystemMeasurementController,
)


def _measurement(
    *,
    overall_quality: float = 0.8,
    rt60: list[float] | None = None,
    coherence: list[float] | None = None,
    magnitude_db: list[float] | None = None,
    positions: int = 6,
) -> MeasurementResult:
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=float)
    magnitude = np.array(magnitude_db or [2.0, 3.5, 2.0, 0.5, 0.0, -0.5, -1.0, -2.0], dtype=float)
    coherence_values = np.array(coherence or [0.9] * len(freqs), dtype=float)
    position_list = [
        MeasurementPosition(position_id=idx, x=float(idx), y=0.0, height=1.4, quality_score=overall_quality)
        for idx in range(positions)
    ]
    return MeasurementResult(
        magnitude_response=FrequencyResponse(
            frequencies=freqs,
            magnitude_db=magnitude,
            phase_deg=np.zeros_like(freqs),
            coherence=coherence_values,
        ),
        phase_response=FrequencyResponse(
            frequencies=freqs,
            magnitude_db=magnitude,
            phase_deg=np.zeros_like(freqs),
            coherence=coherence_values,
        ),
        rt60=RT60Data(
            bands=[63, 125, 250, 500, 1000, 2000, 4000, 8000],
            rt60=rt60 or [1.1] * 8,
        ),
        room_analysis=RoomAnalysisResult(),
        positions=position_list,
        overall_quality=overall_quality,
    )


def test_room_compensation_plan_is_dry_run_and_bounded():
    planner = RoomCompensationPlanner()
    measurement = _measurement(magnitude_db=[6.0, 6.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    corrections = [
        EQCorrection(frequency=125.0, gain_db=-8.5, q=3.2, type="peak"),
        EQCorrection(frequency=4000.0, gain_db=4.5, q=2.8, type="peak"),
        EQCorrection(frequency=8000.0, gain_db=1.2, q=1.2, type="high_shelf"),
    ]

    plan = planner.plan(measurement, corrections)

    assert plan.dry_run_only is True
    assert plan.applied is False
    assert plan.status == "dry_run"
    assert plan.send_status == "not_sent"
    assert plan.blocked is False
    assert plan.recommended_target == "group"
    assert len(plan.filters) <= plan.filter_cap
    assert max(abs(item.gain_db) for item in plan.filters) <= plan.max_cut_db
    assert max(item.q for item in plan.filters) <= 2.6
    assert "Dry-run only" in plan.safe_override_guidance[0]


def test_room_compensation_plan_blocks_low_quality_measurement():
    planner = RoomCompensationPlanner()
    measurement = _measurement(overall_quality=0.2, coherence=[0.2] * 8, positions=2)

    plan = planner.plan(measurement, [])

    assert plan.blocked is True
    assert plan.blocked_reason == "measurement_quality_too_low"
    assert plan.filters == []
    assert "No console write was sent." in plan.message_for_user()


def test_room_compensation_operator_payload_exposes_non_applied_status_and_quality():
    planner = RoomCompensationPlanner()
    measurement = _measurement(
        rt60=[2.2] * 8,
        magnitude_db=[0.0, 0.5, 0.5, 0.5, 0.0, -0.5, -1.0, -1.5],
    )
    corrections = [
        EQCorrection(frequency=500.0, gain_db=-3.5, q=1.8, type="peak"),
        EQCorrection(frequency=4000.0, gain_db=2.5, q=1.8, type="peak"),
    ]

    plan = planner.plan(measurement, corrections)
    payload = build_room_compensation_operator_payload(plan, event_type="room_compensation_plan")

    assert payload["type"] == "room_compensation_plan"
    assert payload["dry_run_only"] is True
    assert payload["applied"] is False
    assert payload["non_applied_status"] == "dry_run"
    assert payload["send_status"] == "not_sent"
    assert payload["room_classification"] == "reverberant"
    assert payload["recommended_target"] == "matrix"
    assert payload["room_quality_indicator"] in {"good", "excellent"}
    assert payload["message_for_user"].startswith("Dry-run room compensation")


def test_system_measurement_controller_returns_room_compensation_payloads():
    controller = SystemMeasurementController()
    measurement = _measurement(magnitude_db=[6.0, 6.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    controller.measurement.analyze = lambda: measurement
    controller.correction_calculator.calculate_corrections = lambda _measurement: [
        EQCorrection(frequency=125.0, gain_db=-7.0, q=2.5, type="peak"),
        EQCorrection(frequency=4000.0, gain_db=3.0, q=1.8, type="peak"),
    ]

    result = controller.analyze_and_calculate()

    assert result["recommended_target"] == "group"
    assert result["room_compensation_plan"]["dry_run_only"] is True
    assert result["room_compensation_operator_payload"]["non_applied_status"] == "dry_run"
    assert result["room_compensation_operator_payload"]["send_status"] == "not_sent"
