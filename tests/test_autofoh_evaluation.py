"""Tests for AutoFOH evaluation policy and rollback helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_analysis import extract_analysis_features
from autofoh_evaluation import (
    AutoFOHEvaluationPolicy,
    PendingActionEvaluation,
    build_rollback_action,
    evaluate_pending_action,
)
from autofoh_models import RuntimeState
from autofoh_safety import ChannelEQMove

import numpy as np


def _sine(
    frequency_hz: float,
    amplitude: float = 1.0,
    sample_rate: int = 48000,
    duration_sec: float = 0.25,
):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def test_control_state_only_evaluation_reports_unobservable_effect():
    action = ChannelEQMove(
        channel_id=4,
        band=2,
        freq_hz=350.0,
        gain_db=-1.0,
        q=1.3,
        reason="Mud cleanup",
    )
    pending = PendingActionEvaluation(
        evaluation_id=1,
        action=action,
        registered_at=0.0,
        due_at=2.0,
        runtime_state=RuntimeState.PRE_SHOW_CHECK,
        channel_id=4,
        pre_features=extract_analysis_features(_sine(320.0, amplitude=0.8)),
        rollback_action=build_rollback_action(action, {"gain_db": 0.0}),
        evaluation_band_name="MUD",
        expected_effect="Reduce sustained mud",
    )

    outcome = evaluate_pending_action(
        pending,
        policy=AutoFOHEvaluationPolicy(
            allow_proxy_audio_evaluation_for_testing=False,
        ),
        control_state_applied=True,
        post_features=extract_analysis_features(_sine(320.0, amplitude=0.4)),
    )

    assert outcome.evaluated is True
    assert outcome.observable is False
    assert outcome.should_rollback is False
    assert "control-state change verified only" in outcome.measured_effect


def test_proxy_audio_evaluation_can_request_rollback_when_error_worsens():
    action = ChannelEQMove(
        channel_id=7,
        band=2,
        freq_hz=350.0,
        gain_db=-1.0,
        q=1.3,
        reason="Mud cleanup",
    )
    pending = PendingActionEvaluation(
        evaluation_id=2,
        action=action,
        registered_at=0.0,
        due_at=2.0,
        runtime_state=RuntimeState.PRE_SHOW_CHECK,
        channel_id=7,
        pre_features=extract_analysis_features(_sine(320.0, amplitude=0.8)),
        rollback_action=build_rollback_action(action, {"gain_db": 0.0}),
        evaluation_band_name="MUD",
        expected_effect="Reduce sustained mud",
    )

    outcome = evaluate_pending_action(
        pending,
        policy=AutoFOHEvaluationPolicy(
            allow_proxy_audio_evaluation_for_testing=True,
            allow_proxy_audio_rollback_for_testing=True,
            worsening_tolerance_db=0.1,
        ),
        control_state_applied=True,
        post_features=extract_analysis_features(_sine(320.0, amplitude=1.0)),
    )

    assert outcome.observable is True
    assert outcome.worsened is True
    assert outcome.should_rollback is True
    assert outcome.rollback_action is not None
    assert outcome.rollback_action.gain_db == 0.0
