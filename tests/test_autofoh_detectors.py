"""Tests for the incremental AutoFOH musical detectors."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_analysis import build_stem_contribution_matrix, extract_analysis_features
from autofoh_detectors import (
    HarshnessExcessDetector,
    LeadMaskingAnalyzer,
    LowEndAnalyzer,
    MudExcessDetector,
    SibilanceExcessDetector,
    aggregate_stem_features,
)
from autofoh_safety import ChannelEQMove, ChannelFaderMove


def _sine(
    frequency_hz: float,
    amplitude: float = 1.0,
    sample_rate: int = 48000,
    duration_sec: float = 0.25,
):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def _build_snapshot(channel_audio, channel_stems):
    channel_features = {
        channel_id: extract_analysis_features(samples)
        for channel_id, samples in channel_audio.items()
    }
    stem_features = aggregate_stem_features(channel_features, channel_stems)
    contribution_matrix = build_stem_contribution_matrix(
        {
            stem_name: feature
            for stem_name, feature in stem_features.items()
            if stem_name != "MASTER"
        }
    )
    return channel_features, stem_features, contribution_matrix


def test_lead_masking_prefers_reducing_masking_stem_before_boosting_lead():
    channel_audio = {
        1: _sine(2200.0, amplitude=0.2),
        2: _sine(2800.0, amplitude=0.9),
    }
    channel_stems = {
        1: ["LEAD"],
        2: ["GUITARS", "MUSIC"],
    }
    channel_features, stem_features, contribution_matrix = _build_snapshot(
        channel_audio,
        channel_stems,
    )
    analyzer = LeadMaskingAnalyzer(
        masking_threshold_db=3.0,
        persistence_required_cycles=2,
    )

    analyzer.analyze(
        channel_features=channel_features,
        channel_stems=channel_stems,
        stem_features=stem_features,
        contribution_matrix=contribution_matrix,
        lead_channel_ids=[1],
        current_faders_db={1: -6.0},
    )
    result = analyzer.analyze(
        channel_features=channel_features,
        channel_stems=channel_stems,
        stem_features=stem_features,
        contribution_matrix=contribution_matrix,
        lead_channel_ids=[1],
        current_faders_db={1: -6.0},
    )

    assert result.problem is not None
    assert result.problem.problem_type == "lead_masking"
    assert result.culprit_stems_by_band["PRESENCE"] == "GUITARS"
    assert isinstance(result.candidate_actions[0], ChannelEQMove)
    assert result.candidate_actions[0].channel_id == 2
    assert result.candidate_actions[0].gain_db < 0.0


def test_mud_detector_identifies_sustained_250_500_hz_excess():
    channel_audio = {
        1: _sine(320.0, amplitude=1.0),
        2: _sine(2200.0, amplitude=0.2),
    }
    channel_stems = {
        1: ["GUITARS", "MUSIC"],
        2: ["LEAD"],
    }
    channel_features, stem_features, contribution_matrix = _build_snapshot(
        channel_audio,
        channel_stems,
    )
    detector = MudExcessDetector(threshold_db=2.5, persistence_required_cycles=2)

    detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )
    recommendation = detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )

    assert recommendation.problem is not None
    assert recommendation.problem.problem_type == "mud_excess"
    assert recommendation.problem.band_name == "MUD"
    assert recommendation.problem.stem == "GUITARS"
    assert isinstance(recommendation.candidate_actions[0], ChannelEQMove)
    assert recommendation.candidate_actions[0].channel_id == 1


def test_harshness_detector_identifies_sustained_3k_6k_excess():
    channel_audio = {
        1: _sine(4200.0, amplitude=1.0),
        2: _sine(2200.0, amplitude=0.2),
    }
    channel_stems = {
        1: ["CYMBALS"],
        2: ["LEAD"],
    }
    channel_features, stem_features, contribution_matrix = _build_snapshot(
        channel_audio,
        channel_stems,
    )
    detector = HarshnessExcessDetector(threshold_db=2.5, persistence_required_cycles=2)

    detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )
    recommendation = detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )

    assert recommendation.problem is not None
    assert recommendation.problem.problem_type == "harshness_excess"
    assert recommendation.problem.band_name == "HARSHNESS"
    assert recommendation.problem.stem == "CYMBALS"


def test_sibilance_detector_identifies_sustained_6k_10k_excess():
    channel_audio = {
        1: _sine(7500.0, amplitude=1.0),
        2: _sine(2200.0, amplitude=0.2),
    }
    channel_stems = {
        1: ["CYMBALS"],
        2: ["LEAD"],
    }
    channel_features, stem_features, contribution_matrix = _build_snapshot(
        channel_audio,
        channel_stems,
    )
    detector = SibilanceExcessDetector(threshold_db=2.5, persistence_required_cycles=2)

    detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )
    recommendation = detector.observe(
        master_features=stem_features["MASTER"],
        contribution_matrix=contribution_matrix,
        channel_features=channel_features,
        channel_stems=channel_stems,
    )

    assert recommendation.problem is not None
    assert recommendation.problem.problem_type == "sibilance_excess"
    assert recommendation.problem.band_name == "SIBILANCE"
    assert recommendation.problem.stem == "CYMBALS"


def test_low_end_analyzer_distinguishes_sub_bass_and_body_issues():
    analyzer = LowEndAnalyzer(persistence_required_cycles=1)

    sub_features, sub_stems, sub_matrix = _build_snapshot(
        {
            1: _sine(45.0, amplitude=1.0),
            2: _sine(2200.0, amplitude=0.2),
        },
        {
            1: ["BASS"],
            2: ["LEAD"],
        },
    )
    sub_result = analyzer.analyze(
        master_features=sub_stems["MASTER"],
        contribution_matrix=sub_matrix,
        channel_features=sub_features,
        channel_stems={
            1: ["BASS"],
            2: ["LEAD"],
        },
    )

    analyzer = LowEndAnalyzer(persistence_required_cycles=1)
    bass_features, bass_stems, bass_matrix = _build_snapshot(
        {
            1: _sine(90.0, amplitude=1.0),
            2: _sine(2200.0, amplitude=0.2),
        },
        {
            1: ["BASS"],
            2: ["LEAD"],
        },
    )
    bass_result = analyzer.analyze(
        master_features=bass_stems["MASTER"],
        contribution_matrix=bass_matrix,
        channel_features=bass_features,
        channel_stems={
            1: ["BASS"],
            2: ["LEAD"],
        },
    )

    analyzer = LowEndAnalyzer(persistence_required_cycles=1)
    body_features, body_stems, body_matrix = _build_snapshot(
        {
            1: _sine(180.0, amplitude=1.0),
            2: _sine(2200.0, amplitude=0.2),
        },
        {
            1: ["BASS"],
            2: ["LEAD"],
        },
    )
    body_result = analyzer.analyze(
        master_features=body_stems["MASTER"],
        contribution_matrix=body_matrix,
        channel_features=body_features,
        channel_stems={
            1: ["BASS"],
            2: ["LEAD"],
        },
    )

    assert sub_result.dominant_issue == "sub_excess"
    assert bass_result.dominant_issue == "bass_excess"
    assert body_result.dominant_issue == "body_excess"
    assert isinstance(body_result.candidate_actions[0], ChannelEQMove)


def test_lead_masking_falls_back_to_small_lead_fader_trim_without_clear_culprit():
    channel_audio = {
        1: _sine(2200.0, amplitude=0.3),
        2: _sine(2600.0, amplitude=0.35),
        3: _sine(3000.0, amplitude=0.35),
    }
    channel_stems = {
        1: ["LEAD"],
        2: ["GUITARS"],
        3: ["KEYS"],
    }
    channel_features, stem_features, contribution_matrix = _build_snapshot(
        channel_audio,
        channel_stems,
    )
    analyzer = LeadMaskingAnalyzer(
        masking_threshold_db=0.2,
        culprit_share_threshold=0.8,
        persistence_required_cycles=2,
    )

    analyzer.analyze(
        channel_features=channel_features,
        channel_stems=channel_stems,
        stem_features=stem_features,
        contribution_matrix=contribution_matrix,
        lead_channel_ids=[1],
        current_faders_db={1: -7.0},
    )
    result = analyzer.analyze(
        channel_features=channel_features,
        channel_stems=channel_stems,
        stem_features=stem_features,
        contribution_matrix=contribution_matrix,
        lead_channel_ids=[1],
        current_faders_db={1: -7.0},
    )

    assert result.problem is not None
    assert isinstance(result.candidate_actions[0], ChannelFaderMove)
    assert result.candidate_actions[0].channel_id == 1
    assert result.candidate_actions[0].is_lead is True
