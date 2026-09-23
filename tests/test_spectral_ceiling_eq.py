import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from heuristics.spectral_ceiling_eq import (
    SpectralCeilingEQAnalyzer,
    SpectralCeilingEQConfig,
    generate_noise_slope_curve,
    guide_slope_db_per_oct,
    load_spectral_ceiling_profiles,
    merge_spectral_proposal_into_eq_bands,
    select_spectral_ceiling_profile,
)
from mix_agent.analysis.loader import infer_stem_role


CLI_TIMEOUT_SEC = 60.0


def _tone(
    frequency_hz: float,
    *,
    amplitude: float = 0.2,
    sample_rate: int = 48000,
    duration_sec: float = 1.0,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def _dual_tone(a_hz: float, b_hz: float, *, a_amp: float = 0.2, b_amp: float = 0.2):
    return _tone(a_hz, amplitude=a_amp) + _tone(b_hz, amplitude=b_amp)


def test_noise_slope_target_curve_supports_white_pink_brown_and_custom():
    freqs = np.asarray([500.0, 1000.0, 2000.0])

    pink = generate_noise_slope_curve(
        freqs,
        slope_db_per_oct=guide_slope_db_per_oct("pink", 0.0),
        reference_freq_hz=1000.0,
        reference_db=0.0,
    )
    brown = generate_noise_slope_curve(
        freqs,
        slope_db_per_oct=guide_slope_db_per_oct("brown", 0.0),
        reference_freq_hz=1000.0,
        reference_db=0.0,
    )
    custom = generate_noise_slope_curve(
        freqs,
        slope_db_per_oct=guide_slope_db_per_oct("custom", -1.5),
        reference_freq_hz=1000.0,
        reference_db=0.0,
    )

    assert np.allclose(pink, [3.0, 0.0, -3.0], atol=0.001)
    assert np.allclose(brown, [6.0, 0.0, -6.0], atol=0.001)
    assert np.allclose(custom, [1.5, 0.0, -1.5], atol=0.001)
    assert guide_slope_db_per_oct("white", -9.0) == 0.0


def test_profile_selection_and_foreground_tilt_for_lead_vocal():
    profiles = load_spectral_ceiling_profiles()
    profile = select_spectral_ceiling_profile("leadVocal", profiles)

    assert profile.role == "lead_vocal"
    assert profile.front_back_position == "foreground"
    assert profile.effective_slope_db_per_oct > profile.slope_db_per_oct


def test_new_drums_synth_percussion_playback_profiles_load_and_alias():
    profiles = load_spectral_ceiling_profiles()

    expected = {
        "tom",
        "floor_tom",
        "hihat",
        "ride",
        "synth",
        "percussion",
        "playback",
        "drums_bus",
    }
    assert expected.issubset(profiles)
    assert select_spectral_ceiling_profile("F Tom", profiles).role == "floor_tom"
    assert select_spectral_ceiling_profile("Hi-Hat", profiles).role == "hihat"
    assert select_spectral_ceiling_profile("Synth+Percussion+Pad", profiles).role == "playback"
    assert select_spectral_ceiling_profile("pad", profiles).role == "playback"
    assert select_spectral_ceiling_profile("Drums Bus", profiles).role == "drums_bus"
    assert select_spectral_ceiling_profile("drums", profiles).role == "drums_bus"


def test_offline_loader_keeps_requested_roles_specific():
    assert infer_stem_role("floor_tom.wav") == "floor_tom"
    assert infer_stem_role("hi-hat_top.wav") == "hihat"
    assert infer_stem_role("ride.wav") == "ride"
    assert infer_stem_role("perc_loop.wav") == "percussion"
    assert infer_stem_role("synth_percussion_pad.wav") == "playback"


def test_dry_run_logs_proposal_without_apply():
    analyzer = SpectralCeilingEQAnalyzer(
        SpectralCeilingEQConfig(dry_run=True, correction_strength=1.0)
    )
    proposal = analyzer.analyze(
        _dual_tone(260.0, 3500.0, a_amp=0.3, b_amp=0.2),
        instrument_role="lead_vocal",
        sample_rate=48000,
        track_id="Lead Vox",
        role_confidence=1.0,
    )

    assert proposal.enabled is True
    assert proposal.dry_run is True
    assert proposal.should_apply is False
    assert any(item["reason"].startswith("dry_run") for item in proposal.skipped)


def test_backing_vocal_demasks_active_lead_vocal_band():
    analyzer = SpectralCeilingEQAnalyzer(SpectralCeilingEQConfig(correction_strength=1.0))
    proposal = analyzer.analyze(
        _tone(3200.0, amplitude=0.3),
        instrument_role="backing_vocal",
        sample_rate=48000,
        track_id="BGV",
        role_confidence=1.0,
        lead_vocal_active=True,
        lead_vocal_confidence=0.9,
    )

    demask = [
        band for band in proposal.bands
        if band.rule == "vocal_priority_demasking"
    ]
    assert demask
    assert demask[0].gain_db < 0.0


def test_master_bus_correction_is_limited_to_one_db():
    analyzer = SpectralCeilingEQAnalyzer(
        SpectralCeilingEQConfig(correction_strength=1.0, master_bus_max_abs_gain_db=1.0)
    )
    proposal = analyzer.analyze(
        _tone(3500.0, amplitude=0.4),
        instrument_role="mix_bus",
        sample_rate=48000,
        track_id="mix",
        role_confidence=1.0,
    )

    assert proposal.bands
    assert all(abs(band.gain_db) <= 1.0 for band in proposal.bands)
    assert proposal.safety["max_abs_gain_db"] == 1.0


def _recovery_noise(seed):
    from scipy.signal import lfilter

    audio = lfilter(
        [0.049922035, -0.095993537, 0.050612699, -0.004408786],
        [1, -2.494956002, 2.017265875, -0.522189400],
        np.random.default_rng(seed).standard_normal(4 * 48000),
    )
    return 0.3 * audio / np.max(np.abs(audio))


def _repair_mix_spectrum(audio, sample_rate):
    from mix_agent.actions.base import apply_parametric_eq
    from mix_agent.agent.decision_loop import _spectral_actions_for_proposal

    proposal = SpectralCeilingEQAnalyzer().analyze(
        audio, instrument_role="mix_bus", sample_rate=sample_rate,
    )
    repaired = audio.copy()
    for action in _spectral_actions_for_proposal(proposal, "mix"):
        if action.mode != "safe_apply":
            continue
        assert action.action_type == "parametric_eq"
        parameters = action.parameters
        repaired = apply_parametric_eq(
            repaired, sample_rate, parameters["frequency_hz"],
            parameters["gain_db"], parameters["q"],
        )
    return repaired


@pytest.mark.parametrize("seed", [7, 23, 91])
def test_mix_bus_recovery_does_not_cut_healthy_bands_around_a_scoop(seed):
    from recovery_trial import run_trial, spectral_distance_db

    original = _recovery_noise(seed)
    report = run_trial(original, 48000, _repair_mix_spectrum)
    by_defect = {result.defect: result for result in report.results}

    assert by_defect["scooped_mids"].recovery_ratio >= -0.01
    for defect in ("mud_300", "harsh_3k", "boxy_500"):
        assert by_defect[defect].recovery_ratio > 0.05
    assert all(result.recovery_ratio >= -0.01 for result in report.results)
    assert spectral_distance_db(
        _repair_mix_spectrum(original, 48000), original, 48000,
    ) < 0.01


def test_mix_bus_recovery_decisions_are_invariant_to_input_gain():
    from recovery_trial import DEFAULT_CORPUS, apply_defect

    analyzer = SpectralCeilingEQAnalyzer()
    damaged = apply_defect(_recovery_noise(7), 48000, DEFAULT_CORPUS[0])
    decisions = []
    for gain_db in (0.0, -6.0, -12.0):
        proposal = analyzer.analyze(
            damaged * 10 ** (gain_db / 20),
            instrument_role="mix_bus", sample_rate=48000,
        )
        assert proposal.should_apply
        decisions.append([(band.name, band.gain_db) for band in proposal.bands])
    assert decisions[0] == decisions[1] == decisions[2]


@pytest.mark.parametrize("slope", [0.0, -3.0, -6.0])
def test_broadband_reference_respects_target_slope_and_local_deficits(slope):
    from heuristics.spectral_ceiling_eq import _reference_db

    frequencies = np.geomspace(80.0, 14000.0, 120)
    spectrum = -24.0 + slope * np.log2(frequencies / 1000.0)
    spectrum[(frequencies > 700.0) & (frequencies < 1400.0)] -= 4.0
    reference = _reference_db(
        frequencies, spectrum, 1000.0, slope_db_per_oct=slope,
    )
    assert reference == pytest.approx(-24.0)


def test_config_disabled_returns_no_applyable_moves():
    proposal = SpectralCeilingEQAnalyzer(
        SpectralCeilingEQConfig(enabled=False)
    ).analyze(
        _tone(260.0, amplitude=0.3),
        instrument_role="lead_vocal",
        sample_rate=48000,
        track_id="Lead",
        role_confidence=1.0,
    )

    assert proposal.enabled is False
    assert proposal.bands == []
    assert proposal.should_apply is False


def test_merge_skips_duplicate_existing_cut_in_same_zone():
    analyzer = SpectralCeilingEQAnalyzer(SpectralCeilingEQConfig(correction_strength=1.0))
    proposal = analyzer.analyze(
        _tone(260.0, amplitude=0.4),
        instrument_role="lead_vocal",
        sample_rate=48000,
        track_id="Lead",
        role_confidence=1.0,
    )
    existing = [
        (90.0, 0.0, 1.0),
        (260.0, -3.0, 1.2),
        (3200.0, 0.0, 1.4),
        (8000.0, 0.0, 1.2),
    ]

    merged, report = merge_spectral_proposal_into_eq_bands(existing, proposal)

    assert merged[1][1] == -3.0
    assert any("already addresses" in item["reason"] for item in report["skipped"])


def test_inspect_spectral_ceiling_cli_prints_human_report(tmp_path):
    audio_path = tmp_path / "lead.wav"
    sf.write(audio_path, _dual_tone(260.0, 3500.0), 48000)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "automixer.tools.inspect_spectral_ceiling",
            "--input",
            str(audio_path),
            "--role",
            "lead_vocal",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=CLI_TIMEOUT_SEC,
    )

    assert "[SPECTRAL_CEILING_EQ]" in result.stdout
    assert "measured_tilt" in result.stdout
