import os
import subprocess
import sys

import numpy as np
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
    )

    assert "[SPECTRAL_CEILING_EQ]" in result.stdout
    assert "measured_tilt" in result.stdout
