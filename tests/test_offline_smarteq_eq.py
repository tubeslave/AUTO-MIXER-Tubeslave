from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from offline_smarteq_eq.integrated_lufs import (
    apply_aiak_integrated_lufs_correction,
    compute_aiak_level_plane_corrections,
)
from offline_smarteq_eq.method import (
    EQBand,
    OfflineSmartEQConfig,
    apply_offline_smarteq_eq,
    design_offline_smarteq_eq,
)
from offline_smarteq_eq.render_mix import run_offline_smarteq_gainstaged_mix


def _sine(freq_hz: float, *, sample_rate: int = 48000, seconds: float = 1.0, amp: float = 0.25):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


def _band_energy(audio: np.ndarray, sample_rate: int, freq_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float32) * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / float(sample_rate))
    mask = (freqs >= freq_hz * 0.85) & (freqs <= freq_hz * 1.15)
    return float(np.mean(spectrum[mask])) if np.any(mask) else 0.0


def test_design_generates_offline_eq_bands_for_boxy_vocal():
    sample_rate = 48000
    signal = (
        _sine(240.0, sample_rate=sample_rate, amp=0.45)
        + _sine(2500.0, sample_rate=sample_rate, amp=0.06)
    ).astype(np.float32)

    config = OfflineSmartEQConfig(fft_size=4096, hop_size=2048, max_active_bands=6)
    report = design_offline_smarteq_eq(
        signal,
        sample_rate,
        "leadVocal",
        config,
    )

    assert report.method_id == "offline_smarteq_style_spectral_eq_v2"
    assert report.band_slots == 24
    assert len(report.active_bands) <= config.max_active_bands
    assert any(band.band_type == "highpass" for band in report.active_bands)
    assert any(band.gain_db < 0.0 for band in report.active_bands)
    peaking = [band for band in report.active_bands if band.band_type == "peaking"]
    for index, band in enumerate(peaking):
        for other in peaking[index + 1 :]:
            spacing = abs(np.log2(band.frequency_hz / other.frequency_hz))
            assert spacing >= config.min_band_spacing_oct - 1e-6


def test_apply_peaking_cut_reduces_target_band_energy():
    sample_rate = 48000
    signal = _sine(250.0, sample_rate=sample_rate, seconds=1.0, amp=0.5)
    before = _band_energy(signal, sample_rate, 250.0)
    processed = apply_offline_smarteq_eq(
        signal,
        sample_rate,
        [EQBand("peaking", 250.0, -6.0, 1.0, "test cut")],
    )
    after = _band_energy(processed, sample_rate, 250.0)
    assert after < before * 0.65


def test_aiak_integrated_lufs_correction_moves_level_toward_target_without_clipping():
    sample_rate = 48000
    signal = np.stack(
        [
            _sine(440.0, sample_rate=sample_rate, seconds=2.0, amp=0.05),
            _sine(660.0, sample_rate=sample_rate, seconds=2.0, amp=0.05),
        ],
        axis=1,
    )
    corrected, report = apply_aiak_integrated_lufs_correction(
        signal,
        sample_rate,
        target_lufs=-18.0,
        max_gain_db=6.0,
        sample_peak_ceiling_dbfs=-1.0,
    )

    assert corrected.shape == signal.shape
    assert report.method_id == "aiak_rolling_integrated_lufs_correction_v1"
    assert 0.0 < report.applied_gain_db <= 6.0
    assert report.post_integrated_lufs > report.pre_integrated_lufs
    assert report.post_sample_peak_dbfs <= -1.0 + 1e-3


def test_aiak_level_plane_correction_uses_integrated_lufs_and_role_targets():
    reports = compute_aiak_level_plane_corrections(
        [
            {"channel": 1, "role": "leadVocal", "integrated_lufs": -28.0},
            {"channel": 2, "role": "overheads", "integrated_lufs": -48.0},
            {"channel": 3, "role": "backingVocal", "integrated_lufs": -100.0},
        ],
        base_lufs=-24.0,
        fader_ceiling_db=0.0,
    )

    by_channel = {report.channel: report for report in reports}
    assert by_channel[1].method_id == "aiak_integrated_lufs_level_plane_correction_v1"
    assert by_channel[1].target_lufs == -22.0
    assert by_channel[1].active is True
    assert by_channel[2].instrument_key == "overheads"
    assert by_channel[2].correction_db <= 4.0
    assert by_channel[3].active is False
    assert by_channel[3].correction_db == 0.0


def test_render_combines_input_trim_gainstaging_and_offline_eq(tmp_path: Path):
    sample_rate = 48000
    input_dir = tmp_path / "SONG  REPA"
    input_dir.mkdir()
    sf.write(str(input_dir / "Vocal.wav"), _sine(250.0, sample_rate=sample_rate, amp=0.25), sample_rate)
    sf.write(str(input_dir / "Bass.wav"), _sine(90.0, sample_rate=sample_rate, amp=0.2), sample_rate)

    output = tmp_path / "render.wav"
    report = tmp_path / "render.json"
    summary = run_offline_smarteq_gainstaged_mix(
        input_dir,
        output,
        report,
        config_path=Path("config/default_config.json"),
        analysis_passes=1,
        chunk_sec=0.2,
        eq_max_active_bands=4,
    )

    assert output.exists()
    assert report.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["methods_applied_in_order"] == [
        "input_trim_preamp_character",
        "offline_smarteq_style_spectral_eq_v2",
        "aiak_integrated_lufs_level_plane_correction_v1",
    ]
    assert summary["track_count"] == 2
    assert data["aiak_level_plane"]["base_lufs"] == -24.0
    assert data["mix_integrated_lufs_correction"] is None
    assert isinstance(data["final_integrated_lufs"], float)
    assert all(track["eq_report"]["active_bands"] for track in data["tracks"])
    assert all(track["aiak_integrated_lufs_level_plane_correction"] for track in data["tracks"])
