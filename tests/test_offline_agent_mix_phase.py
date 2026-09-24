"""Signal-level regression tests for the actual offline drum phase stage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_offline_agent_mix_classification import load_offline_agent_mix


def hits(sr, seed=7):
    rng = np.random.default_rng(seed)
    audio = np.zeros(6 * sr)
    t = np.arange(int(0.1 * sr)) / sr
    for start_sec in np.arange(0.25, 5.8, 0.45):
        start = int(start_sec * sr)
        audio[start:start + len(t)] = 0.1 * rng.normal(size=len(t)) * np.exp(-40 * t)
    return audio.astype(np.float32)


def setup_pair(monkeypatch, sr, close, far):
    mod = load_offline_agent_mix()
    plans = {}
    for channel, name, role in [(1, "SNARE", "snare"), (2, "OH L", "overhead"),
                                (3, "OH R", "overhead")]:
        plans[channel] = mod.ChannelPlan(
            path=Path(name + ".wav"), name=name, instrument=role,
            pan=0.0, hpf=40.0, target_rms_db=-25.0,
        )
    samples = {plans[1].path: close, plans[2].path: far, plans[3].path: far * 0.8}
    monkeypatch.setattr(mod, "read_mono", lambda path: (samples[path], sr))
    return mod, plans


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
@pytest.mark.parametrize("polarity", [1, -1])
def test_phase_stage_recovers_known_delay_and_polarity(monkeypatch, sr, polarity):
    close = hits(sr)
    offset = int(round(0.003 * sr))
    far = np.pad(close, (offset, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, polarity * close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "applied"
    corrected = mod.delay_signal(polarity * close, sr, plans[1].delay_ms)
    if plans[1].phase_invert:
        corrected = -corrected
    assert plans[1].delay_ms == pytest.approx(offset * 1000 / sr, abs=1000 / sr)
    assert plans[1].phase_invert is (polarity == -1)
    assert mod._norm_corr(far, corrected) > 0.99
    assert all(plans[ch].delay_ms == 0 for ch in (2, 3))
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "unchanged"
    assert plans[1].delay_ms == pytest.approx(offset * 1000 / sr, abs=1000 / sr)
    assert plans[1].phase_invert is (polarity == -1)


@pytest.mark.parametrize("seed", [7, 23, 91])
def test_unrelated_sources_do_not_receive_phase_changes(monkeypatch, seed):
    sr = 48000
    mod, plans = setup_pair(monkeypatch, sr, hits(sr, seed), hits(sr, seed + 1))
    mod.apply_drum_phase_alignment(plans, sr)
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False


def test_inconsistent_arrivals_preserve_previous_phase_state(monkeypatch):
    sr = 48000
    close = hits(sr)
    far = np.pad(close, (144, 0))[:len(close)]
    far[len(far) // 2:] = np.pad(close, (384, 0))[:len(close)][len(far) // 2:]
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    plans[1].delay_ms = 0.5
    plans[1].phase_invert = True
    mod.apply_drum_phase_alignment(plans, sr)
    assert plans[1].delay_ms == 0.5
    assert plans[1].phase_invert is True


def test_verified_absolute_delay_can_replace_an_overshooting_previous_value(monkeypatch):
    sr = 48000
    close = hits(sr)
    far = np.pad(close, (144, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    plans[1].delay_ms = 5.0
    mod.apply_drum_phase_alignment(plans, sr)
    assert plans[1].delay_ms == pytest.approx(3.0, abs=1000 / sr)


@pytest.mark.parametrize("silent", [False, True])
def test_aligned_or_silent_sources_remain_unchanged(monkeypatch, silent):
    sr = 48000
    close = np.zeros(6 * sr) if silent else hits(sr)
    mod, plans = setup_pair(monkeypatch, sr, close, close)
    mod.apply_drum_phase_alignment(plans, sr)
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False


@pytest.mark.parametrize("gain", [0.01, 0.1, 4.0])
def test_delay_and_polarity_recovery_is_gain_independent(monkeypatch, gain):
    sr = 48000
    close = hits(sr)
    far = np.pad(close, (144, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, -close * gain, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "applied"
    assert plans[1].delay_ms == pytest.approx(3.0, abs=1000 / sr)
    assert plans[1].phase_invert is True


@pytest.mark.parametrize("delay_ms", [0.2, 11.0])
def test_bounded_delays_recover_at_sample_resolution(monkeypatch, delay_ms):
    sr = 48000
    close = hits(sr)
    samples = round(delay_ms * sr / 1000)
    far = np.pad(close, (samples, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "applied"
    assert plans[1].delay_ms == pytest.approx(samples * 1000 / sr, abs=1000 / sr)


@pytest.mark.parametrize("delay_ms", [-3.0, 12.0, 17.0])
def test_advance_or_boundary_or_excess_delay_does_not_mutate(monkeypatch, delay_ms):
    sr = 48000
    close = hits(sr)
    shifted = np.pad(close, (round(abs(delay_ms) * sr / 1000), 0))[:len(close)]
    target, far = (shifted, close) if delay_ms < 0 else (close, shifted)
    mod, plans = setup_pair(monkeypatch, sr, target, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "skipped"
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False


def test_insufficient_hits_preserve_existing_state(monkeypatch):
    sr = 48000
    close = hits(sr)[:2 * sr]
    far = np.pad(close, (144, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    plans[1].delay_ms, plans[1].phase_invert = 0.5, True
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["reason"] == "insufficient_valid_hits"
    assert plans[1].delay_ms == 0.5
    assert plans[1].phase_invert is True


def test_inconsistent_polarity_is_not_applied(monkeypatch):
    sr = 48000
    close = hits(sr)
    far = np.pad(close, (144, 0))[:len(close)]
    far[len(far) // 2:] *= -1
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["reason"] == "weak_or_inconsistent_correlation"
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False


def test_strong_consistent_phat_peak_is_not_enough_for_unrelated_sources(monkeypatch):
    sr = 48000
    mod, plans = setup_pair(monkeypatch, sr, hits(sr, 7), hits(sr, 23))
    monkeypatch.setattr(mod.GCCPHATAnalyzer, "compute_delay",
                        lambda *args: SimpleNamespace(delay_ms=-3.0, psr=10.0))
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["reason"] == "weak_or_inconsistent_correlation"
    assert plans[1].delay_ms == 0.0


def test_low_band_sum_vetoes_broadband_correlation_improvement(monkeypatch):
    sr = 48000
    high = load_offline_agent_mix().highpass(hits(sr), sr, 1500.0)
    low = np.zeros_like(high)
    t = np.arange(int(0.1 * sr)) / sr
    for sec in np.arange(0.25, 5.8, 0.45):
        start = int(sec * sr)
        low[start:start + len(t)] = 0.04 * np.sin(2 * np.pi * 90 * t) * np.exp(-40 * t)
    close = high + low
    far = np.pad(high, (144, 0))[:len(high)] + low
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["corr_candidate"] > report[0]["corr_current"]
    assert report[0]["reason"] == "low_band_sum_worsened"
    assert min(report[0]["sum_band_gain_db"]) < -0.05
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False


@pytest.mark.parametrize("bad_reference", [False, True])
def test_nonfinite_audio_preserves_state_and_has_json_safe_report(monkeypatch, bad_reference):
    import json

    sr = 48000
    close, far = hits(sr), hits(sr)
    (far if bad_reference else close)[sr] = np.nan
    mod, plans = setup_pair(monkeypatch, sr, close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["reason"] == "nonfinite_audio"
    assert plans[1].delay_ms == 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_delay_cap_rejected_before_mutation(monkeypatch, cap):
    mod, plans = setup_pair(monkeypatch, 48000, hits(48000), hits(48000))
    with pytest.raises(ValueError, match="max_delay_ms"):
        mod.apply_drum_phase_alignment(plans, 48000, cap)
    assert not plans[1].phase_notes


@pytest.mark.parametrize("polarity", [1, -1])
def test_recorded_snare_recovers_injected_delay(monkeypatch, polarity):
    import soundfile as sf

    corpus = Path(__file__).resolve().parents[1] / "external/automix-toolkit/drums-test-rock/tracks"
    files = list(corpus.glob("02_snare*.wav"))
    if not files:
        pytest.skip("local recorded snare corpus unavailable")
    close, sr = sf.read(files[0], dtype="float32")
    offset = round(0.003 * sr)
    far = np.pad(close, (offset, 0))[:len(close)]
    mod, plans = setup_pair(monkeypatch, sr, polarity * close, far)
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["status"] == "applied"
    corrected = mod.delay_signal(polarity * close, sr, plans[1].delay_ms)
    if plans[1].phase_invert:
        corrected = -corrected
    assert mod._norm_corr(far, corrected) > 0.99


@pytest.mark.parametrize("field", ["delay_ms", "psr"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_phat_measurement_is_rejected(monkeypatch, field, value):
    sr = 48000
    close = hits(sr)
    mod, plans = setup_pair(monkeypatch, sr, close, np.pad(close, (144, 0))[:len(close)])
    values = dict(delay_ms=-3.0, psr=10.0)
    values[field] = value
    monkeypatch.setattr(mod.GCCPHATAnalyzer, "compute_delay",
                        lambda *args: SimpleNamespace(**values))
    report = mod.apply_drum_phase_alignment(plans, sr)
    assert report[0]["valid_hits"] == 0
    assert plans[1].delay_ms == 0.0
    assert plans[1].phase_invert is False
