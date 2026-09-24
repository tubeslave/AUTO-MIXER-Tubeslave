import numpy as np
import pytest

from audio_workbench.mixing.bass_compression_director import (
    _event_windows, measure_event_body, propose_bass_candidates, technical_screen,
)
from audio_workbench.mixing.compression import CompressorConfig


def bass_source(sr=8000, seconds=8):
    n = sr * seconds
    x = np.zeros(n, dtype=np.float32)
    for i, start in enumerate(range(sr // 2, n - sr // 2, sr // 2)):
        length = min(sr // 3, n - start)
        t = np.arange(length) / sr
        amp = .08 + .025 * (i % 5)
        env = np.exp(-7 * t)
        x[start:start + length] += (amp * env * np.sin(2 * np.pi * 82.4 * t)).astype(np.float32)
    return x


def test_proposer_returns_bounded_unranked_human_only_family():
    x = bass_source()
    result = propose_bass_candidates(x, 8000)
    assert result['schema'] == 'bass-compression-director-v1'
    assert len(result['candidates']) == 3
    assert 'winner' not in result and 'ranking' not in result
    assert result['requires_human_listening'] and not result['baseline_eligible']
    for item in result['candidates']:
        cfg = CompressorConfig(**item['compressor'])
        assert 5 <= cfg.attack_ms <= 25
        assert 75 <= cfg.release_ms <= 180
        assert cfg.max_gr_db == 5
        assert item['calibration']['max_gr_unchanged'] is True
        assert item['requires_human_listening'] and not item['baseline_eligible']


def test_fixed_event_measurement_is_level_invariant_for_spread_and_contrast():
    x = bass_source(); windows = _event_windows(x, 8000)
    a = measure_event_body(x, windows)
    b = measure_event_body(x * np.float32(.25), windows)
    assert a['valid_windows'] >= 8
    assert b['body_level_p90_minus_p10_db'] == pytest.approx(a['body_level_p90_minus_p10_db'], abs=1e-5)
    assert b['median_attack_peak_to_body_rms_db'] == pytest.approx(a['median_attack_peak_to_body_rms_db'], abs=1e-5)


def test_technical_screen_rejects_deliberate_note_body_instability():
    x = bass_source(); bad = x.copy(); windows = _event_windows(x, 8000)
    assert len(windows) >= 8
    for index, (_, _, start, end) in enumerate(windows):
        if index % 2:
            bad[start:end] *= np.float32(.2)
    result = technical_screen(x, bad, x, 8000)
    assert not result['passed']
    assert 'body_stability_regressed' in result['failures']
    assert not result['baseline_eligible']


def test_technical_screen_accepts_identical_only_for_contextual_audition():
    x = bass_source()
    result = technical_screen(x, x.copy(), x, 8000)
    assert result['passed']
    assert result['requires_full_mix_rerender']
    assert result['requires_human_listening']
    assert not result['baseline_eligible']


def test_silence_fails_closed_as_insufficient_evidence():
    x = np.zeros(8000 * 2, dtype=np.float32)
    result = technical_screen(x, x, x, 8000)
    assert not result['passed']
    assert 'insufficient_event_evidence' in result['failures']
