import numpy as np

from audio_workbench.mixing.compression import CompressorConfig, LinkedCompressor
from audio_workbench.mixing.kick_compression_director import (
    KickGatePolicy,
    assess_against_baseline,
    kick_event_centers,
    kick_transient_evidence,
    propose_baseline_aware_candidates,
)


def source(seconds=12):
    sr = 44100
    t = np.arange(sr * seconds) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    amplitudes = np.linspace(.45, .9, int(seconds / .4))
    for index, point in enumerate(np.arange(.2, seconds - .1, .4)):
        start = int(point * sr)
        count = min(int(.28 * sr), len(audio) - start)
        local = np.arange(count) / sr
        audio[start:start + count] += amplitudes[index % len(amplitudes)] * (
            np.sin(2 * np.pi * 62 * local) * np.exp(-local * 16)
            + .22 * np.sin(2 * np.pi * 2500 * local) * np.exp(-local * 75)
        )
    return audio


def baseline_cfg():
    return CompressorConfig(
        threshold_dbfs=-30,
        ratio=3,
        attack_ms=22,
        release_ms=115,
        knee_db=5,
        max_gr_db=4.5,
        rms_ms=3,
    )


def test_event_windows_are_fixed_and_metrics_finite():
    audio = source()
    centers = kick_event_centers(audio, 44100)
    assert len(centers) >= 24
    baseline, _ = LinkedCompressor(44100, baseline_cfg()).process(audio)
    evidence = kick_transient_evidence(audio, baseline, 44100)
    assert evidence['event_count'] >= 24
    assert all(np.isfinite(evidence[key]) for key in (
        'body_level_spread_db',
        'median_attack_body_db',
        'quiet_hit_level_dbfs',
        'between_hit_floor_dbfs',
    ))


def test_gate_accepts_clear_body_improvement_without_attack_or_spill_damage():
    audio = source()
    baseline, _ = LinkedCompressor(44100, baseline_cfg()).process(audio)
    centers = kick_event_centers(audio, 44100)
    candidate = baseline.copy()
    bodies = []
    for center in centers:
        low = center + int(.025 * 44100)
        high = min(len(candidate), center + int(.095 * 44100))
        if high > low:
            bodies.append(np.sqrt(np.mean(candidate[low:high].astype(float) ** 2) + 1e-30))
    target = np.median(bodies)
    for center, body_rms in zip(centers, bodies):
        low = center + int(.025 * 44100)
        high = min(len(candidate), center + int(.095 * 44100))
        gain = np.clip(target / max(body_rms, 1e-12), .85, 1.18)
        candidate[low:high] *= gain
    result = assess_against_baseline(
        audio,
        baseline,
        candidate,
        44100,
        policy=KickGatePolicy(
            min_body_spread_improvement_db=.01,
            max_attack_body_loss_db=2.0,
            max_quiet_hit_loss_db=2.0,
            max_between_hit_floor_increase_db=2.0,
        ),
    )
    assert result['body_spread_delta_db'] < 0
    assert result['technically_survives']


def test_gate_rejects_attack_loss_and_candidate_family_is_bounded():
    audio = source()
    baseline, _ = LinkedCompressor(44100, baseline_cfg()).process(audio)
    candidate = baseline.copy()
    for center in kick_event_centers(audio, 44100):
        low = max(0, center - int(.008 * 44100))
        high = min(len(candidate), center + int(.018 * 44100))
        candidate[low:high] *= .65
    result = assess_against_baseline(
        audio,
        baseline,
        candidate,
        44100,
        policy=KickGatePolicy(min_body_spread_improvement_db=-100),
    )
    assert 'kick_attack_body_contrast_reduced' in result['failures']
    family = propose_baseline_aware_candidates(44100, baseline_cfg())
    assert [candidate['id'] for candidate in family['candidates']] == [
        'more_punch', 'tighter_body', 'longer_body'
    ]
    for candidate in family['candidates']:
        assert candidate['compressor']['threshold_dbfs'] == baseline_cfg().threshold_dbfs
        assert candidate['compressor']['ratio'] == baseline_cfg().ratio
        assert not candidate['baseline_eligible']
