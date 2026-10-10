import numpy as np
from audio_workbench.mixing.compression import CompressorConfig
from audio_workbench.mixing.guitar_compression_director import (
    GuitarGatePolicy,
    assess_against_baseline,
    baseline_actionability,
    guitar_dynamics_evidence,
    propose_candidates,
)


def guitar_like(sr=44100, seconds=30, unstable=False):
    n = sr * seconds
    t = np.arange(n) / sr
    if unstable:
        phase = ((t * 4).astype(int) % 2).astype(np.float32)
        amp = 0.055 + phase * 0.085
    else:
        amp = np.full(n, 0.10, np.float32)
    x = amp * np.sin(2 * np.pi * 220 * t) + amp * 0.30 * np.sin(2 * np.pi * 2200 * t)
    for c in np.arange(.1, seconds, .125):
        i = int(c * sr)
        j = min(n, i + int(.02 * sr))
        if j > i:
            w = np.hanning((j - i) * 2)[:j-i]
            x[i:j] += w * .04 * np.sin(2*np.pi*3000*np.arange(j-i)/sr)
    return x.astype(np.float32)


def baseline_cfg():
    return CompressorConfig(
        threshold_dbfs=-24,
        ratio=2,
        attack_ms=25,
        release_ms=180,
        knee_db=5,
        max_gr_db=2.8,
    )


def test_no_change_first_when_no_actionable_problem():
    x = guitar_like(unstable=False)
    ev = guitar_dynamics_evidence(x, x, 44100)
    assert ev['active_block_count'] >= 12
    action = baseline_actionability(x, x, 44100)
    proposal = propose_candidates(44100, baseline_cfg(), action)
    assert action['actionable'] is False
    assert 'no_actionable_local_dynamics_problem' in action['failures']
    assert proposal['decision'] == 'no_change'
    assert proposal['candidates'] == []


def test_actionable_problem_opens_only_bounded_timing_candidates():
    x = guitar_like(unstable=True)
    action = baseline_actionability(x, x, 44100)
    assert action['actionable'] is True
    proposal = propose_candidates(44100, baseline_cfg(), action)
    assert proposal['decision'] == 'evaluate_bounded_candidates'
    assert [c['id'] for c in proposal['candidates']] == ['preserve_pick', 'tighten_body', 'longer_sustain']
    for c in proposal['candidates']:
        cfg = c['compressor']
        assert cfg['threshold_dbfs'] == -24
        assert cfg['ratio'] == 2
        assert cfg['knee_db'] == 5
        assert cfg['max_gr_db'] == 2.8


def test_identical_candidate_cannot_fake_a_stability_win():
    x = guitar_like(unstable=True)
    result = assess_against_baseline(x, x, x.copy(), 44100, policy=GuitarGatePolicy())
    assert result['technically_survives'] is False
    assert 'guitar_local_stability_not_improved' in result['failures']
    assert result['requires_full_session_rerender'] is False
