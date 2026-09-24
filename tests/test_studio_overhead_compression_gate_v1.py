from __future__ import annotations

import numpy as np
import pytest

from audio_workbench.mixing.compression import CompressorConfig
from audio_workbench.mixing.overhead_director import (
    OverheadPolicy,
    assess_against_baseline,
    baseline_actionability,
    dynamics_evidence,
    propose_compression_candidates,
    render_candidate,
)


def _synthetic_oh(sr: int = 16000, seconds: float = 24.0, *, unstable: bool) -> np.ndarray:
    n = int(sr * seconds)
    rng = np.random.default_rng(9281 if unstable else 9280)
    t = np.arange(n) / sr
    common = rng.normal(0, 0.005, n)
    left = common + rng.normal(0, 0.002, n)
    right = 0.88 * common + rng.normal(0, 0.0023, n)
    event_times = np.arange(0.8, seconds - 0.6, 0.48)
    for i, et in enumerate(event_times):
        c = int(et * sr)
        m = min(n - c, int(.36 * sr))
        u = np.arange(m) / sr
        env = np.exp(-u / .13)
        carrier_l = rng.normal(0, 1, m)
        carrier_r = .65 * carrier_l + .76 * rng.normal(0, 1, m)
        if unstable:
            amp = [0.055, 0.085, 0.16, 0.32][i % 4]
            if i % 11 == 0:
                amp *= 1.45
        else:
            amp = 0.12 * (1.0 + 0.035 * np.sin(i * .7))
        left[c:c+m] += amp * env * carrier_l
        right[c:c+m] += amp * env * carrier_r
    left += 0.002 * np.sin(2*np.pi*430*t)
    right += 0.002 * np.sin(2*np.pi*437*t + .3)
    return np.column_stack([left, right]).astype(np.float32)


def test_stable_overheads_stop_at_no_change():
    sr = 16000
    x = _synthetic_oh(sr, unstable=False)
    policy = OverheadPolicy(actionable_peak_excess_p95_db=4.0,
                            actionable_peak_excess_spread_db=2.4)
    result = baseline_actionability(x, x, sr, policy=policy)
    assert result["actionable"] is False
    assert result["failures"]
    cfg = CompressorConfig(threshold_dbfs=-24, ratio=2, attack_ms=12, release_ms=180,
                           knee_db=5, max_gr_db=2.5)
    proposals = propose_compression_candidates(sr, cfg, result)
    assert proposals["decision"] == "no_change"
    assert proposals["candidates"] == []


def test_unstable_bright_events_are_actionable_and_keep_fixed_event_evidence():
    sr = 16000
    x = _synthetic_oh(sr, unstable=True)
    policy = OverheadPolicy(actionable_peak_excess_p95_db=4.0,
                            actionable_peak_excess_spread_db=2.4)
    evidence = dynamics_evidence(x, x, sr)
    assert evidence["event_count"] >= policy.min_events
    assert evidence["active_block_count"] >= policy.min_active_blocks
    result = baseline_actionability(x, x, sr, policy=policy)
    assert result["actionable"] is True, result


def test_proposals_change_only_attack_release():
    sr = 16000
    x = _synthetic_oh(sr, unstable=True)
    policy = OverheadPolicy(actionable_peak_excess_p95_db=4.0,
                            actionable_peak_excess_spread_db=2.4)
    action = baseline_actionability(x, x, sr, policy=policy)
    base = CompressorConfig(threshold_dbfs=-25.5, ratio=2.2, attack_ms=14,
                            release_ms=220, knee_db=4, max_gr_db=2.4,
                            detector="rms", rms_ms=4, sidechain_hpf_hz=180)
    result = propose_compression_candidates(sr, base, action)
    assert result["decision"] == "review_candidates"
    assert len(result["candidates"]) == 3
    for item in result["candidates"]:
        cfg = item["config"]
        for key in ["threshold_dbfs", "ratio", "knee_db", "max_gr_db", "detector",
                    "rms_ms", "sidechain_hpf_hz", "bypass"]:
            assert cfg[key] == getattr(base, key)
        assert item["change_scope"] == "attack_release_only"


def test_linked_render_preserves_stereo_layout_and_reports_gr():
    sr = 16000
    x = _synthetic_oh(sr, seconds=4, unstable=True)
    cfg = CompressorConfig(threshold_dbfs=-27, ratio=2.0, attack_ms=10,
                           release_ms=150, knee_db=5, max_gr_db=2.0)
    y, report = render_candidate(x, sr, cfg)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert report["linked_stereo"] is True
    assert 0 <= report["max_gr_db"] <= cfg.max_gr_db + 1e-5
    mask = (np.abs(x[:, 0]) > 1e-6) & (np.abs(x[:, 1]) > 1e-6)
    gl = y[mask, 0] / x[mask, 0]
    gr = y[mask, 1] / x[mask, 1]
    assert np.max(np.abs(gl - gr)) < 2e-5


def test_assessment_rejects_stereo_image_cheat():
    sr = 16000
    x = _synthetic_oh(sr, unstable=True)
    policy = OverheadPolicy(actionable_peak_excess_p95_db=4.0,
                            actionable_peak_excess_spread_db=2.4)
    mid = x.mean(axis=1)
    bad = np.column_stack([mid, mid]).astype(np.float32)
    result = assess_against_baseline(x, x, bad, sr, policy=policy)
    assert result["technically_survives"] is False
    assert any("stereo" in f for f in result["failures"])


def test_assessment_requires_real_peak_improvement_not_just_safe_stereo():
    sr = 16000
    x = _synthetic_oh(sr, unstable=True)
    policy = OverheadPolicy(actionable_peak_excess_p95_db=4.0,
                            actionable_peak_excess_spread_db=2.4)
    result = assess_against_baseline(x, x, x.copy(), sr, policy=policy)
    assert result["technically_survives"] is False
    assert "bright_peak_consistency_not_improved" in result["failures"]


def test_invalid_layout_fails_closed():
    sr = 16000
    mono = np.ones(sr, dtype=np.float32) * .01
    with pytest.raises(ValueError, match="explicit stereo"):
        baseline_actionability(mono, mono, sr)
