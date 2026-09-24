import numpy as np

from audio_workbench.mixing.compression_director import propose_candidates, render_candidate


def test_director_candidate_runs_through_offline_dynamics_and_stays_human_gated():
    sr = 48000
    t = np.arange(sr * 2) / sr
    envelope = np.where((t % .4) < .12, np.exp(-((t % .4) / .04)), 0.0)
    x = (.18 * np.sin(2 * np.pi * 180 * t) * envelope).astype("float32")
    source = x.copy()
    proposal = propose_candidates(x, sr, "bass")
    candidate = next(item for item in proposal["candidates"] if item["id"] == "balanced")
    y, report = render_candidate(x, sr, "bass", candidate, enable_ride=False)
    np.testing.assert_array_equal(x, source)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert report["compression_director_candidate_id"] == "balanced"
    assert report["requires_human_review"] is True
    assert report["requires_human_listening"] is True
    assert report["baseline_eligible"] is False
    assert report["max_gr_db"] > 0
