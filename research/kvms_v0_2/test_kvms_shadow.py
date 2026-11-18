import numpy as np
import pytest

from kvms_shadow import SceneContext, score_scene, propose_masker_gain_reduction

FS = 48000
T = np.arange(FS, dtype=np.float64) / FS
VOCAL = .15 * np.sin(2 * np.pi * 440 * T)
GUITAR = .15 * np.sin(2 * np.pi * 440 * T + .4)
CONTEXT = SceneContext("chorus", "vocal", "foreground", "TARGET", .99, "human")


def stems():
    return {"vocal": VOCAL.copy(), "guitar": GUITAR.copy()}


def test_overlapping_tracks_produce_nonzero_proxy():
    r = score_scene(stems(), CONTEXT, FS)
    assert r.status == "valid" and 0.1 < r.overlap_proxy < 0.9


def test_silent_masker_produces_zero_overlap():
    r = score_scene({"vocal": VOCAL, "guitar": np.zeros_like(GUITAR)}, CONTEXT, FS)
    assert r.status == "valid" and r.overlap_proxy == 0.


def test_silent_target_is_not_interpreted_as_clearly_audible():
    r = score_scene({"vocal": np.zeros_like(VOCAL), "guitar": GUITAR}, CONTEXT, FS)
    assert r.status == "abstain" and r.overlap_proxy is None


def test_blend_role_does_not_incentivize_source_separation():
    c = SceneContext("chorus", "vocal", "blend", "TARGET", .99, "human")
    assert score_scene(stems(), c, FS).status == "abstain"


@pytest.mark.parametrize("status,confidence",[("UNKNOWN",.99),("BLEED",.99),("TARGET+BLEED",.99),("TARGET",.70)])
def test_unverified_source_never_gets_valid_score(status, confidence):
    c = SceneContext("chorus", "vocal", "foreground", status, confidence, "human")
    assert score_scene(stems(), c, FS).status == "abstain"


def test_proposal_is_advisory_and_decreases_overlap_proxy():
    p = propose_masker_gain_reduction(stems(), CONTEXT, "guitar", FS, attribution_labels={"guitar":"TARGET"})
    assert p.status == "candidate_shadow"
    assert p.auto_apply is False
    assert p.delta_db == -1.
    assert p.candidate_overlap_proxy < p.baseline_overlap_proxy


def test_masker_attribution_unchanged_means_abstention():
    p = propose_masker_gain_reduction(stems(), CONTEXT, "guitar", FS, attribution_labels={"guitar":"UNKNOWN"})
    assert p.status == "abstain" and p.auto_apply is False


def test_bound_excludes_larger_or_positive_gain_changes():
    for db in [-2., 0., 1., float('nan')]:
        p = propose_masker_gain_reduction(stems(), CONTEXT, "guitar", FS, delta_db=db, attribution_labels={"guitar":"TARGET"})
        assert p.status == "abstain" and p.auto_apply is False


def test_common_gain_scaling_yields_same_overlap_proxy():
    x = stems()
    a = score_scene(x, CONTEXT, FS)
    b = score_scene({k:v*1.4 for k,v in x.items()}, CONTEXT, FS)
    assert abs(a.overlap_proxy-b.overlap_proxy) < 1e-9


def test_bad_audio_input_rejected():
    x = stems()
    x["vocal"][50] = float('nan')
    with pytest.raises(ValueError):
        score_scene(x, CONTEXT, FS)


def test_misaligned_audio_rejected():
    with pytest.raises(ValueError):
        score_scene({"vocal": VOCAL, "guitar": GUITAR[:-1]}, CONTEXT, FS)


def test_no_mix_or_live_device_interface_in_proposal():
    p = propose_masker_gain_reduction(stems(), CONTEXT, "guitar", FS, attribution_labels={"guitar":"TARGET"})
    assert not any(k in p.to_dict() for k in ["osc", "mixer", "write", "commit", "set_fader"])
