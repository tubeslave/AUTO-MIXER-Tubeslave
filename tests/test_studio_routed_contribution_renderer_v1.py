import numpy as np
import pytest

from audio_workbench import causal
from audio_workbench.routed_contribution import (
    SessionRender, _sha256, evaluate_routed_compression_context,
)


def _prepared(n, sr=48000, target="vocal_intelligibility"):
    plan = causal.make_plan(
        observation="bounded routed compression candidate",
        hypothesis="candidate may improve full mix context",
        target=target,
        interventions=[{"type":"compressor", "id":"balanced", "params":{}}],
        expected_effect=f"improve {target}",
        protected_metrics=["density","width_db","foreground_db","harshness",
                           "vocal_intelligibility","punch_db","climax_lift_db"],
        confidence={"cause":.9,"intervention":.9},
    )
    plan["requires_human_review"] = True
    return {
        "schema":"studio-compression-iteration-bridge-v1",
        "source":{"sample_rate":sr,"frames":n,"channels":1},
        "target":target,
        "plan":plan,
        "candidates":[{
            "id":"balanced", "source_audio_sha256":"source",
            "candidate_audio_sha256":"candidate", "objective_failures":[],
            "objective_gate_passed":True,
        }],
    }


def _session_renderer(sr, n, *, nondeterministic=False):
    t=np.arange(n)/sr
    other=np.column_stack([
        .075*np.sin(2*np.pi*500*t)+.025*np.sin(2*np.pi*2100*t),
        .070*np.sin(2*np.pi*520*t)+.024*np.sin(2*np.pi*2050*t),
    ]).astype("float32")
    calls={"n":0}

    def render(overrides):
        calls["n"] += 1
        src=np.asarray(overrides["lead"],dtype="float32")
        if src.ndim != 1 or len(src) != n:
            raise ValueError("lead override has wrong shape")
        dry=np.column_stack([src*.72,src*.68]).astype("float32")
        delayed=np.zeros_like(src)
        d=round(.011*sr)
        delayed[d:]=src[:-d]
        room=np.column_stack([delayed*.19,delayed*.23]).astype("float32")
        vocal=(dry+room).astype("float32")
        pre=(other+vocal).astype("float64")
        # Shared nonlinear bus makes a fixed dry-source subtraction non-authoritative.
        mix=(np.tanh(pre*1.18)/1.18).astype("float32")
        if nondeterministic:
            mix=(mix+np.float32(calls["n"]*2e-5)).astype("float32")
        return SessionRender(
            mix=mix, vocal_bus=vocal, early_room=room,
            metadata={"graph":"synthetic_send_return_shared_nonlinear_bus"},
        )
    return render


def _source_pair(sr, n):
    t=np.arange(n)/sr
    original=(.070*np.sin(2*np.pi*250*t)+.025*np.sin(2*np.pi*1500*t)).astype("float32")
    candidate=(.055*np.sin(2*np.pi*250*t)+.040*np.sin(2*np.pi*1500*t)).astype("float32")
    return original,candidate


def test_authoritative_rerender_includes_send_return_and_shared_bus_without_subtraction():
    sr=48000;n=sr*3
    original,candidate=_source_pair(sr,n)
    renderer=_session_renderer(sr,n)
    original_copy=original.copy();candidate_copy=candidate.copy()
    report,audition=evaluate_routed_compression_context(
        _prepared(n,sr),"balanced","lead",original,candidate,sr,renderer,
        source_group="vocal",evaluation_confidence=.95,
    )
    assert report["routing_authority"] == "full_session_rerender"
    assert report["direct_source_subtraction_used_for_candidate_mix"] is False
    assert report["renderer_invocations"] == 4
    assert report["rerender_identity"]["hash_equal"] is True
    assert report["rerender_identity"]["max_abs_error"] == 0
    assert report["context_objective_gate_passed"] is True
    assert report["baseline_promoted"] is False
    assert report["evaluation"]["transition"]["promote_baseline"] is False
    assert "not an additive stem" in report["counterfactual_marginal"]["meaning"]
    np.testing.assert_array_equal(original,original_copy)
    np.testing.assert_array_equal(candidate,candidate_copy)
    if report["audition_export_allowed"]:
        assert report["status"] == "pending_human_review"
        assert set(audition) == {"reference","candidate"}
    else:
        assert audition == {}


def test_candidate_mix_hash_is_from_full_rerender_not_naive_dry_replacement():
    sr=48000;n=sr*2
    original,candidate=_source_pair(sr,n)
    renderer=_session_renderer(sr,n)
    report,_=evaluate_routed_compression_context(
        _prepared(n,sr),"balanced","lead",original,candidate,sr,renderer,
        source_group="vocal",evaluation_confidence=.95,
    )
    gain=10**(float(report["source_match"]["required_gain_db"])/20)
    matched=(candidate*np.float32(gain)).astype("float32")
    exact=renderer({"lead":matched}).mix
    assert report["renders"]["candidate_mix_sha256"] == _sha256(exact,sr)
    baseline=renderer({"lead":original}).mix
    naive=(baseline + np.column_stack([(matched-original)*.72,(matched-original)*.68])).astype("float32")
    assert _sha256(naive,sr) != report["renders"]["candidate_mix_sha256"]
    assert np.max(np.abs(naive-exact)) > 1e-5


def test_nondeterministic_session_renderer_is_objectively_rejected():
    sr=48000;n=sr
    original,candidate=_source_pair(sr,n)
    report,audition=evaluate_routed_compression_context(
        _prepared(n,sr),"balanced","lead",original,candidate,sr,
        _session_renderer(sr,n,nondeterministic=True),
        source_group="vocal",evaluation_confidence=.95,rerender_tolerance=1e-7,
    )
    assert "session_rerender_not_deterministic" in report["context_failures"]
    assert report["context_objective_gate_passed"] is False
    assert report["evaluation"]["transition"]["status"] == "rejected"
    assert report["evaluation"]["transition"]["next_action"] == "rollback_candidate"
    assert report["audition_export_allowed"] is False
    assert audition == {}


def test_missing_rerendered_anchor_bus_fails_closed():
    sr=48000;n=sr
    original,candidate=_source_pair(sr,n)
    base=_session_renderer(sr,n)
    def no_anchor(overrides):
        r=base(overrides)
        return SessionRender(r.mix, early_room=r.early_room)
    report,_=evaluate_routed_compression_context(
        _prepared(n,sr),"balanced","lead",original,candidate,sr,no_anchor,
        source_group="vocal",evaluation_confidence=.95,
    )
    assert "vocal_anchor_bus_unavailable" in report["context_failures"]
    assert report["evaluation"]["transition"]["status"] == "rejected"


def test_renderer_contract_and_prepared_timeline_fail_closed():
    sr=48000;n=sr
    original,candidate=_source_pair(sr,n)
    with pytest.raises(TypeError):
        evaluate_routed_compression_context(
            _prepared(n,sr),"balanced","lead",original,candidate,sr,
            lambda overrides: np.zeros((n,2),dtype="float32"),
        )
    bad=_prepared(n+1,sr)
    with pytest.raises(ValueError):
        evaluate_routed_compression_context(
            bad,"balanced","lead",original,candidate,sr,_session_renderer(sr,n),
        )
