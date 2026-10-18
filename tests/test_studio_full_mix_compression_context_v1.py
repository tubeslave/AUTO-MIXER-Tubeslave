import numpy as np
import pytest

from audio_workbench import causal
from audio_workbench.compression_mix_context import evaluate_full_mix_compression_context


def _prepared(n, sr=48000, target="vocal_intelligibility"):
    plan = causal.make_plan(
        observation="bounded compression candidate",
        hypothesis="candidate may improve mix context",
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


def _center(x):
    return np.column_stack([x,x]).astype("float32") / np.float32(np.sqrt(2))


def test_whole_mix_survivor_exports_audition_but_never_promotes():
    sr=48000;n=sr*4;t=np.arange(n)/sr
    rest=np.column_stack([
        .08*np.sin(2*np.pi*500*t)+.03*np.sin(2*np.pi*2000*t),
        .08*np.sin(2*np.pi*520*t)+.03*np.sin(2*np.pi*2100*t),
    ]).astype("float32")
    original=(.07*np.sin(2*np.pi*250*t)+.025*np.sin(2*np.pi*1500*t)).astype("float32")
    candidate=(.055*np.sin(2*np.pi*250*t)+.04*np.sin(2*np.pi*1500*t)).astype("float32")
    original=_center(original);candidate=_center(candidate)
    baseline=(rest+original).astype("float32")
    report,audition=evaluate_full_mix_compression_context(
        _prepared(n,sr),"balanced",baseline,original,candidate,sr,
        source_group="vocal",vocal_bus=original,evaluation_confidence=.95,
    )
    assert report["context_objective_gate_passed"] is True
    assert report["evaluation"]["critic"]["target_improvement"] > .02
    assert report["evaluation"]["transition"]["status"] == "pending_human_review"
    assert report["baseline_promoted"] is False
    assert report["audition_export_allowed"] is True
    assert set(audition) == {"reference","candidate"}
    assert audition["reference"].shape == baseline.shape
    assert report["replacement"]["baseline_reconstruction_max_error"] <= 1e-7
    assert report["replacement"]["candidate_rest_max_error"] <= 1e-7


def test_gain_only_candidate_collapses_to_no_change_and_is_rejected():
    sr=48000;n=sr*2;t=np.arange(n)/sr
    original=_center((.08*np.sin(2*np.pi*900*t)).astype("float32"))
    rest=_center((.10*np.sin(2*np.pi*350*t)).astype("float32"))
    baseline=(rest+original).astype("float32")
    report,audition=evaluate_full_mix_compression_context(
        _prepared(n,sr),"balanced",baseline,original,original*2,sr,
        source_group="vocal",vocal_bus=original,evaluation_confidence=.95,
    )
    assert "candidate_mix_identical_to_baseline" in report["context_failures"]
    assert report["evaluation"]["transition"]["status"] == "rejected"
    assert report["evaluation"]["transition"]["next_action"] == "rollback_candidate"
    assert report["audition_export_allowed"] is False
    assert audition == {}


def test_non_target_rest_is_exactly_reused():
    sr=48000;n=sr;t=np.arange(n)/sr
    original=_center((.04*np.sin(2*np.pi*200*t)).astype("float32"))
    candidate=_center((.04*np.sin(2*np.pi*700*t)).astype("float32"))
    rest=np.column_stack([.1*np.sin(2*np.pi*300*t),.09*np.sin(2*np.pi*330*t)]).astype("float32")
    baseline=(rest+original).astype("float32")
    report,_=evaluate_full_mix_compression_context(
        _prepared(n,sr,target="harshness"),"balanced",baseline,original,candidate,sr,
        source_group="other",evaluation_confidence=.95,
    )
    assert report["replacement"]["other_mix_contributions_reused"] is True
    assert report["replacement"]["candidate_rest_max_error"] <= 1e-7
    assert report["no_change_counterfactual_sha256"] == report["replacement"]["baseline_mix_sha256"]


@pytest.mark.parametrize("group",["vocal","drums"])
def test_anchor_group_requires_corresponding_bus(group):
    x=np.zeros((4800,2),dtype="float32");x[:,0]=.01;x[:,1]=.01
    with pytest.raises(ValueError):
        evaluate_full_mix_compression_context(
            _prepared(len(x)),"balanced",x,x*.2,x*.19,48000,
            source_group=group,
        )


def test_invalid_layout_and_prepared_timeline_fail_closed():
    mono=np.zeros(4800,dtype="float32")
    with pytest.raises(ValueError):
        evaluate_full_mix_compression_context(
            _prepared(len(mono)),"balanced",mono,mono,mono,48000,
        )
    stereo=np.zeros((4800,2),dtype="float32");stereo[:,0]=.01;stereo[:,1]=.01
    bad=_prepared(len(stereo)+1)
    with pytest.raises(ValueError):
        evaluate_full_mix_compression_context(
            bad,"balanced",stereo,stereo*.2,stereo*.19,48000,
        )
