import copy
import json

import numpy as np

from audio_workbench.compression_iteration import (
    VARIANT_IDS,
    evaluate_compression_candidate,
    prepare_compression_iteration,
)
from audio_workbench.mixing.perceptual_critic import PerceptualSnapshot


def _source(sr=12000, seconds=2.4):
    t=np.arange(int(sr*seconds))/sr
    phase=t % .32
    env=np.where(phase<.12,np.exp(-phase/.045),0.0)
    return (.22*np.sin(2*np.pi*180*t)*env).astype("float32")


def _snapshot(**changes):
    values=dict(foreground_db=-5.0,vocal_intelligibility=.50,punch_db=6.0,
                harshness=.45,density=.50,depth_proxy=.50,width_db=-8.0,
                climax_lift_db=1.5)
    values.update(changes)
    return PerceptualSnapshot(**values)


def test_prepare_has_exact_three_variants_plus_one_no_change_and_preserves_source():
    x=_source(); original=x.copy()
    report,renders=prepare_compression_iteration(x,12000,"bass","punch")
    np.testing.assert_array_equal(x,original)
    assert tuple(report["candidate_order"])==VARIANT_IDS
    assert tuple(renders)==VARIANT_IDS
    assert [c["id"] for c in report["candidates"]]==list(VARIANT_IDS)
    bypass=[c for c in report["plan"]["candidates"] if c.get("type")=="bypass"]
    compressors=[c for c in report["plan"]["candidates"] if c.get("type")=="compressor"]
    assert len(bypass)==1 and bypass[0]["label"]=="no_change"
    assert [c["id"] for c in compressors]==list(VARIANT_IDS)
    assert report["source_unchanged"] is True
    assert report["baseline_eligible"] is False
    json.dumps(report,allow_nan=False)
    for evidence in report["candidates"]:
        y=renders[evidence["id"]]
        assert y.shape==x.shape and np.isfinite(y).all()
        assert evidence["source_audio_sha256"]==report["source"]["audio_sha256"]
        assert evidence["candidate_audio_sha256"]!=report["source"]["audio_sha256"]
        assert evidence["requires_human_listening"] is True
        assert evidence["baseline_eligible"] is False
        assert evidence["frames"]==len(x) and evidence["sample_rate"]==12000


def test_machine_safe_perceptual_result_still_waits_for_human_and_keeps_baseline():
    report,_=prepare_compression_iteration(_source(),12000,"bass","punch")
    assert any(c["objective_gate_passed"] for c in report["candidates"])
    cid=next(c["id"] for c in report["candidates"] if c["objective_gate_passed"])
    result=evaluate_compression_candidate(
        report,cid,_snapshot(),_snapshot(punch_db=6.7),
        evaluation_confidence=.95,baseline_id="mix-A",
    )
    assert result["critic"]["machine_decision"]=="machine_safe"
    assert result["transition"]["status"]=="pending_human_review"
    assert result["transition"]["next_action"]=="human_listening"
    assert result["transition"]["promote_baseline"] is False
    assert result["baseline_after"]=="mix-A"


def test_natural_objective_failure_vetoes_even_improved_perceptual_proxy_and_rolls_back():
    silent=np.zeros(12000,dtype="float32")
    report,_=prepare_compression_iteration(silent,12000,"bass","punch")
    evidence=report["candidates"][0]
    assert evidence["objective_gate_passed"] is False
    assert "candidate_identical_to_source" in evidence["objective_failures"]
    result=evaluate_compression_candidate(
        report,evidence["id"],_snapshot(),_snapshot(punch_db=7.0),
        evaluation_confidence=.99,baseline_id="mix-A",
    )
    assert result["critic"]["machine_decision"]=="rejected"
    assert result["transition"]["status"]=="rejected"
    assert result["transition"]["rollback_candidate"] is True
    assert result["transition"]["baseline_after"]=="mix-A"


def test_objective_veto_is_not_overridden_by_candidate_or_plan_mutation():
    report,_=prepare_compression_iteration(_source(),12000,"bass","punch")
    tampered=copy.deepcopy(report)
    entry=tampered["candidates"][1]
    entry["objective_gate_passed"]=False
    entry["objective_failures"]=["synthetic_test_veto"]
    result=evaluate_compression_candidate(
        tampered,"balanced",_snapshot(),_snapshot(punch_db=7.0),
        evaluation_confidence=.99,baseline_id="mix-A",
    )
    assert result["transition"]["rollback_candidate"] is True
    assert result["baseline_promoted"] is False


def test_invalid_family_target_or_candidate_fails_closed():
    x=_source()
    try:
        prepare_compression_iteration(x,12000,"bass","overall_quality")
    except ValueError as exc:
        assert "unsupported perceptual target" in str(exc)
    else:
        raise AssertionError("unsupported target must fail")
    report,_=prepare_compression_iteration(x,12000,"bass","punch")
    try:
        evaluate_compression_candidate(
            report,"winner",_snapshot(),_snapshot(punch_db=7),evaluation_confidence=.9
        )
    except ValueError as exc:
        assert "unknown or duplicate" in str(exc)
    else:
        raise AssertionError("unknown candidate must fail")
