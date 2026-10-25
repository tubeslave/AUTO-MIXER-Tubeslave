import json

import numpy as np
import pytest

from audio_workbench.compression_iteration import VARIANT_IDS, prepare_compression_iteration
from audio_workbench.routed_compression_family import evaluate_routed_compression_family
from audio_workbench.routed_contribution import SessionRender


def _source(sr=12000, seconds=2.4):
    t=np.arange(int(sr*seconds))/sr
    phase=t % .32
    env=np.where(phase<.12,np.exp(-phase/.045),0.0)
    return (.22*np.sin(2*np.pi*180*t)*env).astype("float32")


def _renderer(sr, n):
    t=np.arange(n)/sr
    bed=np.column_stack([
        .07*np.sin(2*np.pi*510*t)+.02*np.sin(2*np.pi*1700*t),
        .065*np.sin(2*np.pi*530*t)+.018*np.sin(2*np.pi*1800*t),
    ]).astype("float32")
    def render(overrides):
        src=np.asarray(overrides["bass"],dtype="float32")
        dry=np.column_stack([src*.70,src*.68]).astype("float32")
        delayed=np.zeros_like(src); d=round(.013*sr); delayed[d:]=src[:-d]
        room=np.column_stack([delayed*.12,delayed*.15]).astype("float32")
        pre=(bed+dry+room).astype("float64")
        mix=(np.tanh(pre*1.12)/1.12).astype("float32")
        return SessionRender(mix=mix,early_room=room,metadata={"graph":"family-test"})
    return render


def test_family_evaluates_exact_three_and_never_selects_or_promotes():
    sr=12000;x=_source(sr);original=x.copy()
    prepared,renders=prepare_compression_iteration(x,sr,"bass","punch")
    report,auditions=evaluate_routed_compression_family(
        prepared,renders,"bass",x,sr,_renderer(sr,len(x)),
        source_group="other",evaluation_confidence=.95,baseline_id="mix-A",
    )
    np.testing.assert_array_equal(x,original)
    assert tuple(report["candidate_order"])==VARIANT_IDS
    assert set(report["reports"])==set(VARIANT_IDS)
    assert set(report["surviving_auditions"]+report["machine_rejected"])==set(VARIANT_IDS)
    assert not set(report["surviving_auditions"]).intersection(report["machine_rejected"])
    assert report["winner"] is None and report["ranking"] is None
    assert report["baseline_promoted"] is False and report["baseline_after"]=="mix-A"
    for cid,item in report["reports"].items():
        assert item["baseline_promoted"] is False
        if cid in report["surviving_auditions"]:
            assert item["audition_export_allowed"] is True
            assert set(auditions[cid])=={"reference","candidate"}
            assert item["evaluation"]["transition"]["next_action"]=="human_listening"
        else:
            assert item["audition_export_allowed"] is False
    json.dumps(report,allow_nan=False)


def test_family_fails_closed_if_render_family_is_incomplete_or_shape_changes():
    sr=12000;x=_source(sr);prepared,renders=prepare_compression_iteration(x,sr,"bass","punch")
    missing=dict(renders);missing.pop("control")
    with pytest.raises(ValueError):
        evaluate_routed_compression_family(prepared,missing,"bass",x,sr,_renderer(sr,len(x)))
    malformed=dict(renders);malformed["control"]=malformed["control"][:-1]
    with pytest.raises(ValueError):
        evaluate_routed_compression_family(prepared,malformed,"bass",x,sr,_renderer(sr,len(x)))
