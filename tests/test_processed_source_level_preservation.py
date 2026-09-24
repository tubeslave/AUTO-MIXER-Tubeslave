import numpy as np
import pytest
from audio_workbench.mixing.level_preservation import match_processed_source


def audio():
    t=np.arange(48001)/48000
    return (.1*np.sin(2*np.pi*997*t)*(1+.3*np.sin(2*np.pi*2*t))).astype('float32')


@pytest.mark.parametrize('gain_db',[-.24,-.12,0,.18])
def test_constant_compensation_and_no_source_mutation(gain_db):
    x=audio();original=x.copy();y=x*np.float32(10**(gain_db/20));yc=y.copy()
    z,r=match_processed_source(x,y,48000,active_percentile=94)
    assert r['compensation_gain_db']==pytest.approx(-gain_db,abs=1e-5)
    assert abs(r['matched_active_rms_delta_db'])<1e-5
    assert r['baseline_eligible'] is False and r['headroom_verified'] is False
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(y,yc)
    np.testing.assert_allclose(z,x,atol=2e-8)


def test_antiphase_stereo_uses_power_not_fold_down():
    x=audio();a=np.column_stack([x,-x]);b=a*.98
    z,r=match_processed_source(a,b,48000)
    np.testing.assert_array_equal(z[:,0],-z[:,1]);assert r['balance_match_passed']


def test_one_gain_does_not_undo_compressed_dynamic_shape():
    x=audio();y=x.copy();y[:24000]*=.95
    z,r=match_processed_source(x,y,48000)
    np.testing.assert_array_equal(z,y*np.float32(10**(r['compensation_gain_db']/20)))
    assert not np.array_equal(z,x)


@pytest.mark.parametrize('bad',[np.zeros(48000,'float32'),np.full(48000,np.nan,'float32')])
def test_bad_candidate_rejected(bad):
    with pytest.raises(ValueError): match_processed_source(audio()[:48000],bad,48000)


def test_excessive_makeup_cannot_silently_pass():
    x=audio()
    with pytest.raises(ValueError):match_processed_source(x,x*.5,48000,max_gain_db=1)


def test_mismatched_shape_rejected():
    x=audio()
    with pytest.raises(ValueError):match_processed_source(x,x[:-1],48000)


def test_level_match_cannot_promote_an_audible_candidate_by_itself():
    from audio_workbench.quality_loop import evaluate_candidate
    x=audio();_,r=match_processed_source(x,x*.98,48000)
    plan={"confidence":{"cause":.9,"intervention":.9}}
    v=evaluate_candidate(plan,{"id":"matched-source",**r},True,[],.95)
    assert v["accepted"] is False
    assert v["acceptance_state"]=="pending_human_review"
