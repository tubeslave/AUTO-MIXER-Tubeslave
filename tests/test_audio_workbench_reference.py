import numpy as np
import soundfile as sf

from audio_workbench.reference import create_profile, compare_to_reference, build_hypotheses

def _write(path, low=.1, high=.02, sr=48000):
    t=np.arange(sr)/sr
    x=low*np.sin(2*np.pi*100*t)+high*np.sin(2*np.pi*4000*t)
    sf.write(path,np.column_stack([x,x]).astype("float32"),sr,subtype="FLOAT")

def test_reference_detects_relative_tonal_difference(tmp_path):
    ref=tmp_path/"ref.wav"; cand=tmp_path/"cand.wav"
    _write(ref,.1,.02); _write(cand,.05,.08)
    p=create_profile(str(ref))
    c=compare_to_reference(str(cand),p)
    assert c["global_delta"]["bands_db"]["presence"] > 3
    assert any(h["domain"]=="tonal_balance" for h in build_hypotheses(c))

def test_level_only_change_does_not_become_tonal_difference(tmp_path):
    ref=tmp_path/"ref.wav"; cand=tmp_path/"cand.wav"
    _write(ref,.1,.02); _write(cand,.05,.01)
    c=compare_to_reference(str(cand),create_profile(str(ref)))
    assert max(abs(v) for v in c["global_delta"]["bands_db"].values()) < .2

def test_section_names_pair_explicitly(tmp_path):
    ref=tmp_path/"ref.wav"; cand=tmp_path/"cand.wav"; _write(ref); _write(cand)
    p=create_profile(str(ref),[{"name":"chorus","start_s":0,"end_s":.5}])
    c=compare_to_reference(str(cand),p,[{"name":"chorus","start_s":.25,"end_s":.75}])
    assert c["section_deltas"][0]["name"]=="chorus"
