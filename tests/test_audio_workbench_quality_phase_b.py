import numpy as np
import soundfile as sf

from audio_workbench.dynamic_masking import dynamic_masking_graph
from audio_workbench.macro import energy_curve, contrast_regression
from audio_workbench.quality_loop import rank_next_problem

def _tone(path,freq,amp=.1,sr=48000,sec=1.0):
    t=np.arange(int(sr*sec))/sr
    x=(amp*np.sin(2*np.pi*freq*t)).astype("float32")
    sf.write(path,np.column_stack([x,x]),sr,subtype="FLOAT")

def test_dynamic_masking_respects_priority(tmp_path):
    a=tmp_path/"a.wav"; b=tmp_path/"b.wav"
    _tone(a,1000); _tone(b,1000)
    g=dynamic_masking_graph([{"name":"Vocal","path":str(a)},{"name":"Guitar","path":str(b)}],
                            {"Vocal":1.0,"Guitar":.5})
    e=g["edges"][0]
    assert e["protect"]=="Vocal" and e["candidate_for_movement"]=="Guitar"

def test_macro_regression_detects_lost_chorus_lift(tmp_path):
    p=tmp_path/"x.wav"; sr=48000
    x=np.concatenate([np.ones(sr)*.05,np.ones(sr)*.2]).astype("float32")
    sf.write(p,np.column_stack([x,x]),sr,subtype="FLOAT")
    sections=[{"name":"verse","start_s":0,"end_s":1},{"name":"chorus","start_s":1,"end_s":2}]
    before=energy_curve(str(p),sections)
    # fabricate after where contrast collapsed
    after={"normalized":[{"name":"verse","lift_db":0},{"name":"chorus","lift_db":2}]}
    r=contrast_regression(before,after,[{"from":"verse","to":"chorus","min_lift_db":5}])
    assert not r["passed"]

def test_problem_priority():
    p=rank_next_problem([
      {"name":"a","importance":.9,"confidence":.9,"expected_impact":.8,"uncertainty":.1},
      {"name":"b","importance":.5,"confidence":.5,"expected_impact":.5,"uncertainty":0},
    ])
    assert p["name"]=="a"
