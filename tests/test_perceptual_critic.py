import numpy as np
from audio_workbench.mixing.perceptual_critic import (
    PerceptualSnapshot,
    accept_candidate,
    diagnose,
    snapshot,
)

def test_snapshot_is_finite():
 sr=8000;t=np.arange(sr*2)/sr
 mix=np.column_stack([np.sin(2*np.pi*220*t),np.sin(2*np.pi*220*t+.1)]).astype("float32")*.1
 s=snapshot(mix,sr)
 assert all(np.isfinite(v) for v in s.__dict__.values())

def test_low_intelligibility_emits_hypothesis():
 sr=12000;t=np.arange(sr)/sr
 music=np.sin(2*np.pi*2000*t).astype("float32")*.2
 vocal=np.sin(2*np.pi*2000*t).astype("float32")*.005
 mix=np.column_stack([music+vocal,music+vocal])
 h=diagnose(snapshot(mix,sr,vocal=vocal))
 assert any(x["target"]=="vocal_intelligibility" for x in h)

def _s(**changes):
 base=dict(foreground_db=-5.0,vocal_intelligibility=.45,punch_db=6.0,harshness=.50,
           density=.55,depth_proxy=.5,width_db=-8.0,climax_lift_db=1.4)
 base.update(changes)
 return PerceptualSnapshot(**base)

def test_accept_candidate_requires_target_improvement():
 r=accept_candidate(_s(),_s(vocal_intelligibility=.48),"vocal_intelligibility")
 assert r["accept"] is True
 assert r["target_improvement"]>0
 r=accept_candidate(_s(),_s(vocal_intelligibility=.455),"vocal_intelligibility")
 assert r["accept"] is False
 assert "target_not_improved" in r["failures"]

def test_accept_candidate_understands_decrease_target():
 r=accept_candidate(_s(harshness=.80),_s(harshness=.72),"harshness")
 assert r["accept"] is True
 assert np.isclose(r["target_improvement"],.08)

def test_accept_candidate_rejects_collateral_width_change():
 r=accept_candidate(_s(),_s(punch_db=6.5,width_db=-10.0),"punch")
 assert r["accept"] is False
 assert "width_regression" in r["failures"]

def test_accept_candidate_rejects_new_harshness_problem():
 r=accept_candidate(_s(),_s(punch_db=6.5,harshness=.58),"punch")
 assert r["accept"] is False
 assert "harshness_regression" in r["failures"]
