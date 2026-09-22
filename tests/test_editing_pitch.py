import numpy as np
from audio_workbench.editing.pitch import estimate_f0_autocorr,correction_plan
def test_f0_tracks_sine():
 sr=8000;t=np.arange(sr)/sr;x=.2*np.sin(2*np.pi*220*t)
 r=estimate_f0_autocorr(x,sr,fmin=100,fmax=400)
 f=r["f0_hz"][np.isfinite(r["f0_hz"])]
 assert len(f)>20 and abs(np.median(f)-220)<8
def test_plan_is_partial_and_bounded():
 s=[{"median_error_cents":30,"confidence":.8,"start_s":0,"end_s":1,"target_midi":60,"p90_abs_error_cents":35}]
 r=correction_plan(s)
 assert 0<abs(r[0]["correction_cents"])<=35
