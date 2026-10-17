import numpy as np
from audio_workbench.mastering.adaptive_maximizer import diagnose,process

def test_adaptive_bands_do_work():
 sr=48000;t=np.arange(sr*3)/sr
 x=np.column_stack([.55*np.sin(2*np.pi*60*t)+.25*np.sin(2*np.pi*1800*t)]*2).astype("float32")
 plan=diagnose(x,sr)
 assert len(plan)==4 and all(p["threshold"]>0 for p in plan)
 y,r=process(x,sr)
 assert y.shape==x.shape and np.isfinite(y).all()
 assert any(b["max_gr_db"]>0 for b in r["bands"])
 assert np.max(np.abs(y)) <= 10**(-1/20)*1.03
