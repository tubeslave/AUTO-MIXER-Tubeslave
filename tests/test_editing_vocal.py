import numpy as np
from audio_workbench.editing.vocal import phrase_activity,cleanup_between_phrases,phrase_level_automation
def test_cleanup_preserves_length_and_attenuates_pause():
 sr=1000;x=np.zeros((sr*4,1),dtype="float32")
 t=np.arange(sr)/sr;x[sr:2*sr,0]=.2*np.sin(2*np.pi*120*t);x[3*sr:,0]=.1*np.sin(2*np.pi*120*t)
 y,r=cleanup_between_phrases(x,sr,8)
 assert y.shape==x.shape and np.isfinite(y).all()
def test_phrase_automation_is_bounded():
 sr=2000;t=np.arange(sr)/sr;x=np.zeros((sr*5,1),dtype="float32")
 x[sr:2*sr,0]=.2*np.sin(2*np.pi*180*t);x[3*sr:4*sr,0]=.05*np.sin(2*np.pi*180*t)
 y,r=phrase_level_automation(x,sr,2)
 assert y.shape==x.shape
 assert all(abs(m["delta_db"])<=2.0 for m in r["moves"])
