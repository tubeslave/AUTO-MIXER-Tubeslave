import numpy as np
from audio_workbench.mixing.perceptual_critic import snapshot,diagnose
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
