import numpy as np
from audio_workbench.mixing.drum_director import self_sample,reinforce
def test_self_sample_and_reinforce_preserve_shape():
 sr=1000;x=np.zeros((3000,1),dtype="float32");hits=np.array([500,1500,2500])
 for h in hits:x[h:h+20,0]=np.linspace(1,0,20)
 s=self_sample(x,hits,sr,pre_ms=5,post_ms=40,count=3)
 y=reinforce(len(x),hits,s,sr,-20)
 assert y.shape==x.shape and np.max(np.abs(y))>0
