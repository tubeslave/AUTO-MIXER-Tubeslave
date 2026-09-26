import numpy as np
from audio_workbench.mixing.bass_director import harmonic_layer,kick_duck_db
def test_harmonic_layer_preserves_shape():
 x=np.sin(np.linspace(0,100,10000)).astype("float32")*.2
 y=harmonic_layer(x)
 assert y.shape==x.shape and np.isfinite(y).all()
def test_kick_duck_is_bounded():
 sr=2000;k=np.zeros(8000,dtype="float32");k[1000:1020]=1;k[4000:4020]=1
 d=kick_duck_db(k,sr,1.4)
 assert np.min(d)>=-1.401 and np.max(d)<=.001
