import numpy as np
from audio_workbench.mixing.guitar_director import wall_layer


def test_wall_is_stereo_and_dense_only():
 sr=1000
 t=np.arange(10000,dtype="float32")/sr
 x=(.1*np.sin(2*np.pi*180*t)).astype("float32")
 d=np.r_[np.zeros(5000),np.ones(5000)]
 y=wall_layer(x,sr,d)
 assert y.shape==(10000,2)
 assert np.sqrt(np.mean(y[6000:]**2))>np.sqrt(np.mean(y[:4000]**2))


def test_wall_fails_closed_when_sample_rate_cannot_represent_band():
 sr=200
 t=np.arange(2000,dtype="float32")/sr
 x=(.1*np.sin(2*np.pi*40*t)).astype("float32")
 d=np.ones_like(x)
 y=wall_layer(x,sr,d)
 assert y.shape==(2000,2)
 assert np.isfinite(y).all()
 assert np.max(np.abs(y))==0
