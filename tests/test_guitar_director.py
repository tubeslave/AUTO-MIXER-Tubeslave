import numpy as np
from audio_workbench.mixing.guitar_director import wall_layer
def test_wall_is_stereo_and_dense_only():
 x=np.ones(10000,dtype="float32")*.1;d=np.r_[np.zeros(5000),np.ones(5000)]
 y=wall_layer(x,1000,d)
 assert y.shape==(10000,2)
 assert np.sqrt(np.mean(y[6000:]**2))>np.sqrt(np.mean(y[:4000]**2))
