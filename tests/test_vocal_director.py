import numpy as np
from audio_workbench.mixing.vocal_director import saturate_level_matched,slow_vocal_ride_db
def test_saturation_is_level_matched():
 x=np.sin(np.linspace(0,100,20000)).astype("float32")*.2
 y=saturate_level_matched(x,1.2)
 ri=np.sqrt(np.mean(x*x));ro=np.sqrt(np.mean(y*y))
 assert abs(20*np.log10(ro/ri))<.05
def test_ride_is_bounded():
 x=np.r_[np.ones(5000)*.02,np.ones(5000)*.2].astype("float32")
 r=slow_vocal_ride_db(x,1000,1.25)
 assert np.max(np.abs(r))<=1.251
