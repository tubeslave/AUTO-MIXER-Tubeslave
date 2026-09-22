import numpy as np
from audio_workbench.mixing.keys_playback_director import playback_section_ride
def test_playback_ride_bounded():
 db=np.array([-30,-24,-28,-25,-29,-26],dtype=float);active=np.ones(len(db),dtype=bool)
 r=playback_section_ride(db,active,.6)
 assert np.max(np.abs(r))<=.601
