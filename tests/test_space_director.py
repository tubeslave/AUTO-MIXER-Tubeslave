import numpy as np
from audio_workbench.mixing.space import common_room_ir,depth_profile,send_gain_db
def test_room_is_stereo_and_bounded():
 ir=common_room_ir(8000)
 assert ir.ndim==2 and ir.shape[1]==2 and np.max(np.abs(ir))<=1.0001
def test_lead_is_closer_than_guitar():
 assert depth_profile("18_VALERA_VOX.wav","vocal").depth < depth_profile("03_GTR.wav","guitar").depth
def test_dense_section_space_move_is_small():
 p=depth_profile("03_GTR.wav","guitar")
 a=send_gain_db(p,np.array([0.,1.]))
 assert abs(a[1]-a[0])<2
