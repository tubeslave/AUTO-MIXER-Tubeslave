from __future__ import annotations
import numpy as np
DEFAULT_ROOM={"pre_delay_ms":22,"send_db":-18.0,"low_cut_hz":180,"high_cut_hz":8500,
              "early_ms":[31,47,73,109,157],"early_gain":[.20,.15,.11,.08,.055]}
def room_policy(density:float)->dict:
    p=dict(DEFAULT_ROOM)
    # Denser sections get slightly less wet level so the room is audible without washing out the mix.
    p["send_db"]=float(-17.5-1.5*np.clip(density,0,1))
    return p
