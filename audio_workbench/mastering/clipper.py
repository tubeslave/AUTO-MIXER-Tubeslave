from __future__ import annotations
import numpy as np
from scipy import signal

def process(x,sr,drive_db=.8,oversample=4):
    up=signal.resample_poly(x,oversample,1,axis=0)
    drive=10**(drive_db/20)
    # normalized tanh soft clip, then downsample. Intended for sub-dB/low-dB peak conditioning.
    y=np.tanh(up*drive)/np.tanh(drive)
    y=signal.resample_poly(y,1,oversample,axis=0)[:len(x)].astype("float32")
    return y,{"drive_db":drive_db,"oversample":oversample}
