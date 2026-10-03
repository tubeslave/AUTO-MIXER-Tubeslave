from __future__ import annotations
import numpy as np
from scipy import signal

def process(x:np.ndarray,sr:int,drive_db:float=.7,mix:float=.08)->tuple[np.ndarray,dict]:
    # Parallel high-mid saturation for density, intentionally subtle.
    sos=signal.butter(2,[700,min(12000,sr*.45)],btype="bandpass",fs=sr,output="sos")
    b=signal.sosfiltfilt(sos,x,axis=0).astype("float32")
    d=10**(drive_db/20);sat=np.tanh(b*d)/d
    harmonic=sat-b
    y=x+harmonic*np.float32(mix)
    return y.astype("float32"),{"drive_db":drive_db,"mix":mix}
