from __future__ import annotations
import numpy as np
from scipy import signal

def process(x:np.ndarray,sr:int,max_width_change:float=.08)->tuple[np.ndarray,dict]:
    M=(x[:,0]+x[:,1])*.5;S=(x[:,0]-x[:,1])*.5
    corr=float(np.corrcoef(x[:,0],x[:,1])[0,1])
    ratio=10*np.log10((np.mean(S*S)+1e-20)/(np.mean(M*M)+1e-20))
    # Conservative autonomy: only widen clearly narrow, highly correlated masters.
    width=1.0
    if corr>.88 and ratio<-10.0: width=1.0+max_width_change
    elif corr<.25 or ratio>-4.0: width=1.0-max_width_change
    # Keep sub/low side essentially unchanged; process side above 180 Hz.
    sos=signal.butter(3,180,btype="highpass",fs=sr,output="sos")
    Shi=signal.sosfiltfilt(sos,S).astype("float32");S2=S+(width-1.0)*Shi
    y=np.column_stack([M+S2,M-S2]).astype("float32")
    corr2=float(np.corrcoef(y[:,0],y[:,1])[0,1])
    return y,{"width_factor":float(width),"correlation_before":corr,"correlation_after":corr2,"side_mid_db_before":float(ratio)}
