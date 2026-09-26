from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def process(x,sr,strength=.18,max_reduction_db=1.5,nfft=2048,hop=512):
    # Perceptual-ish adaptive spectral smoothing: attenuate local spectral protrusions only.
    Z=[]
    for c in range(x.shape[1]):
        f,t,z=signal.stft(x[:,c],fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary="zeros")
        Z.append(z)
    Z=np.stack(Z); mag=np.mean(np.abs(Z),axis=0)+1e-10
    log=20*np.log10(mag); smooth=ndimage.gaussian_filter(log,sigma=(3.0,1.2))
    excess=np.clip(log-smooth-1.0,0,max_reduction_db)
    gain=np.power(10,(-strength*excess)/20).astype("float32")
    for c in range(2): Z[c]*=gain
    ys=[]
    for c in range(2):
        _,y=signal.istft(Z[c],fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary=True);ys.append(y[:len(x)])
    return np.column_stack(ys).astype("float32"),{"strength":strength,"max_reduction_db":max_reduction_db}
