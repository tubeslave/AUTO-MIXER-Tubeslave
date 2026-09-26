import numpy as np
from audio_workbench.mastering.musical_critic import evaluate

def test_identical_passes():
 sr=48000;t=np.arange(sr*2)/sr
 x=np.column_stack([.2*np.sin(2*np.pi*70*t)+.05*np.sin(2*np.pi*2500*t)]*2).astype("float32")
 r=evaluate(x,x,sr)
 assert r["accept"] and r["spectral_shift"]["max_abs_shift_db"]<1e-5

def test_spectral_damage_is_detected():
 sr=48000;t=np.arange(sr*2)/sr
 x=np.column_stack([.2*np.sin(2*np.pi*70*t)+.08*np.sin(2*np.pi*3000*t)]*2).astype("float32")
 y=np.column_stack([.2*np.sin(2*np.pi*70*t)+.015*np.sin(2*np.pi*3000*t)]*2).astype("float32")
 r=evaluate(x,y,sr)
 assert "spectral_shift" in r["failures"]
