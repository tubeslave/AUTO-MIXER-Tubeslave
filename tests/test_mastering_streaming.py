import numpy as np, soundfile as sf
from audio_workbench.mastering.streaming import four_band_maximizer,render_file

def test_four_band_finite_and_ceiling():
 sr=48000;t=np.arange(sr*2)/sr
 x=np.column_stack([.9*np.sin(2*np.pi*60*t)+.4*np.sin(2*np.pi*2200*t)]*2).astype("float32")
 y,r=four_band_maximizer(x,sr,-1,2)
 assert y.shape==x.shape and np.isfinite(y).all()
 assert np.max(np.abs(y)) <= 10**(-1/20)*1.02

def test_streaming_length(tmp_path):
 sr=48000;x=np.zeros((sr*3,2),dtype="float32");x[:,0]=.2*np.sin(2*np.pi*440*np.arange(len(x))/sr);x[:,1]=x[:,0]
 a=tmp_path/"in.wav";b=tmp_path/"out.wav";sf.write(a,x,sr,subtype="FLOAT")
 render_file(str(a),str(b),chunk_s=1.0,overlap_s=.2)
 assert sf.info(a).frames==sf.info(b).frames
