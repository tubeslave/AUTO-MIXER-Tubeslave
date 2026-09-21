import numpy as np
from audio_workbench.mastering import MasteringDirector,MasteringConfig
from audio_workbench.mastering.analyzer import analyze

def tone(sr=48000,n=48000):
 t=np.arange(n)/sr
 return np.column_stack([.25*np.sin(2*np.pi*80*t)+.08*np.sin(2*np.pi*3000*t)]*2).astype("float32")

def test_analyzer_finite():
 a=analyze(tone(),48000)
 assert all(np.isfinite([a["crest_db"],a["side_mid_db"],a["correlation"]]))

def test_pipeline_finite_and_same_shape():
 x=tone();y,r=MasteringDirector().render(x,48000)
 assert y.shape==x.shape and np.isfinite(y).all()
 assert len(r["events"])==5

def test_maximizer_respects_ceiling_approximately():
 x=tone()*3
 y,r=MasteringDirector(MasteringConfig(stabilizer=False,clarity=False,impact=False,clipper=False,ceiling_db=-1,maximizer_drive_db=2)).render(x,48000)
 assert np.max(np.abs(y)) <= 10**(-1/20)*1.03
