import numpy as np
from audio_workbench.mastering.decision import choose_modules,accept
from audio_workbench.mastering.stereo_director import process as stereo
from audio_workbench.mastering.bass_director import process as bass

def test_decision_is_bounded():
 a={"crest_db":18,"correlation":.95,"side_mid_db":-12,"bands_db":{"sub":-18,"low":-16,"mid":-20,"presence":-22,"air":-29}}
 d=choose_modules(a)
 assert d["impact"] and d["stereo_director"] and d["maximizer"]

def test_stereo_preserves_shape():
 sr=48000;t=np.arange(sr)/sr
 x=np.column_stack([np.sin(2*np.pi*440*t),np.sin(2*np.pi*440*t)]).astype("float32")*.1
 y,r=stereo(x,sr);assert y.shape==x.shape and np.isfinite(y).all()

def test_accept_rejects_overprocessing():
 b={"crest_db":18,"side_mid_db":-10,"correlation":.8}
 a={"crest_db":10,"side_mid_db":-13,"correlation":.8}
 assert not accept(b,a)["accept"]
