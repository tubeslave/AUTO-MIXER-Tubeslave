import numpy as np
from audio_workbench.mixing.artifact_critic import detect,compare
def test_clicks_are_detected_as_regression():
 sr=8000;x=np.sin(np.arange(sr*2)*2*np.pi*220/sr).astype("float32")*.1
 y=x.copy();y[::400]=1
 a=detect(x,sr);b=detect(y,sr)
 assert b["click_rate"]>a["click_rate"]
 assert not compare(a,b)["accept"]
