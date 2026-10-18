import numpy as np
from audio_workbench.editing.drum_performance import contextual_outliers
def test_contextual_outlier_is_partial():
 hits=[{"time_s":float(i),"confidence":.8} for i in range(12)]
 levels=[-20]*12;levels[6]=-12
 r=contextual_outliers(hits,levels,20,4,1.5)
 assert len(r)==1 and r[0]["gain_db"]==-1.5
