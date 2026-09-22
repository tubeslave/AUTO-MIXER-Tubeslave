import numpy as np
from audio_workbench.editing.performance import robust_outliers,bounded_event_gain,click_candidates
def test_outlier_and_gain_are_bounded():
 x=np.array([-10,-10.2,-9.8,-10.1,-10,0.])
 m=robust_outliers(x,2.5)
 assert m[-1]
 assert abs(bounded_event_gain(0,-10,2))<=2
def test_click_candidate():
 x=np.zeros(1000,dtype="float32");x[500]=1
 assert len(click_candidates(x,1000))>=1
