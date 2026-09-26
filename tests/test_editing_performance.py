import numpy as np
from audio_workbench.editing.performance import robust_outliers,bounded_event_gain,click_candidates


def test_outlier_and_gain_are_bounded():
 x=np.array([-10,-10.2,-9.8,-10.1,-10,0.])
 m=robust_outliers(x,2.5)
 assert m[-1]
 assert m.sum()==1
 assert abs(bounded_event_gain(0,-10,2))<=2


def test_small_stable_phrase_is_not_overcorrected():
 x=np.array([-10.0,-10.1,-9.9,-10.05,-9.95,-10.0])
 assert not robust_outliers(x,2.5).any()


def test_too_short_phrase_remains_diagnose_only():
 x=np.array([-10.0,-10.0,-10.0,-10.0,0.0])
 assert not robust_outliers(x,2.5).any()


def test_click_candidate():
 x=np.zeros(1000,dtype="float32");x[500]=1
 assert click_candidates(x,1000)==[500]


def test_click_candidate_step_discontinuity_is_detected():
 x=np.zeros(1000,dtype="float32");x[500:]=1
 assert click_candidates(x,1000)==[500]


def test_broad_transient_ramp_is_not_mislabeled_as_click():
 x=np.zeros(1000,dtype="float32")
 x[500:510]=np.linspace(0,1,10,dtype="float32")
 x[510:]=1
 assert click_candidates(x,1000)==[]


def test_smooth_periodic_signal_has_no_click_candidate():
 t=np.arange(1000,dtype="float32")/1000
 x=np.sin(2*np.pi*10*t).astype("float32")
 assert click_candidates(x,1000)==[]
