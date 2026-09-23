import numpy as np
from audio_workbench.mastering import MasteringDirector,MasteringConfig
from audio_workbench.mastering.analyzer import analyze
from audio_workbench.mastering.decision import MasteringSafetyPolicy,evaluate

def tone(sr=48000,n=48000):
 t=np.arange(n)/sr
 return np.column_stack([.25*np.sin(2*np.pi*80*t)+.08*np.sin(2*np.pi*3000*t)]*2).astype("float32")

def test_analyzer_finite():
 a=analyze(tone(),48000)
 assert all(np.isfinite([a["crest_db"],a["side_mid_db"],a["correlation"]]))

def test_analyzer_true_peak_is_explicit_and_not_below_sample_peak():
 a=analyze(tone(),48000,include_true_peak=True)
 assert np.isfinite(a["true_peak_dbtp"])
 assert a["true_peak_dbtp"] >= a["sample_peak_dbfs"]-.05

def test_pipeline_finite_and_same_shape():
 x=tone();y,r=MasteringDirector().render(x,48000)
 assert y.shape==x.shape and np.isfinite(y).all()
 assert len(r["events"])==5
 assert "true_peak_dbtp" in r["after"]
 assert r["safety"]["requires_human_listening"] is True
 assert r["safety"]["baseline_eligible"] is False

def test_maximizer_respects_ceiling_approximately():
 x=tone()*3
 y,r=MasteringDirector(MasteringConfig(stabilizer=False,clarity=False,impact=False,clipper=False,ceiling_db=-1,maximizer_drive_db=2)).render(x,48000)
 assert np.max(np.abs(y)) <= 10**(-1/20)*1.03

def test_mastering_safety_rejects_true_peak_violation():
 before={"crest_db":12.0,"side_mid_db":-8.0,"correlation":.8}
 after={"crest_db":10.0,"side_mid_db":-8.2,"correlation":.8,"true_peak_dbtp":-.7}
 r=evaluate(before,after,MasteringSafetyPolicy(true_peak_ceiling_dbtp=-1.0))
 assert r["verdict"]=="rejected"
 assert r["machine_safe"] is False
 assert "true_peak_ceiling_exceeded" in r["protected_regressions"]

def test_safe_master_still_waits_for_human_listening():
 before={"crest_db":12.0,"side_mid_db":-8.0,"correlation":.8}
 after={"crest_db":10.0,"side_mid_db":-8.2,"correlation":.8,"true_peak_dbtp":-1.2}
 r=evaluate(before,after,MasteringSafetyPolicy(true_peak_ceiling_dbtp=-1.0))
 assert r["machine_safe"] is True
 assert r["verdict"]=="pending_human_review"
 assert r["baseline_eligible"] is False
