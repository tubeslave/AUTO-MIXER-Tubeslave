from __future__ import annotations
from dataclasses import dataclass,asdict

@dataclass
class MasteringSafetyPolicy:
    true_peak_ceiling_dbtp: float=-1.0
    true_peak_tolerance_db: float=.05
    max_crest_loss_db: float=4.0
    max_width_shift_db: float=1.0
    min_correlation: float=0.0
    target_lufs: float|None=None
    loudness_tolerance_lu: float=.5
    require_limiter_evidence: bool=False
    max_final_limiter_gr_db: float=3.0
    max_band_limiter_gr_db: float=4.0
    require_human_listening: bool=True

def choose_modules(a:dict)->dict:
    return {
      "stabilizer":True,
      "clarity":True,
      "bass_director": a["bands_db"]["sub"] > a["bands_db"]["mid"]+1.0 or a["bands_db"]["low"] > a["bands_db"]["mid"]+3.5,
      "stereo_director": a["correlation"]>.88 or a["correlation"]<.25,
      "impact": a["crest_db"]>13.0,
      "exciter": a["bands_db"]["air"] < a["bands_db"]["presence"]-4.0,
      "clipper":True,
      "maximizer":True,
    }

def evaluate(before:dict,after:dict,policy:MasteringSafetyPolicy|None=None,processing_evidence:dict|None=None)->dict:
    """Return an auditable machine-safety verdict for an audible mastering candidate."""
    p=policy or MasteringSafetyPolicy();fail=[];uncertainty=[];processing_evidence=processing_evidence or {}
    tp=after.get("true_peak_dbtp")
    if tp is None:
        fail.append("true_peak_unmeasured");uncertainty.append("true_peak_unmeasured")
    elif tp>p.true_peak_ceiling_dbtp+p.true_peak_tolerance_db:
        fail.append("true_peak_ceiling_exceeded")
    crest_loss=before["crest_db"]-after["crest_db"]
    if crest_loss>p.max_crest_loss_db: fail.append("excessive_crest_loss")
    width_shift=after["side_mid_db"]-before["side_mid_db"]
    if abs(width_shift)>p.max_width_shift_db: fail.append("excessive_width_shift")
    if after["correlation"]<p.min_correlation: fail.append("negative_stereo_correlation")

    measured_lufs=after.get("integrated_lufs")
    lufs_method=after.get("integrated_lufs_method")
    loudness_error_lu=None
    if p.target_lufs is not None:
        if measured_lufs is None or lufs_method!="pyloudnorm":
            fail.append("loudness_target_unmeasured")
            uncertainty.append("loudness_target_unmeasured")
        else:
            loudness_error_lu=float(measured_lufs-p.target_lufs)
            if abs(loudness_error_lu)>p.loudness_tolerance_lu:
                fail.append("loudness_target_missed")

    limiter=processing_evidence.get("maximizer")
    final_limiter_gr_db=None;worst_band_limiter_gr_db=None
    if p.require_limiter_evidence:
        if not isinstance(limiter,dict) or "final_max_gr_db" not in limiter or not isinstance(limiter.get("bands"),list):
            fail.append("limiter_gr_unmeasured")
            uncertainty.append("limiter_gr_unmeasured")
        else:
            final_limiter_gr_db=float(limiter["final_max_gr_db"])
            band_gr=[float(b.get("max_gr_db",0.0)) for b in limiter["bands"]]
            worst_band_limiter_gr_db=max(band_gr,default=0.0)
            if final_limiter_gr_db>p.max_final_limiter_gr_db:
                fail.append("final_limiter_gr_budget_exceeded")
            if worst_band_limiter_gr_db>p.max_band_limiter_gr_db:
                fail.append("band_limiter_gr_budget_exceeded")

    machine_safe=not fail
    requires_human=bool(p.require_human_listening)
    verdict="rejected" if fail else ("pending_human_review" if requires_human else "machine_safe")
    uncertainty_score=1.0 if uncertainty else (0.25 if requires_human else 0.0)
    return {
      "verdict":verdict,"machine_safe":machine_safe,"baseline_eligible":machine_safe and not requires_human,
      "requires_human_listening":requires_human,"protected_regressions":fail,
      "uncertainty":{"score":uncertainty_score,"reasons":uncertainty},
      "evidence":{
        "true_peak_dbtp":tp,"true_peak_ceiling_dbtp":p.true_peak_ceiling_dbtp,
        "crest_loss_db":crest_loss,"max_crest_loss_db":p.max_crest_loss_db,
        "width_shift_db":width_shift,"max_width_shift_db":p.max_width_shift_db,
        "correlation":after["correlation"],"min_correlation":p.min_correlation,
        "target_lufs":p.target_lufs,"integrated_lufs":measured_lufs,"integrated_lufs_method":lufs_method,
        "loudness_error_lu":loudness_error_lu,"loudness_tolerance_lu":p.loudness_tolerance_lu,
        "final_limiter_gr_db":final_limiter_gr_db,"max_final_limiter_gr_db":p.max_final_limiter_gr_db,
        "worst_band_limiter_gr_db":worst_band_limiter_gr_db,"max_band_limiter_gr_db":p.max_band_limiter_gr_db,
      },
      "policy":asdict(p),
    }

def accept(before:dict,after:dict)->dict:
    failures=[]
    if after["correlation"] < 0.0: failures.append("negative_stereo_correlation")
    if after["crest_db"] < before["crest_db"]-5.0: failures.append("excessive_crest_loss")
    if abs(after["side_mid_db"]-before["side_mid_db"])>2.0: failures.append("excessive_width_shift")
    return {"accept":not failures,"failures":failures}
