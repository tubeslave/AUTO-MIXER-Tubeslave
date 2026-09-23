from __future__ import annotations
from dataclasses import dataclass,asdict

@dataclass
class MasteringSafetyPolicy:
    true_peak_ceiling_dbtp: float=-1.0
    true_peak_tolerance_db: float=.05
    max_crest_loss_db: float=4.0
    max_width_shift_db: float=1.0
    min_correlation: float=0.0
    require_human_listening: bool=True

def choose_modules(a:dict)->dict:
    # Bounded, explainable v0.2 policy. No genre/reference target required.
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

def evaluate(before:dict,after:dict,policy:MasteringSafetyPolicy|None=None)->dict:
    """Return an auditable machine-safety verdict for an audible mastering candidate.

    Machine safety is intentionally distinct from subjective acceptance: a technically
    safe mastered candidate remains pending until a human level-matched listen accepts it.
    """
    p=policy or MasteringSafetyPolicy();fail=[];uncertainty=[]
    tp=after.get("true_peak_dbtp")
    if tp is None:
        fail.append("true_peak_unmeasured")
        uncertainty.append("true_peak_unmeasured")
    elif tp>p.true_peak_ceiling_dbtp+p.true_peak_tolerance_db:
        fail.append("true_peak_ceiling_exceeded")
    crest_loss=before["crest_db"]-after["crest_db"]
    if crest_loss>p.max_crest_loss_db: fail.append("excessive_crest_loss")
    width_shift=after["side_mid_db"]-before["side_mid_db"]
    if abs(width_shift)>p.max_width_shift_db: fail.append("excessive_width_shift")
    if after["correlation"]<p.min_correlation: fail.append("negative_stereo_correlation")
    machine_safe=not fail
    requires_human=bool(p.require_human_listening)
    verdict="rejected" if fail else ("pending_human_review" if requires_human else "machine_safe")
    uncertainty_score=1.0 if uncertainty else (0.25 if requires_human else 0.0)
    return {
      "verdict":verdict,
      "machine_safe":machine_safe,
      "baseline_eligible":machine_safe and not requires_human,
      "requires_human_listening":requires_human,
      "protected_regressions":fail,
      "uncertainty":{"score":uncertainty_score,"reasons":uncertainty},
      "evidence":{
        "true_peak_dbtp":tp,
        "true_peak_ceiling_dbtp":p.true_peak_ceiling_dbtp,
        "crest_loss_db":crest_loss,
        "max_crest_loss_db":p.max_crest_loss_db,
        "width_shift_db":width_shift,
        "max_width_shift_db":p.max_width_shift_db,
        "correlation":after["correlation"],
        "min_correlation":p.min_correlation,
      },
      "policy":asdict(p),
    }

def accept(before:dict,after:dict)->dict:
    """Legacy technical gate retained for compatibility."""
    failures=[]
    if after["correlation"] < 0.0: failures.append("negative_stereo_correlation")
    if after["crest_db"] < before["crest_db"]-5.0: failures.append("excessive_crest_loss")
    if abs(after["side_mid_db"]-before["side_mid_db"])>2.0: failures.append("excessive_width_shift")
    return {"accept":not failures,"failures":failures}
