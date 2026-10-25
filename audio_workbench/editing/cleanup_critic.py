from __future__ import annotations
def evaluate(metrics:dict)->dict:
    fail=[]
    if metrics.get("new_clicks",0)>0:fail.append("new_clicks")
    if metrics.get("transient_loss_db",0)>1.0:fail.append("transient_damage")
    if metrics.get("active_note_rms_change_db",0)<-1.2:fail.append("musical_material_removed")
    if metrics.get("phase_delta",0)>.08:fail.append("phase_regression")
    if metrics.get("artifact_reduction_db",0)<1.0:fail.append("insufficient_cleanup")
    return {"accept":not fail,"failures":fail}
