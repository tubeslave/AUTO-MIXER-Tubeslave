from __future__ import annotations

def classify(human_audible: bool|None, musical_critic:dict,
             loudness_gain_lu:float, quality_metric_gain:float|None=None)->dict:
    """Separate quality upgrades from delivery-only changes."""
    failures=musical_critic.get("failures",[])
    if failures:
        return {"class":"reject","promote_quality_baseline":False,"reason":"musical regression: "+",".join(failures)}
    if human_audible is False:
        return {"class":"delivery_variant","promote_quality_baseline":False,
                "reason":"safe loudness increase but no meaningful level-matched audible quality change",
                "loudness_gain_lu":loudness_gain_lu}
    if human_audible is True:
        return {"class":"quality_candidate","promote_quality_baseline":True,
                "reason":"musical guards pass and level-matched improvement is audible"}
    return {"class":"unverified_candidate","promote_quality_baseline":False,
            "reason":"musical guards pass; human level-matched significance not yet verified"}
