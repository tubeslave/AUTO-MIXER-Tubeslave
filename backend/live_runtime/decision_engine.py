from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .contracts import ChannelFeatures,MixFeatures,ProposedAction

@dataclass(frozen=True)
class LiveHypothesis:
    name:str
    target:str
    reason:str
    confidence:float
    action:ProposedAction
    verify_metric:str
    min_improvement:float

def _by_name(channels:Iterable[ChannelFeatures],needle:str)->list[ChannelFeatures]:
    n=needle.lower()
    return [c for c in channels if n in c.name.lower()]

def propose_one(features:MixFeatures,roles:dict[int,str],
                masking:dict[tuple[int,int],float]|None=None)->LiveHypothesis|None:
    """Studio-style live reasoning: one evidence-backed bounded hypothesis at a time."""
    masking=masking or {}
    candidates:list[LiveHypothesis]=[]
    leads=[c for c in features.channels if roles.get(c.channel)=="lead_vocal"]
    music=[c for c in features.channels if roles.get(c.channel) in {"guitar","keys","playback","music_bus"}]

    # Vocal masking: low vocal audibility + measured overlap, not 'vocal dominates' heuristic.
    for v in leads:
        if v.activity<.45: continue
        for m in music:
            overlap=float(masking.get((v.channel,m.channel),0.))
            if overlap<.35: continue
            # harshness here is used only as supporting evidence on the masker.
            conf=min(.94,.58+.28*overlap+.08*max(0.,m.harshness or 0.))
            candidates.append(LiveHypothesis(
                "vocal_masking_release",f"ch:{v.channel}",
                f"{m.name} overlaps active lead vocal; release presence on masker, not boost master.",
                conf,
                ProposedAction(f"ch:{m.channel}","eq_gain_db",-0.7,
                    f"free lead vocal from {m.name} masking",conf,max_step=1.0,risk="low"),
                "vocal_intelligibility",.03))

    # Headroom: source/group correction is preferred; main cut is a last safety move.
    if features.main_peak_dbfs>-2.0:
        conf=min(.99,.82+(features.main_peak_dbfs+2)*.04)
        candidates.append(LiveHypothesis(
            "main_headroom_protection","main:1","Main peak headroom is below live corridor.",
            conf,ProposedAction("main:1","fader_db",-0.5,"restore main headroom",conf,max_step=.5,risk="low"),
            "main_peak_dbfs",.25))

    # Harsh source correction. Do not touch inactive channels.
    for c in features.channels:
        if c.activity>.45 and (c.harshness or 0)>.78:
            conf=min(.9,.62+.3*(c.harshness or 0))
            candidates.append(LiveHypothesis(
                "source_harshness",f"ch:{c.channel}",f"{c.name} has persistent upper-presence excess.",
                conf,ProposedAction(f"ch:{c.channel}","eq_gain_db",-0.6,
                    "bounded dynamic/PEQ presence reduction",conf,max_step=.8,risk="low"),
                "harshness",.04))

    return max(candidates,key=lambda h:h.confidence) if candidates else None

def verify(before:dict,after:dict,h:LiveHypothesis)->dict:
    """Accept only if the intended metric improves without live collateral damage."""
    fail=[]
    b=before.get(h.verify_metric);a=after.get(h.verify_metric)
    if b is None or a is None:
        fail.append("verification_metric_missing")
    else:
        if h.verify_metric=="main_peak_dbfs":
            improved=(a<=b-h.min_improvement)
        elif h.name=="source_harshness":
            improved=(a<=b-h.min_improvement)
        else:
            improved=(a>=b+h.min_improvement)
        if not improved: fail.append("target_not_improved")
    if after.get("main_peak_dbfs",-99)>-1.0: fail.append("headroom_regression")
    if after.get("feedback_risk",0)>before.get("feedback_risk",0)+.08: fail.append("feedback_risk_regression")
    if after.get("operator_touch",False): fail.append("operator_took_control")
    return {"accept":not fail,"failures":fail,"rollback":bool(fail)}
