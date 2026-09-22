from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Any

@dataclass
class Hypothesis:
    name:str
    target:str
    rationale:str
    max_change_db:float
    priority:float
    params:dict

def choose_hypothesis(diagnostics:dict)->Hypothesis|None:
    """Choose one bounded problem at a time. No kitchen-sink iteration."""
    candidates=[]
    if diagnostics.get("vocal_section_spread_db",0)>3.0:
        candidates.append(Hypothesis("vocal_section_ride","vocal",
            "Vocal prominence varies excessively between active song sections.",.8,
            diagnostics["vocal_section_spread_db"],{}))
    if abs(diagnostics.get("kick_bass_section_drift_db",0))>2.0:
        candidates.append(Hypothesis("low_end_section_rebalance","bass",
            "Kick/bass relationship drifts across sections.",.7,
            abs(diagnostics["kick_bass_section_drift_db"]),{}))
    if diagnostics.get("dense_section_width_deficit_db",0)>.8:
        candidates.append(Hypothesis("dense_width_support","guitar",
            "Dense sections do not gain enough width relative to sparse sections.",.6,
            diagnostics["dense_section_width_deficit_db"],{}))
    if diagnostics.get("section_loudness_jump_db",0)>1.2:
        candidates.append(Hypothesis("section_level_smoothing","mix",
            "Section transition changes broadband level too abruptly.",.5,
            diagnostics["section_loudness_jump_db"],{}))
    return max(candidates,key=lambda x:x.priority) if candidates else None

def accept_candidate(before:dict,after:dict,h:Hypothesis)->dict[str,Any]:
    fail=[]
    # Target metric must improve without collateral regressions.
    target_map={
      "vocal_section_ride":"vocal_section_spread_db",
      "low_end_section_rebalance":"kick_bass_section_drift_db",
      "dense_width_support":"dense_section_width_deficit_db",
      "section_level_smoothing":"section_loudness_jump_db",
    }
    key=target_map[h.name]
    b=abs(before.get(key,0));a=abs(after.get(key,0))
    if a>=b-.08: fail.append("target_not_improved")
    if after.get("peak_dbfs",-99)>-1.0: fail.append("headroom_regression")
    if abs(after.get("lufs_i",0)-before.get("lufs_i",0))>.35: fail.append("loudness_cheat")
    if after.get("broad_spectral_shift_db",0)>.45: fail.append("tonal_regression")
    return {"accept":not fail,"failures":fail,"target_before":b,"target_after":a}
