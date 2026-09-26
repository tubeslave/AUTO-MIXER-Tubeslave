from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Any

@dataclass
class LoudnessSearchConfig:
    start_drive_db: float=2.0
    step_db: float=.5
    max_drive_db: float=5.0
    max_crest_loss_db: float=3.5
    max_width_shift_db: float=1.0
    max_band_gr_db: float=1.5
    max_final_gr_db: float=1.25

def regression_ok(base:dict,candidate:dict,diagnostics:dict,cfg:LoudnessSearchConfig)->tuple[bool,list[str]]:
    fail=[]
    if base["crest_db"]-candidate["crest_db"]>cfg.max_crest_loss_db: fail.append("crest_loss")
    if abs(candidate["side_mid_db"]-base["side_mid_db"])>cfg.max_width_shift_db: fail.append("width_shift")
    if diagnostics.get("max_band_gr_db",0)>cfg.max_band_gr_db: fail.append("band_gr")
    if diagnostics.get("final_max_gr_db",0)>cfg.max_final_gr_db: fail.append("final_gr")
    if candidate.get("correlation",1)<0: fail.append("negative_correlation")
    return not fail,fail

def search(render_and_measure:Callable[[float],dict],base:dict,cfg:LoudnessSearchConfig|None=None)->dict[str,Any]:
    cfg=cfg or LoudnessSearchConfig();drive=cfg.start_drive_db;accepted=[];rejected=[]
    while drive<=cfg.max_drive_db+1e-9:
        result=render_and_measure(drive)
        ok,fail=regression_ok(base,result["metrics"],result.get("diagnostics",{}),cfg)
        row={"drive_db":drive,**result,"accepted":ok,"failures":fail}
        (accepted if ok else rejected).append(row)
        if not ok: break
        drive+=cfg.step_db
    chosen=accepted[-1] if accepted else None
    return {"chosen":chosen,"accepted":accepted,"rejected":rejected,"config":cfg.__dict__}
