from __future__ import annotations
from dataclasses import dataclass,asdict
from itertools import product
from typing import Callable,Any

@dataclass(frozen=True)
class LoudnessBudget:
    pregain_db: float
    bus_max_gr_db: float
    multiband_max_gr_db: float
    clip_max_reduction_db: float
    limiter_max_gr_db: float=3.0

@dataclass
class SearchConfig:
    target_loud_section_lufs: float=-8.0
    tolerance_lu: float=.25
    limiter_hard_max_gr_db: float=3.0
    bus_grid: tuple=(.6,.9,1.2,1.5)
    multiband_grid: tuple=(.5,.8,1.0,1.25,1.5)
    clip_grid: tuple=(1.0,1.5,2.0,2.5,3.0)
    pregain_grid: tuple=(4.5,5.0,5.5,6.0,6.25,6.5,6.75,7.0)

def candidates(cfg:SearchConfig):
    for pg,b,m,c in product(cfg.pregain_grid,cfg.bus_grid,cfg.multiband_grid,cfg.clip_grid):
        yield LoudnessBudget(pg,b,m,c,cfg.limiter_hard_max_gr_db)

def _cost(row:dict,cfg:SearchConfig)->float:
    b=row["budget"];d=row["diagnostics"]
    target_error=abs(row["loud_section_lufs"]-cfg.target_loud_section_lufs)
    # Prefer distributed, low-depth processing. Final limiter is deliberately the most expensive stage.
    process=(.35*b["bus_max_gr_db"]+.45*b["multiband_max_gr_db"]+
             .55*b["clip_max_reduction_db"]+1.25*d["limiter_max_gr_db"])
    artifact=(1.0*d.get("spectral_shift_rms_db",0)+
              .8*max(0,-d.get("kick_attack_change_db",0))+
              .6*max(0,-d.get("snare_attack_change_db",0)))
    return 4.0*target_error+process+artifact

def search(evaluate:Callable[[LoudnessBudget],dict],cfg:SearchConfig|None=None)->dict[str,Any]:
    cfg=cfg or SearchConfig();rows=[]
    for b in candidates(cfg):
        r=evaluate(b);diag=r["diagnostics"]
        feasible=(diag["limiter_max_gr_db"]<=cfg.limiter_hard_max_gr_db+.02 and
                  abs(r["loud_section_lufs"]-cfg.target_loud_section_lufs)<=cfg.tolerance_lu and
                  r.get("regression_pass",True))
        row={"budget":asdict(b),**r,"feasible":feasible}
        row["cost"]=_cost(row,cfg);rows.append(row)
    feasible=[r for r in rows if r["feasible"]]
    feasible.sort(key=lambda r:r["cost"])
    return {"chosen":feasible[0] if feasible else None,
            "shortlist":feasible[:5],"evaluated":len(rows),"config":asdict(cfg),
            "rule":"Optimization minimizes bounded processing cost under loudness/limiter/regression constraints; human listening remains final acceptance."}
