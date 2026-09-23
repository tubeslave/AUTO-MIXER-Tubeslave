from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from .analyzer import analyze
from . import stabilizer,clarity,impact,clipper,maximizer
from .decision import MasteringSafetyPolicy,evaluate

@dataclass
class MasteringConfig:
    stabilizer: bool=True
    clarity: bool=True
    impact: bool=True
    clipper: bool=True
    maximizer: bool=True
    clarity_strength: float=.18
    clip_drive_db: float=.8
    maximizer_drive_db: float=2.0
    ceiling_db: float=-1.0

class MasteringDirector:
    def __init__(self,config:MasteringConfig|None=None): self.config=config or MasteringConfig()
    def render(self,x:np.ndarray,sr:int):
        y=np.asarray(x,dtype=np.float32); before=analyze(y,sr,include_true_peak=True);events=[]
        if self.config.stabilizer:
            y,s=stabilizer.process(y,sr,before);events.append({"module":"stabilizer",**s})
        if self.config.clarity:
            y,s=clarity.process(y,sr,self.config.clarity_strength);events.append({"module":"clarity",**s})
        if self.config.impact:
            y,s=impact.process(y,sr);events.append({"module":"impact",**s})
        if self.config.clipper:
            y,s=clipper.process(y,sr,self.config.clip_drive_db);events.append({"module":"clipper",**s})
        if self.config.maximizer:
            y,s=maximizer.process(y,sr,self.config.ceiling_db,self.config.maximizer_drive_db);events.append({"module":"maximizer",**s})
        after=analyze(y,sr,include_true_peak=True)
        regression={"crest_change_db":after["crest_db"]-before["crest_db"],
          "side_mid_change_db":after["side_mid_db"]-before["side_mid_db"],
          "correlation_change":after["correlation"]-before["correlation"]}
        safety=evaluate(before,after,MasteringSafetyPolicy(true_peak_ceiling_dbtp=self.config.ceiling_db))
        return y,{"config":asdict(self.config),"before":before,"after":after,"events":events,"regression":regression,"safety":safety}
