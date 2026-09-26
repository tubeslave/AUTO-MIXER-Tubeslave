from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class LearnedEvidence:
    name:str
    value:float
    confidence:float
    detail:dict

class PerceptualAdapter(Protocol):
    def analyze(self,audio:np.ndarray,sr:int,**context)->list[LearnedEvidence]: ...

class AdapterFusion:
    """Learned evidence can strengthen/weakly challenge a deterministic hypothesis,
    but cannot authorize a destructive edit by itself."""
    def fuse(self,hypothesis:dict,evidence:list[LearnedEvidence])->dict:
        related=[e for e in evidence if e.name==hypothesis.get("target")]
        support=np.mean([e.value*e.confidence for e in related]) if related else 0.
        out=dict(hypothesis)
        out["learned_support"]=float(support)
        out["requires_render_and_guardrails"]=True
        out["requires_human_ab_during_development"]=True
        return out
