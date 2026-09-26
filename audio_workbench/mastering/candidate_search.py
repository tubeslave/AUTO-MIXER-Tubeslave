from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Callable,Any

@dataclass(frozen=True)
class Candidate:
    name:str
    changes:dict[str,Any]
    hypothesis:str

DEFAULT_CANDIDATES=(
 Candidate("clarity_light",{"clarity_strength":.10},"Less spectral smoothing may preserve texture while retaining cleanup."),
 Candidate("impact_light",{"impact_max_gr_db":.75},"Less transient conditioning may preserve drum attack."),
 Candidate("clip_light",{"clip_drive_db":.40},"Less clipping may preserve microdynamics."),
 Candidate("bass_light",{"bass_max_db":.45},"Less low-end sustain correction may preserve weight."),
 Candidate("stabilizer_half",{"stabilizer_scale":.50},"Smaller tonal correction may preserve source character."),
 Candidate("max_drive_plus",{"maximizer_drive_db":2.50},"More band-limited loudness may improve density without broadband pumping."),
)

def classify(critic:dict,metrics:dict,baseline_metrics:dict)->str:
    if not critic.get("accept",False): return "reject"
    crest_loss=baseline_metrics["crest_db"]-metrics["crest_db"]
    if crest_loss>2.5:return "reject"
    # Passing a critic does not prove an audible improvement.
    return "survivor"

def search(render:Callable[[Candidate],dict],baseline_metrics:dict,
           candidates=DEFAULT_CANDIDATES)->dict:
    rows=[]
    for c in candidates:
        r=render(c);status=classify(r["critic"],r["metrics"],baseline_metrics)
        rows.append({"candidate":asdict(c),"status":status,**r})
    survivors=[r for r in rows if r["status"]=="survivor"]
    # Pareto-ish shortlist: smallest musical deviation first. Never label a winner.
    survivors.sort(key=lambda r:(r["critic"]["spectral_shift"]["rms_shift_db"],
                                 abs(r["critic"]["low_end_punch"]["median_punch_change_db"]),
                                 abs(r["metrics"]["crest_db"]-baseline_metrics["crest_db"])))
    return {"rows":rows,"shortlist":survivors[:3],
            "rule":"Shortlist is low-regression, not a quality ranking. Human level-matched A/B decides."}
