from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class Goal:
    id:str
    title:str
    objective:str
    acceptance:list[str]
    blocked_by:list[str]
    priority:int

ROADMAP=[
 Goal("live-soundcheck-v2","Live Soundcheck Pipeline v2",
      "Build a separate realtime pipeline: local multichannel audio capture, WING state/control, live directors, safety governor and verified USB/Dante routing.",
      ["USB capture MVP","WING snapshot/readback","proposal safety governor","soundcheck FSM","HIL test"],[],98),
 Goal("cleanup-model-validation","Validate model-based source cleanup",
      "Select per-source pretrained cleanup only after RAW/CLEAN/REMOVED listening and artifact checks.",
      ["full-song cleaned mix rendered","removed residual audited","no gate-like fallback","accepted strengths recorded"],[],100),
 Goal("perceptual-critic","Perceptual Mix Critic v2",
      "Measure foreground/background, intelligibility, punch, harshness, depth, width and climax without a fake overall score.",
      ["artifact critic integrated","section context integrated","foreground/background evidence implemented","tests pass"],[],95),
 Goal("autonomous-iteration-v2","Autonomous Iteration v2",
      "Let evidence select one bounded mix hypothesis, render B, then accept or roll back.",
      ["one change per iteration","artifact guardrail","loudness guardrail","rollback log","iteration budget"],["perceptual-critic"],90),
 Goal("mastering-director","Mastering Director",
      "Prepare loud sections toward -8 LUFS while keeping final limiter GR normally <=3 dB.",
      ["pre-limiter crest management","clip/compression stages bounded","true-peak guard","A/B loudness matched"],["autonomous-iteration-v2"],80),
 Goal("live-transfer","Live Automixer transfer",
      "Map validated offline decisions into supervised low-latency live control.",
      ["bounded writes","hysteresis","rollback","HIL test","latency budget"],["autonomous-iteration-v2"],70),
]

def load_state(path:Path)->dict:
    if path.exists(): return json.loads(path.read_text())
    return {"completed":[],"active":None,"history":[]}

def choose_goal(state:dict)->Goal|None:
    done=set(state.get("completed",[]))
    candidates=[g for g in ROADMAP if g.id not in done and all(x in done for x in g.blocked_by)]
    return max(candidates,key=lambda g:g.priority) if candidates else None

def assign(path:Path)->dict:
    state=load_state(path);g=choose_goal(state)
    state["active"]=asdict(g) if g else None
    if g: state["history"].append({"event":"assigned","goal":g.id})
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    return state
