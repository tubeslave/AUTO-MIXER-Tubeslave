from __future__ import annotations
from .contracts import LiveMode,ProposedAction

DEFAULT_LIMITS={
    "fader_db":1.0,
    "eq_gain_db":1.5,
    "eq_freq_ratio":1.35,
    "comp_threshold_db":2.0,
    "send_db":1.0,
    "pan":0.10,
}

BLOCKED_IN_AUTO={"routing","source","phantom","sample_rate","clock","scene_recall","preamp_gain"}

def authorize(action:ProposedAction,mode:LiveMode,manual_freeze:bool=False)->tuple[bool,str]:
    if manual_freeze or mode==LiveMode.FREEZE:
        return False,"frozen"
    if mode in (LiveMode.OBSERVE,LiveMode.PROPOSE):
        return False,"writes_disabled"
    if mode==LiveMode.AUTO_SAFE and action.parameter in BLOCKED_IN_AUTO:
        return False,"parameter_not_allowlisted"
    if mode==LiveMode.AUTO_SAFE and (action.risk!="low" or not action.reversible):
        return False,"risk_not_allowed"
    if action.confidence<.75 and mode==LiveMode.AUTO_SAFE:
        return False,"confidence_too_low"
    return True,"authorized"
