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
    # BENCH_TEST is an explicit engineering mode requested by the operator:
    # normal production allowlists, confidence gates and bounded-step policies are bypassed
    # so every console decision is visible on WING. Readback/audit remain enabled.
    if mode==LiveMode.BENCH_TEST:
        return True,"bench_test_unrestricted"
    if mode==LiveMode.AUTO_SAFE and action.parameter in BLOCKED_IN_AUTO:
        return False,"parameter_not_allowlisted"
    if mode==LiveMode.AUTO_SAFE and (action.risk!="low" or not action.reversible):
        return False,"risk_not_allowed"
    if action.confidence<.75 and mode==LiveMode.AUTO_SAFE:
        return False,"confidence_too_low"
    return True,"authorized"


def authorize_rollback(mode:LiveMode,manual_freeze:bool=False)->tuple[bool,str]:
    """Authorize restoration of a value captured by the control plane.

    A rollback is not a new musical decision: it restores the fresh pre-write
    value recorded by a previous verified execution. Production confidence,
    parameter allowlists and original action risk therefore do not block the
    restoration. Explicit no-write states still win, including manual freeze.
    """
    if manual_freeze or mode==LiveMode.FREEZE:
        return False,"frozen"
    if mode in (LiveMode.OBSERVE,LiveMode.PROPOSE):
        return False,"writes_disabled"
    return True,"rollback_authorized"
