from __future__ import annotations
from typing import Any

def significance_gate(delta_rms_dbfs: float, affected_fraction: float,
                      perceptual_verdict: str | None=None,
                      min_delta_rms_dbfs: float=-50.0,
                      min_affected_fraction: float=.02) -> dict[str,Any]:
    """Stop spending iterations on vanishingly small changes.
    This is an engineering relevance gate, not a universal audibility model.
    """
    tiny = float(delta_rms_dbfs) < min_delta_rms_dbfs or float(affected_fraction) < min_affected_fraction
    human_tie = perceptual_verdict in ("tie","uncertain")
    meaningful = not tiny and not human_tie
    reason = "meaningful_candidate"
    if human_tie: reason="blind_ab_tie_or_uncertain"
    elif tiny: reason="effect_below_engineering_significance_floor"
    return {"meaningful":meaningful,"reason":reason,
            "delta_rms_dbfs":float(delta_rms_dbfs),"affected_fraction":float(affected_fraction),
            "thresholds":{"min_delta_rms_dbfs":min_delta_rms_dbfs,
                          "min_affected_fraction":min_affected_fraction},
            "policy":"below-threshold changes are not auto-optimized repeatedly; escalate to a larger causal hypothesis or stop"}

def update_problem_after_tie(problem: dict[str,Any]) -> dict[str,Any]:
    q=dict(problem)
    q["status"]="deprioritized"
    q["reason"]="blind_ab_no_meaningful_difference"
    q["expected_impact"]=min(float(q.get("expected_impact",.5)),.2)
    q["uncertainty"]=max(float(q.get("uncertainty",0)),.7)
    return q


def assess_blind_trials(choices: list[str], key: list[dict[str,str]],
                        proposed_labels: set[str]) -> dict[str,Any]:
    if len(choices)!=len(key):
        raise ValueError("choices/key length mismatch")
    resolved=[]
    for choice,mapping in zip(choices,key):
        if choice in ("tie","uncertain"):
            resolved.append(choice)
        else:
            if choice not in mapping:
                raise ValueError(f"unknown blind label: {choice}")
            resolved.append(mapping[choice])
    baseline=sum(1 for x in resolved if x=="baseline")
    proposed=sum(1 for x in resolved if x in proposed_labels)
    ties=sum(1 for x in resolved if x in ("tie","uncertain"))
    n=len(resolved)
    if baseline==n:
        verdict="reject_intervention_family"
    elif proposed>baseline and proposed>=max(2,(n+1)//2):
        verdict="perceptually_significant_candidate"
    else:
        verdict="insufficient_or_mixed"
    return {"resolved":resolved,"baseline_votes":baseline,"proposed_votes":proposed,
            "tie_uncertain_votes":ties,"verdict":verdict,
            "policy":"consistent baseline preference is evidence against processing, not a request for a subtler version"}

def effect_budget(previous_results: list[dict[str,Any]]) -> dict[str,Any]:
    rejected=sum(1 for r in previous_results if r.get("verdict")=="reject_intervention_family")
    weak=sum(1 for r in previous_results if r.get("verdict")=="insufficient_or_mixed")
    action="change_problem_or_intervention_family" if rejected else ("raise_minimum_effect_size" if weak>=2 else "continue")
    return {"rejected_families":rejected,"weak_results":weak,"next_action":action}


def minimum_effect_policy(history: list[dict[str,Any]]) -> dict[str,Any]:
    """Escalate away from micro-polish after repeated inaudible/uncertain experiments."""
    weak=sum(1 for r in history if r.get("result") in ("tie","uncertain","inaudible"))
    rejected=sum(1 for r in history if r.get("result")=="rejected")
    if weak>=2:
        return {"mode":"macro_or_source_rebalance",
                "minimum_expected_audibility":"clearly_audible",
                "forbid":["sub_db_micro_eq","sub_db_section_gain","more_mix_bus_density"],
                "reason":"repeated blind tests were below useful perceptual significance"}
    if rejected:
        return {"mode":"change_intervention_family","minimum_expected_audibility":"audible",
                "forbid":[],"reason":"previous family lost blind preference"}
    return {"mode":"normal","minimum_expected_audibility":"small_but_audible","forbid":[]}
