from __future__ import annotations
from typing import Any

def accumulate(baseline:dict[str,Any],accepted:list[dict[str,Any]],
               max_changes:int=3)->dict[str,Any]:
    """Combine small non-conflicting survivor deltas, then require a fresh regression pass.
    This never assumes individually safe deltas are safe in combination."""
    merged=dict(baseline);used=[];keys=set()
    for row in accepted:
        change=row.get("changes",{})
        if keys.intersection(change): continue
        merged.update(change);keys.update(change);used.append(row.get("name","candidate"))
        if len(used)>=max_changes: break
    return {"config":merged,"components":used,
            "requires_fresh_render":True,"requires_fresh_musical_critic":True,
            "requires_level_matched_human_check":True}

def cumulative_accept(individual_pass:bool,compound_critic:dict)->bool:
    return bool(individual_pass and compound_critic.get("accept",False))
