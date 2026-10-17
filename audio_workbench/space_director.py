from __future__ import annotations
from typing import Any

ROLES=("dry","medium","large")

def profile(role: str) -> dict[str,float]:
    presets={
      "dry":{"send_db":-6.0,"decay_s":.55,"predelay_ms":10.0,"late_ratio":.35,"width":.65},
      "medium":{"send_db":-3.0,"decay_s":.95,"predelay_ms":18.0,"late_ratio":.55,"width":.82},
      "large":{"send_db":0.0,"decay_s":1.55,"predelay_ms":28.0,"late_ratio":.75,"width":1.0},
    }
    if role not in presets: raise ValueError(f"unknown space role: {role}")
    return dict(presets[role])

def section_plan(sections: list[dict[str,Any]], role_map: dict[str,str]) -> dict[str,Any]:
    rows=[]
    for s in sections:
        name=s["name"]
        role=role_map.get(name)
        if role is None: continue
        rows.append({"section":name,"start_s":float(s["start_s"]),"end_s":float(s["end_s"]),
                     "role":role,"profile":profile(role)})
    return {"sections":rows,
            "policy":"space is section-dependent; presets are experiment starting points, not automatic artistic truth"}

def validate_transition(plan: dict[str,Any], max_send_jump_db: float=6.0,
                        min_crossfade_ms: float=100.0) -> dict[str,Any]:
    rows=plan.get("sections",[])
    issues=[]
    for a,b in zip(rows,rows[1:]):
        jump=abs(float(b["profile"]["send_db"])-float(a["profile"]["send_db"]))
        if jump>max_send_jump_db:
            issues.append({"from":a["section"],"to":b["section"],"send_jump_db":jump})
    return {"passed":not issues,"issues":issues,"min_crossfade_ms":min_crossfade_ms}

def hypotheses(plan: dict[str,Any]) -> list[dict[str,Any]]:
    return [{"section":r["section"],"hypothesis":f"{r['role']} depth better supports this section role",
             "interventions":[{"type":"bypass","params":{}},
                              {"type":"space_profile","params":r["profile"]}],
             "protected":["punch","vocal_clarity","mono_compatibility","macro_contrast"]}
            for r in plan.get("sections",[])]
