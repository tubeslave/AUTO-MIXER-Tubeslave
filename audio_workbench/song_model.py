from __future__ import annotations
from typing import Any

def build_hierarchy(manifest: dict[str,Any]) -> dict[str,Any]:
    """Build a conservative hierarchy from manifest role guesses.
    Filename/role guesses are descriptive only; section intent can override priority.
    """
    groups={}
    for t in manifest.get("tracks",[]):
        role=t.get("role_guess","unknown")
        parent={"kick":"drums","snare":"drums","toms":"drums","overheads":"drums","drums":"drums",
                "bass":"bass","guitar":"guitars","keys":"keys","vocal":"vocals",
                "backing_vocal":"vocals","fx":"fx"}.get(role,"other")
        groups.setdefault(parent,[]).append({"id":t["name"],"name":t["name"],"role":role,
                                             "path":t["path"]})
    return {"schema_version":2,"root":"mix","groups":groups,
            "limitations":["role guesses are not artistic intent","explicit project context overrides guesses"]}

def section_priorities(manifest: dict[str,Any], section_name: str) -> dict[str,float]:
    defaults={"vocal":1.0,"bass":.9,"kick":.9,"snare":.85,"guitar":.7,"keys":.6,
              "overheads":.65,"drums":.7,"backing_vocal":.55,"fx":.35,"unknown":.5}
    p={t["name"]:defaults.get(t.get("role_guess","unknown"),.5) for t in manifest.get("tracks",[])}
    for s in manifest.get("sections",[]):
        if s.get("name")==section_name:
            p.update({k:float(v) for k,v in s.get("role_priorities",{}).items()})
    return p

def contributors(measurements: dict[str,float], hierarchy: dict[str,Any]) -> dict[str,Any]:
    total=sum(max(0,float(v)) for v in measurements.values())+1e-20
    groups={}
    for group,tracks in hierarchy["groups"].items():
        value=sum(max(0,float(measurements.get(t["name"],0))) for t in tracks)
        groups[group]={"value":value,"share":value/total}
    ranked=sorted(groups.items(),key=lambda kv:kv[1]["value"],reverse=True)
    return {"groups":dict(ranked),"total":total,
            "policy":"contribution is measurement attribution, not permission to attenuate"}
