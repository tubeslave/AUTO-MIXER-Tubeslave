from __future__ import annotations

from typing import Any

def role_priorities(manifest: dict[str,Any], section_name: str | None=None) -> dict[str,Any]:
    tracks=manifest.get("tracks",[])
    priorities={}
    for t in tracks:
        role=t.get("role_guess","unknown")
        base={"vocal":1.0,"kick":.9,"snare":.85,"bass":.9,"guitar":.7,"keys":.6,
              "overheads":.65,"drums":.7,"backing_vocal":.55,"fx":.35,"unknown":.5}.get(role,.5)
        priorities[t["name"]]=base
    # Explicit section context may override guesses.
    for s in manifest.get("sections",[]):
        if section_name and s.get("name")==section_name:
            for name,val in s.get("role_priorities",{}).items():
                priorities[name]=float(val)
    return {
      "section":section_name,"priorities":priorities,
      "limitations":["defaults are only starting priors; explicit musical intent overrides them",
                     "priority never authorizes automatic attenuation of another source"]
    }
