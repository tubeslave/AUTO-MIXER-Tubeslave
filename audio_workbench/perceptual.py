from __future__ import annotations
import hashlib
from typing import Any

AXES=("balance","punch","clarity","vocal_placement","low_end","harshness",
      "depth","stereo","groove","macro_contrast","overall")

def blind_assignment(a: str,b: str,trial_id: str) -> dict[str,str]:
    h=hashlib.sha256((trial_id+"|"+a+"|"+b).encode()).digest()[0]
    return {"A":a,"B":b} if h%2==0 else {"A":b,"B":a}

def validate_verdict(verdict: dict[str,Any]) -> dict[str,Any]:
    pref=verdict.get("preference")
    if pref not in ("A","B","tie","uncertain"):
        raise ValueError("preference must be A/B/tie/uncertain")
    conf=float(verdict.get("confidence",0))
    if not 0<=conf<=1: raise ValueError("confidence outside 0..1")
    axes=verdict.get("axes",{})
    unknown=set(axes)-set(AXES)
    if unknown: raise ValueError(f"unknown perceptual axes: {sorted(unknown)}")
    for k,v in axes.items():
        if v not in ("A","B","tie","uncertain"):
            raise ValueError(f"invalid axis verdict {k}: {v}")
    return {"preference":pref,"confidence":conf,"axes":axes,
            "reason":verdict.get("reason","")}

def aggregate(verdicts: list[dict[str,Any]], min_confidence: float=.65) -> dict[str,Any]:
    valid=[validate_verdict(v) for v in verdicts]
    weighted={"A":0.0,"B":0.0}
    uncertain=0
    for v in valid:
        if v["preference"] in ("tie","uncertain") or v["confidence"]<min_confidence:
            uncertain+=1; continue
        weighted[v["preference"]]+=v["confidence"]
    total=weighted["A"]+weighted["B"]
    if total<=0:
        pref="uncertain"; conf=0.0
    else:
        margin=abs(weighted["A"]-weighted["B"])/total
        pref=max(weighted,key=weighted.get) if margin>=.2 else "uncertain"
        conf=margin
    return {"preference":pref,"confidence":conf,"weighted":weighted,
            "uncertain_votes":uncertain,"n":len(valid)}

def catch_trial_result(file_a_sha: str,file_b_sha: str,verdict: dict[str,Any]) -> dict[str,Any]:
    v=validate_verdict(verdict)
    identical=file_a_sha==file_b_sha
    passed=not identical or v["preference"] in ("tie","uncertain") or v["confidence"]<.5
    return {"identical":identical,"passed":passed,
            "policy":"confident preference on identical audio disables judge authority"}
