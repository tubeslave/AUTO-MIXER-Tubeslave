from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

def sha256(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def state_path(root: str) -> Path:
    return Path(root)/".awb_checkpoints.json"

def load(root: str) -> dict[str,Any]:
    p=state_path(root)
    return json.loads(p.read_text()) if p.exists() else {"artifacts":{}}

def commit(root: str, key: str, path: str, inputs: dict[str,str],
           params: dict[str,Any] | None=None) -> dict[str,Any]:
    p=Path(path).resolve()
    if not p.exists(): raise FileNotFoundError(p)
    s=load(root); ident=sha256(str(p))
    row={"path":str(p),"sha256":ident,"inputs":dict(inputs),"params":params or {}}
    s["artifacts"][key]=row
    sp=state_path(root);sp.parent.mkdir(parents=True,exist_ok=True)
    sp.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding="utf-8")
    return row

def valid(root: str, key: str, inputs: dict[str,str],
          params: dict[str,Any] | None=None) -> dict[str,Any]:
    row=load(root).get("artifacts",{}).get(key)
    if not row: return {"valid":False,"reason":"missing_checkpoint"}
    p=Path(row["path"])
    if not p.exists(): return {"valid":False,"reason":"artifact_missing"}
    if row.get("inputs")!=dict(inputs) or row.get("params",{})!=(params or {}):
        return {"valid":False,"reason":"dependency_or_parameter_change"}
    current=sha256(str(p))
    if current!=row.get("sha256"): return {"valid":False,"reason":"artifact_changed"}
    return {"valid":True,"reason":"current","artifact":row}

def pending(root: str, graph: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for node in graph:
        r=valid(root,node["key"],node.get("inputs",{}),node.get("params",{}))
        if not r["valid"]: out.append({**node,"checkpoint_status":r})
    return out
