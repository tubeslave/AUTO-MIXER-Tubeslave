from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal

from . import core
from .compare import compare

def _read(path: str):
    return sf.read(path, always_2d=True, dtype="float32")

def _write(path: Path, y: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(y).all():
        raise ValueError("candidate contains NaN/Inf")
    # Preserve overload evidence. FLOAT avoids silently hard-clipping probes at +/-1.
    sf.write(path, y.astype("float32"), sr, subtype="FLOAT")

def _gain(y, sr, p):
    return y * (10.0 ** (float(p.get("db", 0.0))/20.0))

def _eq_bell(y, sr, p):
    f=float(p["freq_hz"]); q=float(p.get("q",0.8)); db=float(p["db"])
    if not 20 < f < sr/2:
        raise ValueError("freq_hz outside audio band")
    # RBJ peaking EQ biquad
    A=10**(db/40); w0=2*np.pi*f/sr; alpha=np.sin(w0)/(2*q); c=np.cos(w0)
    b=np.array([1+alpha*A,-2*c,1-alpha*A])
    a=np.array([1+alpha/A,-2*c,1-alpha/A])
    b/=a[0]; a/=a[0]
    return signal.lfilter(b,a,y,axis=0).astype("float32")

def _compress(y, sr, p):
    threshold=float(p.get("threshold_db",-18)); ratio=max(1.0,float(p.get("ratio",2)))
    attack=max(.0001,float(p.get("attack_ms",20))/1000)
    release=max(.001,float(p.get("release_ms",100))/1000)
    env=np.max(np.abs(y),axis=1)+1e-12
    env_db=20*np.log10(env)
    over=np.maximum(env_db-threshold,0)
    gr_db=-(1-1/ratio)*over
    target=10**(gr_db/20)
    out=np.empty_like(target); out[0]=target[0]
    aa=np.exp(-1/(sr*attack)); ar=np.exp(-1/(sr*release))
    for i in range(1,len(target)):
        coeff=aa if target[i]<out[i-1] else ar
        out[i]=coeff*out[i-1]+(1-coeff)*target[i]
    return y*out[:,None]

PROCESSORS: dict[str, Callable] = {"gain":_gain,"eq_bell":_eq_bell,"compressor":_compress}

def render_candidate(input_path: str, output_path: str, action: dict[str,Any]) -> dict[str,Any]:
    src=Path(input_path).expanduser().resolve()
    dst=Path(output_path).expanduser().resolve()
    if src == dst:
        raise ValueError("refusing to overwrite source audio")
    y,sr=_read(str(src))
    kind=action["type"]
    if kind == "bypass":
        out=y
    elif kind in PROCESSORS:
        out=PROCESSORS[kind](y,sr,action.get("params",{}))
    else:
        raise ValueError(f"unsupported action: {kind}")
    _write(dst,out,sr)
    return {"output_path":str(Path(output_path).resolve()),"action":action,"analysis":core.analyze(output_path)}

def run_experiment(project_root: str, input_path: str, hypothesis: str,
                   actions: list[dict[str,Any]]) -> dict[str,Any]:
    root=Path(project_root).resolve()
    exp_id=uuid.uuid4().hex[:12]
    d=root/"experiments"/exp_id
    d.mkdir(parents=True,exist_ok=True)
    baseline=d/"baseline.wav"
    shutil.copy2(input_path,baseline)
    base=core.register(str(root),str(baseline))
    rows=[]
    for i,action in enumerate([{"type":"bypass","params":{}}]+actions):
        out=d/f"candidate_{i:02d}.wav"
        r=render_candidate(str(baseline),str(out),action)
        ident=core.register(str(root),str(out))
        rows.append({
          "candidate":i,"render_sha":ident["sha256"],"path":str(out),
          "action":action,"signal_analysis":r["analysis"],
          "delta_vs_baseline":compare(str(baseline),str(out),True)
        })
    manifest={"experiment_id":exp_id,"hypothesis":hypothesis,"baseline_sha":base["sha256"],
              "baseline_path":str(baseline),"candidates":rows,"status":"awaiting_evaluation"}
    (d/"experiment.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest

def choose_candidate(project_root: str, experiment_id: str, candidate: int,
                     reason: str, evaluator: str="human_or_calibrated_judge") -> dict[str,Any]:
    p=Path(project_root).resolve()/"experiments"/experiment_id/"experiment.json"
    m=json.loads(p.read_text(encoding="utf-8"))
    row=next(x for x in m["candidates"] if x["candidate"]==candidate)
    cov=core.coverage(project_root,row["render_sha"])
    if not cov["finalizable"]:
        raise RuntimeError(f"candidate has incomplete/stale checks: {cov['blockers']}")
    m["status"]="selected"; m["selected_candidate"]=candidate
    m["selection_reason"]=reason; m["evaluator"]=evaluator
    p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
    core.log_decision(project_root,row["render_sha"],m["hypothesis"],row["action"],"accepted")
    return {"experiment_id":experiment_id,"selected":row,"reason":reason,"evaluator":evaluator}
