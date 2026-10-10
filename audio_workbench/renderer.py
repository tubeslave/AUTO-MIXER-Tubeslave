from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

def _pan_gains(pan: float) -> tuple[float,float]:
    pan=max(-1.0,min(1.0,float(pan)))
    angle=(pan+1.0)*math.pi/4.0
    return math.cos(angle), math.sin(angle)

def render_mix(project_root: str, output_path: str, track_settings: dict[str,dict[str,Any]] | None=None) -> dict[str,Any]:
    root=Path(project_root).expanduser().resolve()
    manifest=json.loads((root/"audio_workbench_project.json").read_text(encoding="utf-8"))
    settings=track_settings or {}
    tracks=manifest["tracks"]
    if not tracks:
        raise ValueError("manifest has no tracks")
    sr=int(tracks[0]["samplerate"]); frames=int(tracks[0]["frames"])
    mix=np.zeros((frames,2),dtype=np.float64)
    rendered=[]
    for t in tracks:
        if int(t["samplerate"]) != sr or int(t["frames"]) != frames:
            raise ValueError(f"unaligned track: {t['name']}")
        cfg=settings.get(t["name"],{})
        if cfg.get("mute",False):
            continue
        y,ysr=sf.read(t["path"],always_2d=True,dtype="float32")
        if ysr != sr or len(y) != frames:
            raise ValueError(f"track changed since manifest: {t['name']}")
        if not np.isfinite(y).all():
            raise ValueError(f"non-finite audio: {t['name']}")
        if bool(cfg.get("invert_polarity",False)):
            y=-y
        gain=10.0**(float(cfg.get("gain_db",0.0))/20.0)
        pan=float(cfg.get("pan",0.0))
        if y.shape[1] == 1:
            gl,gr=_pan_gains(pan)
            stereo=np.column_stack((y[:,0]*gl,y[:,0]*gr))
        elif y.shape[1] == 2:
            if abs(pan)>1e-9:
                raise ValueError(f"pan for stereo track is ambiguous: {t['name']}; use source processing or leave pan=0")
            stereo=y
        else:
            raise ValueError(f"unsupported channel count {y.shape[1]}: {t['name']}")
        mix += stereo*gain
        rendered.append({"name":t["name"],"gain_db":float(cfg.get("gain_db",0.0)),"pan":pan})
    out=Path(output_path).expanduser().resolve()
    if any(Path(t["path"]).resolve()==out for t in tracks):
        raise ValueError("refusing to overwrite source track")
    out.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(out),mix.astype("float32"),sr,subtype="FLOAT")
    return {"output_path":str(out),"samplerate":sr,"frames":frames,"tracks_rendered":rendered,
            "peak":float(np.max(np.abs(mix))),"clipped_samples_if_pcm":int(np.sum(np.abs(mix)>1.0)),
            "note":"float render preserves overload evidence; mastering/output conversion is separate"}

def save_mix_state(project_root: str, name: str, track_settings: dict[str,dict[str,Any]]) -> str:
    root=Path(project_root).expanduser().resolve()
    p=root/"mix_states"/f"{name}.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(track_settings,ensure_ascii=False,indent=2),encoding="utf-8")
    return str(p)
