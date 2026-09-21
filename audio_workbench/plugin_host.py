from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf

def inspect_plugin(plugin_path: str) -> dict[str,Any]:
    try:
        from pedalboard import load_plugin
    except ImportError as exc:
        raise RuntimeError("pedalboard is required") from exc
    p=load_plugin(plugin_path)
    params={}
    for name,param in p.parameters.items():
        params[name]={
          "raw_value":param.raw_value,
          "min_value":param.min_value,
          "max_value":param.max_value,
          "default_value":param.default_value,
        }
    return {"path":str(Path(plugin_path).resolve()),"parameters":params,
            "warning":"third-party plugins can crash or retain state; compatibility must be calibrated before autonomous use"}

def render_plugin(input_path: str, output_path: str, plugin_path: str,
                  parameters: dict[str,Any] | None=None, buffer_size: int=8192) -> dict[str,Any]:
    from pedalboard import load_plugin
    y,sr=sf.read(input_path,always_2d=True,dtype="float32")
    p=load_plugin(plugin_path)
    for k,v in (parameters or {}).items():
        if k not in p.parameters:
            raise KeyError(f"unknown plugin parameter: {k}")
        setattr(p,k,v)
    # Pedalboard convention: channels x samples.
    out=p(y.T,sr,buffer_size=buffer_size,reset=True)
    out=np.asarray(out,dtype="float32").T
    sf.write(output_path,out,sr,subtype="PCM_24")
    return {"output_path":str(Path(output_path).resolve()),"plugin_path":str(Path(plugin_path).resolve()),
            "parameters":parameters or {}}

def calibration_plan(plugin_path: str) -> dict[str,Any]:
    info=inspect_plugin(plugin_path)
    return {
      "plugin":info,
      "required_tests":["bypass/null","repeatability","tail completion","latency","finite samples",
                        "gain invariance where expected","parameter min/default/max smoke test"],
      "autonomous_write_allowed":False
    }
