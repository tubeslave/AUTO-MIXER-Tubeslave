from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def make_observer_exam(project_root: str, clean_path: str,
                       perturbations: list[dict[str,Any]]) -> dict[str,Any]:
    """Create a machine-readable calibration plan. Rendering is delegated to experiment engine."""
    plan={
      "clean_path":str(Path(clean_path).resolve()),
      "tests":[],
      "rules":[
        "observer must identify controlled direction before its output can influence decisions",
        "A/B order must be tested both ways",
        "identical-file catch trials are required",
        "failure on a capability disables that capability rather than averaging it away"
      ]
    }
    for i,p in enumerate(perturbations):
        plan["tests"].append({
          "id":i,"change":p,
          "expected_detection":p.get("expected_detection"),
          "capability":p.get("capability","unspecified"),
          "status":"not_run"
        })
    out=Path(project_root).resolve()/"observer_calibration.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"path":str(out),"plan":plan}
