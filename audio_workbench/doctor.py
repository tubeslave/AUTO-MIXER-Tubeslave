from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

def status(project_root: str | None=None) -> dict[str,Any]:
    root=Path(project_root).expanduser().resolve() if project_root else None
    deps={n: importlib.util.find_spec(n) is not None for n in
          ("numpy","scipy","soundfile","librosa","fastmcp","mcp","optuna","pedalboard","transformers","torch")}
    result={"python":sys.version.split()[0],"ffmpeg":shutil.which("ffmpeg"),"dependencies":deps}
    if root:
        result["project_root"]=str(root)
        manifest=root/"audio_workbench_project.json"
        result["manifest_exists"]=manifest.exists()
        if manifest.exists():
            m=json.loads(manifest.read_text(encoding="utf-8"))
            missing=[t["path"] for t in m.get("tracks",[]) if not Path(t["path"]).exists()]
            result["track_count"]=len(m.get("tracks",[])); result["missing_tracks"]=missing
        result["state_db_exists"]=(root/".audio_workbench.sqlite3").exists()
        result["free_bytes"]=shutil.disk_usage(root).free if root.exists() else None
    return result
