from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

DB_NAME = ".audio_workbench.sqlite3"

CHECKS = (
    "integrity", "loudness", "dynamics", "spectrum",
    "stereo_phase", "transients", "song_context", "delivery"
)

DEPENDENCIES = {
    "gain": ("loudness", "dynamics", "spectrum", "song_context", "delivery"),
    "eq": ("spectrum", "dynamics", "stereo_phase", "song_context", "delivery"),
    "compression": ("loudness", "dynamics", "transients", "spectrum", "song_context", "delivery"),
    "stereo": ("stereo_phase", "spectrum", "song_context", "delivery"),
    "reverb": ("loudness", "spectrum", "stereo_phase", "song_context", "delivery"),
    "edit": CHECKS,
    "routing": CHECKS,
    "master": CHECKS,
}

@dataclass(frozen=True)
class AudioId:
    sha256: str
    path: str
    frames: int
    samplerate: int
    channels: int

def _db(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / DB_NAME)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS renders(
      sha256 TEXT PRIMARY KEY, path TEXT NOT NULL, frames INTEGER,
      samplerate INTEGER, channels INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS checks(
      render_sha TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
      result_json TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(render_sha,name)
    );
    CREATE TABLE IF NOT EXISTS decisions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, render_sha TEXT NOT NULL,
      hypothesis TEXT NOT NULL, action_json TEXT, outcome TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    return con

def fingerprint(path: str) -> AudioId:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    info = sf.info(str(p))
    return AudioId(h.hexdigest(), str(p), info.frames, info.samplerate, info.channels)

def register(root: str, path: str) -> dict[str, Any]:
    aid = fingerprint(path)
    con = _db(Path(root))
    con.execute("INSERT OR REPLACE INTO renders(sha256,path,frames,samplerate,channels) VALUES(?,?,?,?,?)",
                (aid.sha256, aid.path, aid.frames, aid.samplerate, aid.channels))
    for name in CHECKS:
        con.execute("INSERT OR IGNORE INTO checks(render_sha,name,status) VALUES(?,?,?)",
                    (aid.sha256, name, "not_run"))
    con.commit()
    return aid.__dict__

def _load(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    if not np.isfinite(y).all():
        raise ValueError("audio contains NaN/Inf")
    return y, sr

def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))

def _band_energy(mono: np.ndarray, sr: int) -> dict[str, float]:
    f, p = signal.welch(mono, sr, nperseg=min(8192, len(mono)))
    bands = {"sub":[20,60], "bass":[60,200], "low_mid":[200,500],
             "mid":[500,2000], "presence":[2000,6000], "air":[6000,min(20000,sr/2)]}
    total = np.trapezoid(p, f) + 1e-20
    out = {}
    for name,(lo,hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        out[name] = float(10*np.log10((np.trapezoid(p[mask], f[mask]) + 1e-20)/total))
    return out

def analyze(path: str) -> dict[str, Any]:
    y, sr = _load(path)
    mono = y.mean(axis=1)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-20))
    crest = _dbfs(peak / max(rms, 1e-12))
    clipped = int(np.sum(np.abs(y) >= 0.9999))
    dc = [float(v) for v in np.mean(y, axis=0)]
    stereo = None
    if y.shape[1] >= 2:
        l, r = y[:,0], y[:,1]
        den = float(np.sqrt(np.sum(l*l)*np.sum(r*r)) + 1e-20)
        corr = float(np.sum(l*r)/den)
        mid = (l+r)*0.5
        side = (l-r)*0.5
        stereo = {"correlation":corr, "side_mid_rms_db":_dbfs(np.sqrt(np.mean(side*side)+1e-20) / max(np.sqrt(np.mean(mid*mid)+1e-20),1e-12))}
    duration = len(y)/sr
    return {
      "path": str(Path(path).resolve()), "samplerate": sr, "channels": int(y.shape[1]),
      "duration_s": duration, "sample_peak_dbfs": _dbfs(peak), "rms_dbfs": _dbfs(rms),
      "crest_db": crest, "clipped_samples": clipped, "dc_offset": dc,
      "band_energy_db_relative": _band_energy(mono, sr), "stereo": stereo,
      "limitations": [
        "sample peak is not true peak",
        "band-energy descriptors are evidence, not mix-quality scores",
        "no musical preference is inferred from these metrics"
      ]
    }

def ffmpeg_loudness(path: str) -> dict[str, Any]:
    cmd = ["ffmpeg","-hide_banner","-nostats","-i",path,"-filter_complex",
           "ebur128=peak=true:framelog=verbose","-f","null","-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stderr
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else text[-4000:]
    return {"returncode":p.returncode, "summary":summary}

def _require_render(con: sqlite3.Connection, render_sha: str) -> None:
    if con.execute("SELECT 1 FROM renders WHERE sha256=?", (render_sha,)).fetchone() is None:
        raise KeyError(f"unknown render_sha: {render_sha}")

def record_check(root: str, render_sha: str, name: str, result: dict[str, Any], status: str="ok") -> dict[str, Any]:
    if name not in CHECKS:
        raise ValueError(f"unknown check: {name}")
    if status not in ("ok", "issue", "not_applicable", "stale", "not_run", "unknown"):
        raise ValueError(f"invalid check status: {status}")
    if status == "ok" and not result:
        raise ValueError("ok check requires non-empty evidence")
    con = _db(Path(root))
    _require_render(con, render_sha)
    con.execute("""INSERT INTO checks(render_sha,name,status,result_json,updated_at)
                   VALUES(?,?,?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(render_sha,name) DO UPDATE SET status=excluded.status,
                   result_json=excluded.result_json, updated_at=CURRENT_TIMESTAMP""",
                (render_sha,name,status,json.dumps(result,ensure_ascii=False)))
    con.commit()
    return {"render_sha":render_sha,"name":name,"status":status}

def invalidate(root: str, render_sha: str, change_type: str) -> dict[str, Any]:
    names = DEPENDENCIES.get(change_type, CHECKS)
    con = _db(Path(root))
    _require_render(con, render_sha)
    con.executemany("UPDATE checks SET status='stale',updated_at=CURRENT_TIMESTAMP WHERE render_sha=? AND name=?",
                    [(render_sha,n) for n in names])
    con.commit()
    return {"render_sha":render_sha,"change_type":change_type,"stale":list(names)}

def coverage(root: str, render_sha: str) -> dict[str, Any]:
    con = _db(Path(root))
    _require_render(con, render_sha)
    rows = con.execute("SELECT name,status,result_json,updated_at FROM checks WHERE render_sha=? ORDER BY name",(render_sha,)).fetchall()
    if len(rows) != len(CHECKS):
        present = {r["name"] for r in rows}
        for name in CHECKS:
            if name not in present:
                con.execute("INSERT OR IGNORE INTO checks(render_sha,name,status) VALUES(?,?,?)",(render_sha,name,"not_run"))
        con.commit()
        rows = con.execute("SELECT name,status,result_json,updated_at FROM checks WHERE render_sha=? ORDER BY name",(render_sha,)).fetchall()
    data = [dict(r) for r in rows]
    blockers = [r["name"] for r in data if r["status"] not in ("ok","not_applicable")]
    return {"render_sha":render_sha,"checks":data,"finalizable":not blockers,"blockers":blockers}

def log_decision(root: str, render_sha: str, hypothesis: str, action: dict[str,Any], outcome: str="proposed") -> dict[str,Any]:
    con = _db(Path(root))
    _require_render(con, render_sha)
    cur = con.execute("INSERT INTO decisions(render_sha,hypothesis,action_json,outcome) VALUES(?,?,?,?)",
                      (render_sha,hypothesis,json.dumps(action,ensure_ascii=False),outcome))
    con.commit()
    return {"decision_id":cur.lastrowid,"render_sha":render_sha,"outcome":outcome}

def next_task(root: str, render_sha: str) -> dict[str,Any]:
    c = coverage(root,render_sha)
    if c["finalizable"]:
        return {"status":"complete","message":"all mandatory checks are current"}
    priority = ["integrity","loudness","dynamics","spectrum","stereo_phase","transients","song_context","delivery"]
    state = {r["name"]:r["status"] for r in c["checks"]}
    for name in priority:
        if state.get(name) not in ("ok","not_applicable"):
            return {"status":"work","check":name,"current_status":state.get(name,"missing")}
    return {"status":"work","check":c["blockers"][0]}
