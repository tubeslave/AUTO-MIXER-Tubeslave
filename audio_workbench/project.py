from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg"}

def _classify(name: str) -> str:
    n = name.lower()
    rules = [
        ("kick", ("kick","bd","bass drum")),
        ("snare", ("snare","sd")),
        ("toms", ("tom","floor")),
        ("overheads", ("oh","overhead")),
        ("drums", ("drum","perc","cymbal","hat","ride")),
        ("bass", ("bass","басс")),
        ("vocal", ("vox","vocal","voice","lead vocal","вокал")),
        ("backing_vocal", ("back","bgv","bv","choir","harmony")),
        ("guitar", ("gtr","guitar","git","гитар")),
        ("keys", ("keys","piano","synth","organ","keyboard")),
        ("fx", ("fx","effect","ambience","room","reverb")),
    ]
    for role, keys in rules:
        if any(k in n for k in keys):
            return role
    return "unknown"

def create_manifest(project_root: str, audio_dir: str, title: str = "") -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    source = Path(audio_dir).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"audio_dir not found: {source}")

    tracks = []
    for p in sorted(source.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        info = sf.info(str(p))
        tracks.append({
            "name": p.stem,
            "path": str(p),
            "role_guess": _classify(p.stem),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_s": info.frames / info.samplerate if info.samplerate else 0.0,
        })

    if not tracks:
        raise ValueError("no supported audio files found")

    samplerates = sorted({t["samplerate"] for t in tracks})
    durations = [t["duration_s"] for t in tracks]
    manifest = {
        "schema_version": 1,
        "title": title or source.name,
        "project_root": str(root),
        "audio_dir": str(source),
        "mode": "dawless",
        "tracks": tracks,
        "track_count": len(tracks),
        "samplerates": samplerates,
        "duration_range_s": [min(durations), max(durations)],
        "warnings": [],
        "sections": [],
        "references": [],
        "notes": [],
    }

    if len(samplerates) > 1:
        manifest["warnings"].append("mixed sample rates detected")
    if max(durations) - min(durations) > 0.1:
        manifest["warnings"].append("track durations differ by more than 100 ms; verify common start/end")

    root.mkdir(parents=True, exist_ok=True)
    out = root / "audio_workbench_project.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(out)
    return manifest

def load_manifest(project_root: str) -> dict[str, Any]:
    p = Path(project_root).expanduser().resolve() / "audio_workbench_project.json"
    if not p.exists():
        raise FileNotFoundError("project manifest does not exist")
    return json.loads(p.read_text(encoding="utf-8"))

def update_context(project_root: str, *, sections: list[dict[str, Any]] | None = None,
                   references: list[str] | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    manifest = load_manifest(project_root)
    if sections is not None:
        manifest["sections"] = sections
    if references is not None:
        manifest["references"] = references
    if notes is not None:
        manifest["notes"] = notes
    p = Path(project_root).expanduser().resolve() / "audio_workbench_project.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
