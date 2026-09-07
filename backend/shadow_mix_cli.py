"""Offline command-line entry points. This module never instantiates a console."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from autofoh_safety import ChannelEQMove, ChannelFaderMove, CompressorAdjust, HighPassAdjust
from shadow_pipeline import ShadowPipeline
from shadow_renderer import BANDS, Compressor, EQBand, RenderChannel, ShadowRenderer, measure
from target_corridor import FEATURE_VERSION, TargetCorridor

ACTION_TYPES = {cls.__name__: cls for cls in
                (ChannelEQMove, ChannelFaderMove, CompressorAdjust, HighPassAdjust)}


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    ShadowRenderer(sr)  # Enforce the same supported sample-rate contract as rendering.
    if data.shape[1] not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Only finite mono/stereo WAV audio is supported")
    return data, sr


def fit(manifest: Path, context: str, output: Path, features: list[str]) -> TargetCorridor:
    records: list[dict[str, Any]] = json.loads(manifest.read_text(encoding="utf-8"))
    seen_audio = set()
    for record in records:
        if record.get("approved") is True and record.get("context") == context and "audio_path" in record:
            audio, sr = _read_audio(manifest.parent / record["audio_path"])
            digest = hashlib.sha256(str((sr, audio.shape)).encode() + audio.tobytes()).hexdigest()
            if digest in seen_audio:
                record["approved"] = False
                continue
            seen_audio.add(digest)
            record["features"] = measure(audio, sr)
            record["feature_version"] = FEATURE_VERSION
    target = TargetCorridor.fit(records, context=context, features=features)
    target.save(output)
    return target


def audit(session_path: Path, target_path: Path, report_path: Path,
          preview_path: Path | None = None) -> bool:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    target = TargetCorridor.load(target_path)
    sr = session["sample_rate"]
    buffers, states = {}, {}
    for item in session["channels"]:
        data, actual_sr = _read_audio(session_path.parent / item["audio_path"])
        if actual_sr != sr:
            raise ValueError("No implicit sample-rate conversion")
        config = dict(item["state"])
        config["eq_bands"] = {int(k): EQBand(**v) for k, v in config.get("eq_bands", {}).items()}
        if config.get("compressor") is not None:
            config["compressor"] = Compressor(**config["compressor"])
        state = RenderChannel(**config)
        if state.channel_id in states:
            raise ValueError("Duplicate channel")
        states[state.channel_id], buffers[state.channel_id] = state, data
    action_config = dict(session["action"])
    action_type = action_config.pop("type")
    action = ACTION_TYPES[action_type](**action_config)
    pipeline = ShadowPipeline({"mode": "shadow", "context": target.context,
                               "guard": {"confirmations": 1}, "max_render_age_sec": 3600}, target)
    decision = pipeline.evaluate(buffers, states, action, sr)
    payload = {"allowed": decision.allowed, "reason": decision.reason, "console_writes": 0,
               "offline_single_observation": True, "action": asdict(action), **decision.report}
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    if decision.allowed and preview_path is not None:
        renderer = ShadowRenderer(sr)
        proposed, _ = renderer.propose(states, action)
        sf.write(preview_path, renderer.render(buffers, proposed).audio, sr, subtype="FLOAT")
    return decision.allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    learn = commands.add_parser("fit", help="Fit only explicitly approved, distinct mixes")
    learn.add_argument("--manifest", required=True, type=Path)
    learn.add_argument("--context", required=True)
    learn.add_argument("--output", required=True, type=Path)
    learn.add_argument("--features", default=",".join((*BANDS, "lufs", "crest_db")))
    check = commands.add_parser("audit", help="Evaluate one offline proposal; never sends OSC/MIDI")
    check.add_argument("--session", required=True, type=Path)
    check.add_argument("--target", required=True, type=Path)
    check.add_argument("--report", required=True, type=Path)
    check.add_argument("--preview", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            target = fit(args.manifest, args.context, args.output, args.features.split(","))
            print(json.dumps({"accepted_mixes": target.sample_count,
                              "rejected_records": target.rejected_count}))
            return 0
        return 0 if audit(args.session, args.target, args.report, args.preview) else 2
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
