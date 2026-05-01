"""Safe offline stem-aware training loop.

This module is intentionally offline-only. It never creates an OSC client and never
sends mixer commands. The default splitter is a deterministic mock splitter so CI
can run without downloading Demucs/ONNX models.

Pipeline:
    mix -> split/mock split -> analyze -> fix -> compare -> write artifacts
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "other")
BANDS = {
    "low_80_160": (80.0, 160.0),
    "low_mid_160_350": (160.0, 350.0),
    "mid_350_700": (350.0, 700.0),
    "presence_700_1500": (700.0, 1500.0),
    "vocal_1500_4000": (1500.0, 4000.0),
    "air_4000_10000": (4000.0, 10000.0),
}


@dataclass
class StemMetrics:
    name: str
    rms_db: float
    peak_db: float
    centroid_hz: float
    band_energy: Dict[str, float]


@dataclass
class LoopResult:
    input_path: str
    output_mix_path: str
    report_path: str
    splitter_mode: str
    score_before: float
    score_after: float
    improvement: float
    actions: list
    osc_disabled: bool = True


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-9))


def generate_fixture(path: Path, sr: int = 44100, seconds: float = 8.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    bass = 0.18 * np.sin(2 * np.pi * 90 * t)
    vocal = 0.10 * np.sin(2 * np.pi * 880 * t) + 0.05 * np.sin(2 * np.pi * 1760 * t)
    harsh = 0.07 * np.sin(2 * np.pi * 3000 * t)
    drums = np.zeros_like(t)
    click_positions = np.arange(0, len(t), sr // 2)
    for pos in click_positions:
        end = min(len(t), pos + 900)
        env = np.exp(-np.linspace(0, 6, end - pos))
        drums[pos:end] += 0.22 * env * np.sin(2 * np.pi * 120 * t[pos:end])
    mix = bass + vocal + harsh + drums
    mix = np.clip(mix, -0.95, 0.95).astype(np.float32)
    sf.write(path, mix, sr)
    return path


def mock_split(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Deterministic FFT mask splitter for tests; not a production separator."""
    mono = _to_mono(audio)
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)

    masks = {
        "bass": freqs < 180,
        "drums": ((freqs >= 80) & (freqs < 350)),
        "vocals": ((freqs >= 700) & (freqs < 4000)),
        "other": ~(((freqs < 180)) | ((freqs >= 700) & (freqs < 4000))),
    }
    stems: Dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        stem_spec = np.zeros_like(spec)
        stem_spec[mask] = spec[mask]
        stem = np.fft.irfft(stem_spec, n=len(mono)).astype(np.float32)
        stems[name] = stem
    return stems


def band_energy_db(audio: np.ndarray, sr: int, f1: float, f2: float) -> float:
    mono = _to_mono(audio)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mask = (freqs >= f1) & (freqs < f2)
    if not np.any(mask):
        return -180.0
    return _db(float(np.sqrt(np.mean(spec[mask] ** 2)) / max(len(mono), 1)))


def analyze_stem(name: str, audio: np.ndarray, sr: int) -> StemMetrics:
    mono = _to_mono(audio)
    rms_db = _db(float(np.sqrt(np.mean(mono ** 2))))
    peak_db = _db(float(np.max(np.abs(mono))))
    mag = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    centroid = float(np.sum(freqs * mag) / max(np.sum(mag), 1e-9))
    bands = {k: band_energy_db(mono, sr, lo, hi) for k, (lo, hi) in BANDS.items()}
    return StemMetrics(name, rms_db, peak_db, centroid, bands)


def score_mix(stems: Dict[str, np.ndarray], sr: int) -> Tuple[float, Dict[str, StemMetrics], Dict[str, float]]:
    metrics = {name: analyze_stem(name, audio, sr) for name, audio in stems.items()}
    vocal_masking = metrics["other"].band_energy["vocal_1500_4000"] - metrics["vocals"].band_energy["vocal_1500_4000"]
    low_conflict = metrics["bass"].band_energy["low_80_160"] - metrics["drums"].band_energy["low_80_160"]
    peak_penalty = max(0.0, max(m.peak_db for m in metrics.values()) + 1.0)
    score = 1.0 - 0.03 * max(0.0, vocal_masking) - 0.02 * abs(low_conflict) - 0.05 * peak_penalty
    diagnostics = {"vocal_masking_index": vocal_masking, "low_conflict_index": low_conflict, "peak_penalty": peak_penalty}
    return float(max(0.0, min(1.0, score))), metrics, diagnostics


def apply_safe_fix(stems: Dict[str, np.ndarray], diagnostics: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], list]:
    fixed = {k: v.copy() for k, v in stems.items()}
    actions = []
    if diagnostics["vocal_masking_index"] > -3.0:
        fixed["vocals"] *= 1.12
        fixed["other"] *= 0.92
        actions.append({"type": "stem_gain", "target": "vocals", "delta_db": 0.98, "reason": "improve vocal clarity"})
        actions.append({"type": "stem_gain", "target": "other", "delta_db": -0.72, "reason": "reduce vocal masking"})
    if abs(diagnostics["low_conflict_index"]) > 8.0:
        fixed["bass"] *= 0.94
        actions.append({"type": "stem_gain", "target": "bass", "delta_db": -0.54, "reason": "reduce low-end conflict"})
    return fixed, actions


def sum_stems(stems: Dict[str, np.ndarray]) -> np.ndarray:
    max_len = max(len(x) for x in stems.values())
    mix = np.zeros(max_len, dtype=np.float32)
    for stem in stems.values():
        mix[: len(stem)] += stem.astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak
    return mix


def run_training_loop(input_path: Path, out_dir: Path, splitter_mode: str = "mock") -> LoopResult:
    if os.environ.get("OSC_DISABLED", "true").lower() != "true":
        raise RuntimeError("Refusing to run: OSC_DISABLED must be true for offline stem training loop")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(input_path, always_2d=False)
    if splitter_mode != "mock":
        raise NotImplementedError("Only mock splitter is enabled in CI. Real splitter must be optional and manual.")
    stems = mock_split(audio, sr)
    score_before, metrics_before, diag_before = score_mix(stems, sr)
    fixed_stems, actions = apply_safe_fix(stems, diag_before)
    score_after, metrics_after, diag_after = score_mix(fixed_stems, sr)
    fixed_mix = sum_stems(fixed_stems)
    output_mix_path = out_dir / "stem_training_fixed_mix.wav"
    sf.write(output_mix_path, fixed_mix, sr)
    report_path = out_dir / "stem_training_report.json"
    report = {
        "input_path": str(input_path),
        "output_mix_path": str(output_mix_path),
        "splitter_mode": splitter_mode,
        "score_before": score_before,
        "score_after": score_after,
        "improvement": score_after - score_before,
        "actions": actions,
        "diagnostics_before": diag_before,
        "diagnostics_after": diag_after,
        "metrics_before": {k: asdict(v) for k, v in metrics_before.items()},
        "metrics_after": {k: asdict(v) for k, v in metrics_after.items()},
        "osc_disabled": True,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return LoopResult(str(input_path), str(output_mix_path), str(report_path), splitter_mode, score_before, score_after, score_after - score_before, actions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline stem-aware training loop")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/stem_training_loop"))
    parser.add_argument("--splitter", default="mock", choices=["mock"])
    parser.add_argument("--generate-fixture", action="store_true")
    args = parser.parse_args()
    input_path = args.input
    if args.generate_fixture or input_path is None:
        input_path = args.out / "stem_training_fixture.wav"
        generate_fixture(input_path)
    result = run_training_loop(input_path, args.out, args.splitter)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
