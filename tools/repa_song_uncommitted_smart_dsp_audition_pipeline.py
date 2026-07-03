#!/usr/bin/env python3
"""Render REPA Song through today's offline soundcheck plus smart-DSP audition.

This pipeline is intentionally offline. It uses the current working tree,
including uncommitted smart-DSP modules, to create a listening file that shows
the present Automixer behavior without granting live WING write authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_AUDIO_DIR = Path.home() / "Desktop" / "Ai MIXING"
DEFAULT_LOG_DIR = Path.home() / "Desktop" / "Ai LOGS"
DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "REPA Song"
FALLBACK_INPUT_DIRS = (
    DEFAULT_INPUT_DIR,
    Path.home()
    / "AUTO-MIXER-Tubeslave-github-20260514"
    / "offline_test_input"
    / "multitrack",
)
PIPELINE_NAME = "repa_song_uncommitted_smart_dsp_audition"
TARGET_LUFS = -15.5
PEAK_CEILING_DBFS = -2.0
RENDER_TIMEOUT_SEC = 3600.0


def timestamp_slug(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%dT%H%M%S")


def default_artifact_stem(now: datetime | None = None) -> str:
    return f"REPA_Song_AUTOMIXER_SMART_DSP_AUDITION_{timestamp_slug(now)}"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def default_output_path(now: datetime | None = None) -> Path:
    return DEFAULT_AUDIO_DIR / f"{default_artifact_stem(now)}.wav"


def default_base_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.base_soundcheck.wav")


def default_report_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.report.json"


def default_base_report_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.base_soundcheck.report.json"


def default_smart_dsp_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.smart_dsp.json"


def default_summary_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.summary.md"


def default_run_log_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.render.log"


def audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        item
        for item in input_dir.iterdir()
        if item.is_file() and item.suffix.lower() in {".wav", ".wave"}
    )


def has_readable_audio(input_dir: Path) -> bool:
    for item in audio_files(input_dir):
        try:
            sf.info(str(item))
            return True
        except Exception:
            continue
    return False


def resolve_input_dir(requested: Path) -> Path:
    candidates = [requested]
    candidates.extend(path for path in FALLBACK_INPUT_DIRS if path != requested)
    for candidate in candidates:
        if has_readable_audio(candidate):
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"No readable REPA Song WAV files found. Checked: {checked}")


def subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    paths = [str(REPO_ROOT), str(BACKEND_DIR)]
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_base_soundcheck(
    *,
    input_dir: Path,
    output_path: Path,
    report_path: Path,
    run_log_path: Path,
    tempo_bpm: float,
    genre: str,
    reference: str,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "channel_triggered_soundcheck_mix.py"),
        "--input-dir",
        str(input_dir),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--tempo-bpm",
        str(float(tempo_bpm)),
    ]
    if genre:
        command.extend(["--genre", genre])
    if reference:
        command.extend(["--reference", reference])

    ensure_parent(output_path)
    ensure_parent(report_path)
    ensure_parent(run_log_path)
    with run_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=RENDER_TIMEOUT_SEC,
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    if amp <= 0.0:
        return -120.0
    return float(20.0 * np.log10(amp))


def to_stereo(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return np.column_stack([data, data]).astype(np.float32)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1).astype(np.float32)
    return data[:, :2].astype(np.float32)


def peak_dbfs(samples: np.ndarray) -> float:
    return amp_to_db(float(np.max(np.abs(samples))) if len(samples) else 0.0)


def integrated_lufs(samples: np.ndarray, sample_rate: int) -> float | None:
    try:
        value = float(pyln.Meter(sample_rate).integrated_loudness(samples))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def true_peak_limited(samples: np.ndarray, ceiling_dbfs: float) -> tuple[np.ndarray, float]:
    ceiling = db_to_amp(ceiling_dbfs)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak <= ceiling or peak <= 0.0:
        return samples.astype(np.float32), 0.0
    gain = ceiling / peak
    return (samples * gain).astype(np.float32), amp_to_db(gain)


def butter_filter(
    samples: np.ndarray,
    sample_rate: int,
    *,
    kind: str,
    frequency: float | tuple[float, float],
    order: int = 3,
) -> np.ndarray:
    sos = signal.butter(order, frequency, kind, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, samples, axis=0).astype(np.float32)


def peaking_eq(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frequency: float,
    q: float,
    gain_db: float,
) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * frequency / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cos_w = np.cos(omega)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a_coeff, samples, axis=0).astype(np.float32)


def high_shelf(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frequency: float,
    gain_db: float,
    slope: float = 0.75,
) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * frequency / sample_rate
    sin_w = np.sin(omega)
    cos_w = np.cos(omega)
    alpha = sin_w / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    sqrt_a = np.sqrt(a)
    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w)
    a2 = (a + 1.0) - (a - 1.0) * cos_w - 2.0 * sqrt_a * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a_coeff, samples, axis=0).astype(np.float32)


def dynamic_band_reduction(
    samples: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float,
    high_hz: float,
    threshold_percentile: float,
    max_reduction_db: float,
    detector_lowpass_hz: float,
) -> tuple[np.ndarray, dict[str, float]]:
    band = butter_filter(
        samples,
        sample_rate,
        kind="bandpass",
        frequency=(low_hz, high_hz),
        order=3,
    )
    detector = np.max(np.abs(band), axis=1)
    if not len(detector):
        return samples, {"max_gain_reduction_db": 0.0}
    sos = signal.butter(2, detector_lowpass_hz, "lowpass", fs=sample_rate, output="sos")
    envelope = signal.sosfiltfilt(sos, detector).astype(np.float32)
    envelope = np.maximum(envelope, 0.0)
    threshold = float(np.percentile(envelope, threshold_percentile))
    top = float(np.percentile(envelope, 99.5))
    if top <= threshold or top <= 1e-9:
        return samples, {"max_gain_reduction_db": 0.0}
    amount = np.clip((envelope - threshold) / (top - threshold), 0.0, 1.0)
    reduction_db = max_reduction_db * np.power(amount, 0.7)
    gain = np.power(10.0, -reduction_db / 20.0).astype(np.float32)
    processed = samples - band * (1.0 - gain[:, None])
    return processed.astype(np.float32), {
        "band_low_hz": float(low_hz),
        "band_high_hz": float(high_hz),
        "threshold_percentile": float(threshold_percentile),
        "max_gain_reduction_db": round(float(np.max(reduction_db)), 3),
    }


def bus_compress(
    samples: np.ndarray,
    sample_rate: int,
    *,
    ratio: float,
    threshold_percentile: float,
    max_reduction_db: float,
) -> tuple[np.ndarray, dict[str, float]]:
    detector = np.max(np.abs(samples), axis=1)
    if not len(detector):
        return samples, {"max_gain_reduction_db": 0.0}
    sos = signal.butter(2, 12.0, "lowpass", fs=sample_rate, output="sos")
    envelope = np.maximum(signal.sosfiltfilt(sos, detector), 1e-9)
    threshold = float(np.percentile(envelope, threshold_percentile))
    level_db = 20.0 * np.log10(envelope)
    threshold_db = amp_to_db(threshold)
    over_db = np.maximum(level_db - threshold_db, 0.0)
    reduction_db = np.minimum(over_db * (1.0 - 1.0 / ratio), max_reduction_db)
    gain = np.power(10.0, -reduction_db / 20.0).astype(np.float32)
    return (samples * gain[:, None]).astype(np.float32), {
        "ratio": float(ratio),
        "threshold_dbfs": round(float(threshold_db), 3),
        "max_gain_reduction_db": round(float(np.max(reduction_db)), 3),
    }


def stereo_width(samples: np.ndarray, factor: float) -> tuple[np.ndarray, dict[str, float]]:
    left = samples[:, 0]
    right = samples[:, 1]
    corr = float(np.corrcoef(left, right)[0, 1]) if len(samples) > 4 else 1.0
    if not np.isfinite(corr):
        corr = 1.0
    safe_factor = factor if corr > 0.05 else 1.0
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * safe_factor
    widened = np.column_stack([mid + side, mid - side]).astype(np.float32)
    return widened, {
        "requested_width_factor": float(factor),
        "applied_width_factor": float(safe_factor),
        "pre_width_correlation": round(corr, 4),
    }


def _source_target_plane_lufs(instrument: str) -> float:
    if instrument == "lead_vocal":
        return -22.0
    if instrument == "backing_vocal":
        return -27.0
    if instrument in {"accordion", "playback", "overhead", "room"}:
        return -30.0
    if instrument in {
        "kick",
        "snare",
        "rack_tom",
        "floor_tom",
        "bass_guitar",
        "electric_guitar",
    }:
        return -25.0
    return -25.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compact_recommendation(proposal: dict[str, Any]) -> dict[str, Any]:
    selected = proposal.get("selected_strategy") or {}
    evidence = proposal.get("evidence") or {}
    audio_evidence = evidence.get("audio_evidence") or {}
    return {
        "target": proposal.get("target"),
        "source_role": evidence.get("source_role") or proposal.get("source_role"),
        "reason": proposal.get("reason"),
        "confidence": proposal.get("confidence"),
        "selected_strategy_id": selected.get("strategy_id"),
        "method_family": selected.get("method_family"),
        "problem": selected.get("problem"),
        "parameters": selected.get("parameters"),
        "blockers": proposal.get("blockers") or [],
        "audio_evidence": {
            "problem_hints": audio_evidence.get("problem_hints") or [],
            "peak_dbfs": audio_evidence.get("peak_dbfs"),
            "estimated_lufs": audio_evidence.get("estimated_lufs"),
            "true_peak_headroom_db": audio_evidence.get("true_peak_headroom_db"),
            "muddiness_score": audio_evidence.get("muddiness_score"),
            "harshness_score": audio_evidence.get("harshness_score"),
            "sibilance_score": audio_evidence.get("sibilance_score"),
            "feedback_risk": audio_evidence.get("feedback_risk"),
            "phase_correlation": audio_evidence.get("phase_correlation"),
        },
    }


def smart_dsp_kwargs(
    *,
    target_id: str,
    source_role: str,
    source_label: str,
    sample_rate: int,
    target_plane_lufs: float,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "source_role": source_role,
        "source_label": source_label,
        "mode": "soundcheck",
        "sample_rate": int(sample_rate),
        "source_identity_confirmed": True,
        "source_identity_confidence": 0.95,
        "current_state_readback_available": True,
        "current_wing_eq_model": "STD",
        "current_wing_eq_enabled": True,
        "current_wing_eq_model_readback_available": True,
        "current_wing_eq_operator_owned": False,
        "current_wing_eq_filter_operator_owned": False,
        "current_wing_dynamics_model": "COMP",
        "current_wing_dynamics_enabled": True,
        "current_wing_dynamics_model_readback_available": True,
        "current_wing_dynamics_operator_owned": False,
        "current_wing_gate_model": "EXP",
        "current_wing_gate_enabled": True,
        "current_wing_gate_slot_readback_available": True,
        "current_wing_gate_slot_operator_owned": False,
        "current_wing_fx_model": "HALL",
        "current_wing_fx_model_readback_available": True,
        "current_wing_fx_operator_owned": False,
        "current_wing_fx_level_operator_owned": False,
        "current_wing_pan_width_operator_owned": False,
        "current_wing_mute_operator_owned": False,
        "current_wing_gain_operator_owned": False,
        "rollback_plan_available": True,
        "allow_inferred_source_identity": True,
        "target_plane_lufs": float(target_plane_lufs),
        "active_performance": False,
    }


def build_smart_dsp_report(
    *,
    input_dir: Path,
    base_mix: np.ndarray,
    sample_rate: int,
    soundcheck_report: dict[str, Any],
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BACKEND_DIR))
    from smart_dsp_evidence_extractor import analyze_smart_dsp_audio

    channel_recommendations: list[dict[str, Any]] = []
    for index, channel_report in enumerate(soundcheck_report.get("final_channels") or [], start=1):
        file_name = str(channel_report.get("file") or "")
        if not file_name:
            continue
        path = input_dir / file_name
        if not path.exists():
            continue
        channel_samples, channel_sr = sf.read(str(path), dtype="float32", always_2d=False)
        instrument = str(channel_report.get("instrument") or "custom")
        recommendation = analyze_smart_dsp_audio(
            channel_samples,
            **smart_dsp_kwargs(
                target_id=f"ch{index}:{Path(file_name).stem}",
                source_role=instrument,
                source_label=file_name,
                sample_rate=int(channel_sr),
                target_plane_lufs=_source_target_plane_lufs(instrument),
            ),
        )
        proposal = json_safe(recommendation.to_operator_proposal())
        channel_recommendations.append(compact_recommendation(proposal))

    master_recommendation = analyze_smart_dsp_audio(
        base_mix,
        **smart_dsp_kwargs(
            target_id="main:REPA_Song",
            source_role="master_bus",
            source_label="base_soundcheck_mix",
            sample_rate=sample_rate,
            target_plane_lufs=TARGET_LUFS,
        ),
    )
    master_proposal = json_safe(master_recommendation.to_operator_proposal())

    return {
        "pipeline": PIPELINE_NAME,
        "mode": "offline_soundcheck_audition",
        "status": "offline_audition_only",
        "notes": [
            "Uses current working-tree smart-DSP logic, including uncommitted modules.",
            "Readback and rollback are simulated for offline audition so the selector can expose correction choices.",
            "The rendered audio is a listening demonstration only; it does not authorize WING writes.",
            "Live use still requires real WING fw3.0.5 OSC mapping, readback, rollback, and operator approval.",
        ],
        "proposal_detail_policy": "compact_recommendations_only",
        "recommendation_count": len(channel_recommendations) + 1,
        "master_recommendation": compact_recommendation(master_proposal),
        "channel_recommendations": channel_recommendations,
    }


def score_from_evidence(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(score):
        return fallback
    return float(np.clip(score, 0.0, 1.0))


def apply_smart_dsp_audition_layer(
    base_mix: np.ndarray,
    sample_rate: int,
    smart_dsp_report: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    before_lufs = integrated_lufs(base_mix, sample_rate)
    before_peak = peak_dbfs(base_mix)
    master = smart_dsp_report.get("master_recommendation") or {}
    evidence = master.get("audio_evidence") or {}
    muddiness = score_from_evidence(evidence.get("muddiness_score"), 0.45)
    harshness = score_from_evidence(evidence.get("harshness_score"), 0.35)
    sibilance = score_from_evidence(evidence.get("sibilance_score"), 0.35)

    processed = to_stereo(base_mix)
    applied: list[dict[str, Any]] = []

    processed = butter_filter(processed, sample_rate, kind="highpass", frequency=28.0, order=3)
    applied.append({"stage": "sub_headroom_highpass", "frequency_hz": 28.0, "slope": "18_db_per_oct"})

    low_mid_cut = -round(0.7 + 1.2 * muddiness, 2)
    processed = peaking_eq(processed, sample_rate, frequency=285.0, q=1.05, gain_db=low_mid_cut)
    applied.append({"stage": "mud_control_eq", "frequency_hz": 285.0, "q": 1.05, "gain_db": low_mid_cut})

    presence_cut = -round(0.35 + 0.95 * harshness, 2)
    processed = peaking_eq(processed, sample_rate, frequency=3150.0, q=1.45, gain_db=presence_cut)
    applied.append({"stage": "harshness_control_eq", "frequency_hz": 3150.0, "q": 1.45, "gain_db": presence_cut})

    air_lift = round(0.35 + 0.35 * max(0.0, 1.0 - harshness), 2)
    processed = high_shelf(processed, sample_rate, frequency=11200.0, gain_db=air_lift)
    applied.append({"stage": "air_recovery_shelf", "frequency_hz": 11200.0, "gain_db": air_lift})

    deess_db = round(1.2 + 2.0 * sibilance, 2)
    processed, deess_report = dynamic_band_reduction(
        processed,
        sample_rate,
        low_hz=5600.0,
        high_hz=9300.0,
        threshold_percentile=74.0,
        max_reduction_db=deess_db,
        detector_lowpass_hz=35.0,
    )
    applied.append({"stage": "dynamic_sibilance_control", **deess_report})

    processed, low_mid_dynamic = dynamic_band_reduction(
        processed,
        sample_rate,
        low_hz=160.0,
        high_hz=420.0,
        threshold_percentile=82.0,
        max_reduction_db=1.4,
        detector_lowpass_hz=18.0,
    )
    applied.append({"stage": "dynamic_low_mid_cleanup", **low_mid_dynamic})

    processed, comp_report = bus_compress(
        processed,
        sample_rate,
        ratio=1.55,
        threshold_percentile=87.0,
        max_reduction_db=2.1,
    )
    applied.append({"stage": "smart_dsp_bus_glue", **comp_report})

    processed, width_report = stereo_width(processed, factor=1.045)
    applied.append({"stage": "pan_width_safety", **width_report})

    current_lufs = integrated_lufs(processed, sample_rate)
    norm_gain_db = 0.0
    if current_lufs is not None:
        norm_gain_db = float(np.clip(TARGET_LUFS - current_lufs, -4.0, 3.0))
        processed = (processed * db_to_amp(norm_gain_db)).astype(np.float32)
    processed, limiter_gain_db = true_peak_limited(processed, PEAK_CEILING_DBFS)
    applied.append(
        {
            "stage": "loudness_and_headroom_trim",
            "target_lufs": TARGET_LUFS,
            "normalization_gain_db": round(norm_gain_db, 3),
            "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "peak_safety_gain_db": round(limiter_gain_db, 3),
        }
    )

    after_lufs = integrated_lufs(processed, sample_rate)
    after_peak = peak_dbfs(processed)
    return processed.astype(np.float32), {
        "status": "applied_to_offline_render",
        "input_metrics": {
            "lufs": round(before_lufs, 3) if before_lufs is not None else None,
            "peak_dbfs": round(before_peak, 3),
        },
        "output_metrics": {
            "lufs": round(after_lufs, 3) if after_lufs is not None else None,
            "peak_dbfs": round(after_peak, 3),
        },
        "driving_master_evidence": {
            "muddiness_score": evidence.get("muddiness_score"),
            "harshness_score": evidence.get("harshness_score"),
            "sibilance_score": evidence.get("sibilance_score"),
            "problem_hints": evidence.get("problem_hints") or [],
        },
        "applied_corrections": applied,
    }


def write_audio(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    ensure_parent(path)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_24")


def write_summary(
    *,
    summary_path: Path,
    output_path: Path,
    base_output_path: Path,
    report_path: Path,
    base_report_path: Path,
    smart_dsp_path: Path,
    run_log_path: Path,
    input_dir: Path,
    final_report: dict[str, Any],
) -> None:
    metrics = final_report.get("audible_smart_dsp_layer", {}).get("output_metrics", {})
    lines = [
        "# REPA Song smart-DSP offline audition",
        "",
        f"- Pipeline: `{PIPELINE_NAME}`",
        f"- Input multitrack: `{input_dir}`",
        f"- Final WAV: `{output_path}`",
        f"- Base soundcheck WAV: `{base_output_path}`",
        f"- Report JSON: `{report_path}`",
        f"- Base report JSON: `{base_report_path}`",
        f"- Smart-DSP JSON: `{smart_dsp_path}`",
        f"- Render log: `{run_log_path}`",
        f"- Final LUFS: `{metrics.get('lufs')}`",
        f"- Final peak: `{metrics.get('peak_dbfs')} dBFS`",
        "",
        "This is an offline listening render. It demonstrates current correction logic,",
        "including uncommitted smart-DSP changes, without live WING write authority.",
    ]
    ensure_parent(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(default_output_path()))
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--genre", default="rock")
    parser.add_argument("--reference", default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output).expanduser()
    base_output_path = default_base_output_path(output_path)
    report_path = default_report_path(output_path)
    base_report_path = default_base_report_path(output_path)
    smart_dsp_path = default_smart_dsp_path(output_path)
    summary_path = default_summary_path(output_path)
    run_log_path = default_run_log_path(output_path)
    input_dir = resolve_input_dir(Path(args.input_dir).expanduser())

    run_base_soundcheck(
        input_dir=input_dir,
        output_path=base_output_path,
        report_path=base_report_path,
        run_log_path=run_log_path,
        tempo_bpm=args.tempo_bpm,
        genre=args.genre,
        reference=args.reference,
    )

    base_mix, sample_rate = sf.read(str(base_output_path), dtype="float32", always_2d=True)
    base_mix = to_stereo(base_mix)
    soundcheck_report = load_json(base_report_path)
    smart_dsp_report = build_smart_dsp_report(
        input_dir=input_dir,
        base_mix=base_mix,
        sample_rate=int(sample_rate),
        soundcheck_report=soundcheck_report,
    )
    final_mix, audition_report = apply_smart_dsp_audition_layer(
        base_mix,
        int(sample_rate),
        smart_dsp_report,
    )
    write_audio(output_path, final_mix, int(sample_rate))

    final_report = {
        "pipeline": PIPELINE_NAME,
        "status": "rendered_offline_audition",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output": str(output_path),
        "base_soundcheck_output": str(base_output_path),
        "sample_rate": int(sample_rate),
        "duration_sec": round(float(len(final_mix) / sample_rate), 3),
        "safety": {
            "live_mixer_writes": False,
            "wing_authority": "none",
            "offline_only": True,
            "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "note": "Real WING apply still requires verified fw3.0.5 OSC mapping, readback, rollback, and operator approval.",
        },
        "base_soundcheck_summary": {
            "final_lufs": soundcheck_report.get("final_lufs"),
            "final_peak_dbfs": soundcheck_report.get("final_peak_dbfs"),
            "final_channels": len(soundcheck_report.get("final_channels") or []),
        },
        "smart_dsp": smart_dsp_report,
        "audible_smart_dsp_layer": audition_report,
        "artifacts": {
            "final_wav": str(output_path),
            "base_wav": str(base_output_path),
            "base_report": str(base_report_path),
            "smart_dsp_report": str(smart_dsp_path),
            "summary": str(summary_path),
            "run_log": str(run_log_path),
        },
        "sha256": {
            "final_wav": sha256_file(output_path),
            "base_wav": sha256_file(base_output_path),
        },
    }
    ensure_parent(smart_dsp_path).write_text(
        json.dumps(json_safe(smart_dsp_report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ensure_parent(report_path).write_text(
        json.dumps(json_safe(final_report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(
        summary_path=summary_path,
        output_path=output_path,
        base_output_path=base_output_path,
        report_path=report_path,
        base_report_path=base_report_path,
        smart_dsp_path=smart_dsp_path,
        run_log_path=run_log_path,
        input_dir=input_dir,
        final_report=final_report,
    )
    print(
        json.dumps(
            json_safe(
                {
                    "pipeline": PIPELINE_NAME,
                    "status": final_report["status"],
                    "output": str(output_path),
                    "report": str(report_path),
                    "summary": str(summary_path),
                    "input_dir": str(input_dir),
                    "duration_sec": final_report["duration_sec"],
                    "base_soundcheck_summary": final_report["base_soundcheck_summary"],
                    "audible_smart_dsp_layer": audition_report,
                    "sha256": final_report["sha256"],
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
