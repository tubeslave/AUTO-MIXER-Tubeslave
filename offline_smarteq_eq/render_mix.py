from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from lufs_gain_staging import INPUT_TRIM_AUTOGAIN_METHOD_ID, LocalInputGainProfile  # noqa: E402
from offline_smarteq_eq.method import (  # noqa: E402
    SMART_EQ_METHOD_ID,
    OfflineSmartEQConfig,
    apply_offline_smarteq_eq,
    design_offline_smarteq_eq,
)
from tools.offline_input_trim_autogain_mix import (  # noqa: E402
    amp_to_db,
    calibration_config,
    db_to_amp,
    infer_track_role,
    iterative_offline_report,
    iter_wavs,
    json_ready,
    load_config,
    mono_sum,
    render_track,
)


DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "SONG  REPA"
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED.wav"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED_report.json"


def run_offline_smarteq_gainstaged_mix(
    input_dir: Path,
    output_path: Path,
    report_path: Path,
    *,
    config_path: Path,
    mode: str = "soundcheck",
    analysis_passes: int = 4,
    chunk_sec: float = 0.4,
    master_ceiling_dbfs: float = -1.0,
    min_signal_presence: Optional[float] = None,
    eq_max_active_bands: int = 8,
) -> Dict[str, Any]:
    wavs = iter_wavs(input_dir)
    if not wavs:
        raise RuntimeError(f"No WAV files found in {input_dir}")

    config = load_config(config_path)
    cal_config = dict(calibration_config(config))
    if min_signal_presence is not None:
        cal_config["min_signal_presence"] = float(min_signal_presence)
    preamp_profile = LocalInputGainProfile.from_config(cal_config.get("local_input_preamp", {}))

    infos = [sf.info(str(path)) for path in wavs]
    sample_rates = {int(info.samplerate) for info in infos}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Sample-rate mismatch: {sorted(sample_rates)}")
    sr = sample_rates.pop()
    target_len = max(int(info.frames) for info in infos)
    mix = np.zeros((target_len, 2), dtype=np.float32)
    tracks: List[Dict[str, Any]] = []
    eq_config = OfflineSmartEQConfig(max_active_bands=int(eq_max_active_bands))

    for channel_id, path in enumerate(wavs, start=1):
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if int(file_sr) != sr:
            raise RuntimeError(f"{path.name}: sample-rate mismatch {file_sr} != {sr}")

        role = infer_track_role(path)
        mono = mono_sum(audio)
        gain_report, pass_history, cumulative_gain_db = iterative_offline_report(
            mono,
            sr,
            channel_id,
            role.preset,
            cal_config,
            preamp_profile,
            mode=mode,
            analysis_passes=analysis_passes,
            chunk_sec=chunk_sec,
        )

        gained = (np.asarray(audio, dtype=np.float32) * db_to_amp(cumulative_gain_db)).astype(np.float32)
        eq_report = design_offline_smarteq_eq(gained, sr, role.preset, eq_config)
        processed = apply_offline_smarteq_eq(gained, sr, eq_report)
        mix += render_track(processed, role.pan, 0.0, target_len)
        tracks.append(
            {
                "channel": channel_id,
                "file": path.name,
                "preset": role.preset,
                "pan": role.pan,
                "gain_staging_method_id": INPUT_TRIM_AUTOGAIN_METHOD_ID,
                "applied_input_trim_gain_db": round(float(cumulative_gain_db), 3),
                "gain_staging_report": gain_report,
                "gain_staging_analysis_passes": pass_history,
                "eq_method_id": SMART_EQ_METHOD_ID,
                "eq_report": eq_report.to_dict(),
            }
        )

    pre_master_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    pre_master_peak_db = amp_to_db(pre_master_peak)
    master_trim_db = min(0.0, float(master_ceiling_dbfs) - pre_master_peak_db)
    if master_trim_db < 0.0:
        mix = (mix * db_to_amp(master_trim_db)).astype(np.float32)
    final_peak_db = amp_to_db(float(np.max(np.abs(mix))) if mix.size else 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix, sr, subtype="PCM_24")

    summary = {
        "type": "offline_smarteq_eq_plus_input_trim_gainstaging_render",
        "methods_applied_in_order": [INPUT_TRIM_AUTOGAIN_METHOD_ID, SMART_EQ_METHOD_ID],
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "config_path": str(config_path),
        "mode": mode,
        "sample_rate": sr,
        "duration_sec": round(float(target_len) / float(sr), 3),
        "track_count": len(tracks),
        "analysis_passes": max(1, int(analysis_passes)),
        "chunk_sec": float(chunk_sec),
        "master_ceiling_dbfs": float(master_ceiling_dbfs),
        "min_signal_presence": float(cal_config.get("min_signal_presence", 0.05)),
        "pre_master_peak_dbfs": round(pre_master_peak_db, 3),
        "master_safety_trim_db": round(master_trim_db, 3),
        "final_peak_dbfs": round(final_peak_db, 3),
        "tracks": tracks,
    }
    report_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render offline input-trim gain staging plus offline spectral EQ.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default_config.json")
    parser.add_argument("--mode", choices=("soundcheck", "live"), default="soundcheck")
    parser.add_argument("--analysis-passes", type=int, default=4)
    parser.add_argument("--chunk-sec", type=float, default=0.4)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=-1.0)
    parser.add_argument("--min-signal-presence", type=float, default=None)
    parser.add_argument("--eq-max-active-bands", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_offline_smarteq_gainstaged_mix(
        args.input_dir.expanduser(),
        args.output.expanduser(),
        args.report.expanduser(),
        config_path=args.config.expanduser(),
        mode=args.mode,
        analysis_passes=args.analysis_passes,
        chunk_sec=args.chunk_sec,
        master_ceiling_dbfs=args.master_ceiling_dbfs,
        min_signal_presence=args.min_signal_presence,
        eq_max_active_bands=args.eq_max_active_bands,
    )
    print(
        json.dumps(
            {
                "output_path": summary["output_path"],
                "report_path": summary["report_path"],
                "track_count": summary["track_count"],
                "duration_sec": summary["duration_sec"],
                "final_peak_dbfs": summary["final_peak_dbfs"],
                "master_safety_trim_db": summary["master_safety_trim_db"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
