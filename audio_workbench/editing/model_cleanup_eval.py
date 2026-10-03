from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf

@dataclass(frozen=True)
class ModelCleanupCandidate:
    name: str
    clean_path: Path
    raw_path: Path

def residual(raw: np.ndarray, clean: np.ndarray) -> np.ndarray:
    n = min(len(raw), len(clean))
    return raw[:n] - clean[:n]

def residual_target_leak_dbfs(raw: np.ndarray, clean: np.ndarray, target_activity: np.ndarray) -> float:
    """Removed RMS during strong intended-source activity. Lower is safer."""
    r = residual(raw, clean)
    a = np.asarray(target_activity[:len(r)], dtype=bool)
    if not np.any(a):
        return -200.0
    rms = np.sqrt(np.mean(r[a].astype("float64") ** 2) + 1e-20)
    return float(20 * np.log10(rms + 1e-20))

def evaluate(raw_path: Path, clean_path: Path, target_activity: np.ndarray) -> dict:
    raw, sr = sf.read(raw_path, always_2d=True, dtype="float32")
    clean, sr2 = sf.read(clean_path, always_2d=True, dtype="float32")
    if sr != sr2:
        raise ValueError("sample-rate mismatch")
    n = min(len(raw), len(clean))
    raw, clean = raw[:n], clean[:n]
    rem = raw - clean
    return {
        "sample_rate": sr,
        "removed_rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(rem.astype("float64") ** 2) + 1e-20) + 1e-20)),
        "target_active_removed_rms_dbfs": residual_target_leak_dbfs(raw, clean, target_activity),
        "peak_error_dbfs": float(20 * np.log10(np.max(np.abs(rem)) + 1e-20)),
    }

def save_triplet(raw_path: Path, clean_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, sr = sf.read(raw_path, always_2d=True, dtype="float32")
    clean, sr2 = sf.read(clean_path, always_2d=True, dtype="float32")
    if sr != sr2:
        raise ValueError("sample-rate mismatch")
    n = min(len(raw), len(clean))
    raw, clean = raw[:n], clean[:n]
    sf.write(out_dir / "RAW.wav", raw, sr, subtype="PCM_24")
    sf.write(out_dir / "CLEAN.wav", clean, sr, subtype="PCM_24")
    sf.write(out_dir / "REMOVED.wav", raw - clean, sr, subtype="PCM_24")

# Development rule: metrics never accept cleanup alone.
# Human audition of RAW / CLEAN / REMOVED is mandatory.
