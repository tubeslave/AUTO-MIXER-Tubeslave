#!/usr/bin/env python3
"""
Сведение стемов без кода из backend/: только numpy/scipy/soundfile/pyloudnorm/pedalboard
и общеинженерные приёмы (headroom, HPF, RBJ bell EQ, компрессия, баланс по LUFS,
мастер: нормализация громкости + ограничение true peak с 4× oversampling).

Не использует AutoMaster, default_config, mixing_rules и т.д. из проекта.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, Pedalboard
from scipy.signal import butter, lfilter, resample_poly, sosfilt

warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")


def _rbj_peak(w0: float, gain_db: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    a = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * max(0.1, q))
    b0 = 1.0 + alpha * a
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def eq_peaks(x: np.ndarray, sr: int, peaks: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    """peaks: (Hz, gain_dB, Q)."""
    y = np.asarray(x, dtype=np.float64)
    for f0, g, q in peaks:
        w0 = 2.0 * np.pi * float(f0) / sr
        b, a = _rbj_peak(w0, float(g), float(q))
        y = lfilter(b, a, y)
    return y


def highpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    if fc <= 0:
        return x
    sos = butter(order, fc, btype="high", fs=sr, output="sos")
    return sosfilt(sos, np.asarray(x, dtype=np.float64))


def trim_peak_dbfs(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) + 1e-12
    return x * (10.0 ** (peak_dbfs / 20.0) / p)


def measure_true_peak_dbtp(audio: np.ndarray, sample_rate: int) -> float:
    """True peak (BS.1770-style detector): 4× oversample, max abs."""
    if audio.ndim == 1:
        os = resample_poly(audio, 4, 1)
    else:
        os = np.column_stack(
            [resample_poly(audio[:, ch], 4, 1) for ch in range(audio.shape[1])]
        )
    lin = float(np.max(np.abs(os)))
    return float(20.0 * np.log10(lin + 1e-12))


def master_lufs_and_tp(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    tp_ceiling_dbtp: float,
) -> np.ndarray:
    """Интегральная громкость → pyloudnorm; затем итеративно снижать до TP."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    meter = pyln.Meter(sr)
    cur = meter.integrated_loudness(x)
    if cur > -70.0:
        x = pyln.normalize.loudness(x, cur, target_lufs)
    x = np.asarray(x, dtype=np.float64)
    for _ in range(8):
        tp = measure_true_peak_dbtp(x, sr)
        if tp <= tp_ceiling_dbtp:
            break
        x = x * (10.0 ** (-(tp - tp_ceiling_dbtp + 0.05) / 20.0))
    return x.astype(np.float32)


@dataclass
class Stem:
    hpf: float
    eq: List[Tuple[float, float, float]]
    thr: float
    ratio: float
    att: float
    rel: float
    makeup_db: float
    target_i: float


# --- Ориентиры «из учебников» / стриминга, не из репозитория AUTO-MIXER ---
# Вокал чуть впереди «бэнда», playback ниже, комната/оверходы тише, суб/удар — в центре.
STEMS: Dict[str, Stem] = {
    "Vox1.wav": Stem(
        85.0,
        [(300.0, -2.0, 1.2), (2800.0, 2.5, 1.0), (11000.0, 1.0, 0.8)],
        -22.0,
        3.0,
        12.0,
        120.0,
        1.0,
        -15.5,
    ),
    "Vox2.wav": Stem(
        85.0,
        [(300.0, -2.0, 1.2), (2800.0, 2.0, 1.0), (11000.0, 0.8, 0.8)],
        -22.0,
        3.0,
        12.0,
        120.0,
        0.5,
        -16.0,
    ),
    "Vox3.wav": Stem(
        85.0,
        [(300.0, -2.0, 1.2), (2800.0, 2.0, 1.0), (11000.0, 0.8, 0.8)],
        -22.0,
        3.0,
        12.0,
        120.0,
        0.5,
        -16.0,
    ),
    "Kick.wav": Stem(
        30.0,
        [(55.0, 2.5, 0.9), (320.0, -3.0, 1.0), (3500.0, 2.0, 1.4)],
        -18.0,
        4.0,
        18.0,
        55.0,
        0.5,
        -17.5,
    ),
    "Snare.wav": Stem(
        90.0,
        [(190.0, 1.5, 1.0), (700.0, -2.5, 1.3), (5000.0, 2.0, 1.2)],
        -16.0,
        3.5,
        7.0,
        95.0,
        1.0,
        -18.0,
    ),
    "Floor Tom.wav": Stem(
        65.0,
        [(140.0, 2.0, 1.0), (450.0, -2.5, 1.2), (3800.0, 1.5, 1.3)],
        -20.0,
        3.0,
        12.0,
        90.0,
        1.5,
        -24.0,
    ),
    "Mid Tom.wav": Stem(
        65.0,
        [(180.0, 2.0, 1.0), (450.0, -2.5, 1.2), (3800.0, 1.5, 1.3)],
        -20.0,
        3.0,
        12.0,
        90.0,
        1.5,
        -24.0,
    ),
    "Hi-Hat.wav": Stem(
        220.0,
        [(9500.0, 1.5, 0.85)],
        -19.0,
        2.0,
        5.0,
        85.0,
        0.0,
        -22.0,
    ),
    "Ride.wav": Stem(
        220.0,
        [(7500.0, 1.0, 0.9)],
        -22.0,
        2.0,
        8.0,
        110.0,
        1.5,
        -25.0,
    ),
    "Overhead L.wav": Stem(
        160.0,
        [(380.0, -1.8, 1.1), (10500.0, 1.2, 0.9)],
        -19.0,
        2.0,
        20.0,
        160.0,
        0.0,
        -23.5,
    ),
    "Overhead R.wav": Stem(
        160.0,
        [(380.0, -1.8, 1.1), (10500.0, 1.2, 0.9)],
        -19.0,
        2.0,
        20.0,
        160.0,
        0.0,
        -23.5,
    ),
    "Drum Room L.wav": Stem(
        110.0,
        [(4500.0, 0.8, 0.9)],
        -21.0,
        1.8,
        25.0,
        200.0,
        0.0,
        -26.5,
    ),
    "Drum Room R.wav": Stem(
        110.0,
        [(4500.0, 0.8, 0.9)],
        -21.0,
        1.8,
        25.0,
        200.0,
        0.0,
        -26.5,
    ),
    "Guitar L.wav": Stem(
        85.0,
        [(250.0, -2.5, 1.2), (2400.0, 1.5, 1.0)],
        -18.0,
        2.2,
        20.0,
        150.0,
        0.5,
        -19.5,
    ),
    "Guitar R.wav": Stem(
        85.0,
        [(250.0, -2.5, 1.2), (2400.0, 1.5, 1.0)],
        -18.0,
        2.2,
        20.0,
        150.0,
        0.5,
        -19.5,
    ),
    "Accordion.wav": Stem(
        75.0,
        [(1000.0, 1.0, 1.0), (3200.0, 1.0, 1.0)],
        -19.0,
        2.5,
        15.0,
        130.0,
        0.5,
        -17.0,
    ),
    "Playback L.wav": Stem(
        45.0,
        [(400.0, -1.5, 1.0)],
        -17.0,
        2.0,
        28.0,
        190.0,
        0.0,
        -21.0,
    ),
    "Playback R.wav": Stem(
        45.0,
        [(400.0, -1.5, 1.0)],
        -17.0,
        2.0,
        28.0,
        190.0,
        0.0,
        -21.0,
    ),
}

CENTER = {
    "Kick.wav",
    "Snare.wav",
    "Floor Tom.wav",
    "Mid Tom.wav",
    "Hi-Hat.wav",
    "Ride.wav",
    "Accordion.wav",
    "Vox1.wav",
    "Vox2.wav",
    "Vox3.wav",
}
LEFT = {"Drum Room L.wav", "Overhead L.wav", "Guitar L.wav", "Playback L.wav"}
RIGHT = {"Drum Room R.wav", "Overhead R.wav", "Guitar R.wav", "Playback R.wav"}


def process_mono(x: np.ndarray, sr: int, stem: Stem) -> np.ndarray:
    y = trim_peak_dbfs(x.astype(np.float64), -16.0)
    y = highpass(y, sr, stem.hpf, order=4 if stem.hpf >= 55 else 2)
    y = eq_peaks(y, sr, stem.eq)
    pb = Pedalboard(
        [
            Compressor(
                threshold_db=stem.thr,
                ratio=stem.ratio,
                attack_ms=stem.att,
                release_ms=stem.rel,
            )
        ]
    )
    y = pb(y.astype(np.float32), sr).astype(np.float64) * (10.0 ** (stem.makeup_db / 20.0))
    meter = pyln.Meter(sr)
    cur = meter.integrated_loudness(y.reshape(-1, 1))
    if cur > -70.0:
        y = pyln.normalize.loudness(y.reshape(-1, 1), cur, stem.target_i).flatten()
    return y.astype(np.float32)


def bus_glue(stereo: np.ndarray, sr: int) -> np.ndarray:
    """Лёгкая «шина»: HPF 28 Hz + мягкое сжатие (педагогический glue, не из backend)."""
    l = stereo[:, 0].astype(np.float64)
    r = stereo[:, 1].astype(np.float64)
    m = 0.5 * (l + r)
    s = 0.5 * (l - r)
    m = highpass(m, sr, 28.0, order=2)
    glue = Pedalboard([Compressor(threshold_db=-14.0, ratio=1.4, attack_ms=35.0, release_ms=220.0)])
    m2 = glue(m.astype(np.float32), sr).astype(np.float64)
    l2 = m2 + s
    r2 = m2 - s
    return np.column_stack([l2, r2]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix_dir", nargs="?", default="/Users/dmitrijvolkov/Desktop/MIX")
    ap.add_argument(
        "-o",
        "--output",
        default="/Users/dmitrijvolkov/Desktop/MIX_KnowledgeOnly.wav",
    )
    ap.add_argument("--master-lufs", type=float, default=-14.0, help="Цель I после суммы")
    ap.add_argument("--tp", type=float, default=-1.0, help="Потолок true peak, dBTP")
    args = ap.parse_args()

    mix_dir = os.path.abspath(args.mix_dir)
    files = sorted(STEMS.keys())
    for f in files:
        if f not in os.listdir(mix_dir):
            print("Нет файла:", f, file=sys.stderr)
            return 1

    _, sr = sf.read(os.path.join(mix_dir, files[0]))
    n = None
    out = None

    for name in files:
        x, r = sf.read(os.path.join(mix_dir, name), dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if r != sr or (n is not None and len(x) != n):
            print("Формат/длина не совпадают", file=sys.stderr)
            return 1
        if n is None:
            n = len(x)
            out = np.zeros((n, 2), dtype=np.float32)

        y = process_mono(x, sr, STEMS[name])
        if name in CENTER:
            out[:, 0] += y
            out[:, 1] += y
        elif name in LEFT:
            out[:, 0] += y
        elif name in RIGHT:
            out[:, 1] += y
        else:
            return 1

    out = bus_glue(out, sr)
    out = master_lufs_and_tp(out, sr, args.master_lufs, args.tp)

    sf.write(os.path.abspath(args.output), out, sr, subtype="PCM_24")
    print("Записано:", os.path.abspath(args.output))
    print(
        "I≈",
        round(pyln.Meter(sr).integrated_loudness(out), 2),
        "LUFS, TP≈",
        round(measure_true_peak_dbtp(out.astype(np.float64), sr), 2),
        "dBTP",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
