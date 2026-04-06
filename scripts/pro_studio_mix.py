#!/usr/bin/env python3
"""
Студийное сведение стемов (без импортов из backend проекта).

Опора на общепринятую практику: gain staging → фильтрация → субтрактивное EQ →
компрессия → сатурация (параллельно) → посылы на реверб → баланс → параллельное
«NY» сжатие по сумме ударных → мастер-шина (полки, клей, лимитер) → громкость
(ITU-R BS.1770) и true peak (4× OS).

Инструменты: numpy, scipy, soundfile, pyloudnorm, Pedalboard (Spotify/JUCE).
Ссылки на практику (вне репозитория): порядок этапов сведения — Black Ghost Audio,
«The 6-Step Mixing Workflow» (https://www.blackghostaudio.com/blog/the-6-step-mixing-workflow);
параллельное / NY-сжатие — Mixing Monster (https://mixingmonster.com/serial-versus-parallel-processing/);
DSP — Pedalboard (JUCE, Spotify, https://github.com/spotify/pedalboard).

Зеркальная / complementary EQ (субтрактивное освобождение места):
- «Yin–yang» / mirror: срез на соседней дорожке даёт тот же эффект ясности, что и
  подъём лида, без лишнего гейна (Learn Audio Engineering, EQ Yin and Yang:
  https://learnaudioengineering.com/eq-yin-yang/).
- Complementary: вырез на одном инструменте на частоте фундамента/тембра другого
  (напр. notch в басу под kick — Johnny Copland, RouteNote Create «kick and bass»).
  Здесь: вырезы под kick (~62–68 Hz), под вокал (2.5–3.5 kHz, 250–400 Hz «mud»),
  под снейр в оверхедах/руме.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Callable, Dict, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Chorus,
    Compressor,
    Distortion,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    NoiseGate,
    Pedalboard,
    PeakFilter,
    Reverb,
)
from scipy.signal import resample_poly

warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")

# --- Утилиты ---


def trim_peak(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) + 1e-12
    return x * (10.0 ** (peak_dbfs / 20.0) / p)


def pb_mono(board: Pedalboard, x: np.ndarray, sr: int) -> np.ndarray:
    y = board(x.astype(np.float32), sr)
    return np.asarray(y, dtype=np.float64).reshape(-1)


def measure_tp_dbtp(audio: np.ndarray, sr: int) -> float:
    if audio.ndim == 1:
        osx = resample_poly(audio, 4, 1)
    else:
        osx = np.column_stack([resample_poly(audio[:, c], 4, 1) for c in range(audio.shape[1])])
    lin = float(np.max(np.abs(osx)))
    return float(20.0 * np.log10(lin + 1e-12))


def normalize_i(x: np.ndarray, sr: int, target: float) -> np.ndarray:
    if x.ndim == 1:
        xm = x.reshape(-1, 1)
    else:
        xm = x
    m = pyln.Meter(sr)
    cur = m.integrated_loudness(xm.astype(np.float32))
    if cur <= -70.0:
        out = xm
    else:
        out = pyln.normalize.loudness(xm.astype(np.float64), cur, target)
    out = np.asarray(out, dtype=np.float64)
    if out.shape[1] == 1:
        return out.reshape(-1)
    return out


def pan_constant_power(m: np.ndarray, pan: float) -> Tuple[np.ndarray, np.ndarray]:
    """pan in [-1, 1], L/R."""
    ang = (float(pan) + 1.0) * (np.pi / 4.0)
    return m * np.cos(ang), m * np.sin(ang)


def stereo_process(pb: Pedalboard, st: np.ndarray, sr: int) -> np.ndarray:
    """st: (samples, 2)"""
    x = st.astype(np.float32).T
    y = pb(x, sr)
    return np.asarray(y, dtype=np.float32).T


def complementary_eq_board(name: str) -> Pedalboard | None:
    """
    Второй проход: только субтрактивные узкие/средние вырезы под маскировку.
    Kick fundamental ~62 Hz; вокал — присутствие ~3 kHz, «тело» 280–400 Hz;
    снейр — ~200–220 Hz в оверхедах.
    """
    f: list = []
    if name in ("Guitar L.wav", "Guitar R.wav"):
        f += [
            PeakFilter(cutoff_frequency_hz=3120.0, gain_db=-2.4, q=2.8),
            PeakFilter(cutoff_frequency_hz=410.0, gain_db=-1.6, q=1.45),
        ]
    if name in ("Playback L.wav", "Playback R.wav"):
        f += [
            PeakFilter(cutoff_frequency_hz=64.0, gain_db=-3.0, q=5.5),
            PeakFilter(cutoff_frequency_hz=295.0, gain_db=-2.2, q=1.35),
            PeakFilter(cutoff_frequency_hz=3050.0, gain_db=-1.8, q=2.2),
        ]
    if name == "Accordion.wav":
        f += [
            PeakFilter(cutoff_frequency_hz=66.0, gain_db=-2.2, q=4.5),
            PeakFilter(cutoff_frequency_hz=2850.0, gain_db=-1.6, q=2.4),
            PeakFilter(cutoff_frequency_hz=380.0, gain_db=-1.2, q=1.2),
        ]
    if name in ("Drum Room L.wav", "Drum Room R.wav"):
        f += [
            PeakFilter(cutoff_frequency_hz=315.0, gain_db=-1.7, q=1.25),
            PeakFilter(cutoff_frequency_hz=2400.0, gain_db=-0.9, q=1.0),
        ]
    if name in ("Overhead L.wav", "Overhead R.wav"):
        f += [
            PeakFilter(cutoff_frequency_hz=218.0, gain_db=-1.35, q=1.35),
            PeakFilter(cutoff_frequency_hz=900.0, gain_db=-0.9, q=1.1),
        ]
    if name == "Floor Tom.wav":
        f.append(PeakFilter(cutoff_frequency_hz=63.0, gain_db=-1.8, q=4.0))
    if name == "Mid Tom.wav":
        f.append(PeakFilter(cutoff_frequency_hz=61.0, gain_db=-1.5, q=3.8))
    if name == "Ride.wav":
        f.append(PeakFilter(cutoff_frequency_hz=2900.0, gain_db=-1.1, q=1.8))
    if name == "Hi-Hat.wav":
        f.append(PeakFilter(cutoff_frequency_hz=3350.0, gain_db=-1.0, q=2.0))
    if name == "Kick.wav":
        f.append(PeakFilter(cutoff_frequency_hz=245.0, gain_db=-1.1, q=1.25))
    if not f:
        return None
    return Pedalboard(f)


def apply_complementary(name: str, y: np.ndarray, sr: int) -> np.ndarray:
    board = complementary_eq_board(name)
    if board is None:
        return y
    return pb_mono(board, y, sr)


# --- Цепочки по типам (студийные ориентиры, не из репозитория) ---


def chain_vocal() -> Pedalboard:
    """
    Лид-вокал: порядок по SOS / iZotope — HPF → субтрактивно (mud/nasal) → присутствие
    → компрессия → de-ess (узкие срезы ~6–8 kHz) → air (high shelf) → лёгкий второй компрессор.
    """
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=86.0),
            PeakFilter(cutoff_frequency_hz=265.0, gain_db=-3.0, q=1.35),
            PeakFilter(cutoff_frequency_hz=520.0, gain_db=-1.5, q=1.45),
            PeakFilter(cutoff_frequency_hz=3050.0, gain_db=2.4, q=1.08),
            Compressor(
                threshold_db=-21.5,
                ratio=3.5,
                attack_ms=8.0,
                release_ms=118.0,
            ),
            PeakFilter(cutoff_frequency_hz=6100.0, gain_db=-3.0, q=4.2),
            PeakFilter(cutoff_frequency_hz=7800.0, gain_db=-1.3, q=3.0),
            HighShelfFilter(cutoff_frequency_hz=11500.0, gain_db=1.45, q=0.707),
            Compressor(
                threshold_db=-19.0,
                ratio=2.1,
                attack_ms=18.0,
                release_ms=205.0,
            ),
        ]
    )


def chain_vocal_sat() -> Pedalboard:
    return Pedalboard([Distortion(drive_db=3.2)])


def chain_kick() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=38.0),
            PeakFilter(cutoff_frequency_hz=58.0, gain_db=2.8, q=0.95),
            PeakFilter(cutoff_frequency_hz=340.0, gain_db=-3.2, q=1.15),
            PeakFilter(cutoff_frequency_hz=3600.0, gain_db=2.2, q=1.35),
            Compressor(
                threshold_db=-17.5,
                ratio=4.2,
                attack_ms=20.0,
                release_ms=58.0,
            ),
        ]
    )


def chain_snare() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=95.0),
            PeakFilter(cutoff_frequency_hz=195.0, gain_db=1.8, q=1.1),
            PeakFilter(cutoff_frequency_hz=720.0, gain_db=-2.4, q=1.35),
            PeakFilter(cutoff_frequency_hz=5100.0, gain_db=2.4, q=1.25),
            Compressor(
                threshold_db=-15.5,
                ratio=3.4,
                attack_ms=6.5,
                release_ms=98.0,
            ),
        ]
    )


def chain_tom() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=68.0),
            PeakFilter(cutoff_frequency_hz=155.0, gain_db=2.0, q=1.05),
            PeakFilter(cutoff_frequency_hz=430.0, gain_db=-2.8, q=1.25),
            PeakFilter(cutoff_frequency_hz=3900.0, gain_db=1.8, q=1.35),
            NoiseGate(threshold_db=-38.0, ratio=8.0, attack_ms=0.5, release_ms=120.0),
            Compressor(
                threshold_db=-21.0,
                ratio=2.9,
                attack_ms=12.0,
                release_ms=88.0,
            ),
        ]
    )


def chain_hat_ride() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=210.0),
            PeakFilter(cutoff_frequency_hz=9800.0, gain_db=1.6, q=0.88),
            Compressor(
                threshold_db=-19.5,
                ratio=2.0,
                attack_ms=4.5,
                release_ms=90.0,
            ),
        ]
    )


def chain_oh() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=155.0),
            PeakFilter(cutoff_frequency_hz=390.0, gain_db=-1.9, q=1.15),
            PeakFilter(cutoff_frequency_hz=10400.0, gain_db=1.4, q=0.82),
            Compressor(
                threshold_db=-19.0,
                ratio=1.9,
                attack_ms=18.0,
                release_ms=155.0,
            ),
        ]
    )


def chain_room() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=105.0),
            PeakFilter(cutoff_frequency_hz=4800.0, gain_db=0.9, q=0.75),
            Compressor(
                threshold_db=-21.5,
                ratio=1.85,
                attack_ms=22.0,
                release_ms=210.0,
            ),
        ]
    )


def chain_guitar() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=82.0),
            PeakFilter(cutoff_frequency_hz=245.0, gain_db=-2.2, q=1.2),
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=1.7, q=1.05),
            Compressor(
                threshold_db=-17.5,
                ratio=2.1,
                attack_ms=19.0,
                release_ms=145.0,
            ),
            Chorus(rate_hz=0.35, depth=0.2, centre_delay_ms=8.0, feedback=0.08, mix=0.12),
        ]
    )


def chain_accordion() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=78.0),
            PeakFilter(cutoff_frequency_hz=1100.0, gain_db=1.0, q=1.0),
            PeakFilter(cutoff_frequency_hz=3300.0, gain_db=1.1, q=1.0),
            Compressor(
                threshold_db=-18.5,
                ratio=2.6,
                attack_ms=14.0,
                release_ms=125.0,
            ),
        ]
    )


def chain_playback() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=42.0),
            PeakFilter(cutoff_frequency_hz=380.0, gain_db=-1.2, q=1.0),
            Compressor(
                threshold_db=-16.0,
                ratio=1.95,
                attack_ms=28.0,
                release_ms=185.0,
            ),
        ]
    )


def chain_drums_parallel_crush() -> Pedalboard:
    return Pedalboard(
        [
            Compressor(
                threshold_db=-22.0,
                ratio=7.5,
                attack_ms=3.0,
                release_ms=95.0,
            ),
            Compressor(
                threshold_db=-18.0,
                ratio=4.0,
                attack_ms=2.0,
                release_ms=75.0,
            ),
        ]
    )


def chain_master() -> Pedalboard:
    return Pedalboard(
        [
            LowShelfFilter(cutoff_frequency_hz=118.0, gain_db=0.55, q=0.72),
            HighShelfFilter(cutoff_frequency_hz=11500.0, gain_db=0.75, q=0.71),
            Compressor(
                threshold_db=-11.5,
                ratio=1.22,
                attack_ms=28.0,
                release_ms=420.0,
            ),
            Limiter(threshold_db=-0.9, release_ms=85.0),
        ]
    )


# --- Маршрутизация ---

STEM_PROCESSORS: Dict[str, Callable[[], Pedalboard]] = {
    "Vox1.wav": chain_vocal,
    "Vox2.wav": chain_vocal,
    "Vox3.wav": chain_vocal,
    "Kick.wav": chain_kick,
    "Snare.wav": chain_snare,
    "Floor Tom.wav": chain_tom,
    "Mid Tom.wav": chain_tom,
    "Hi-Hat.wav": chain_hat_ride,
    "Ride.wav": chain_hat_ride,
    "Overhead L.wav": chain_oh,
    "Overhead R.wav": chain_oh,
    "Drum Room L.wav": chain_room,
    "Drum Room R.wav": chain_room,
    "Guitar L.wav": chain_guitar,
    "Guitar R.wav": chain_guitar,
    "Accordion.wav": chain_accordion,
    "Playback L.wav": chain_playback,
    "Playback R.wav": chain_playback,
}

# Панорамы (константная мощность)
PAN: Dict[str, float] = {
    "Kick.wav": 0.0,
    "Snare.wav": 0.06,
    "Floor Tom.wav": -0.28,
    "Mid Tom.wav": 0.28,
    "Hi-Hat.wav": -0.38,
    "Ride.wav": 0.42,
    "Overhead L.wav": -1.0,
    "Overhead R.wav": 1.0,
    "Drum Room L.wav": -0.92,
    "Drum Room R.wav": 0.92,
    "Vox1.wav": 0.0,
    "Vox2.wav": -0.14,
    "Vox3.wav": 0.14,
    "Guitar L.wav": -0.95,
    "Guitar R.wav": 0.95,
    "Playback L.wav": -0.88,
    "Playback R.wav": 0.88,
    "Accordion.wav": 0.0,
}

# Относительные «фейдеры» после обработки
GAIN: Dict[str, float] = {
    "Vox1.wav": 1.08,
    "Vox2.wav": 1.02,
    "Vox3.wav": 1.02,
    "Kick.wav": 1.0,
    "Snare.wav": 1.02,
    "Floor Tom.wav": 1.0,
    "Mid Tom.wav": 1.0,
    "Hi-Hat.wav": 0.95,
    "Ride.wav": 0.92,
    "Overhead L.wav": 0.88,
    "Overhead R.wav": 0.88,
    "Drum Room L.wav": 0.82,
    "Drum Room R.wav": 0.82,
    "Guitar L.wav": 0.94,
    "Guitar R.wav": 0.94,
    "Accordion.wav": 1.0,
    "Playback L.wav": 0.76,
    "Playback R.wav": 0.76,
}

DRUM_SET = {
    "Kick.wav",
    "Snare.wav",
    "Floor Tom.wav",
    "Mid Tom.wav",
    "Hi-Hat.wav",
    "Ride.wav",
    "Overhead L.wav",
    "Overhead R.wav",
    "Drum Room L.wav",
    "Drum Room R.wav",
}

VOC_SET = {"Vox1.wav", "Vox2.wav", "Vox3.wav"}

TARGET_I: Dict[str, float] = {
    "Vox1.wav": -15.8,
    "Vox2.wav": -16.2,
    "Vox3.wav": -16.2,
    "Kick.wav": -17.2,
    "Snare.wav": -17.8,
    "Floor Tom.wav": -23.5,
    "Mid Tom.wav": -23.5,
    "Hi-Hat.wav": -21.5,
    "Ride.wav": -24.0,
    "Overhead L.wav": -22.8,
    "Overhead R.wav": -22.8,
    "Drum Room L.wav": -26.0,
    "Drum Room R.wav": -26.0,
    "Guitar L.wav": -19.2,
    "Guitar R.wav": -19.2,
    "Accordion.wav": -16.5,
    "Playback L.wav": -20.5,
    "Playback R.wav": -20.5,
}


def process_generic(
    name: str, x: np.ndarray, sr: int, use_complementary: bool = False
) -> np.ndarray:
    x = trim_peak(x, -17.0)
    pb = STEM_PROCESSORS[name]()
    y = pb_mono(pb, x, sr)
    if use_complementary:
        y = apply_complementary(name, y, sr)
    y = normalize_i(y, sr, TARGET_I[name])
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix_dir", nargs="?", default="/Users/dmitrijvolkov/Desktop/MIX")
    ap.add_argument("-o", "--output", default="/Users/dmitrijvolkov/Desktop/MIX_ProStudio.wav")
    ap.add_argument("--lufs", type=float, default=-10.5, help="Целевой integrated LUFS мастера")
    ap.add_argument("--tp", type=float, default=-1.0, help="True peak ceiling dBTP")
    ap.add_argument(
        "--mirror-eq",
        action="store_true",
        help="Complementary / зеркальные вырезы под вокал, kick, снейр (см. docstring)",
    )
    ap.add_argument(
        "--no-mastering",
        action="store_true",
        help="Без мастер-шины: не shelf/комп/лимитер на сумме, без нормализации по LUFS и TP; "
        "только масштаб по пику (см. --peak-headroom-db).",
    )
    ap.add_argument(
        "--peak-headroom-db",
        type=float,
        default=-3.0,
        help="При --no-mastering: целевой sample peak на LR (dBFS), например -3",
    )
    args = ap.parse_args()

    mix_dir = os.path.abspath(args.mix_dir)
    names = sorted(STEM_PROCESSORS.keys())
    for n in names:
        if n not in os.listdir(mix_dir):
            print("Отсутствует:", n, file=sys.stderr)
            return 1

    path0 = os.path.join(mix_dir, names[0])
    _, sr = sf.read(path0)
    n_samp = None
    drum_lr = None
    rest_lr = None

    vox_sum = None
    sat_pb = chain_vocal_sat()
    vox_pb = chain_vocal()

    for name in names:
        x, r = sf.read(os.path.join(mix_dir, name), dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if r != sr or (n_samp is not None and len(x) != n_samp):
            print("Несовпадение формата", file=sys.stderr)
            return 1
        if n_samp is None:
            n_samp = len(x)
            drum_lr = np.zeros((n_samp, 2), dtype=np.float64)
            rest_lr = np.zeros((n_samp, 2), dtype=np.float64)
            vox_sum = np.zeros(n_samp, dtype=np.float64)

        if name in VOC_SET:
            y = trim_peak(x, -17.0)
            y = pb_mono(vox_pb, y, sr)
            ys = pb_mono(sat_pb, trim_peak(x, -17.0), sr)
            y = 0.88 * y + 0.12 * ys
            if args.mirror_eq:
                y = apply_complementary(name, y, sr)
            y = normalize_i(y, sr, TARGET_I[name])
        else:
            y = process_generic(name, x, sr, use_complementary=args.mirror_eq)

        y = y.astype(np.float64) * GAIN[name]
        l, r = pan_constant_power(y, PAN[name])

        if name in DRUM_SET:
            drum_lr[:, 0] += l
            drum_lr[:, 1] += r
        else:
            rest_lr[:, 0] += l
            rest_lr[:, 1] += r

        if name in VOC_SET:
            vox_sum += y

    # Параллельное сжатие ударных (стерео)
    crush = chain_drums_parallel_crush()
    squashed = stereo_process(crush, drum_lr.astype(np.float32), sr).astype(np.float64)
    drums_out = 0.72 * drum_lr + 0.28 * squashed

    # Реверб с посыла вокала (mono wet → в оба канала)
    vox_sum = np.clip(vox_sum, -1.0, 1.0)
    send = vox_sum * 0.38
    revb = Pedalboard(
        [
            Reverb(
                room_size=0.58,
                damping=0.45,
                wet_level=1.0,
                dry_level=0.0,
                width=0.85,
            )
        ]
    )
    wet = pb_mono(revb, send.astype(np.float32), sr).astype(np.float64)
    wet_lr = np.column_stack([wet * 0.52, wet * 0.52])

    mix = drums_out + rest_lr + wet_lr

    if args.no_mastering:
        mix = np.asarray(mix, dtype=np.float64)
        peak = float(np.max(np.abs(mix))) + 1e-12
        target = 10.0 ** (args.peak_headroom_db / 20.0)
        mix = mix * (target / peak)
    else:
        # Мастер (Pedalboard: вход shape (ch, samples))
        mm = mix.astype(np.float32)
        if mm.ndim == 2 and mm.shape[1] == 2:
            mm = mm.T
        mix = chain_master()(mm, sr)
        mix = np.asarray(mix, dtype=np.float64)
        if mix.ndim == 2 and mix.shape[0] == 2:
            mix = mix.T
        if mix.ndim == 1:
            mix = np.column_stack([mix, mix])

        mix = normalize_i(mix, sr, args.lufs)
        for _ in range(12):
            tp = measure_tp_dbtp(mix, sr)
            if tp <= args.tp:
                break
            mix = mix * (10.0 ** (-(tp - args.tp + 0.06) / 20.0))

    out = mix.astype(np.float32)
    sf.write(os.path.abspath(args.output), out, sr, subtype="PCM_24")

    m = pyln.Meter(sr)
    print("Файл:", os.path.abspath(args.output))
    if args.no_mastering:
        print(
            "Режим без мастеринга: sample peak ≈",
            f"{args.peak_headroom_db:.1f}",
            "dBFS; I ≈",
            round(m.integrated_loudness(out), 2),
            "LUFS; TP ≈",
            round(measure_tp_dbtp(out, sr), 2),
            "dBTP",
        )
    else:
        print(
            "Итог: I ≈",
            round(m.integrated_loudness(out), 2),
            "LUFS; TP ≈",
            round(measure_tp_dbtp(out, sr), 2),
            "dBTP",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
