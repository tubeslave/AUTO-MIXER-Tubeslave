#!/usr/bin/env python3
"""
Офлайн-сведение в духе канальной цепочки Behringer WING (см. wing_addresses).

Правила, которые учитываются (без пакета backend/):
  - presets/method_presets_eq.json, method_presets_compressor.json, method_presets_gain.json
  - .cursorrules / CLAUDE.md: безопасность уровня на экспорте (потолок по пику / true peak)
  - docs/CONVENTIONS.md: ориентир true peak с передискретизацией (минимум 4×), здесь 4× через scipy

Явно не используется: импорты из backend/ (server, handlers, wing_client, bleed, RAG из
backend/ai и т.д.) — логика живого контура и OSC не подмешивается.

Папка MIX: все *.wav; пары «… L.wav» / «… R.wav» — один инструмент, L влево, R вправо
(constant power). Опечатка «Playbacks R.wav» рядом с «Playback L.wav» учитывается.

Сумма: подгруппа ударных (parallel crush) + остальное, нормализация с учётом true peak
(если доступен scipy), иначе по sample peak.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, NoiseGate, Pedalboard, PeakFilter

_ROOT = Path(__file__).resolve().parents[1]

# Опорный пик из method_presets_gain (категория instruments) для относительного fader
_GAIN_REF_PEAK_DBFS = -12.0

DRUM_INST_IDS = frozenset({"kick", "snare", "tom", "hi_hat", "ride", "overhead"})


def load_json(name: str) -> Dict[str, Any]:
    p = _ROOT / "presets" / name
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_inst(data: Dict[str, Any], inst_id: str) -> Optional[Dict[str, Any]]:
    for row in data.get("instruments", []):
        if row.get("id") == inst_id:
            return row
    return None


def build_channel_strip(eq_data: Dict[str, Any], comp_data: Dict[str, Any], inst_id: str) -> Pedalboard:
    """HPF (lcf) + PEQ из пресета + компрессор base task."""
    row = find_inst(eq_data, inst_id)
    if row is None:
        row = find_inst(eq_data, "custom")
    assert row is not None
    p = row["params"]
    comps = find_inst(comp_data, inst_id) or find_inst(comp_data, "custom")
    assert comps is not None
    t = comps["params"]["tasks"]["base"]

    chain: List[Any] = []
    lcf = float(p.get("low_cut_freq", 80))
    chain.append(HighpassFilter(cutoff_frequency_hz=min(2000.0, max(20.0, lcf))))

    for item in p.get("cut_frequencies", [])[:4]:
        f, g, q = float(item[0]), float(item[1]), float(item[2])
        chain.append(
            PeakFilter(
                cutoff_frequency_hz=f,
                gain_db=max(-15.0, min(15.0, g)),
                q=max(0.44, min(10.0, q)),
            )
        )
    for item in p.get("boost_frequencies", [])[:4]:
        f, g, q = float(item[0]), float(item[1]), float(item[2])
        chain.append(
            PeakFilter(
                cutoff_frequency_hz=f,
                gain_db=max(-15.0, min(15.0, g)),
                q=max(0.44, min(10.0, q)),
            )
        )

    chain.append(
        Compressor(
            threshold_db=float(t["threshold"]),
            ratio=float(t["ratio"]),
            attack_ms=float(t["attack_ms"]),
            release_ms=float(t["release_ms"]),
        )
    )
    return Pedalboard(chain)


def gate_tom_snare(inst_id: str) -> Optional[NoiseGate]:
    if inst_id in ("tom", "snare"):
        return NoiseGate(threshold_db=-36.0, ratio=10.0, attack_ms=0.5, release_ms=100.0)
    return None


def pb_mono(board: Pedalboard, x: np.ndarray, sr: int) -> np.ndarray:
    y = board(x.astype(np.float32), sr)
    return np.asarray(y, dtype=np.float64).reshape(-1)


def trim_peak(x: np.ndarray, dbfs: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) + 1e-12
    return x * (10.0 ** (dbfs / 20.0) / p)


def pan_wing(pan_cent: float, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pan -100..100 → constant power."""
    p = float(np.clip(pan_cent, -100.0, 100.0)) / 100.0
    ang = (p + 1.0) * (np.pi / 4.0)
    return m * np.cos(ang), m * np.sin(ang)


def fader_db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def filename_to_inst_id(filename: str) -> str:
    """Сопоставление имени файла с id пресета (порядок правил важен)."""
    n = os.path.basename(filename).lower()
    if n.endswith(".wav"):
        n = n[:-4]
    # «bass» до «guitar», «snare» до «tom» (избегаем false positive в «bottom»)
    if "bass" in n:
        return "bass"
    if "playback" in n or "playbacks" in n:
        return "playback"
    if "overhead" in n:
        return "overhead"
    if "accordion" in n:
        return "accordion"
    if "kick" in n:
        return "kick"
    if "snare" in n or "snate" in n:
        return "snare"
    if "tom" in n:
        return "tom"
    if "vox" in n or "vocal" in n or "voice" in n:
        return "lead_vocal"
    if "guitar" in n:
        return "electric_guitar"
    return "custom"


def find_right_channel(left_file: str, wavs: set[str]) -> Optional[str]:
    """Для «Name L.wav» ищем «Name R.wav» или «Names R.wav» (опечатка)."""
    if not left_file.endswith(" L.wav"):
        return None
    base = left_file[: -len(" L.wav")]
    for candidate in (f"{base} R.wav", f"{base}s R.wav"):
        if candidate in wavs:
            return candidate
    return None


def build_track_plan(wavs: set[str]) -> List[Tuple[str, str, float]]:
    """
    Список (имя_файла, inst_id, pan).
    Стереопары обрабатываются как два канала одного inst_id.
    """
    plan: List[Tuple[str, str, float]] = []
    consumed: set[str] = set()

    for left in sorted(wavs):
        if left in consumed:
            continue
        if not left.endswith(" L.wav"):
            continue
        right = find_right_channel(left, wavs)
        if right is None:
            continue
        inst_id = filename_to_inst_id(left)
        plan.append((left, inst_id, -100.0))
        plan.append((right, inst_id, 100.0))
        consumed.add(left)
        consumed.add(right)

    for w in sorted(wavs):
        if w in consumed:
            continue
        if w.endswith(" R.wav"):
            # Потенциально осиротевший R — не дублируем, если есть пара слева
            base_guess = w.replace(" R.wav", "")
            if f"{base_guess} L.wav" in wavs:
                continue
            if base_guess.endswith("s") and f"{base_guess[:-1]} L.wav" in wavs:
                continue
        plan.append((w, filename_to_inst_id(w), 0.0))
        consumed.add(w)

    return plan


def true_peak_max_linear(mix_lr: np.ndarray, sr: int, up: int = 4) -> float:
    """Макс. |сигнал| после 4× передискретизации (оценка true peak), на канал."""
    try:
        from scipy import signal
    except ImportError:
        return float(np.max(np.abs(mix_lr)))
    peak = 0.0
    for c in range(mix_lr.shape[1]):
        up_sig = signal.resample_poly(mix_lr[:, c].astype(np.float64), up, 1)
        peak = max(peak, float(np.max(np.abs(up_sig))))
    return peak + 1e-15


def normalize_to_true_peak(mix_lr: np.ndarray, sr: int, target_dbfs: float) -> np.ndarray:
    """Масштабировать так, чтобы оценка true peak (4×) была ≤ target_dbfs."""
    p = true_peak_max_linear(mix_lr, sr)
    tgt = 10.0 ** (target_dbfs / 20.0)
    return mix_lr * (tgt / p)


def fader_db_from_gain_json(gain_data: Dict[str, Any], inst_id: str) -> float:
    row = find_inst(gain_data, inst_id) or find_inst(gain_data, "custom")
    assert row is not None
    peak = float(row["params"]["peak_dbfs"])
    return peak - _GAIN_REF_PEAK_DBFS


def process_stem(
    x: np.ndarray,
    sr: int,
    inst_id: str,
    eq_data: Dict[str, Any],
    comp_data: Dict[str, Any],
) -> np.ndarray:
    x = trim_peak(x, -16.0)
    strip = build_channel_strip(eq_data, comp_data, inst_id)
    y = pb_mono(strip, x, sr)
    g = gate_tom_snare(inst_id)
    if g is not None:
        y = pb_mono(Pedalboard([g]), y.astype(np.float32), sr)
    return y


def remix_literature_presets(
    mix_dir: str,
    eq_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    gain_data: Dict[str, Any],
    peak_dbfs: float,
    output: str,
    use_true_peak: bool,
) -> int:
    wavs = {
        f
        for f in os.listdir(mix_dir)
        if f.lower().endswith(".wav") and not f.startswith(".")
    }
    if not wavs:
        print("В папке нет .wav файлов", file=sys.stderr)
        return 1

    plan = build_track_plan(wavs)
    names = [t[0] for t in plan]

    info0 = sf.info(os.path.join(mix_dir, names[0]))
    sr = info0.samplerate
    n_samp: Optional[int] = None
    for name in names:
        x, r = sf.read(os.path.join(mix_dir, name), dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if r != sr:
            print(f"SR mismatch: {name}", file=sys.stderr)
            return 1
        if n_samp is None:
            n_samp = len(x)
        elif len(x) != n_samp:
            print(f"Length mismatch: {name}", file=sys.stderr)
            return 1
    assert n_samp is not None

    drum_lr = np.zeros((n_samp, 2), dtype=np.float64)
    rest_lr = np.zeros((n_samp, 2), dtype=np.float64)

    print("═══ План сведения (только пресеты) ═══", file=sys.stderr)
    for name, inst_id, pan in plan:
        fader_db = fader_db_from_gain_json(gain_data, inst_id)
        print(f"  {name:28s}  id={inst_id:16s}  pan={pan:+.0f}  fader={fader_db:+.1f} dB", file=sys.stderr)

    for name, inst_id, pan in plan:
        x, r = sf.read(os.path.join(mix_dir, name), dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x[:n_samp]
        y = process_stem(x, sr, inst_id, eq_data, comp_data)
        fader_db = fader_db_from_gain_json(gain_data, inst_id)
        y = y * fader_db_to_lin(fader_db)
        l_ch, r_ch = pan_wing(pan, y)
        if inst_id in DRUM_INST_IDS:
            drum_lr[:, 0] += l_ch
            drum_lr[:, 1] += r_ch
        else:
            rest_lr[:, 0] += l_ch
            rest_lr[:, 1] += r_ch

    crush = Pedalboard(
        [Compressor(threshold_db=-20.0, ratio=5.5, attack_ms=4.0, release_ms=90.0)]
    )
    d_st = drum_lr.astype(np.float32).T
    d_sq = crush(d_st, sr)
    d_sq = np.asarray(d_sq, dtype=np.float64).T
    drums_out = 0.74 * drum_lr + 0.26 * d_sq

    mix = drums_out + rest_lr
    if use_true_peak:
        try:
            import scipy.signal  # noqa: F401
        except ImportError:
            print("scipy недоступен — нормализация по sample peak", file=sys.stderr)
            use_true_peak = False
    if use_true_peak:
        mix = normalize_to_true_peak(mix, sr, peak_dbfs)
        tp_db = 20.0 * np.log10(true_peak_max_linear(mix, sr))
        print(f"  True peak (4×): {tp_db:.2f} dBFS (цель {peak_dbfs:.1f})", file=sys.stderr)
    else:
        peak = float(np.max(np.abs(mix))) + 1e-12
        tgt = 10.0 ** (peak_dbfs / 20.0)
        mix = mix * (tgt / peak)

    out_path = os.path.abspath(output)
    sf.write(out_path, mix.astype(np.float32), sr, subtype="PCM_24")
    print("Записано:", out_path, file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="WING-like offline mix: только method_presets EQ/comp/gain, авто-стемы"
    )
    ap.add_argument("mix_dir", nargs="?", default="/Users/dmitrijvolkov/Desktop/MIX")
    ap.add_argument(
        "-o",
        "--output",
        default="/Users/dmitrijvolkov/Desktop/MIX_WING_NoBackend.wav",
    )
    ap.add_argument(
        "--peak-dbfs",
        type=float,
        default=-1.0,
        help="Целевой пик (true peak при --true-peak, иначе sample peak), dBFS",
    )
    ap.add_argument(
        "--true-peak",
        action="store_true",
        default=True,
        help="Нормализовать по оценке true peak 4× (scipy); по умолчанию включено",
    )
    ap.add_argument(
        "--no-true-peak",
        action="store_false",
        dest="true_peak",
        help="Только sample peak",
    )
    args = ap.parse_args()

    mix_dir = os.path.abspath(args.mix_dir)
    eq_data = load_json("method_presets_eq.json")
    comp_data = load_json("method_presets_compressor.json")
    gain_data = load_json("method_presets_gain.json")

    return remix_literature_presets(
        mix_dir,
        eq_data,
        comp_data,
        gain_data,
        args.peak_dbfs,
        args.output,
        args.true_peak,
    )


if __name__ == "__main__":
    raise SystemExit(main())
