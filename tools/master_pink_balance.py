#!/usr/bin/env python3
"""Балансировка общего микса по розовому шуму (директива оператора 2026-07-06).

Замкнутый контур: мастер-петля Main LR -> USB 47/48 -> третьоктавный
спектр -> сравнение с розовой кривой (равная энергия на октаву) ->
решатель pa_v3 (prefer-cut, кламп −6/+3, ≤6 фильтров) -> EQ мейна
`/main/1/eq` с readback -> контрольный замер после.

SETUP-фаза по явному запуску оператора; rollback-файл пишется до первой
записи. Цифровая петля -> когерентность 1 (сетевые/акустические артефакты
исключены по построению).

Запуск:
    PYTHONPATH=backend python3 tools/master_pink_balance.py \
        --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
    PYTHONPATH=backend python3 tools/master_pink_balance.py --rollback <file>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import sounddevice as sd  # noqa: E402
import yaml  # noqa: E402

from heuristics.pa_v3 import solve_system_eq, target_curve_db  # noqa: E402
from wing_client import WingClient  # noqa: E402

LIVE_CONFIRM = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
REPORTS = REPO_ROOT / "reports"
SR = 48000
CAPTURE_S = 20.0
MAIN_EQ_BANDS = 6          # PEQ-полосы main LR


def capture_master(seconds: float = CAPTURE_S) -> np.ndarray:
    dev = next(i for i, d in enumerate(sd.query_devices())
               if d["name"] == "WING" and d["max_input_channels"] >= 48)
    a = sd.rec(int(seconds * SR), samplerate=SR, channels=48, device=dev,
               dtype="float32")
    sd.wait()
    return 0.5 * (a[:, 46].astype(np.float64) + a[:, 47].astype(np.float64))


def third_octave_spectrum(mono: np.ndarray):
    """Центры и медианные уровни третьоктавных полос 40 Гц - 16 кГц."""
    n = len(mono)
    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mag_db = 20.0 * np.log10(spec + 1e-12)
    edges = 40.0 * (2.0 ** np.arange(0, 8.7, 1.0 / 3.0))
    centers, levels = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            centers.append(float(np.sqrt(lo * hi)))
            levels.append(float(np.median(mag_db[m])))
    return np.asarray(centers), np.asarray(levels)


def pink_deviation(centers: np.ndarray, levels: np.ndarray) -> np.ndarray:
    target = target_curve_db(centers, "pink")
    return ((levels - np.median(levels))
            - (target - np.median(target)))


def main() -> None:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: master EQ writes are disabled"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-live-wing-write", default="")
    parser.add_argument("--rollback", default="")
    args = parser.parse_args()

    mixer = yaml.safe_load(open(REPO_ROOT / "config" / "automixer.yaml"))["mixer"]
    client = WingClient(ip=mixer["ip"], port=int(mixer.get("receive_port", 2223)))
    if not client.connect(timeout=5.0):
        raise SystemExit("пульт не отвечает")
    client.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(0.3)

    def rb(addr, timeout=0.9):
        r = client.query_parameter(addr, timeout=timeout)
        if r["status"] != "received":
            r = client.query_parameter(addr, timeout=timeout)
        return r["value"] if r["status"] == "received" else None

    def wr(addr, value, settle=0.15):
        client.send(addr, value)
        time.sleep(settle)
        got = rb(addr)
        if got is None or (isinstance(value, (int, float))
                           and isinstance(got, (int, float))
                           and abs(got - value) > 0.5):
            time.sleep(0.25)
            client.send(addr, value)
            time.sleep(settle)
            got = rb(addr)
        return got

    try:
        if args.rollback:
            snap = json.loads(Path(args.rollback).read_text())["original"]
            for addr, val in snap.items():
                if val is not None:
                    wr(addr, val)
            print("Откат main-EQ завершён")
            return
        if args.confirm_live_wing_write != LIVE_CONFIRM:
            raise SystemExit(f"нужно --confirm-live-wing-write {LIVE_CONFIRM}")

        # мьют-вахта петли (действующее разрешение оператора)
        wr("/main/1/mute", 0)

        # снимок main-EQ до записи (полосы + шельфы)
        addrs = ["/main/1/eq/on"]
        for b in range(1, MAIN_EQ_BANDS + 1):
            addrs += [f"/main/1/eq/{b}g", f"/main/1/eq/{b}f", f"/main/1/eq/{b}q"]
        addrs += [f"/main/1/eq/{t}" for t in ("lf", "lg", "hf", "hg")]
        original = {a: rb(a) for a in addrs}
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rollback_path = REPORTS / f"master_pink_balance_rollback_{stamp}.json"
        rollback_path.write_text(json.dumps({"original": original},
                                            ensure_ascii=False, indent=1))
        print(f"Rollback-файл: {rollback_path.name}", flush=True)

        print(f"Замер мастера {CAPTURE_S:.0f} с...", flush=True)
        mono = capture_master()
        if float(np.sqrt(np.mean(mono ** 2))) < 1e-4:
            raise SystemExit("мастер-петля молчит (mute/роутинг) — стоп")
        centers, levels = third_octave_spectrum(mono)
        dev_before = pink_deviation(centers, levels)
        print(f"Отклонение от розового ДО: медиана |dev| "
              f"{np.median(np.abs(dev_before)):.1f} дБ")

        journal = {"stamp": stamp, "filters": [], "before": {
            "median_abs_dev_db": float(np.median(np.abs(dev_before)))}}

        # СТУПЕНЬ 1: наклон лечится ШЕЛЬФАМИ (колокола наклон не меняют —
        # live-урок 2026-07-06: 6 колоколов по −6 не сдвинули медиану)
        low_m = centers < 250.0
        high_m = centers > 4000.0
        dev_low = float(np.mean(dev_before[low_m])) if low_m.any() else 0.0
        dev_high = float(np.mean(dev_before[high_m])) if high_m.any() else 0.0
        lg = float(np.clip(-dev_low, -6.0, 3.0))
        hg = float(np.clip(-dev_high, -3.0, 3.0))
        journal["shelves"] = {
            "lf": wr("/main/1/eq/lf", 180.0), "lg": wr("/main/1/eq/lg", round(lg, 1)),
            "hf": wr("/main/1/eq/hf", 5000.0), "hg": wr("/main/1/eq/hg", round(hg, 1)),
        }
        journal["eq_on"] = wr("/main/1/eq/on", 1)
        print(f"Шельфы: LF 180 Гц {lg:+.1f} дБ | HF 5 кГц {hg:+.1f} дБ")

        # СТУПЕНЬ 2: остаточные пики — колоколами по свежему замеру
        time.sleep(1.0)
        mono_mid = capture_master(12.0)
        c_mid, l_mid = third_octave_spectrum(mono_mid)
        result = solve_system_eq(
            c_mid, l_mid, target_curve_db(c_mid, "pink"),
            coherence=np.ones_like(c_mid),
            consistent=np.ones(len(c_mid), dtype=bool),
            max_filters=4,
        )
        for i, flt in enumerate(result["filters"], start=1):
            got_f = wr(f"/main/1/eq/{i}f", round(float(flt["freq_hz"]), 1))
            got_g = wr(f"/main/1/eq/{i}g", round(float(flt["gain_db"]), 1))
            got_q = wr(f"/main/1/eq/{i}q", round(float(flt["q"]), 2))
            journal["filters"].append(
                {"band": i, "freq": got_f, "gain": got_g, "q": got_q})
            print(f"  band{i}: {got_f} Гц {got_g:+.1f} дБ Q{got_q}")
        # неиспользованные полосы — в ноль
        for i in range(len(result["filters"]) + 1, MAIN_EQ_BANDS + 1):
            wr(f"/main/1/eq/{i}g", 0.0)

        time.sleep(1.0)
        mono2 = capture_master(12.0)
        c2, l2 = third_octave_spectrum(mono2)
        dev_after = pink_deviation(c2, l2)
        journal["after"] = {
            "median_abs_dev_db": float(np.median(np.abs(dev_after)))}
        print(f"Отклонение от розового ПОСЛЕ: медиана |dev| "
              f"{np.median(np.abs(dev_after)):.1f} дБ")
        out = REPORTS / f"master_pink_balance_applied_{stamp}.json"
        out.write_text(json.dumps(journal, ensure_ascii=False, indent=1,
                                  default=str))
        print(f"Журнал: {out.name}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
