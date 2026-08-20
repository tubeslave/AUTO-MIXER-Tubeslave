#!/usr/bin/env python3
"""Саундчек v2 — стадия 3: FX и фаза/задержки с onset-gated GCC-PHAT.

1. FX (AutoFXPlanner, backend/auto_fx.py): 4 шины эффектов из шаблона
   планировщика — Vocal Plate (FX1/bus13), Drum Room (FX2/bus14),
   Tempo Delay (FX3/bus15, время от оценённого темпа), Mod Doubler
   (FX4/bus16) + пост-фейдерные посылы по инструментам. Планировщик
   закарантинен на прямую запись — этот скрипт реализует требуемый
   supervised-контракт: лимиты плана (посылы -36..-12 dB, возвраты
   <= -3 dB, только post-fader), физический readback каждой записи,
   rollback JSON. Темп оценивается по onset-интервалам бочки.

2. ФАЗА/ЗАДЕРЖКИ: onset-gated GCC-PHAT — вместо замера по всей записи
   (bleed рушит когерентность) окна берутся только вокруг УДАРОВ
   близкого микрофона, где его SNR максимален. Медиана лага по окнам +
   требование согласованности (MAD). Задержка близких микрофонов
   барабанов к OH (0.5..15 мс), полярность SNARE B по знаку корреляции
   в onset-окнах SNARE T.

Настройки остаются на пульте. Запуск:
    python3 tools/live_wing_soundcheck_v2_stage3.py --dry-run
    python3 tools/live_wing_soundcheck_v2_stage3.py \
        --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import yaml  # noqa: E402
import sounddevice as sd  # noqa: E402

from wing_client import WingClient  # noqa: E402
from lufs_gain_staging import LUFSMeter  # noqa: E402
from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402
from auto_fx import AutoFXPlanner  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
NUM_CH, SR, BLOCK = 40, 48000, 4800
CAPTURE_S = 12.0
ACTIVE_ST_LUFS = -45.0
READBACK_TIMEOUT = 1.2
GCC_WIN = 8192                      # окно GCC-PHAT (~170 мс)
ONSET_PRE_S = 0.010
DELAY_MIN_MS, DELAY_MAX_MS = 0.5, 15.0
SEND_MIN_DB, SEND_MAX_DB = -36.0, -12.0
RETURN_MAX_DB = -3.0

EXCLUDE_NAME_RE = re.compile(r"master|mstr|\bref\b|refer|talk|\btb\b|mtx|matrix|\bbus\b|\bmon\b", re.I)
NAME_ROLE = [
    (re.compile(r"kick|\bbd\b|бочк", re.I), "kick"),
    (re.compile(r"snare|\bsn\b|малый", re.I), "snare"),
    (re.compile(r"f\.?\s?tom|floor", re.I), "floor_tom"),
    (re.compile(r"\btom\b|том", re.I), "rack_tom"),
    (re.compile(r"\boh\b|over", re.I), "overhead"),
    (re.compile(r"hat|\bhh\b|ride|crash|cymb|тарел", re.I), "hi_hat"),
    (re.compile(r"bass|\bbs\b|бас", re.I), "bass"),
    (re.compile(r"g(ui)?t|guitar|гитар", re.I), "electric_guitar"),
    (re.compile(r"accord|аккорд|акорд", re.I), "accordion"),
    (re.compile(r"keys?|piano|synth|pad|клав|органн", re.I), "keys"),
    (re.compile(r"\bback\b|\bbv\b|back.*voc", re.I), "backing_vocal"),
    (re.compile(r"voc|vox|vcl|lead|sing|голос|вок", re.I), "lead_vocal"),
    (re.compile(r"play|track|\bpb\b|минус|фонограм", re.I), "playback"),
]


def role_for(name: str) -> str:
    for rx, role in NAME_ROLE:
        if name and rx.search(name):
            return role
    return "line"


def readback(client, addr):
    r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    if r["status"] != "received":
        r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    return r["value"] if r["status"] == "received" else None


def st_lufs_max(x: np.ndarray) -> float:
    lm = LUFSMeter(sample_rate=SR)
    vals = [float(lm.process(x[i:i + BLOCK].astype(np.float64)))
            for i in range(0, len(x) - BLOCK + 1, BLOCK)]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else -144.0


def detect_onsets(x: np.ndarray, min_gap_s: float = 0.15, max_onsets: int = 24) -> list[int]:
    """Индексы сэмплов ударов: скачок огибающей над локальной медианой."""
    hop = 480  # 10 мс
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    env_db = 20 * np.log10(env + 1e-10)
    med = np.median(env_db)
    onsets, last = [], -10 ** 9
    for i in range(2, len(env_db)):
        if env_db[i] > med + 12 and env_db[i] - env_db[i - 2] > 8:
            idx = i * hop
            if idx - last >= int(min_gap_s * SR):
                onsets.append(idx)
                last = idx
                if len(onsets) >= max_onsets:
                    break
    return onsets


BODY_BAND = {  # полоса «тела» инструмента для префильтрации GCC
    "kick": (50.0, 150.0), "snare": (150.0, 450.0),
    "rack_tom": (80.0, 300.0), "floor_tom": (60.0, 250.0),
}


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x)


def onset_gated_gcc(close: np.ndarray, oh: np.ndarray, band: tuple | None = None) -> dict:
    """GCC-PHAT только в окнах вокруг ударов близкого микрофона.

    band: полоса тела инструмента — оба сигнала фильтруются, чтобы bleed
    вне полосы не рушил когерентность (PHAT-взвешивание чувствительно к нему).
    """
    onsets = detect_onsets(close)
    if band:
        close_f = _bandpass(close.astype(np.float64), *band)
        oh_f = _bandpass(oh.astype(np.float64), *band)
    else:
        close_f, oh_f = close, oh
    lags, corrs = [], []
    an = GCCPHATAnalyzer(sample_rate=SR, fft_size=GCC_WIN, max_delay_ms=20.0)
    pre = int(ONSET_PRE_S * SR)
    for o in onsets:
        s = max(0, o - pre)
        if s + GCC_WIN > len(close_f):
            break
        cw = np.asarray(close_f[s:s + GCC_WIN], dtype=np.float64)
        ow = np.asarray(oh_f[s:s + GCC_WIN], dtype=np.float64)
        dm = an.compute_delay(ref_signal=cw, tgt_signal=ow)
        if dm.correlation_peak < 0.25:
            continue
        lags.append(float(dm.delay_ms))
        # полярность на найденном лаге
        lag_smp = int(round(dm.delay_samples))
        n = GCC_WIN - abs(lag_smp)
        if n > 1000:
            if lag_smp >= 0:
                a, b = cw[:n], ow[lag_smp:lag_smp + n]
            else:
                a, b = cw[-lag_smp:-lag_smp + n], ow[:n]
            denom = (np.std(a) * np.std(b)) or 1e-12
            corrs.append(float(np.mean((a - a.mean()) * (b - b.mean())) / denom))
    if len(lags) < 5:
        return {"ok": False, "n": len(lags), "onsets": len(onsets)}
    lags_a = np.array(lags)
    med = float(np.median(lags_a))
    mad = float(np.median(np.abs(lags_a - med)))
    return {"ok": mad <= 0.5, "n": len(lags), "onsets": len(onsets),
            "lag_ms": med, "mad_ms": mad,
            "corr": float(np.median(corrs)) if corrs else 0.0}


def estimate_tempo(kick: np.ndarray) -> float:
    onsets = detect_onsets(kick, min_gap_s=0.2, max_onsets=48)
    if len(onsets) < 4:
        return 120.0
    iois = np.diff(onsets) / SR
    iois = iois[(iois > 0.25) & (iois < 2.0)]
    if not len(iois):
        return 120.0
    beat = float(np.median(iois))
    bpm = 60.0 / beat
    while bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return round(bpm, 1)


def main() -> int:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: use the supervised control plane"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-live-wing-write", default="")
    args = ap.parse_args()
    if not args.dry_run and args.confirm_live_wing_write != LIVE_WING_CONFIRMATION:
        print(f"FAIL: нужен --confirm-live-wing-write {LIVE_WING_CONFIRMATION}", file=sys.stderr)
        return 2

    mixer = yaml.safe_load(open(REPO_ROOT / "config" / "automixer.yaml"))["mixer"]
    client = WingClient(ip=mixer["ip"], port=int(mixer.get("receive_port", 2223)))
    print(f"Подключение к WING {mixer['ip']}...")
    if not client.connect(timeout=5.0):
        print("FAIL: пульт не отвечает")
        return 2
    client.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(1.0)
    names = client.get_all_channel_names(num_channels=NUM_CH)

    devs = sd.query_devices()
    dev = next(i for i, d in enumerate(devs) if d["name"] == "WING" and d["max_input_channels"] >= NUM_CH)
    print(f"Захват {CAPTURE_S:.0f} с...")
    audio = sd.rec(int(CAPTURE_S * SR), samplerate=SR, channels=NUM_CH, device=dev, dtype="float32")
    sd.wait()

    active = {}
    for ch in range(1, NUM_CH + 1):
        name = (names.get(ch) or "").strip()
        if EXCLUDE_NAME_RE.search(name):
            continue
        x = audio[:, ch - 1]
        if st_lufs_max(x) < ACTIVE_ST_LUFS:
            continue
        active[ch] = {"name": name, "role": role_for(name), "x": x}
    print(f"Активные каналы: { {ch: c['role'] for ch, c in sorted(active.items())} }")

    # ===== 1. Фаза/задержки: onset-gated GCC-PHAT =====
    oh_ch = next((ch for ch, c in sorted(active.items()) if c["role"] == "overhead"), None)
    snare_t = next((ch for ch, c in active.items() if c["name"].upper().startswith("SNARE T")), None)
    snare_b = next((ch for ch, c in active.items() if c["name"].upper().startswith("SNARE B")), None)
    phase_actions = []   # (ch, invert|None, delay_ms|None, why)

    if snare_t and snare_b:
        r = onset_gated_gcc(audio[:, snare_t - 1], audio[:, snare_b - 1], band=BODY_BAND["snare"])
        if r.get("ok"):
            inv = 1 if r["corr"] < -0.2 else None
            why = (f"SNARE B vs T: {r['n']} ударов, лаг {r['lag_ms']:.2f}±{r['mad_ms']:.2f} мс, "
                   f"corr {r['corr']:+.2f} -> {'ИНВЕРСИЯ' if inv else 'фаза согласована'}")
            phase_actions.append((snare_b, inv, None, why))
        else:
            phase_actions.append((snare_b, None, None,
                                  f"SNARE B vs T: недостаточно согласованных ударов ({r})"))
    if oh_ch:
        for ch, c in sorted(active.items()):
            if c["role"] not in ("kick", "snare", "rack_tom", "floor_tom") or ch == snare_b:
                continue
            r = onset_gated_gcc(audio[:, ch - 1], audio[:, oh_ch - 1], band=BODY_BAND.get(c["role"]))
            if not r.get("ok"):
                phase_actions.append((ch, None, None, f"{c['name']} -> OH: несогласованно ({r})"))
                continue
            lag = r["lag_ms"]
            inv = 1 if r["corr"] < -0.25 else None
            dly = round(lag, 2) if DELAY_MIN_MS <= lag <= DELAY_MAX_MS else None
            why = (f"{c['name']} -> OH: {r['n']} ударов, лаг {lag:.2f}±{r['mad_ms']:.2f} мс, "
                   f"corr {r['corr']:+.2f} -> "
                   f"{'delay ' + str(dly) + ' мс' if dly else 'без задержки'}"
                   f"{', ИНВЕРСИЯ' if inv else ''}")
            phase_actions.append((ch, inv, dly, why))

    print("\n--- Фаза/задержки (onset-gated GCC-PHAT) ---")
    for ch, inv, dly, why in phase_actions:
        print(f"  ch{ch:2d}: {why}")

    # ===== 2. FX план =====
    kick_ch = next((ch for ch, c in active.items() if c["role"] == "kick"), None)
    bpm = estimate_tempo(audio[:, kick_ch - 1]) if kick_ch else 120.0
    print(f"\nОценённый темп: {bpm} BPM")
    planner = AutoFXPlanner(tempo_bpm=bpm, vocal_priority=True)
    plan = planner.create_plan({ch: c["role"] for ch, c in active.items()})
    calls = planner._build_apply_calls(plan)
    # лимиты из контракта планировщика
    for s in plan.sends:
        assert SEND_MIN_DB <= s.send_db <= SEND_MAX_DB and s.post_fader, f"send вне лимитов: {s}"
    ret_levels = {b.bus_id: min(b.return_level_db, RETURN_MAX_DB) for b in plan.buses}

    print("--- FX шины ---")
    for b in plan.buses:
        print(f"  bus{b.bus_id} {b.name}: {b.fx_slot}={b.model}, return {ret_levels[b.bus_id]:.1f} dB, {b.params}")
    print("--- FX посылы ---")
    for s in plan.sends:
        print(f"  ch{s.channel_id:2d} {active[s.channel_id]['name']:12s} -> bus{s.bus_id} {s.send_db:.0f} dB ({s.reason})")

    if args.dry_run:
        print("\nDRY-RUN: записи на пульт не было.")
        client.disconnect()
        return 0

    # ===== Откат =====
    touched = {}
    addrs = []
    for ch, inv, dly, _ in phase_actions:
        addrs += [f"/ch/{ch}/in/set/inv", f"/ch/{ch}/in/set/dly", f"/ch/{ch}/in/set/dlyon",
                  f"/ch/{ch}/in/set/dlymode"]
    for b in plan.buses:
        n = b.fx_slot.replace("FX", "")
        addrs += [f"/fx/{n}/mdl", f"/fx/{n}/fxmix"] + [f"/fx/{n}/{i}" for i in range(1, 7)]
        addrs.append(f"/bus/{b.bus_id}/fdr")
    for s in plan.sends:
        addrs += [f"/ch/{s.channel_id}/send/{s.bus_id}/lvl", f"/ch/{s.channel_id}/send/{s.bus_id}/on",
                  f"/ch/{s.channel_id}/send/{s.bus_id}/mode"]
    for a in addrs:
        touched[a] = readback(client, a)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rollback = reports / f"live_wing_soundcheck_v2s3_original_{stamp}.json"
    rollback.write_text(json.dumps({"mixer_ip": mixer["ip"], "saved_at": stamp, "original": touched},
                                   ensure_ascii=False, indent=2))
    print(f"\nОткат: {rollback}")

    fails = []

    def wr(addr, setter, value, tol=0.25, rel=False):
        setter(value)
        time.sleep(0.07)
        act = readback(client, addr)
        ok = act is not None and (
            (isinstance(value, str) and str(act).strip().upper() == str(value).upper())
            or (not isinstance(value, str)
                and abs(float(act) - float(value)) <= (max(tol, 0.06 * abs(float(value))) if rel else tol)))
        if not ok:
            fails.append((addr, value, act))
        return ok

    # фаза/задержки
    for ch, inv, dly, _ in phase_actions:
        if inv is not None:
            wr(f"/ch/{ch}/in/set/inv", lambda v: client.set_channel_phase_invert(ch, v), 1, tol=0.5)
        if dly is not None:
            client.send(f"/ch/{ch}/in/set/dlymode", "MS")
            time.sleep(0.05)
            wr(f"/ch/{ch}/in/set/dly", lambda v: client.send(f"/ch/{ch}/in/set/dly", v), float(dly), tol=0.1)
            wr(f"/ch/{ch}/in/set/dlyon", lambda v: client.send(f"/ch/{ch}/in/set/dlyon", v), 1, tol=0.5)

    # FX шины
    FX_MODEL_FIX = {"STEREO_DELAY": "ST-DL"}  # официальные имена моделей WING FW 3.1
    for b in plan.buses:
        n = b.fx_slot.replace("FX", "")
        model = FX_MODEL_FIX.get(b.model, b.model)
        wr(f"/fx/{n}/mdl", lambda v: client.set_fx_model(b.fx_slot, v), model)
        # адреса /fx/N/on в протоколе WING нет: слот активен при загруженной модели
        client.send(f"/fx/{n}/fxmix", 100.0)
        # маршрутизация: шина должна иметь insert FX-слота, иначе посылы беззвучны
        wr(f"/bus/{b.bus_id}/preins/ins", lambda v: client.send(f"/bus/{b.bus_id}/preins/ins", v), b.fx_slot)
        wr(f"/bus/{b.bus_id}/preins/on", lambda v: client.send(f"/bus/{b.bus_id}/preins/on", v), 1, tol=0.5)
        for param, value in planner._wing_parameter_map(b).items():
            client.set_fx_parameter(b.fx_slot, param, float(value))
            time.sleep(0.03)
        wr(f"/bus/{b.bus_id}/fdr", lambda v: client.set_bus_fader(b.bus_id, v), float(ret_levels[b.bus_id]))
    # посылы
    for s in plan.sends:
        wr(f"/ch/{s.channel_id}/send/{s.bus_id}/lvl",
           lambda v: client.set_channel_send(s.channel_id, s.bus_id, level=v), float(s.send_db), tol=0.3)
        wr(f"/ch/{s.channel_id}/send/{s.bus_id}/on",
           lambda v: client.set_channel_send(s.channel_id, s.bus_id, on=int(v)), 1, tol=0.5)
        client.set_channel_send(s.channel_id, s.bus_id, mode="POST")
        time.sleep(0.03)

    applied = reports / f"live_wing_soundcheck_v2s3_applied_{stamp}.json"
    applied.write_text(json.dumps(
        {"applied_at": stamp, "tempo_bpm": bpm, "fx_plan": plan.to_dict(),
         "phase_actions": [(ch, inv, dly, why) for ch, inv, dly, why in phase_actions],
         "return_levels": ret_levels},
        ensure_ascii=False, indent=2, default=float))
    client.disconnect()

    print("\n================ ИТОГ (стадия 3) ================")
    if fails:
        print(f"Не подтвердились readback'ом ({len(fails)}):")
        for a, v, act in fails[:12]:
            print(f"  {a}: послано {v!r}, readback {act!r}")
    else:
        print("Все записи подтверждены readback'ом.")
    print(f"Настройки ОСТАВЛЕНЫ. План: {applied}\nОткат: {rollback}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
