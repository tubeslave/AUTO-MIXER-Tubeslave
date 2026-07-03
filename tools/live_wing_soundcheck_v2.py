#!/usr/bin/env python3
"""Саундчек v2 — живая настройка WING на НОВОМ пайплайне консолидированного проекта.

Отличия от устаревшего live_wing_soundcheck_setup:
- Измерения: LUFSMeter (ITU-R BS.1770, K-weighting) и TruePeakMeter (4x
  oversampling) из lufs_gain_staging — вместо наивного RMS/пика.
- Классификация: channel_classifier_heuristic (MIR-признаки: центроид,
  rolloff, crest, ZCR, flatness, harmonicity) + имя канала как приор.
- Спектр: SpectralCeilingEQAnalyzer — главный спектральный инструмент
  нового пайплайна (CLAUDE.md); предложения мержатся в 4 консольные полосы
  через merge_spectral_proposal_into_eq_bands с защитой notch-полос.
- Решения: ActionPlan/ActionDecision (Decision Engine v2 models), каждое
  решение проходит SafetyGate v2 (evaluate_plan) с саундчек-лимитами.
- Компрессия: bounded, от измеренного crest factor (контроль crest, не
  пресеты), makeup ограничен по правилу CLAUDE.md.
- Trim: открытый контур от pre-trim замера (USB tap ДО трима — правило
  CLAUDE.md), boost ограничен.
- Фейдеры: плоскости Ayaic с сохранением относительного баланса, потолок 0 dBFS.

Применение — с физическим readback; исходное состояние в reports/ (rollback).
Настройки остаются на пульте.

Запуск:
    python3 tools/live_wing_soundcheck_v2.py --dry-run
    python3 tools/live_wing_soundcheck_v2.py \
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
from channel_classifier_heuristic import classify_audio  # noqa: E402
from lufs_gain_staging import LUFSMeter, TruePeakMeter  # noqa: E402
from heuristics.spectral_ceiling_eq import (  # noqa: E402
    SpectralCeilingEQAnalyzer,
    merge_spectral_proposal_into_eq_bands,
)
from automixer.decision.models import (  # noqa: E402
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ActionDecision,
    ActionPlan,
)
from automixer.safety.safety_gate import SafetyGate, SafetyGateConfig  # noqa: E402

LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"

NUM_CH = 40
SR = 48000
CAPTURE_S = 8.0
BLOCK = 4800                       # 100 мс для ST-LUFS
ACTIVE_ST_LUFS = -45.0             # активный канал: max short-term LUFS выше
TRIM_TARGET_DBTP = -18.0           # цель true peak после трима (открытый контур)
TRIM_BOOST_CAP = 6.0
FADER_FLOOR = -40.0
FADER_CEIL = 0.0
READBACK_TIMEOUT = 1.2

WING_RATIOS = ["1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0",
               "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"]
EXCLUDE_NAME_RE = re.compile(r"master|mstr|\bref\b|refer|talk|\btb\b|mtx|matrix|\bbus\b|\bmon\b", re.I)

# имя канала -> роль (приор над классификатором)
NAME_ROLE = [
    (re.compile(r"kick|\bbd\b|бочк", re.I), "kick"),
    (re.compile(r"snare|\bsn\b|малый", re.I), "snare"),
    (re.compile(r"f\.?\s?tom|floor", re.I), "floor_tom"),
    (re.compile(r"\btom\b|том", re.I), "tom"),
    (re.compile(r"\boh\b|over", re.I), "overheads"),
    (re.compile(r"hat|\bhh\b|ride|crash|cymb|тарел", re.I), "hihat"),
    (re.compile(r"bass|\bbs\b|бас", re.I), "bass"),
    (re.compile(r"g(ui)?t|guitar|гитар", re.I), "guitar"),
    (re.compile(r"keys?|piano|synth|pad|клав|органн", re.I), "keys"),
    (re.compile(r"\bbv\b|back.*voc", re.I), "backing_vocal"),
    (re.compile(r"voc|vox|vcl|lead|sing|голос|вок", re.I), "lead_vocal"),
    (re.compile(r"play|track|\bpb\b|минус|фонограм", re.I), "playback"),
    (re.compile(r"accord|аккорд|акорд", re.I), "keys"),
]

# классификатор -> роль spectral-ceiling
CLASSIFIER_TO_ROLE = {
    "kick": "kick", "snare": "snare", "tom": "tom", "hihat": "hihat",
    "cymbals": "overheads", "overheads": "overheads", "bass": "bass",
    "guitar": "guitar", "keys": "keys", "leadVocal": "lead_vocal",
    "backingVocal": "backing_vocal",
}

# Плоскости Ayaic (CLAUDE.md)
AYAIC_PLANE = {
    "lead_vocal": -22.0, "backing_vocal": -27.0,
    "kick": -25.0, "snare": -25.0, "tom": -28.0, "floor_tom": -28.0,
    "hihat": -28.0, "overheads": -28.0,
    "bass": -25.0, "guitar": -25.0, "keys": -30.0,
    "playback": -25.0, "line": -27.0,
}
HPF_BY_ROLE = {
    "lead_vocal": 100.0, "backing_vocal": 120.0, "kick": 30.0, "snare": 100.0,
    "tom": 60.0, "floor_tom": 45.0, "hihat": 300.0, "overheads": 150.0,
    "bass": 30.0, "guitar": 80.0, "keys": 40.0, "playback": 20.0, "line": 60.0,
}
GATE_ROLES = {"kick", "snare", "tom", "floor_tom"}

# Саундчек-лимиты SafetyGate v2 (не mid-show микрошаги: базовая настройка)
SOUNDCHECK_GATE_CONFIG = {
    "max_gain_change_db": 15.0,
    "max_live_gain_increase_db": TRIM_BOOST_CAP,
    "min_gain_change_db": 0.3,
    "max_eq_boost_db": 2.0,
    "max_eq_cut_db": 6.0,
    "max_eq_q": 4.0,
    "max_compression_ratio": 6.0,
    "max_compression_threshold_change_db": 30.0,
    "min_interval_sec": 0.0,
    "cooldown_sec": 0.0,
    "hysteresis_db": 0.25,
    "live_mode_blocks_abrupt_changes": True,
    "true_peak_ceiling_dbtp": -1.0,
}


def nearest_ratio(r: float) -> str:
    return min(WING_RATIOS, key=lambda s: abs(float(s) - r))


def readback(client: WingClient, addr: str):
    r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    if r["status"] != "received":
        r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    return r["value"] if r["status"] == "received" else None


def measure_channel(x: np.ndarray) -> dict:
    """Новый DSP: short-term LUFS серия (K-weighting) + true peak (4x)."""
    lufs_meter = LUFSMeter(sample_rate=SR)
    tp_meter = TruePeakMeter(sample_rate=SR)
    st: list[float] = []
    tp = -144.0
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        blk = x[i:i + BLOCK].astype(np.float64)
        st.append(float(lufs_meter.process(blk)))
        tp = max(tp, float(tp_meter.process(blk)))
    st_arr = np.array([v for v in st if np.isfinite(v)])
    st_max = float(np.max(st_arr)) if st_arr.size else -144.0
    gated = st_arr[st_arr > max(st_max - 20.0, -60.0)]
    est_lufs = float(np.mean(gated)) if gated.size else st_max
    crest = tp - est_lufs
    return {"st_max_lufs": st_max, "est_lufs": est_lufs, "true_peak_dbtp": tp, "crest_db": crest}


def role_for(name: str, audio: np.ndarray) -> tuple[str, float, str]:
    for rx, role in NAME_ROLE:
        if name and rx.search(name):
            return role, 1.0, f"имя {name!r}"
    res = classify_audio(audio.astype(np.float64), sample_rate=SR)
    role = CLASSIFIER_TO_ROLE.get(res.instrument, "line")
    return role, float(res.confidence), f"классификатор {res.instrument} ({res.confidence:.2f})"


def main() -> int:
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
    print(f"Захват {CAPTURE_S:.0f} с (pre-trim USB tap, 40 каналов)...")
    audio = sd.rec(int(CAPTURE_S * SR), samplerate=SR, channels=NUM_CH, device=dev, dtype="float32")
    sd.wait()

    spectral = SpectralCeilingEQAnalyzer()

    channels: list[dict] = []
    for ch in range(1, NUM_CH + 1):
        name = (names.get(ch) or "").strip()
        if EXCLUDE_NAME_RE.search(name):
            continue
        x = audio[:, ch - 1]
        meas = measure_channel(x)
        if meas["st_max_lufs"] < ACTIVE_ST_LUFS:
            continue
        role, conf, how = role_for(name, x)
        channels.append({"ch": ch, "name": name, "role": role, "conf": conf, "how": how,
                         "meas": meas, "audio": x})

    if not channels:
        print("Активных каналов не найдено.")
        client.disconnect()
        return 1

    print(f"\nАктивные каналы ({len(channels)}), измерения BS.1770:")
    for c in channels:
        m = c["meas"]
        print(f"  ch{c['ch']:2d} {c['name'] or '(без имени)':12s} -> {c['role']:13s} [{c['how']}] "
              f"ST-LUFS {m['est_lufs']:6.1f}, TP {m['true_peak_dbtp']:6.1f} dBTP, crest {m['crest_db']:4.1f}")

    # ===== Решения (Decision Engine v2 модели) =====
    decisions: list[ActionDecision] = []
    lead_active = any(c["role"] == "lead_vocal" for c in channels)

    for c in channels:
        ch, role, m = c["ch"], c["role"], c["meas"]
        target = f"ch{ch}"
        cur_trim = float(readback(client, f"/ch/{ch}/in/set/trim") or 0.0)
        c["cur_trim"] = cur_trim

        # 1) GAIN (trim, открытый контур от pre-trim TP)
        delta = min(TRIM_TARGET_DBTP - m["true_peak_dbtp"], TRIM_BOOST_CAP)
        new_trim = float(np.clip(cur_trim + delta, -18.0, 18.0))
        c["new_trim"] = new_trim
        decisions.append(ActionDecision(
            id=f"trim.ch{ch}", action_type=ACTION_GAIN, target=target,
            parameters={"gain_db": round(new_trim - cur_trim, 2), "absolute_trim_db": round(new_trim, 1)},
            reason=f"gain staging: TP {m['true_peak_dbtp']:.1f} dBTP -> цель {TRIM_TARGET_DBTP}",
            confidence=c["conf"], risk_level="low", safe_to_apply=True,
            source_modules=["lufs_gain_staging.TruePeakMeter"],
        ))

        # 2) EQ: spectral ceiling proposal -> консольные полосы (с защитой notch)
        prop = spectral.analyze(
            c["audio"].astype(np.float64), instrument_role=role, sample_rate=SR,
            track_id=f"ch{ch}", role_confidence=c["conf"],
            lead_vocal_active=lead_active, lead_vocal_confidence=1.0 if lead_active else 0.0,
        )
        cur_bands = []
        for b in ("1", "2", "3", "4"):
            f = readback(client, f"/ch/{ch}/eq/{b}f") or 1000.0
            g = readback(client, f"/ch/{ch}/eq/{b}g") or 0.0
            q = readback(client, f"/ch/{ch}/eq/{b}q") or 2.0
            cur_bands.append((float(f), float(g), float(q)))
        c["cur_bands"] = cur_bands
        merged, rep = merge_spectral_proposal_into_eq_bands(cur_bands, prop)
        eq_moves = []
        for idx, (orig, new) in enumerate(zip(cur_bands, merged)):
            if orig == new:
                continue
            if orig[1] <= -4.0:  # защита существующего notch (CLAUDE.md)
                continue
            eq_moves.append((idx + 1, new))
            decisions.append(ActionDecision(
                id=f"eq.ch{ch}.b{idx+1}", action_type=ACTION_EQ, target=target,
                parameters={"band": idx + 1, "frequency_hz": round(new[0], 1),
                            "gain_db": round(new[1] - orig[1], 2), "absolute_gain_db": round(new[1], 2),
                            "q": round(new[2], 2)},
                reason=f"spectral_ceiling_eq: {prop.target_profile}, tilt {prop.measured_tilt_db_per_oct:.1f}"
                       f"->{prop.target_tilt_db_per_oct:.1f} dB/oct",
                confidence=float(prop.confidence), risk_level="low", safe_to_apply=True,
                source_modules=["heuristics.spectral_ceiling_eq"],
            ))
        c["eq_moves"] = eq_moves
        c["hpf"] = prop.low_cut_hz or HPF_BY_ROLE.get(role, 60.0)

        # 3) Компрессия от crest factor (bounded)
        crest = m["crest_db"]
        if crest > 12.0:
            ratio = 4.0 if role in GATE_ROLES else (3.0 if crest > 18 else 2.5)
            thr = float(np.clip(round(m["est_lufs"] + 4.0), -40.0, -8.0))
            makeup = float(min(2.0, max(0.0, (crest - 14.0) / 4.0)))
            decisions.append(ActionDecision(
                id=f"comp.ch{ch}", action_type=ACTION_COMPRESSION, target=target,
                parameters={"ratio": ratio, "threshold_db": thr, "makeup_db": makeup,
                            "attack_ms": 5.0 if role not in GATE_ROLES else 2.0,
                            "release_ms": 150.0 if role not in GATE_ROLES else 90.0},
                reason=f"crest {crest:.1f} dB -> bounded compression для контроля crest factor",
                confidence=c["conf"], risk_level="low", safe_to_apply=True,
                source_modules=["crest_control"],
            ))

    # 4) Фейдеры: плоскости Ayaic, относительная нормализация, потолок 0
    raw = {}
    for c in channels:
        est_post_trim = c["meas"]["est_lufs"] + (c["new_trim"] - c["cur_trim"])
        raw[c["ch"]] = AYAIC_PLANE.get(c["role"], -27.0) - est_post_trim
    shift = max(raw.values()) - FADER_CEIL
    for c in channels:
        fader = float(np.clip(raw[c["ch"]] - shift, FADER_FLOOR, FADER_CEIL))
        c["fader"] = round(fader, 1)
        rb_fader = readback(client, f"/ch/{c['ch']}/fdr")
        if rb_fader is None:
            rb_fader = client.get_channel_fader(c["ch"])  # фолбэк: кэш состояния
        cur_fader = float(rb_fader if rb_fader is not None else 0.0)
        c["cur_fader"] = cur_fader
        decisions.append(ActionDecision(
            id=f"fader.ch{c['ch']}", action_type=ACTION_GAIN, target=f"ch{c['ch']}.fader",
            parameters={"gain_db": round(c["fader"] - cur_fader, 2),
                        "absolute_fader_db": c["fader"], "is_fader_plane": True},
            reason=f"плоскость Ayaic {AYAIC_PLANE.get(c['role'], -27.0)} LUFS, сдвиг {-shift:+.1f}",
            confidence=1.0, risk_level="low", safe_to_apply=True,
            source_modules=["ayaic_planes"],
        ))

    # ===== SafetyGate v2 =====
    gate = SafetyGate(SafetyGateConfig.from_mapping(SOUNDCHECK_GATE_CONFIG))
    plan = ActionPlan(plan_id=f"soundcheck_v2_{time.strftime('%Y%m%dT%H%M%S')}",
                      mode="soundcheck", decisions=decisions,
                      source_modules=["live_wing_soundcheck_v2"])
    state = {f"ch{c['ch']}": {"true_peak_dbtp": c["meas"]["true_peak_dbtp"]} for c in channels}
    gated = gate.evaluate_plan(plan, current_state=state, live_mode=False, dry_run=args.dry_run)
    allowed_ids = {d.id for d in gated.allowed_plan.decisions}
    print(f"\nSafetyGate v2: разрешено {gated.allowed_plan and len(gated.allowed_plan.decisions)}"
          f"/{len(decisions)}, заблокировано {len(gated.blocked)}")
    for b in gated.blocked:
        print(f"  BLOCKED {b['id']}: {b['reason']}")

    # План по каналам
    print("\nПлан:")
    for c in channels:
        eqs = ", ".join(f"b{i}:{v[1]:+.1f}dB@{v[0]:.0f}Hz" for i, v in c["eq_moves"]) or "—"
        print(f"  ch{c['ch']:2d} {c['name']:12s}: trim {c['cur_trim']:+.1f}->{c['new_trim']:+.1f}, "
              f"HPF {c['hpf']:.0f} Hz, EQ [{eqs}], fader ->{c['fader']}")

    if args.dry_run:
        print("\nDRY-RUN: записи на пульт не было.")
        client.disconnect()
        return 0

    # ===== Снятие отката и применение =====
    touched = {}
    for c in channels:
        ch = c["ch"]
        addrs = [f"/ch/{ch}/in/set/trim", f"/ch/{ch}/fdr", f"/ch/{ch}/flt/lc", f"/ch/{ch}/flt/lcf",
                 f"/ch/{ch}/dyn/on", f"/ch/{ch}/dyn/thr", f"/ch/{ch}/dyn/ratio", f"/ch/{ch}/dyn/att",
                 f"/ch/{ch}/dyn/rel", f"/ch/{ch}/dyn/gain", f"/ch/{ch}/eq/on",
                 f"/ch/{ch}/gate/on", f"/ch/{ch}/gate/thr", f"/ch/{ch}/gate/range"]
        for b in ("1", "2", "3", "4"):
            addrs += [f"/ch/{ch}/eq/{b}f", f"/ch/{ch}/eq/{b}g", f"/ch/{ch}/eq/{b}q"]
        for a in addrs:
            touched[a] = readback(client, a)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rollback = reports / f"live_wing_soundcheck_v2_original_{stamp}.json"
    rollback.write_text(json.dumps({"mixer_ip": mixer["ip"], "saved_at": stamp, "original": touched},
                                   ensure_ascii=False, indent=2))
    print(f"\nОткат: {rollback}")

    results = []
    for c in channels:
        ch = c["ch"]
        ok_all = True

        def wr(addr, setter, value, tol=0.25, rel=False):
            nonlocal ok_all
            setter(value)
            time.sleep(0.07)
            act = readback(client, addr)
            ok = act is not None and (
                (isinstance(value, str) and (str(act).strip() == value or abs(float(act) - float(value)) < 1e-6))
                or (not isinstance(value, str) and
                    abs(float(act) - float(value)) <= (max(tol, 0.06 * abs(float(value))) if rel else tol))
            )
            ok_all &= ok
            return ok

        if f"trim.ch{ch}" in allowed_ids:
            wr(f"/ch/{ch}/in/set/trim", lambda v: client.set_channel_gain(ch, v), c["new_trim"])
        wr(f"/ch/{ch}/flt/lc", lambda v: client.send(f"/ch/{ch}/flt/lc", v), 1, tol=0.5)
        wr(f"/ch/{ch}/flt/lcf", lambda v: client.send(f"/ch/{ch}/flt/lcf", v), float(c["hpf"]), rel=True)
        if c["eq_moves"]:
            wr(f"/ch/{ch}/eq/on", lambda v: client.set_eq_on(ch, v), 1, tol=0.5)
            for band_idx, (f, g, q) in c["eq_moves"]:
                if f"eq.ch{ch}.b{band_idx}" not in allowed_ids:
                    continue
                b = str(band_idx)
                wr(f"/ch/{ch}/eq/{b}f", lambda v: client.set_eq_band_frequency(ch, f"{b}f", v), float(f), rel=True)
                wr(f"/ch/{ch}/eq/{b}g", lambda v: client.set_eq_band_gain(ch, f"{b}g", v), float(g))
                wr(f"/ch/{ch}/eq/{b}q", lambda v: client.set_eq_band_q(ch, f"{b}q", v), float(q), rel=True)
        comp = next((d for d in gated.allowed_plan.decisions if d.id == f"comp.ch{ch}"), None)
        if comp:
            p = comp.parameters
            wr(f"/ch/{ch}/dyn/on", lambda v: client.set_compressor_on(ch, v), 1, tol=0.5)
            wr(f"/ch/{ch}/dyn/thr", lambda v: client.set_compressor_threshold(ch, v), float(p["threshold_db"]))
            wr(f"/ch/{ch}/dyn/ratio", lambda v: client.set_compressor_ratio(ch, v), nearest_ratio(float(p["ratio"])))
            wr(f"/ch/{ch}/dyn/att", lambda v: client.set_compressor_attack(ch, v), float(p["attack_ms"]), tol=1.0)
            wr(f"/ch/{ch}/dyn/rel", lambda v: client.set_compressor_release(ch, v), float(p["release_ms"]), rel=True)
            wr(f"/ch/{ch}/dyn/gain", lambda v: client.set_compressor_gain(ch, v), float(p["makeup_db"]))
        if c["role"] in GATE_ROLES:
            client.set_gate(ch, threshold=float(c["meas"]["est_lufs"] - 12.0), range_db=24.0,
                            attack=1.0, hold=20.0, release=150.0)
            time.sleep(0.07)
            wr(f"/ch/{ch}/gate/on", lambda v: client.set_gate_on(ch, v), 1, tol=0.5)
        if f"fader.ch{ch}" in allowed_ids:
            wr(f"/ch/{ch}/fdr", lambda v: client.set_channel_fader(ch, v), c["fader"])
        results.append((ch, c["name"], c["role"], ok_all))
        print(f"  {'PASS' if ok_all else 'PARTIAL'} ch{ch:2d} {c['name']:12s} ({c['role']})")

    applied = reports / f"live_wing_soundcheck_v2_applied_{stamp}.json"
    applied.write_text(json.dumps(
        {"applied_at": stamp, "plan": gated.allowed_plan.to_dict(), "blocked": gated.blocked,
         "channels": [{k: v for k, v in c.items() if k != "audio"} for c in channels]},
        ensure_ascii=False, indent=2, default=float))
    client.disconnect()

    ok = sum(1 for *_, s in results if s)
    print("\n================ ИТОГ (пайплайн v2) ================")
    print(f"Настроено: {ok}/{len(results)}; настройки ОСТАВЛЕНЫ. План/blocked: {applied}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
