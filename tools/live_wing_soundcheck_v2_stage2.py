#!/usr/bin/env python3
"""Саундчек v2 — стадия 2: фазирование, задержки, панорама, mirror EQ, вокал.

Дополняет стадию 1 (live_wing_soundcheck_v2) остальным новым DSP:

1. ОЖИДАНИЕ ВОКАЛА: циклический захват, пока лид-вокал не станет активным
   (или таймаут) — затем его полная настройка (trim/HPF/spectral EQ/comp)
   и фейдер, вписанный в ТЕКУЩИЙ баланс относительно плоскостей Ayaic.
2. ФАЗА/ЗАДЕРЖКИ: GCCPHATAnalyzer (GCC-PHAT, суб-сэмпловая точность,
   метрики PSR/coherence). Пары: SNARE B->SNARE T (полярность),
   близкие микрофоны барабанов -> OH (выравнивающая задержка 0.5..15 мс).
   Полярность — по знаку корреляции на найденном лаге.
   ПОРЯДОК АРГУМЕНТОВ по правилу CLAUDE.md: gcc_phat(target, reference).
3. ПАНОРАМА: шаблоны auto_panner_adaptive (Genre.ROCK) + стереопары по
   именам (L/R): гитары L45/R45, OH L67/R67, playback L90/R90 и т.д.
4. МАСКИРОВАНИЕ (EQ v3): ERB-полосы, маскер = СУММА остальных каналов
   (Hafezi & Reiss 2015), cut-only с защитой вокала, кэп −6 дБ — заменяет
   парный mirror EQ. Мержится в СВОБОДНЫЕ консольные полосы (не трогает
   notch и полосы, занятые spectral ceiling стадии 1).

Все решения проходят SafetyGate v2. Откат в reports/. Настройки остаются.

Запуск:
    python3 tools/live_wing_soundcheck_v2_stage2.py --dry-run
    python3 tools/live_wing_soundcheck_v2_stage2.py --wait-vocal 90 \
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
from lufs_gain_staging import LUFSMeter, TruePeakMeter  # noqa: E402
from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402
from auto_panner_adaptive import (  # noqa: E402
    Genre,
    GenreTemplate,
    PanPosition,
    lf_energy_ratio,
    should_center_low_frequency,
)
from heuristics.erb_masking import (  # noqa: E402
    allocate_complementary_cuts,
    band_energies_db,
    erb_band_edges,
)
from heuristics.spectral_ceiling_eq import (  # noqa: E402
    SpectralCeilingEQAnalyzer,
    merge_spectral_proposal_into_eq_bands,
)
from automixer.decision.models import (  # noqa: E402
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_PAN,
    ActionDecision,
    ActionPlan,
)
from automixer.safety.safety_gate import SafetyGate, SafetyGateConfig  # noqa: E402

LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
NUM_CH, SR, BLOCK = 40, 48000, 4800
CAPTURE_S = 8.0
ACTIVE_ST_LUFS = -45.0
READBACK_TIMEOUT = 1.2
DELAY_MIN_MS, DELAY_MAX_MS = 0.5, 15.0

EXCLUDE_NAME_RE = re.compile(r"master|mstr|\bref\b|refer|talk|\btb\b|mtx|matrix|\bbus\b|\bmon\b", re.I)

NAME_ROLE = [
    (re.compile(r"kick|\bbd\b|бочк", re.I), "kick"),
    (re.compile(r"snare|\bsn\b|малый", re.I), "snare"),
    (re.compile(r"f\.?\s?tom|floor", re.I), "floor_tom"),
    (re.compile(r"\btom\b|том", re.I), "tom"),
    (re.compile(r"\boh\b|over", re.I), "overheads"),
    (re.compile(r"hat|\bhh\b|ride|crash|cymb|тарел", re.I), "hihat"),
    (re.compile(r"bass|\bbs\b|бас", re.I), "bass"),
    (re.compile(r"g(ui)?t|guitar|гитар", re.I), "guitar"),
    (re.compile(r"keys?|piano|synth|pad|клав|органн|accord|аккорд", re.I), "keys"),
    (re.compile(r"\bback\b|\bbv\b|back.*voc", re.I), "backing_vocal"),
    (re.compile(r"voc|vox|vcl|lead|sing|голос|вок", re.I), "lead_vocal"),
    (re.compile(r"play|track|\bpb\b|минус|фонограм", re.I), "playback"),
]

AYAIC_PLANE = {
    "lead_vocal": -22.0, "backing_vocal": -27.0, "kick": -25.0, "snare": -25.0,
    "tom": -28.0, "floor_tom": -28.0, "hihat": -28.0, "overheads": -28.0,
    "bass": -25.0, "guitar": -25.0, "keys": -30.0, "playback": -25.0, "line": -27.0,
}
# панорама: роль -> позиция(и); стереопары раскладываются симметрично
PAN_ROLE = {
    "kick": 0, "snare": 0, "bass": 0, "lead_vocal": 0,
    "guitar": (PanPosition.L45.value, PanPosition.R45.value),
    "overheads": (PanPosition.L67.value, PanPosition.R67.value),
    "playback": (PanPosition.L90.value, PanPosition.R90.value),
    "backing_vocal": (PanPosition.L45.value, PanPosition.R45.value),
    "keys": PanPosition.L22.value,
    "tom": PanPosition.L22.value, "floor_tom": PanPosition.R22.value,
    "hihat": PanPosition.R45.value, "line": 0,
}

GATE_CONFIG = {
    "max_gain_change_db": 15.0, "max_live_gain_increase_db": 6.0,
    "min_gain_change_db": 0.3, "max_eq_boost_db": 2.0, "max_eq_cut_db": 6.0,
    "max_eq_q": 4.0, "max_compression_ratio": 6.0,
    "max_compression_threshold_change_db": 30.0, "min_interval_sec": 0.0,
    "cooldown_sec": 0.0, "hysteresis_db": 0.25,
    "live_mode_blocks_abrupt_changes": True, "true_peak_ceiling_dbtp": -1.0,
    "max_pan_step": 1.0, "max_pan_abs": 1.0,
}


def readback(client, addr):
    r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    if r["status"] != "received":
        r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    return r["value"] if r["status"] == "received" else None


def role_for(name: str) -> str:
    for rx, role in NAME_ROLE:
        if name and rx.search(name):
            return role
    return "line"


def measure(x: np.ndarray) -> dict:
    lm, tm = LUFSMeter(sample_rate=SR), TruePeakMeter(sample_rate=SR)
    st, tp = [], -144.0
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        blk = x[i:i + BLOCK].astype(np.float64)
        st.append(float(lm.process(blk)))
        tp = max(tp, float(tm.process(blk)))
    arr = np.array([v for v in st if np.isfinite(v)])
    st_max = float(np.max(arr)) if arr.size else -144.0
    gated = arr[arr > max(st_max - 20.0, -60.0)]
    return {"st_max_lufs": st_max,
            "est_lufs": float(np.mean(gated)) if gated.size else st_max,
            "true_peak_dbtp": tp}


def polarity_at_lag(ref: np.ndarray, tgt: np.ndarray, lag_samples: int) -> float:
    """Знак корреляции target против reference на найденном лаге."""
    n = min(len(ref), len(tgt)) - abs(lag_samples)
    if n < SR:
        return 0.0
    if lag_samples >= 0:
        a, b = ref[lag_samples:lag_samples + n], tgt[:n]
    else:
        a, b = ref[:n], tgt[-lag_samples:-lag_samples + n]
    denom = (np.std(a) * np.std(b)) or 1e-12
    return float(np.mean((a - a.mean()) * (b - b.mean())) / denom)


def capture(dev_idx: int, seconds: float) -> np.ndarray:
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=NUM_CH, device=dev_idx, dtype="float32")
    sd.wait()
    return audio


def main() -> int:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: use the supervised control plane"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-live-wing-write", default="")
    ap.add_argument("--wait-vocal", type=float, default=90.0, help="сколько секунд ждать активности вокала")
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
    roles = {ch: role_for((names.get(ch) or "").strip()) for ch in range(1, NUM_CH + 1)}
    vocal_chs = [ch for ch, r in roles.items() if r == "lead_vocal"
                 and not EXCLUDE_NAME_RE.search(names.get(ch) or "")]

    devs = sd.query_devices()
    dev = next(i for i, d in enumerate(devs) if d["name"] == "WING" and d["max_input_channels"] >= NUM_CH)

    # ===== 1. Захват с ожиданием вокала =====
    deadline = time.time() + max(0.0, args.wait_vocal)
    audio, vocal_active = None, False
    attempt = 0
    while True:
        attempt += 1
        print(f"Захват {CAPTURE_S:.0f} с (попытка {attempt})...")
        audio = capture(dev, CAPTURE_S)
        if vocal_chs:
            vmeas = measure(audio[:, vocal_chs[0] - 1])
            vocal_active = vmeas["st_max_lufs"] > ACTIVE_ST_LUFS
            print(f"  вокал ch{vocal_chs[0]}: ST-LUFS max {vmeas['st_max_lufs']:.1f} -> "
                  f"{'АКТИВЕН' if vocal_active else 'молчит'}")
        if vocal_active or time.time() >= deadline or not vocal_chs:
            break
        print(f"  ждём вокал (осталось {max(0, deadline - time.time()):.0f} с)...")
        time.sleep(2.0)

    active: dict[int, dict] = {}
    for ch in range(1, NUM_CH + 1):
        name = (names.get(ch) or "").strip()
        if EXCLUDE_NAME_RE.search(name):
            continue
        x = audio[:, ch - 1]
        m = measure(x)
        if m["st_max_lufs"] < ACTIVE_ST_LUFS:
            continue
        active[ch] = {"name": name, "role": roles[ch], "meas": m, "x": x}
    print(f"\nАктивные каналы: {sorted(active)}")

    decisions: list[ActionDecision] = []
    spectral = SpectralCeilingEQAnalyzer()

    # ===== 2. Вокал: полная настройка, если активен =====
    vocal_plan = None
    if vocal_active and vocal_chs[0] in active:
        vch = vocal_chs[0]
        c = active[vch]
        cur_trim = float(readback(client, f"/ch/{vch}/in/set/trim") or 0.0)
        delta = min(-18.0 - c["meas"]["true_peak_dbtp"], 6.0)
        new_trim = float(np.clip(cur_trim + delta, -18.0, 18.0))
        # фейдер: вписать в ТЕКУЩИЙ баланс (медиана по другим активным каналам)
        offsets = []
        for ch, o in active.items():
            if ch == vch:
                continue
            f = readback(client, f"/ch/{ch}/fdr")
            if f is None:
                continue
            offsets.append(float(f) + (AYAIC_PLANE["lead_vocal"] - AYAIC_PLANE.get(o["role"], -27.0))
                           + (o["meas"]["est_lufs"] - (c["meas"]["est_lufs"] + (new_trim - cur_trim))))
        fader = float(np.clip(np.median(offsets) if offsets else -3.0, -40.0, 0.0))
        prop = spectral.analyze(c["x"].astype(np.float64), instrument_role="lead_vocal",
                                sample_rate=SR, track_id=f"ch{vch}", role_confidence=1.0,
                                lead_vocal_active=True, lead_vocal_confidence=1.0)
        cur_bands = []
        for b in "1234":
            cur_bands.append((float(readback(client, f"/ch/{vch}/eq/{b}f") or 1000.0),
                              float(readback(client, f"/ch/{vch}/eq/{b}g") or 0.0),
                              float(readback(client, f"/ch/{vch}/eq/{b}q") or 2.0)))
        merged, _ = merge_spectral_proposal_into_eq_bands(cur_bands, prop)
        eq_moves = [(i + 1, nb) for i, (ob, nb) in enumerate(zip(cur_bands, merged))
                    if ob != nb and ob[1] > -4.0]
        vocal_plan = {"ch": vch, "trim": new_trim, "cur_trim": cur_trim, "fader": fader,
                      "hpf": prop.low_cut_hz or 100.0, "eq_moves": eq_moves,
                      "comp": {"threshold_db": float(np.clip(round(c["meas"]["est_lufs"] + 4.0), -26, -8)),
                               "ratio": 3.0, "attack_ms": 5.0, "release_ms": 150.0,
                               "makeup_db": 1.0}}
        decisions.append(ActionDecision(
            id=f"vocal.trim.ch{vch}", action_type=ACTION_GAIN, target=f"ch{vch}",
            parameters={"gain_db": round(new_trim - cur_trim, 2)},
            reason="гейн-стейджинг вокала (дождались активности)",
            confidence=1.0, risk_level="low", safe_to_apply=True))
        for bi, (f, g, q) in eq_moves:
            og = cur_bands[bi - 1][1]
            decisions.append(ActionDecision(
                id=f"vocal.eq.ch{vch}.b{bi}", action_type=ACTION_EQ, target=f"ch{vch}",
                parameters={"frequency_hz": f, "gain_db": round(g - og, 2), "q": q},
                reason="spectral ceiling лид-вокала", confidence=float(prop.confidence),
                risk_level="low", safe_to_apply=True))

    # ===== 3. Фаза/задержки (GCC-PHAT) =====
    phase_plan = []  # (ch, invert|None, delay_ms|None, why)
    oh = [ch for ch, c in active.items() if c["role"] == "overheads"]
    snare_t = next((ch for ch, c in active.items() if c["name"].upper().startswith("SNARE T")), None)
    snare_b = next((ch for ch, c in active.items() if c["name"].upper().startswith("SNARE B")), None)

    def gcc(ref_ch: int, tgt_ch: int):
        an = GCCPHATAnalyzer(sample_rate=SR)
        # ПОРЯДОК: (target, reference) по правилу CLAUDE.md -> в API это
        # compute_delay(ref_signal=REF, tgt_signal=TGT); положительный
        # delay = target отстаёт от reference.
        return an.compute_delay(ref_signal=active[ref_ch]["x"].astype(np.float64),
                                tgt_signal=active[tgt_ch]["x"].astype(np.float64))

    if snare_t and snare_b:
        dm = gcc(snare_t, snare_b)
        pol = polarity_at_lag(active[snare_t]["x"].astype(np.float64),
                              active[snare_b]["x"].astype(np.float64),
                              int(round(dm.delay_samples)))
        if dm.is_valid(min_correlation=0.4, min_psr=3.0) and pol < -0.15:
            phase_plan.append((snare_b, 1, None,
                               f"SNARE B против SNARE T: corr {pol:.2f} на лаге {dm.delay_ms:.2f} мс -> инверсия"))
        else:
            phase_plan.append((snare_b, None, None,
                               f"SNARE B: corr {pol:.2f}, лаг {dm.delay_ms:.2f} мс (conf {dm.confidence:.2f}) — без изменений"))
    if oh:
        ohr = oh[0]
        for ch, c in active.items():
            if c["role"] not in ("kick", "snare", "tom", "floor_tom"):
                continue
            if ch == snare_b:
                continue
            dm = gcc(ch, ohr)  # насколько OH отстаёт от близкого микрофона
            if dm.is_valid(min_correlation=0.35, min_psr=3.0) and DELAY_MIN_MS <= dm.delay_ms <= DELAY_MAX_MS:
                phase_plan.append((ch, None, round(float(dm.delay_ms), 2),
                                   f"{c['name']} -> OH: задержка {dm.delay_ms:.2f} мс "
                                   f"(conf {dm.confidence:.2f}, coh {dm.coherence:.2f})"))
            else:
                phase_plan.append((ch, None, None,
                                   f"{c['name']} -> OH: лаг {dm.delay_ms:.2f} мс, conf {dm.confidence:.2f} — не применяем"))

    # ===== 4. Панорама =====
    pan_plan = []  # (ch, pan_value)
    by_role: dict[str, list[int]] = {}
    for ch, c in sorted(active.items()):
        by_role.setdefault(c["role"], []).append(ch)
    for role, chans in by_role.items():
        spec = PAN_ROLE.get(role, 0)
        if isinstance(spec, tuple) and len(chans) >= 2:
            pan_plan.append((chans[0], float(spec[0])))
            pan_plan.append((chans[1], float(spec[1])))
            for extra in chans[2:]:
                pan_plan.append((extra, 0.0))
        else:
            v = float(spec[0]) if isinstance(spec, tuple) else float(spec)
            for ch in chans:
                pan_plan.append((ch, v))
    # LCR-правило: канал с энергией преимущественно ниже ~120 Гц — в центр
    # (энергия и фазовая устойчивость важнее стереокартины), независимо от
    # роли-шаблона (например, суб-контент в playback/floor tom).
    pan_plan = [
        (ch, 0.0 if should_center_low_frequency(
            lf_energy_ratio(active[ch]["x"], SR)) else pan)
        for ch, pan in pan_plan
    ]
    for ch, pan in pan_plan:
        decisions.append(ActionDecision(
            id=f"pan.ch{ch}", action_type=ACTION_PAN, target=f"ch{ch}",
            parameters={"pan": pan / 100.0, "pan_step": 1.0, "absolute_pan": pan},
            reason=f"шаблон панорамы ({active[ch]['role']})", confidence=1.0,
            risk_level="low", safe_to_apply=True))

    # ===== 5. Маскирование (EQ v3: ERB против суммы) =====
    # EQ v3: маскирование по ERB против СУММЫ остальных каналов (не парное),
    # cut-only с защитой вокала — заменяет парный mirror EQ (CrossAdaptiveEQ):
    # попарная оптимизация могла улучшать одну пару, ухудшая другую, а boost
    # маскируемого снижал запас до самовозбуждения.
    erb_edges = erb_band_edges()
    erb_bands = {ch: band_energies_db(c["x"].astype(np.float64), SR, erb_edges)
                 for ch, c in active.items()}
    erb_roles = {ch: c["role"] for ch, c in active.items()}
    masking_cuts = allocate_complementary_cuts(erb_bands, erb_roles, edges=erb_edges)
    mirror_moves: dict[int, list] = {}
    for cut in masking_cuts:
        ch = int(cut.channel)
        moves = mirror_moves.setdefault(ch, [])
        if len(moves) >= 2 or abs(cut.gain_db) < 0.3:
            continue
        moves.append((cut.freq_hz, cut.gain_db, cut.q, f"erb{cut.band_index}",
                      ch, cut.masked_channel))
        decisions.append(ActionDecision(
            id=f"mirror.ch{ch}.erb{cut.band_index}.{len(moves)}",
            action_type=ACTION_EQ, target=f"ch{ch}",
            parameters={"frequency_hz": cut.freq_hz, "gain_db": cut.gain_db,
                        "q": float(np.clip(cut.q, 0.5, 4.0))},
            reason=(f"ERB-маскирование: ch{ch} маскирует ch{cut.masked_channel} "
                    f"({cut.mask_before_db:.1f} -> {cut.mask_after_db:.1f} dB)"),
            confidence=0.8, risk_level="low", safe_to_apply=True))

    # ===== SafetyGate =====
    gate = SafetyGate(SafetyGateConfig.from_mapping(GATE_CONFIG))
    plan = ActionPlan(plan_id=f"soundcheck_v2s2_{time.strftime('%Y%m%dT%H%M%S')}",
                      mode="soundcheck", decisions=decisions,
                      source_modules=["live_wing_soundcheck_v2_stage2"])
    state = {f"ch{ch}": {"true_peak_dbtp": c["meas"]["true_peak_dbtp"]} for ch, c in active.items()}
    gated = gate.evaluate_plan(plan, current_state=state, live_mode=False, dry_run=args.dry_run)
    allowed = {d.id for d in gated.allowed_plan.decisions}
    print(f"\nSafetyGate v2: разрешено {len(gated.allowed_plan.decisions)}/{len(decisions)}, "
          f"заблокировано {len(gated.blocked)}")
    for b in gated.blocked:
        print(f"  BLOCKED {b['id']}: {b['reason']}")

    print("\n--- Фаза/задержки (GCC-PHAT) ---")
    for ch, inv, dly, why in phase_plan:
        print(f"  ch{ch:2d}: {why}")
    print("--- Панорама ---")
    for ch, pan in sorted(pan_plan):
        print(f"  ch{ch:2d} {active[ch]['name']:12s} -> pan {pan:+.0f}")
    print("--- Mirror EQ ---")
    for ch, moves in sorted(mirror_moves.items()):
        for f, g, q, band, masker, masked in moves:
            print(f"  ch{ch:2d} {active[ch]['name']:12s}: {g:+.1f} dB @ {f:.0f} Hz Q{q:.1f} "
                  f"[{band}; masker ch{masker}, masked ch{masked}]")
    if vocal_plan:
        vs = ", ".join(f"b{i}:{v[1]:+.1f}dB@{v[0]:.0f}Hz" for i, v in vocal_plan["eq_moves"]) or "—"
        print(f"--- Вокал ch{vocal_plan['ch']} ---\n  trim {vocal_plan['cur_trim']:+.1f}->{vocal_plan['trim']:+.1f}, "
              f"HPF {vocal_plan['hpf']:.0f} Hz, EQ [{vs}], comp {vocal_plan['comp']['threshold_db']}/3:1, "
              f"fader ->{vocal_plan['fader']:.1f}")
    elif vocal_chs:
        print(f"--- Вокал ch{vocal_chs[0]}: НЕ дождались активности за {args.wait_vocal:.0f} с ---")

    if args.dry_run:
        print("\nDRY-RUN: записи на пульт не было.")
        client.disconnect()
        return 0

    # ===== Откат =====
    touched = {}
    addrs = []
    for ch in active:
        addrs += [f"/ch/{ch}/pan", f"/ch/{ch}/in/set/inv", f"/ch/{ch}/in/set/dly",
                  f"/ch/{ch}/in/set/dlyon", f"/ch/{ch}/in/set/dlymode"]
        for b in "1234":
            addrs += [f"/ch/{ch}/eq/{b}f", f"/ch/{ch}/eq/{b}g", f"/ch/{ch}/eq/{b}q"]
        addrs.append(f"/ch/{ch}/eq/on")
    if vocal_plan:
        vch = vocal_plan["ch"]
        addrs += [f"/ch/{vch}/in/set/trim", f"/ch/{vch}/fdr", f"/ch/{vch}/flt/lc",
                  f"/ch/{vch}/flt/lcf", f"/ch/{vch}/dyn/on", f"/ch/{vch}/dyn/thr",
                  f"/ch/{vch}/dyn/ratio", f"/ch/{vch}/dyn/att", f"/ch/{vch}/dyn/rel",
                  f"/ch/{vch}/dyn/gain"]
    for a in addrs:
        touched[a] = readback(client, a)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rollback = reports / f"live_wing_soundcheck_v2s2_original_{stamp}.json"
    rollback.write_text(json.dumps({"mixer_ip": mixer["ip"], "saved_at": stamp, "original": touched},
                                   ensure_ascii=False, indent=2))
    print(f"\nОткат: {rollback}")

    fails = []

    def wr(addr, setter, value, tol=0.25, rel=False):
        setter(value)
        time.sleep(0.07)
        act = readback(client, addr)
        ok = act is not None and (
            (isinstance(value, str) and str(act).strip() == value)
            or (not isinstance(value, str) and
                abs(float(act) - float(value)) <= (max(tol, 0.06 * abs(float(value))) if rel else tol)))
        if not ok:
            fails.append((addr, value, act))
        return ok

    # применение: фаза/задержки
    for ch, inv, dly, _ in phase_plan:
        if inv is not None:
            wr(f"/ch/{ch}/in/set/inv", lambda v: client.set_channel_phase_invert(ch, v), 1, tol=0.5)
        if dly is not None:
            client.send(f"/ch/{ch}/in/set/dlymode", "MS")
            time.sleep(0.05)
            wr(f"/ch/{ch}/in/set/dly", lambda v: client.send(f"/ch/{ch}/in/set/dly", v), float(dly), tol=0.1)
            wr(f"/ch/{ch}/in/set/dlyon", lambda v: client.send(f"/ch/{ch}/in/set/dlyon", v), 1, tol=0.5)
    # панорама
    for ch, pan in pan_plan:
        if f"pan.ch{ch}" in allowed:
            wr(f"/ch/{ch}/pan", lambda v: client.set_channel_pan(ch, v), float(pan), tol=2.0)
    # mirror EQ — в свободные консольные полосы
    for ch, moves in mirror_moves.items():
        cur = []
        for b in "1234":
            cur.append((float(readback(client, f"/ch/{ch}/eq/{b}f") or 1000.0),
                        float(readback(client, f"/ch/{ch}/eq/{b}g") or 0.0),
                        float(readback(client, f"/ch/{ch}/eq/{b}q") or 2.0)))
        free = [i for i, (f, g, q) in enumerate(cur) if abs(g) < 0.2]
        applied_moves = 0
        for k, (f, g, q, band, *_ ) in enumerate(moves, 1):
            if f"mirror.ch{ch}.{band}.{k}" not in allowed or not free:
                continue
            bi = min(free, key=lambda i: abs(np.log2(cur[i][0] / f)))
            free.remove(bi)
            b = str(bi + 1)
            wr(f"/ch/{ch}/eq/{b}f", lambda v: client.set_eq_band_frequency(ch, f"{b}f", v), float(f), rel=True)
            wr(f"/ch/{ch}/eq/{b}g", lambda v: client.set_eq_band_gain(ch, f"{b}g", v), float(g))
            wr(f"/ch/{ch}/eq/{b}q", lambda v: client.set_eq_band_q(ch, f"{b}q", v), float(np.clip(q, 0.5, 4.0)), rel=True)
            applied_moves += 1
        if applied_moves:
            wr(f"/ch/{ch}/eq/on", lambda v: client.set_eq_on(ch, v), 1, tol=0.5)
    # вокал
    if vocal_plan:
        vch = vocal_plan["ch"]
        if f"vocal.trim.ch{vch}" in allowed:
            wr(f"/ch/{vch}/in/set/trim", lambda v: client.set_channel_gain(vch, v), vocal_plan["trim"])
        wr(f"/ch/{vch}/flt/lc", lambda v: client.send(f"/ch/{vch}/flt/lc", v), 1, tol=0.5)
        wr(f"/ch/{vch}/flt/lcf", lambda v: client.send(f"/ch/{vch}/flt/lcf", v), float(vocal_plan["hpf"]), rel=True)
        for bi, (f, g, q) in vocal_plan["eq_moves"]:
            if f"vocal.eq.ch{vch}.b{bi}" not in allowed:
                continue
            b = str(bi)
            wr(f"/ch/{vch}/eq/{b}f", lambda v: client.set_eq_band_frequency(vch, f"{b}f", v), float(f), rel=True)
            wr(f"/ch/{vch}/eq/{b}g", lambda v: client.set_eq_band_gain(vch, f"{b}g", v), float(g))
            wr(f"/ch/{vch}/eq/{b}q", lambda v: client.set_eq_band_q(vch, f"{b}q", v), float(q), rel=True)
        wr(f"/ch/{vch}/eq/on", lambda v: client.set_eq_on(vch, v), 1, tol=0.5)
        cp = vocal_plan["comp"]
        wr(f"/ch/{vch}/dyn/on", lambda v: client.set_compressor_on(vch, v), 1, tol=0.5)
        wr(f"/ch/{vch}/dyn/thr", lambda v: client.set_compressor_threshold(vch, v), cp["threshold_db"])
        wr(f"/ch/{vch}/dyn/ratio", lambda v: client.set_compressor_ratio(vch, v), "3.0")
        wr(f"/ch/{vch}/dyn/att", lambda v: client.set_compressor_attack(vch, v), cp["attack_ms"], tol=1.0)
        wr(f"/ch/{vch}/dyn/rel", lambda v: client.set_compressor_release(vch, v), cp["release_ms"], rel=True)
        wr(f"/ch/{vch}/dyn/gain", lambda v: client.set_compressor_gain(vch, v), cp["makeup_db"])
        wr(f"/ch/{vch}/fdr", lambda v: client.set_channel_fader(vch, v), vocal_plan["fader"])

    applied = reports / f"live_wing_soundcheck_v2s2_applied_{stamp}.json"
    applied.write_text(json.dumps(
        {"applied_at": stamp, "plan": gated.allowed_plan.to_dict(), "blocked": gated.blocked,
         "phase_plan": [(ch, inv, dly, why) for ch, inv, dly, why in phase_plan],
         "pan_plan": pan_plan,
         "mirror": {str(ch): [m[:4] for m in mv] for ch, mv in mirror_moves.items()},
         "vocal": {k: v for k, v in (vocal_plan or {}).items() if k != "x"}},
        ensure_ascii=False, indent=2, default=float))
    client.disconnect()
    print("\n================ ИТОГ (стадия 2) ================")
    if fails:
        print(f"Не подтвердились readback'ом ({len(fails)}):")
        for a, v, act in fails[:10]:
            print(f"  {a}: послано {v!r}, readback {act!r}")
    else:
        print("Все записи подтверждены readback'ом.")
    print(f"Настройки ОСТАВЛЕНЫ. План: {applied}\nОткат: {rollback}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
