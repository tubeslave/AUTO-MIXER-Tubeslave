#!/usr/bin/env python3
"""Непрерывный режим саундчека: мониторинг пульта + коррекции до остановки.

Цикл (каждые ~12 с), пока звукорежиссёр не остановит:
  1. Захват аудио со всех каналов WING USB (pre-trim tap).
  2. Измерения BS.1770 (LUFSMeter) + true peak (TruePeakMeter).
  3. FEEDBACK FIRST (CLAUDE.md): детекция устойчивого узкополосного
     растущего пика — немедленное снижение фейдера канала на 3 dB,
     приоритет над всеми другими решениями.
  4. Инкрементальные коррекции:
     - trim: открытый контур к цели -18 dBTP (сходится за цикл, дальше
       глушится гистерезисом);
     - фейдеры: шаги <= 1.5 dB/цикл к относительному балансу Ayaic;
     - EQ (spectral ceiling): предложение применяется ТОЛЬКО если оно
       устойчиво три цикла подряд (persistence), затем cooldown.
  5. Каждое решение — через SafetyGate v2 с live-лимитами (микрошаги,
     гистерезис, интервалы). Заблокированное = «предложено, не применено».
  6. Применение с физическим readback; журнал цикла в reports/.

Остановка звукорежиссёром:
  - Ctrl+C, либо
  - создать stop-файл:  touch reports/SOUNDCHECK_STOP
  - или задать --duration/--max-cycles для ограниченного теста.

Запуск:
    python3 tools/live_wing_soundcheck_daemon.py --dry-run --max-cycles 2
    python3 tools/live_wing_soundcheck_daemon.py --duration 120 \
        --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
"""

from __future__ import annotations

import argparse
import json
import re
import signal
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
from heuristics.spectral_ceiling_eq import (  # noqa: E402
    SpectralCeilingEQAnalyzer,
    merge_spectral_proposal_into_eq_bands,
)
from automixer.decision.models import (  # noqa: E402
    ACTION_EQ,
    ACTION_GAIN,
    ActionDecision,
    ActionPlan,
)
from automixer.safety.safety_gate import SafetyGate, SafetyGateConfig  # noqa: E402

LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
NUM_CH, SR, BLOCK = 40, 48000, 4800
CAPTURE_S = 8.0
CYCLE_PERIOD_S = 12.0
ACTIVE_ST_LUFS = -45.0
TRIM_TARGET_DBTP = -18.0
FADER_STEP_MAX = 1.5              # dB за цикл
FADER_FLOOR, FADER_CEIL = -40.0, 0.0
EQ_PERSISTENCE = 3                # циклов подряд до применения EQ
EQ_COOLDOWN_S = 60.0
READBACK_TIMEOUT = 1.2
STOP_FILE = REPO_ROOT / "reports" / "SOUNDCHECK_STOP"

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
    (re.compile(r"accord|аккорд|акорд", re.I), "keys"),
    (re.compile(r"keys?|piano|synth|pad|клав|органн", re.I), "keys"),
    (re.compile(r"\bback\b|\bbv\b|back.*voc", re.I), "backing_vocal"),
    (re.compile(r"voc|vox|vcl|lead|sing|голос|вок", re.I), "lead_vocal"),
    (re.compile(r"play|track|\bpb\b|минус|фонограм", re.I), "playback"),
]
AYAIC_PLANE = {
    "lead_vocal": -22.0, "backing_vocal": -27.0, "kick": -25.0, "snare": -25.0,
    "tom": -28.0, "floor_tom": -28.0, "hihat": -28.0, "overheads": -28.0,
    "bass": -25.0, "guitar": -25.0, "keys": -30.0, "playback": -25.0, "line": -27.0,
}

# live-лимиты непрерывного режима: микрошаги + гистерезис + интервалы
LIVE_GATE_CONFIG = {
    "max_gain_change_db": FADER_STEP_MAX,
    "max_live_gain_increase_db": 1.0,
    "min_gain_change_db": 0.3,
    "max_eq_boost_db": 1.5,
    "max_eq_cut_db": 4.0,
    "max_eq_q": 4.0,
    "min_interval_sec": CYCLE_PERIOD_S * 0.8,
    "cooldown_sec": 5.0,
    "hysteresis_db": 0.4,
    "live_mode_blocks_abrupt_changes": True,
    "true_peak_ceiling_dbtp": -1.0,
}

_running = True


def _sigint(_s, _f):
    global _running
    _running = False
    print("\n[STOP] Получен сигнал остановки от оператора (Ctrl+C)")


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


def detect_feedback(x: np.ndarray) -> dict | None:
    """Устойчивый узкополосный растущий пик = заводка.

    Признаки: доминирующий спектральный пик > 18 dB над медианой спектра
    в >= 4 последовательных блоках по 0.5 с, на стабильной частоте (+-2%),
    с неубывающей амплитудой.
    """
    win = SR // 2
    peaks = []
    for i in range(0, len(x) - win + 1, win):
        seg = x[i:i + win].astype(np.float64) * np.hanning(win)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win, 1 / SR)
        m = freqs > 80
        spec, freqs = spec[m], freqs[m]
        db = 20 * np.log10(spec + 1e-10)
        j = int(np.argmax(db))
        f_peak, a_peak = float(freqs[j]), float(db[j])
        prominence = a_peak - float(np.median(db))
        # гармоники: у музыкальной ноты энергия на 2f/3f близка к пику;
        # у заводки — чистый тон, гармоники слабее на 25+ dB
        harm = -200.0
        for mult in (2.0, 3.0):
            hm = (freqs > f_peak * mult * 0.97) & (freqs < f_peak * mult * 1.03)
            if hm.any():
                harm = max(harm, float(np.max(db[hm])))
        harmonic_like = harm > a_peak - 25.0
        peaks.append((f_peak, a_peak, prominence, harmonic_like))
    run = 0
    for k in range(1, len(peaks)):
        f0, a0, p0, h0 = peaks[k - 1]
        f1, a1, p1, h1 = peaks[k]
        stable = abs(f1 - f0) / max(f0, 1.0) < 0.02
        growing = a1 >= a0 + 1.0        # заводка монотонно растёт
        if p1 > 18.0 and stable and growing and not h1:
            run += 1
            if run >= 3:
                return {"freq_hz": round(f1, 1), "prominence_db": round(p1, 1)}
        else:
            run = 0
    return None


def main() -> int:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: continuous write authority is disabled"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-live-wing-write", default="")
    ap.add_argument("--duration", type=float, default=0.0, help="сек; 0 = до остановки оператором")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = без лимита")
    args = ap.parse_args()
    if not args.dry_run and args.confirm_live_wing_write != LIVE_WING_CONFIRMATION:
        print(f"FAIL: нужен --confirm-live-wing-write {LIVE_WING_CONFIRMATION}", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    STOP_FILE.unlink(missing_ok=True)

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

    devs = sd.query_devices()
    dev = next(i for i, d in enumerate(devs) if d["name"] == "WING" and d["max_input_channels"] >= NUM_CH)

    gate = SafetyGate(SafetyGateConfig.from_mapping(LIVE_GATE_CONFIG))
    spectral = SpectralCeilingEQAnalyzer()
    eq_streak: dict[tuple, int] = {}          # (ch, band_idx, направление) -> подряд циклов
    eq_last_applied: dict[int, float] = {}    # ch -> момент применения EQ
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_path = REPO_ROOT / "reports" / f"soundcheck_daemon_log_{stamp}.jsonl"
    log_f = open(log_path, "a")
    print(f"Журнал: {log_path}")
    print(f"Остановка: Ctrl+C или touch {STOP_FILE}")

    t_start = time.time()
    cycle = 0
    totals = {"applied": 0, "proposed_blocked": 0, "feedback_events": 0}

    while _running:
        if STOP_FILE.exists():
            print(f"\n[STOP] Обнаружен stop-файл {STOP_FILE} — остановка оператором")
            break
        if args.duration and time.time() - t_start >= args.duration:
            print(f"\n[STOP] Достигнут лимит --duration {args.duration:.0f} с")
            break
        if args.max_cycles and cycle >= args.max_cycles:
            print(f"\n[STOP] Достигнут лимит --max-cycles {args.max_cycles}")
            break
        cycle += 1
        t0 = time.time()
        print(f"\n===== ЦИКЛ {cycle} (t+{t0 - t_start:.0f}s) =====")
        audio = sd.rec(int(CAPTURE_S * SR), samplerate=SR, channels=NUM_CH, device=dev, dtype="float32")
        sd.wait()

        active = {}
        for ch in range(1, NUM_CH + 1):
            name = (names.get(ch) or "").strip()
            if EXCLUDE_NAME_RE.search(name):
                continue
            x = audio[:, ch - 1]
            m = measure(x)
            if m["st_max_lufs"] < ACTIVE_ST_LUFS:
                continue
            active[ch] = {"name": name, "role": roles[ch], "meas": m, "x": x}
        if not active:
            print("  нет активных каналов — мониторинг продолжается")
            time.sleep(max(0.0, CYCLE_PERIOD_S - (time.time() - t0)))
            continue

        # ---- FEEDBACK FIRST ----
        cycle_log = {"cycle": cycle, "t": round(t0 - t_start, 1),
                     "active": sorted(active), "feedback": [], "applied": [], "blocked": []}
        feedback_hit = False
        for ch, c in active.items():
            fb = detect_feedback(c["x"])
            if not fb:
                continue
            feedback_hit = True
            totals["feedback_events"] += 1
            cur = readback(client, f"/ch/{ch}/fdr")
            cur = float(cur) if cur is not None else 0.0
            new = max(FADER_FLOOR, cur - 3.0)
            print(f"  [FEEDBACK] ch{ch} {c['name']}: {fb['freq_hz']} Hz "
                  f"(+{fb['prominence_db']} dB) -> fader {cur:.1f} -> {new:.1f}")
            cycle_log["feedback"].append({"ch": ch, **fb, "fader_from": cur, "fader_to": new})
            if not args.dry_run:
                client.set_channel_fader(ch, new)
                time.sleep(0.1)
        if feedback_hit:
            log_f.write(json.dumps(cycle_log, ensure_ascii=False, default=float) + "\n")
            log_f.flush()
            time.sleep(max(0.0, CYCLE_PERIOD_S - (time.time() - t0)))
            continue  # цикл отдан безопасности; коррекции — в следующем

        # ---- инкрементальные решения ----
        decisions = []
        ctx = {}
        # trim: АБСОЛЮТНАЯ цель. USB tap стоит ДО трима (правило CLAUDE.md),
        # замер не отражает изменение трима — относительное приращение
        # превращается в храповик. trim_abs = TARGET - TP_pre идемпотентен.
        for ch, c in active.items():
            cur_trim = readback(client, f"/ch/{ch}/in/set/trim")
            cur_trim = float(cur_trim) if cur_trim is not None else 0.0
            trim_abs = float(np.clip(TRIM_TARGET_DBTP - c["meas"]["true_peak_dbtp"], -18.0, 18.0))
            delta = float(np.clip(trim_abs - cur_trim, -FADER_STEP_MAX, 1.0))
            ctx[f"trim.ch{ch}"] = (ch, cur_trim, float(np.clip(cur_trim + delta, -18.0, 18.0)))
            decisions.append(ActionDecision(
                id=f"trim.ch{ch}", action_type=ACTION_GAIN, target=f"ch{ch}",
                parameters={"gain_db": round(delta, 2)},
                reason=f"TP {c['meas']['true_peak_dbtp']:.1f} dBTP vs цель {TRIM_TARGET_DBTP}",
                confidence=0.9, risk_level="low", safe_to_apply=True))
        # фейдеры: шаг к балансу Ayaic
        raw = {}
        faders = {}
        for ch, c in active.items():
            f = readback(client, f"/ch/{ch}/fdr")
            faders[ch] = float(f) if f is not None else 0.0
            raw[ch] = AYAIC_PLANE.get(c["role"], -27.0) - c["meas"]["est_lufs"]
        shift = max(raw.values()) - FADER_CEIL  # якорь: максимум плана на потолке (не зависит от текущих фейдеров)
        for ch, c in active.items():
            target = float(np.clip(raw[ch] - shift, FADER_FLOOR, FADER_CEIL))
            step = float(np.clip(target - faders[ch], -FADER_STEP_MAX, FADER_STEP_MAX))
            ctx[f"fader.ch{ch}"] = (ch, faders[ch], faders[ch] + step)
            decisions.append(ActionDecision(
                id=f"fader.ch{ch}", action_type=ACTION_GAIN, target=f"ch{ch}.fdr",
                parameters={"gain_db": round(step, 2)},
                reason=f"баланс Ayaic: цель {target:.1f}, текущий {faders[ch]:.1f}",
                confidence=0.85, risk_level="low", safe_to_apply=True))
        # EQ: только устойчивые предложения
        now = time.time()
        for ch, c in active.items():
            if now - eq_last_applied.get(ch, 0.0) < EQ_COOLDOWN_S:
                continue
            prop = spectral.analyze(c["x"].astype(np.float64), instrument_role=c["role"],
                                    sample_rate=SR, track_id=f"ch{ch}", role_confidence=0.9)
            cur_bands = []
            for b in "1234":
                cur_bands.append((float(readback(client, f"/ch/{ch}/eq/{b}f") or 1000.0),
                                  float(readback(client, f"/ch/{ch}/eq/{b}g") or 0.0),
                                  float(readback(client, f"/ch/{ch}/eq/{b}q") or 2.0)))
            merged, _ = merge_spectral_proposal_into_eq_bands(cur_bands, prop)
            for idx, (ob, nb) in enumerate(zip(cur_bands, merged)):
                if ob == nb or ob[1] <= -4.0:
                    continue
                direction = "cut" if nb[1] < ob[1] else "boost"
                key = (ch, idx, direction)
                eq_streak[key] = eq_streak.get(key, 0) + 1
                stale = [k for k in eq_streak if k[0] == ch and k != key and k[1] == idx]
                for k in stale:
                    eq_streak.pop(k, None)
                if eq_streak[key] >= EQ_PERSISTENCE:
                    ctx[f"eq.ch{ch}.b{idx+1}"] = (ch, idx + 1, nb)
                    decisions.append(ActionDecision(
                        id=f"eq.ch{ch}.b{idx+1}", action_type=ACTION_EQ, target=f"ch{ch}",
                        parameters={"frequency_hz": nb[0], "gain_db": round(nb[1] - ob[1], 2), "q": nb[2]},
                        reason=f"spectral ceiling устойчиво {eq_streak[key]} циклов",
                        confidence=float(prop.confidence), risk_level="low", safe_to_apply=True))

        plan = ActionPlan(plan_id=f"daemon_c{cycle}", mode="soundcheck_live", decisions=decisions)
        state = {f"ch{ch}": {"true_peak_dbtp": c["meas"]["true_peak_dbtp"]} for ch, c in active.items()}
        gated = gate.evaluate_plan(plan, current_state=state, live_mode=True, dry_run=args.dry_run)
        allowed = {d.id for d in gated.allowed_plan.decisions}
        n_noop = len(gated.blocked)
        totals["proposed_blocked"] += n_noop

        applied_now = 0
        for d in gated.allowed_plan.decisions:
            if d.id.startswith("trim.") and d.id in ctx:
                ch, cur, new = ctx[d.id]
                if not args.dry_run:
                    client.set_channel_gain(ch, new)
                    time.sleep(0.06)
                print(f"  APPLY {d.id}: trim {cur:+.1f} -> {new:+.1f} ({d.reason})")
                applied_now += 1
            elif d.id.startswith("fader.") and d.id in ctx:
                ch, cur, new = ctx[d.id]
                if not args.dry_run:
                    client.set_channel_fader(ch, new)
                    time.sleep(0.06)
                print(f"  APPLY {d.id}: fader {cur:+.1f} -> {new:+.1f} ({d.reason})")
                applied_now += 1
            elif d.id.startswith("eq.") and d.id in ctx:
                ch, bi, (f, g, q) = ctx[d.id]
                b = str(bi)
                if not args.dry_run:
                    client.set_eq_band_frequency(ch, f"{b}f", float(f))
                    client.set_eq_band_gain(ch, f"{b}g", float(g))
                    client.set_eq_band_q(ch, f"{b}q", float(q))
                    client.set_eq_on(ch, 1)
                    time.sleep(0.1)
                eq_last_applied[ch] = time.time()
                for k in [k for k in eq_streak if k[0] == ch]:
                    eq_streak.pop(k, None)
                print(f"  APPLY {d.id}: {g:+.1f} dB @ {f:.0f} Hz Q{q:.1f} ({d.reason})")
                applied_now += 1
        totals["applied"] += applied_now
        if not applied_now:
            print(f"  стабильно: 0 применено, {n_noop} предложений внутри гистерезиса/лимитов")
        else:
            print(f"  применено {applied_now}, отложено gate'ом {n_noop}")

        cycle_log["applied"] = [d.to_dict() for d in gated.allowed_plan.decisions if d.id in ctx]
        cycle_log["blocked"] = gated.blocked
        log_f.write(json.dumps(cycle_log, ensure_ascii=False, default=float) + "\n")
        log_f.flush()
        time.sleep(max(0.0, CYCLE_PERIOD_S - (time.time() - t0)))

    log_f.close()
    client.disconnect()
    print("\n================ ИТОГ НЕПРЕРЫВНОГО САУНДЧЕКА ================")
    print(f"Циклов: {cycle}; применено коррекций: {totals['applied']}; "
          f"предложено-но-удержано gate'ом: {totals['proposed_blocked']}; "
          f"feedback-событий: {totals['feedback_events']}")
    print(f"Журнал: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
