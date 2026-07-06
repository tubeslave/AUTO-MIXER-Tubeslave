#!/usr/bin/env python3
"""Проба правила звукорежиссуры на реальном пульте + счёт обучения.

Пайплайн обучения (директива оператора 2026-07-06):
  собрать правило из источников -> ПОПРОБОВАТЬ на реальном WING bounded
  -> оценить мастер-петлёй до/после -> вердикт оператора ушами ->
  счёт helped/hurt -> провалившиеся правила понижаются, победившие
  промоутятся в active.

Применение:
  # проба (журнал + rollback-файл до первой записи):
  PYTHONPATH=backend python tools/mixing_rule_trial.py \
      --rule eq.tonal_targets.kick --channel 1

  # откат последней пробы правила:
  ... --rule eq.tonal_targets.kick --channel 1 --rollback

  # операторский вердикт (обновляет счёт и статус правила):
  ... --rule eq.tonal_targets.kick --verdict helped --note "kick читается"

Только tonal-target правила EQ (action_type=eq_tonal_target). Boost
консервативен (<= +2 дБ), свободная полоса обязательна; занятые полосы
(резонанс-нотчи и ручные) не перестраиваются, регион пропускается.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

TRIALS_PATH = REPO / "backend/source_knowledge/data/rule_trials.jsonl"
RULES_PATH = REPO / "backend/source_knowledge/data/rules.jsonl"
BANDS = ("1", "2", "3", "4")
BOOST_CAP_DB = 2.0
CUT_CAP_DB = -3.0
CAPTURE_S = 8.0
PROMOTE_AFTER_HELPED = 2
DEMOTE_AFTER_HURT = 2


def load_rule(rule_id: str):
    from source_knowledge.store import SourceKnowledgeStore
    st = SourceKnowledgeStore()
    rule = st.get_rule(rule_id)
    assert rule is not None, f"правило {rule_id} не найдено"
    return rule


def capture_master(seconds: float = CAPTURE_S):
    import sounddevice as sd
    dev = None
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == "WING" and d["max_input_channels"] >= 48:
            dev = i
            break
    assert dev is not None, "устройство WING не найдено"
    sr = 48000
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=48,
                 device=dev, dtype="float32")
    sd.wait()
    return rec[:, 46], rec[:, 47], sr   # USB 47/48


def evaluate(left, right, sr):
    from master_mix_eval import evaluate_master_mix
    return evaluate_master_mix(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64), sr,
        target_curve="pink", target_lufs=-20.0).to_dict()


def run_trial(rule_id: str, channel: int) -> int:
    rule = load_rule(rule_id)
    templates = [t for t in rule.action_templates
                 if t.get("action_type") == "eq_tonal_target"]
    assert templates, "правило не содержит eq_tonal_target"
    regions = templates[0].get("parameters", {}).get("regions", [])

    from wing_client import WingClient
    cfg = yaml.safe_load(open(REPO / "config/automixer.yaml"))["mixer"]
    c = WingClient(ip=cfg["ip"], port=int(cfg.get("receive_port", 2223)))
    assert c.connect(timeout=5.0), "пульт не отвечает"
    c.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(0.3)

    def rb(addr, timeout=0.9):
        r = c.query_parameter(addr, timeout=timeout)
        if r["status"] != "received":
            r = c.query_parameter(addr, timeout=timeout)
        return r["value"] if r["status"] == "received" else None

    stamp = time.strftime("%Y%m%dT%H%M%S")
    p = f"/ch/{channel}"
    eq_addrs = [f"{p}/eq/on"] + [f"{p}/eq/{b}{t}" for b in BANDS
                                 for t in ("g", "f", "q")]
    eq_addrs += [f"{p}/flt/lc", f"{p}/flt/lcf"]
    before_state = {a: rb(a) for a in eq_addrs}
    rb_path = REPO / "reports" / f"rule_trial_rollback_{rule_id}_{stamp}.json"
    rb_path.write_text(json.dumps(before_state, ensure_ascii=False, indent=1,
                                  default=str))
    print(f"[rollback] {rb_path.name}", flush=True)

    # план: занятые полосы не трогаем
    free = [b for b in BANDS
            if abs(float(before_state.get(f"{p}/eq/{b}g") or 0.0)) < 0.1]
    used = set()
    writes, skipped = [], []
    for reg in regions:
        purpose = reg.get("purpose", "?")
        lo, hi = float(reg["hz"][0]), float(reg["hz"][1])
        center = math.sqrt(lo * hi)
        if purpose == "hpf":
            writes += [(f"{p}/flt/lc", 1), (f"{p}/flt/lcf", center)]
            continue
        g_lo, g_hi = float(reg["gain_db"][0]), float(reg["gain_db"][1])
        gain = (max(CUT_CAP_DB, g_lo) if g_hi <= 0.0
                else min(BOOST_CAP_DB, g_hi))
        # полоса с частотой уже в регионе — перенастраиваем её же;
        # уже назначенные этой пробой полосы недоступны (баг первой
        # пробы: box_cut перехватил полосу, отданную body)
        band = next((b for b in BANDS
                     if b not in used
                     and lo <= float(before_state.get(f"{p}/eq/{b}f") or 0)
                     <= hi), None)
        if band is None:
            free = [b for b in free if b not in used]
            if not free:
                skipped.append(purpose)
                continue
            band = free.pop(0)
        elif band in free:
            free.remove(band)
        used.add(band)
        writes += [(f"{p}/eq/{band}f", center), (f"{p}/eq/{band}g", gain),
                   (f"{p}/eq/{band}q", 1.6 if gain < 0 else 1.2)]
    writes.append((f"{p}/eq/on", 1))

    print(f"[before] захват мастера {CAPTURE_S:.0f} c...", flush=True)
    l0, r0, sr = capture_master()
    verdict_before = evaluate(l0, r0, sr)

    applied, mismatch = [], []
    for addr, val in writes:
        ok = c.send(addr, val)
        if ok is False:
            time.sleep(0.15)
            c.send(addr, val)
        time.sleep(0.03)
        applied.append({"addr": addr, "value": val})
    time.sleep(0.5)
    for addr, val in writes:
        got = rb(addr)
        if got is None or (isinstance(val, float)
                           and abs(float(got) - val) > max(0.11, val * 0.02)):
            mismatch.append({"addr": addr, "want": val, "got": got})

    print(f"[after] захват мастера {CAPTURE_S:.0f} c...", flush=True)
    l1, r1, sr = capture_master()
    verdict_after = evaluate(l1, r1, sr)
    c.disconnect()

    record = {
        "stamp": stamp, "rule_id": rule_id, "channel": channel,
        "writes": applied, "readback_mismatches": mismatch,
        "skipped_regions": skipped, "rollback_file": rb_path.name,
        "verdict_before": verdict_before, "verdict_after": verdict_after,
        "delta": {k: round(float(verdict_after.get(k) or 0)
                           - float(verdict_before.get(k) or 0), 2)
                  for k in ("lufs_integrated", "tilt_deviation_db",
                            "vocal_presence_db", "true_peak_dbtp")},
        "operator_verdict": None,
    }
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRIALS_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: record[k] for k in
                      ("skipped_regions", "readback_mismatches", "delta")},
                     ensure_ascii=False, indent=1), flush=True)
    print(f"[trial] записана проба {rule_id} ch{channel}; вердикт оператора: "
          f"--verdict helped|hurt|neutral", flush=True)
    return 0 if not mismatch else 1


def rollback_trial(rule_id: str, channel: int) -> int:
    files = sorted((REPO / "reports").glob(
        f"rule_trial_rollback_{rule_id}_*.json"))
    assert files, "rollback-файл не найден"
    state = json.loads(files[-1].read_text())
    from wing_client import WingClient
    cfg = yaml.safe_load(open(REPO / "config/automixer.yaml"))["mixer"]
    c = WingClient(ip=cfg["ip"], port=int(cfg.get("receive_port", 2223)))
    assert c.connect(timeout=5.0)
    time.sleep(0.3)
    for addr, val in state.items():
        if val is None:
            continue
        ok = c.send(addr, val)
        if ok is False:
            time.sleep(0.15)
            c.send(addr, val)
        time.sleep(0.03)
    c.disconnect()
    print(f"[rollback] восстановлен {files[-1].name}", flush=True)
    return 0


def apply_verdict(rule_id: str, verdict: str, note: str) -> int:
    assert verdict in ("helped", "hurt", "neutral")
    # вердикт в последнюю пробу правила
    lines = TRIALS_PATH.read_text().splitlines() if TRIALS_PATH.exists() else []
    records = [json.loads(l) for l in lines if l.strip()]
    target = None
    for r in reversed(records):
        if r["rule_id"] == rule_id:
            target = r
            break
    assert target is not None, "проб этого правила ещё не было"
    target["operator_verdict"] = verdict
    target["operator_note"] = note
    TRIALS_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str)
                  for r in records) + "\n")
    # счёт и статус правила
    helped = sum(1 for r in records
                 if r["rule_id"] == rule_id
                 and r.get("operator_verdict") == "helped")
    hurt = sum(1 for r in records
               if r["rule_id"] == rule_id
               and r.get("operator_verdict") == "hurt")
    rules = [json.loads(l) for l in RULES_PATH.read_text().splitlines()
             if l.strip()]
    for rule in rules:
        if rule["rule_id"] != rule_id:
            continue
        delta = {"helped": 0.05, "hurt": -0.1, "neutral": 0.0}[verdict]
        rule["confidence"] = round(
            min(0.95, max(0.1, rule.get("confidence", 0.5) + delta)), 2)
        if hurt >= DEMOTE_AFTER_HURT:
            rule["status"] = "rejected"
        elif helped >= PROMOTE_AFTER_HELPED:
            rule["status"] = "active"
            rule["mode"] = "advisory"
        print(f"[score] {rule_id}: helped={helped} hurt={hurt} "
              f"confidence={rule['confidence']} status={rule['status']}",
              flush=True)
    RULES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rules) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--verdict", choices=["helped", "hurt", "neutral"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.verdict:
        return apply_verdict(args.rule, args.verdict, args.note)
    assert args.channel > 0, "--channel обязателен для пробы/отката"
    if args.rollback:
        return rollback_trial(args.rule, args.channel)
    return run_trial(args.rule, args.channel)


if __name__ == "__main__":
    raise SystemExit(main())
