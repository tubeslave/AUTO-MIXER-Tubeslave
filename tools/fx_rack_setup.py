#!/usr/bin/env python3
"""SETUP-раннер FX-рэка WING: план AutoFXPlanner -> консоль, с readback.

Операторская SETUP-фаза (вне песенной автоматики, доктрина fx_v3/pa_v3):
  - премиум-модели ТОЛЬКО в слоты 1-8 (правило прошивки 2026-07-05);
  - каждая запись: write -> readback; слот с несовпавшим readback
    откатывается в NONE;
  - rollback-файл пишется ДО первой записи;
  - шины плана: 13 Vocal Plate (FX1), 14 Drum Room (FX2),
    15 Tempo Delay (FX3); insert: /bus/N/preins/ins=FXn + preins/on=1
    (FX-возврат работает только через insert шины — правило CLAUDE.md);
  - выход шины в мейн включается, фейдер шины = return_level_db плана.

Запуск:
    PYTHONPATH=backend python3 tools/fx_rack_setup.py \
        --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
    PYTHONPATH=backend python3 tools/fx_rack_setup.py --rollback <file.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import yaml  # noqa: E402

from wing_client import WingClient  # noqa: E402
from auto_fx import AutoFXPlanner  # noqa: E402

LIVE_CONFIRM = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
REPORTS = REPO_ROOT / "reports"

# Карты токенов, подтверждённые readback'ом на реальном пульте 2026-07-05
REVERB_TOKENS = ("pdel", "dcy", "damp", "lc", "hc")
DELAY_TOKENS = ("time", "feed", "flc", "fhc", "mode")


def connect() -> WingClient:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: FX rack writes are disabled"
    )
    mixer = yaml.safe_load(open(REPO_ROOT / "config" / "automixer.yaml"))["mixer"]
    client = WingClient(ip=mixer["ip"], port=int(mixer.get("receive_port", 2223)))
    if not client.connect(timeout=5.0):
        raise SystemExit("пульт не отвечает")
    client.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(0.3)
    return client


def make_rw(client: WingClient):
    def rb(addr: str, timeout: float = 0.9):
        r = client.query_parameter(addr, timeout=timeout)
        if r["status"] != "received":
            r = client.query_parameter(addr, timeout=timeout)
        return r["value"] if r["status"] == "received" else None

    def wr(addr: str, value, settle: float = 0.15):
        client.send(addr, value)
        time.sleep(settle)
        got = rb(addr)
        if got is None or (isinstance(got, float) and isinstance(value, (int, float))
                           and abs(got - float(value)) > 0.5):
            time.sleep(0.25)  # ретрай поверх интервала OSC-троттлера
            client.send(addr, value)
            time.sleep(settle)
            got = rb(addr)
        return got

    return rb, wr


def snapshot_targets(rb, decisions) -> dict:
    snap: dict = {"fx": {}, "buses": {}}
    for d in decisions:
        slot = int(d.fx_slot.replace("FX", ""))
        snap["fx"][slot] = {"mdl": rb(f"/fx/{slot}/mdl")}
        snap["buses"][d.bus_id] = {
            "preins/ins": rb(f"/bus/{d.bus_id}/preins/ins"),
            "preins/on": rb(f"/bus/{d.bus_id}/preins/on"),
            "fdr": rb(f"/bus/{d.bus_id}/fdr"),
            "mute": rb(f"/bus/{d.bus_id}/mute"),
            "name": rb(f"/bus/{d.bus_id}/name"),
            "main/1/on": rb(f"/bus/{d.bus_id}/main/1/on"),
        }
    return snap


def apply_plan(client: WingClient, decisions, journal: dict) -> None:
    rb, wr = make_rw(client)
    for d in decisions:
        slot = int(d.fx_slot.replace("FX", ""))
        if slot not in range(1, 9):
            journal.setdefault("skipped", []).append(
                {"slot": slot, "reason": "premium_models_only_slots_1_8"})
            continue
        entry: dict = {"slot": slot, "bus": d.bus_id, "model": d.model, "writes": {}}

        got_mdl = wr(f"/fx/{slot}/mdl", d.model, settle=0.8)
        entry["writes"]["mdl"] = got_mdl
        if got_mdl != d.model:
            # тихий отказ прошивки — откат слота, шина не трогается
            wr(f"/fx/{slot}/mdl", "NONE", settle=0.5)
            entry["verdict"] = "model_rejected_slot_rolled_back"
            journal["decisions"].append(entry)
            continue

        if d.fx_type == "reverb":
            params = {
                "pdel": float(d.params.get("predelay_ms", 30.0)),
                "dcy": float(d.params.get("decay_s", 1.4)),
                "lc": float(d.hpf_hz or 100.0),
                "hc": float(d.lpf_hz or 8000.0),
            }
        else:  # delay
            params = {
                "time": float(d.params.get("left_delay_ms", 300.0)),
                "feed": float(d.params.get("feedback", 0.2)) * 100.0,
                "flc": float(d.hpf_hz or 200.0),
                "fhc": float(d.lpf_hz or 6000.0),
            }
        for tok, value in params.items():
            entry["writes"][tok] = wr(f"/fx/{slot}/{tok}", round(value, 2))

        entry["writes"]["preins/ins"] = wr(f"/bus/{d.bus_id}/preins/ins", f"FX{slot}")
        entry["writes"]["preins/on"] = wr(f"/bus/{d.bus_id}/preins/on", 1)
        entry["writes"]["bus_name"] = wr(f"/bus/{d.bus_id}/name", d.name)
        entry["writes"]["bus_main_on"] = wr(f"/bus/{d.bus_id}/main/1/on", 1)
        entry["writes"]["bus_fdr"] = wr(f"/bus/{d.bus_id}/fdr",
                                        float(d.return_level_db))
        if d.fx_type == "delay":
            # темпо-дилей вне сетки трека читается как «второй мультитрек»
            # (live 2026-07-06): шина ставится ЗАМЬЮЧЕННОЙ, оператор
            # снимает мьют после подтверждения темпа (или ре-синк fx_v3)
            entry["writes"]["bus_mute"] = wr(f"/bus/{d.bus_id}/mute", 1)
            entry["delay_muted_until_tempo_confirmed"] = True
        entry["verdict"] = ("applied" if entry["writes"]["preins/on"] == 1
                            else "insert_not_confirmed")
        journal["decisions"].append(entry)
        print(f"FX{slot} {d.model} -> bus {d.bus_id} '{d.name}': "
              f"{entry['verdict']}; params={ {k: v for k, v in entry['writes'].items()} }",
              flush=True)


def rollback(client: WingClient, snap: dict) -> None:
    rb, wr = make_rw(client)
    for slot, vals in snap.get("fx", {}).items():
        wr(f"/fx/{int(slot)}/mdl", vals.get("mdl") or "NONE", settle=0.6)
    for bus, vals in snap.get("buses", {}).items():
        bus = int(bus)
        for key in ("preins/ins", "preins/on", "fdr", "mute", "name", "main/1/on"):
            if vals.get(key) is not None:
                wr(f"/bus/{bus}/{key}", vals[key])
    print("Откат завершён", flush=True)


def main() -> None:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: FX rack writes are disabled"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-live-wing-write", default="")
    parser.add_argument("--rollback", default="",
                        help="путь к rollback-файлу для отката")
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--venue-rt60-s", type=float, default=None)
    args = parser.parse_args()

    client = connect()
    try:
        if args.rollback:
            snap = json.loads(Path(args.rollback).read_text())["original"]
            rollback(client, snap)
            return
        if args.confirm_live_wing_write != LIVE_CONFIRM:
            raise SystemExit(
                f"нужно --confirm-live-wing-write {LIVE_CONFIRM}")

        planner = AutoFXPlanner(tempo_bpm=args.tempo_bpm, vocal_priority=True,
                                venue_rt60_s=args.venue_rt60_s)
        decisions = [d for d in planner._default_buses()
                     if d.fx_type in ("reverb", "delay")][:3]

        rb, _ = make_rw(client)
        original = snapshot_targets(rb, decisions)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rollback_path = REPORTS / f"fx_rack_setup_rollback_{stamp}.json"
        rollback_path.write_text(json.dumps(
            {"original": original}, ensure_ascii=False, indent=1, default=str))
        print(f"Rollback-файл: {rollback_path.name}", flush=True)

        journal = {"stamp": stamp, "decisions": []}
        apply_plan(client, decisions, journal)
        out = REPORTS / f"fx_rack_setup_applied_{stamp}.json"
        out.write_text(json.dumps(journal, ensure_ascii=False, indent=1,
                                  default=str))
        applied = sum(1 for e in journal["decisions"]
                      if e.get("verdict") == "applied")
        print(f"ИТОГ: {applied}/{len(journal['decisions'])} FX-шин поставлено. "
              f"Журнал: {out.name}", flush=True)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
