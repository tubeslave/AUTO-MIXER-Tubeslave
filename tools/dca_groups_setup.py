#!/usr/bin/env python3
"""SETUP DCA-групп WING: инструменты по группам, финальный баланс — DCA.

Директива оператора 2026-07-06. Назначение канала в DCA на WING — тег
`#DN` в `/ch/N/tags` (протокол FW 3.1: «Tags #D1..#D16 are reserved for
DCA1..16 assignment»). Раннер:
  - именует DCA, ставит их фейдеры в 0, снимает мьют;
  - прописывает каналам теги групп (сохраняя прочие теги);
  - rollback-файл до первой записи, readback каждой записи.

Запуск:
    PYTHONPATH=backend python3 tools/dca_groups_setup.py \
        --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
    PYTHONPATH=backend python3 tools/dca_groups_setup.py --rollback <file>
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

LIVE_CONFIRM = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"
REPORTS = REPO_ROOT / "reports"

# Группы этого шоу: имя DCA -> каналы
DCA_GROUPS = {
    1: ("DRUMS", [1, 2, 3, 4, 5, 6, 7]),
    2: ("BASS", [8]),
    3: ("GTRS", [9, 10]),
    4: ("PLAYBK", [11, 12]),
    5: ("KEYS", [13]),
    6: ("VOX", [16]),
}


def main() -> None:
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
        if got != value and not (isinstance(value, float)
                                 and isinstance(got, float)
                                 and abs(got - value) <= 0.3):
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
            print("Откат DCA-групп завершён")
            return
        if args.confirm_live_wing_write != LIVE_CONFIRM:
            raise SystemExit(f"нужно --confirm-live-wing-write {LIVE_CONFIRM}")

        addrs = []
        for dca, (_, members) in DCA_GROUPS.items():
            addrs += [f"/dca/{dca}/name", f"/dca/{dca}/fdr", f"/dca/{dca}/mute"]
            addrs += [f"/ch/{ch}/tags" for ch in members]
        original = {a: rb(a) for a in dict.fromkeys(addrs)}
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rollback_path = REPORTS / f"dca_groups_rollback_{stamp}.json"
        rollback_path.write_text(json.dumps({"original": original},
                                            ensure_ascii=False, indent=1))
        print(f"Rollback-файл: {rollback_path.name}", flush=True)

        journal = {"stamp": stamp, "groups": {}}
        for dca, (name, members) in DCA_GROUPS.items():
            entry = {
                "name": wr(f"/dca/{dca}/name", name),
                "fdr": wr(f"/dca/{dca}/fdr", 0.0),
                "mute": wr(f"/dca/{dca}/mute", 0),
                "members": {},
            }
            for ch in members:
                existing = original.get(f"/ch/{ch}/tags") or ""
                tags = [t for t in existing.split() if not t.startswith("#D")]
                tags.append(f"#D{dca}")
                entry["members"][ch] = wr(f"/ch/{ch}/tags", " ".join(tags))
            journal["groups"][dca] = entry
            print(f"DCA{dca} '{name}': fdr {entry['fdr']} | каналы "
                  f"{ {c: t for c, t in entry['members'].items()} }", flush=True)
        out = REPORTS / f"dca_groups_applied_{stamp}.json"
        out.write_text(json.dumps(journal, ensure_ascii=False, indent=1,
                                  default=str))
        print(f"Журнал: {out.name}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
