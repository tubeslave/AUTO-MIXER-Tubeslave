#!/usr/bin/env python3
"""SETUP-раннер: стем-шины + полный сброс обработки + измерительные тапы.

Директива оператора 2026-07-06:
  1. Сбросить всю обработку на каналах 1-16 и мастере (EQ/gate/comp/
     delay/polarity/trim). Фейдеры, мьюты, пан и FX-посылы не трогаем.
  2. Группы инструментов на шинах 1-6 (DRUMS/BASS/GTRS/PLAYBK/KEYS/VOX),
     каналы шлются в группы посылом в режиме GRP (post-fader subgroup),
     группы — в Main LR; прямой посыл каналов в Main отключается.
  3. USB-выходы 33-44 = шины 1-6 (L/R, post-fader) — стемы для DCA-баланса.
     USB 45-46 = Main LR PRE-fader через USR-сигналы 17/18 — замер микса
     даже при замьюченном мастере.

Токены по WING Remote Protocols V3.1: send mode GRP, /io/in/USR/N/user/
{grp,in,tap,lr} (tap PRE/POST), /io/out/USB/N/{grp,in} (запись 1-базная,
readback 0-базный). Rollback-журнал пишется ДО первой записи.

Последовательность переключения безопасна для ушей: шины строятся
ЗАМЬЮЧЕННЫМИ, каналы снимаются с мастера, потом шины размьючиваются —
короткий провал звука вместо удвоения уровня (+6 дБ).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import yaml  # noqa: E402

from wing_client import WingClient  # noqa: E402

STAMP = time.strftime("%Y%m%dT%H%M%S")
OUT = REPO / "reports"

# группы соответствуют config autofoh.dca_balance / tools/dca_groups_setup.py
STEM_BUSES = {
    1: ("DRUMS", [1, 2, 3, 4, 5, 6, 7]),
    2: ("BASS", [8]),
    3: ("GTRS", [9, 10]),
    4: ("PLAYBK", [11, 12]),
    5: ("KEYS", [13]),
    6: ("VOX", [16]),
}
RESET_CHANNELS = list(range(1, 17))
# USR-слоты 1-16 заняты операторскими post-fader тапами каналов
USR_MAIN_PRE = {17: ("MAIN PRE L", "L"), 18: ("MAIN PRE R", "R")}

THROTTLE_GAP_S = 0.15


def main() -> int:
    raise RuntimeError(
        "legacy_live_mutation_surface_quarantined: broad routing rewrites are disabled"
    )
    mixer = yaml.safe_load(open(REPO / "config" / "automixer.yaml"))["mixer"]
    c = WingClient(ip=mixer["ip"], port=int(mixer.get("receive_port", 2223)))
    assert c.connect(timeout=5.0), "пульт не отвечает"
    c.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(0.3)

    def rb(addr, timeout=0.8):
        r = c.query_parameter(addr, timeout=timeout)
        if r["status"] != "received":
            r = c.query_parameter(addr, timeout=timeout)
        return r["value"] if r["status"] == "received" else None

    # ── план записей (адрес, значение) в порядке применения ──
    writes = []

    # 1. сброс обработки каналов 1-16
    for ch in RESET_CHANNELS:
        p = f"/ch/{ch}"
        writes += [(f"{p}/eq/on", 0)]
        writes += [(f"{p}/eq/{b}g", 0.0) for b in range(1, 7)]
        writes += [
            (f"{p}/gate/on", 0),
            (f"{p}/dyn/on", 0),
            (f"{p}/dyn/gain", 0.0),
            (f"{p}/flt/lc", 0),
            (f"{p}/flt/hc", 0),
            (f"{p}/in/set/dly", 0.0),
            (f"{p}/in/set/inv", 0),
            (f"{p}/in/set/trim", 0.0),
        ]
    # мастер: EQ (6 полос + шельфы pink-баланса) + динамика
    writes += [("/main/1/eq/on", 0), ("/main/1/dyn/on", 0)]
    writes += [(f"/main/1/eq/{b}g", 0.0) for b in range(1, 7)]
    writes += [("/main/1/eq/lg", 0.0), ("/main/1/eq/hg", 0.0)]

    # 2. шины-группы: имя, фейдер 0, ЗАМЬЮЧЕНЫ, в Main LR
    for bus, (name, _members) in STEM_BUSES.items():
        p = f"/bus/{bus}"
        writes += [
            (f"{p}/name", name),
            (f"{p}/fdr", 0.0),
            (f"{p}/mute", 1),
            (f"{p}/main/1/on", 1),
            (f"{p}/main/1/lvl", 0.0),
        ]

    # 3. посылы канал -> шина в режиме GRP (post-fader subgroup)
    for bus, (_name, members) in STEM_BUSES.items():
        for ch in members:
            p = f"/ch/{ch}/send/{bus}"
            writes += [
                (f"{p}/mode", "GRP"),
                (f"{p}/lvl", 0.0),
                (f"{p}/plink", 1),
                (f"{p}/on", 1),
            ]

    # 4. переключение: каналы групп уходят с прямого Main
    grouped = sorted({ch for _n, m in STEM_BUSES.values() for ch in m})
    writes += [(f"/ch/{ch}/main/1/on", 0) for ch in grouped]
    # ...и группы открываются
    writes += [(f"/bus/{bus}/mute", 0) for bus in STEM_BUSES]

    # 5. USR 17/18 = Main LR pre-fader
    for usr, (name, lr) in USR_MAIN_PRE.items():
        p = f"/io/in/USR/{usr}"
        writes += [
            (f"{p}/name", name),
            (f"{p}/user/grp", "MAIN"),
            (f"{p}/user/in", 1),
            (f"{p}/user/tap", "PRE"),
            (f"{p}/user/lr", lr),
        ]

    # 6. USB-выходы: 33-44 = шины 1-6 L/R, 45/46 = USR 17/18
    for i, bus in enumerate(sorted(STEM_BUSES)):
        for side in (0, 1):  # L, R
            n = 33 + i * 2 + side
            writes += [
                (f"/io/out/USB/{n}/grp", "BUS"),
                (f"/io/out/USB/{n}/in", bus * 2 - 1 + side),  # 1-базная
            ]
    for n, usr in ((45, 17), (46, 18)):
        writes += [
            (f"/io/out/USB/{n}/grp", "USR"),
            (f"/io/out/USB/{n}/in", usr),  # 1-базная
        ]

    # ── rollback-журнал ДО записей ──
    print(f"[pre-read] {len(writes)} записей, читаю прежние значения...",
          flush=True)
    rollback = {}
    for addr, _val in writes:
        if addr not in rollback:
            rollback[addr] = rb(addr)
    rb_path = OUT / f"stem_bus_routing_rollback_{STAMP}.json"
    rb_path.write_text(json.dumps(rollback, ensure_ascii=False, indent=1,
                                  default=str))
    print(f"[rollback] {rb_path.name}", flush=True)

    # ── применение ──
    applied, failed = [], []
    for addr, val in writes:
        ok = c.send(addr, val)
        if ok is False:  # OSC-троттл: ретрай с паузой больше интервала
            time.sleep(THROTTLE_GAP_S)
            ok = c.send(addr, val)
        time.sleep(0.02)
        (applied if ok is not False else failed).append((addr, val))
    print(f"[apply] отправлено {len(applied)}, отказов {len(failed)}",
          flush=True)

    # ── readback-верификация ──
    time.sleep(0.5)
    mismatches = []
    verify = {}
    for addr, val in writes:
        got = rb(addr)
        verify[addr] = {"want": val, "got": got}
        if got is None:
            mismatches.append((addr, val, None))
            continue
        if isinstance(val, float) or isinstance(got, float):
            try:
                if abs(float(got) - float(val)) > 0.11:
                    mismatches.append((addr, val, got))
            except (TypeError, ValueError):
                mismatches.append((addr, val, got))
        elif addr.endswith("/in") and "/io/out/" in addr:
            # запись 1-базная, readback 0-базный
            if got not in (val, val - 1):
                mismatches.append((addr, val, got))
        elif str(got) != str(val):
            mismatches.append((addr, val, got))

    journal = {
        "stamp": STAMP,
        "writes": [{"addr": a, "value": v} for a, v in writes],
        "failed_sends": [{"addr": a, "value": v} for a, v in failed],
        "verify": verify,
        "mismatches": [
            {"addr": a, "want": w, "got": g} for a, w, g in mismatches],
    }
    j_path = OUT / f"stem_bus_routing_applied_{STAMP}.json"
    j_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1,
                                 default=str))
    c.disconnect()
    print(f"[journal] {j_path.name}", flush=True)
    print(f"[verify] несовпадений: {len(mismatches)}", flush=True)
    for a, w, g in mismatches[:25]:
        print(f"  MISMATCH {a}: want={w} got={g}", flush=True)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
