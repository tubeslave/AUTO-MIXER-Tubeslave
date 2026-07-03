#!/usr/bin/env python3
"""Живой тест применения ВСЕХ методов коррекции на реальный пульт WING.

Без заглушек: пишет напрямую через WingClient (OSC) и проверяет каждое
значение физическим readback'ом с пульта. Схема безопасности:

1. Канал на время теста мьютируется (исходный mute восстанавливается).
2. Перед каждой записью исходное значение сохраняется.
3. После прогона ВСЕ параметры восстанавливаются и восстановление
   тоже проверяется readback'ом.

Методы коррекции, покрываемые тестом (по одному на подсистему автомикса):
  fader          — auto_fader (пост-фейдерная коррекция баланса)
  trim           — gain staging (входной трим)
  eq_band        — auto_eq / feedback notch (параметрическая полоса: gain+freq+Q)
  low_cut        — auto_eq HPF
  compressor     — auto_compressor (threshold/ratio/attack/release/makeup)
  pan            — auto_panner
  phase_invert   — auto_phase
  delay          — phase alignment (входная задержка)

Запуск:
    python3 tools/live_wing_correction_test.py \
      --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
    python3 tools/live_wing_correction_test.py --channel 40 \
      --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
    python3 tools/live_wing_correction_test.py --keep-mute \
      --confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import yaml  # noqa: E402

from wing_client import WingClient  # noqa: E402

READBACK_TIMEOUT = 1.5
SETTLE_S = 0.15
TOL_DB = 0.11          # допуск для dB-параметров
TOL_FREQ_REL = 0.06    # относительный допуск для частот (WING квантует)
TOL_Q_REL = 0.12
LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"


def load_mixer_config() -> dict:
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "automixer.yaml"))
    return cfg["mixer"]


def readback(client: WingClient, address: str):
    res = client.query_parameter(address, timeout=READBACK_TIMEOUT)
    if res["status"] == "received":
        return res["value"]
    # одна повторная попытка — UDP
    res = client.query_parameter(address, timeout=READBACK_TIMEOUT)
    return res["value"] if res["status"] == "received" else None


def close_enough(expected, actual, kind: str) -> bool:
    if actual is None:
        return False
    if kind == "int":
        try:
            return int(round(float(actual))) == int(expected)
        except (TypeError, ValueError):
            return str(actual).strip() == str(expected)
    if kind == "str":
        try:
            return abs(float(actual) - float(expected)) < 1e-6 or str(actual).strip() == str(expected)
        except (TypeError, ValueError):
            return str(actual).strip() == str(expected)
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    if kind == "db" or kind == "ms":
        return abs(a - e) <= TOL_DB if kind == "db" else abs(a - e) <= max(0.5, 0.05 * abs(e))
    if kind == "freq":
        return abs(a - e) <= max(2.0, TOL_FREQ_REL * abs(e))
    if kind == "q":
        return abs(a - e) <= max(0.05, TOL_Q_REL * abs(e))
    if kind == "pan":
        return abs(a - e) <= 2.0
    return abs(a - e) <= 0.11


def pick_safe_channel(client: WingClient, explicit: int | None) -> tuple[int, str]:
    if explicit is not None:
        return explicit, "указан оператором"
    names = client.get_all_channel_names(num_channels=40)
    candidates = []
    for ch in range(40, 0, -1):
        name = (names.get(ch) or "").strip().lower()
        mute = client.query(f"/ch/{ch}/mute")
        fdr = client.query(f"/ch/{ch}/fdr")
        unnamed = name in ("", f"ch {ch}", f"ch{ch}")
        spare = any(t in name for t in ("test", "spare", "free", "-", "тест"))
        muted = mute in (1, 1.0, "1")
        fader_down = isinstance(fdr, (int, float)) and fdr <= -60.0
        score = (2 if (unnamed or spare) else 0) + (1 if muted else 0) + (1 if fader_down else 0)
        candidates.append((score, ch, name, muted, fdr))
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    score, ch, name, muted, fdr = candidates[0]
    why = f"автовыбор: имя={name!r}, mute={muted}, fader={fdr}, счёт безопасности={score}/4"
    return ch, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=None, help="номер канала 1..40")
    ap.add_argument("--keep-mute", action="store_true", help="оставить канал замьюченным после теста")
    ap.add_argument(
        "--no-restore",
        action="store_true",
        help="НЕ восстанавливать исходное состояние: коррекции остаются на пульте, "
             "исходные значения сохраняются в reports/ для ручного отката",
    )
    ap.add_argument(
        "--unmute-after",
        action="store_true",
        help="вместе с --no-restore: снять safety-мьют в конце, оставив коррекции слышимыми",
    )
    ap.add_argument(
        "--confirm-live-wing-write",
        default="",
        help=f"обязательная фраза подтверждения: {LIVE_WING_CONFIRMATION}",
    )
    args = ap.parse_args()

    if args.confirm_live_wing_write != LIVE_WING_CONFIRMATION:
        print(
            "FAIL: этот скрипт пишет широкий набор параметров в живой WING. "
            f"Повтори с --confirm-live-wing-write {LIVE_WING_CONFIRMATION}",
            file=sys.stderr,
        )
        return 2

    mixer = load_mixer_config()
    print(f"Подключение к WING {mixer['ip']} (OSC {mixer.get('receive_port', 2223)})...")
    client = WingClient(ip=mixer["ip"], port=int(mixer.get("receive_port", 2223)))
    if not client.connect(timeout=5.0):
        print("FAIL: пульт не ответил на handshake WING?/OSC")
        return 2
    # тест обходит все программные заглушки, но троттлинг OSC оставляем мягким
    client.set_osc_throttle(enabled=True, hz=50.0)
    time.sleep(1.0)  # дать скану состояния прогреться

    ch, why = pick_safe_channel(client, args.channel)
    print(f"Тестовый канал: {ch} ({why})")

    C = f"/ch/{ch}"
    # (метод, адрес, тестовое значение, вид сравнения, setter)
    plan = [
        ("mute (safety)",        f"{C}/mute",            1,       "int",  lambda v: client.set_channel_mute(ch, v)),
        ("fader [auto_fader]",   f"{C}/fdr",             -20.0,   "db",   lambda v: client.set_channel_fader(ch, v)),
        ("trim [gain staging]",  f"{C}/in/set/trim",     2.0,     "db",   lambda v: client.set_channel_gain(ch, v)),
        ("eq on [auto_eq]",      f"{C}/eq/on",           1,       "int",  lambda v: client.set_eq_on(ch, v)),
        ("eq b3 gain [notch]",   f"{C}/eq/3g",           -6.0,    "db",   lambda v: client.set_eq_band_gain(ch, "3g", v)),
        ("eq b3 freq [notch]",   f"{C}/eq/3f",           3150.0,  "freq", lambda v: client.set_eq_band_frequency(ch, "3f", v)),
        ("eq b3 Q [notch]",      f"{C}/eq/3q",           8.0,     "q",    lambda v: client.set_eq_band_q(ch, "3q", v)),
        ("low cut on [auto_eq]", f"{C}/flt/lc",          1,       "int",  lambda v: client.send(f"{C}/flt/lc", v)),
        ("low cut freq",         f"{C}/flt/lcf",         80.0,    "freq", lambda v: client.send(f"{C}/flt/lcf", v)),
        ("comp on [auto_comp]",  f"{C}/dyn/on",          1,       "int",  lambda v: client.set_compressor_on(ch, v)),
        ("comp threshold",       f"{C}/dyn/thr",         -18.0,   "db",   lambda v: client.set_compressor_threshold(ch, v)),
        ("comp ratio",           f"{C}/dyn/ratio",       "3.0",   "str",  lambda v: client.set_compressor_ratio(ch, v)),
        ("comp attack",          f"{C}/dyn/att",         30.0,    "ms",   lambda v: client.set_compressor_attack(ch, v)),
        ("comp release",         f"{C}/dyn/rel",         200.0,   "ms",   lambda v: client.set_compressor_release(ch, v)),
        ("comp makeup gain",     f"{C}/dyn/gain",        2.0,     "db",   lambda v: client.set_compressor_gain(ch, v)),
        ("pan [auto_panner]",    f"{C}/pan",             10.0,    "pan",  lambda v: client.set_channel_pan(ch, v)),
        ("phase inv [auto_phase]", f"{C}/in/set/inv",    1,       "int",  lambda v: client.set_channel_phase_invert(ch, v)),
        ("delay on [phase align]", f"{C}/in/set/dlyon",  1,       "int",  lambda v: client.send(f"{C}/in/set/dlyon", v)),
        ("delay value ms",       f"{C}/in/set/dly",      1.5,     "ms",   lambda v: client.send(f"{C}/in/set/dly", v)),
    ]

    # 1. Снять исходное состояние
    original: dict[str, object] = {}
    for _, addr, _, _, _ in plan:
        original[addr] = readback(client, addr)
    print(f"Исходное состояние снято ({sum(v is not None for v in original.values())}/{len(plan)} параметров прочитано)")

    # 2. Применить и проверить
    results = []
    for name, addr, value, kind, setter in plan:
        setter(value)
        time.sleep(SETTLE_S)
        actual = readback(client, addr)
        ok = close_enough(value, actual, kind)
        results.append((name, addr, value, actual, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {addr:22s} -> {value!r:>8}  readback={actual!r}")

    # 3. Восстановить (или сохранить откат и выйти, если --no-restore)
    restore_fail = []
    if args.no_restore:
        import json
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rollback_path = reports_dir / f"live_wing_correction_original_ch{ch}_{stamp}.json"
        rollback_path.write_text(json.dumps(
            {"channel": ch, "mixer_ip": mixer["ip"], "saved_at": stamp,
             "original": {a: v for a, v in original.items()}},
            ensure_ascii=False, indent=2))
        print(f"Восстановление ПРОПУЩЕНО (--no-restore). Откат сохранён: {rollback_path}")
        if args.unmute_after:
            orig_mute = original.get(f"{C}/mute")
            client.set_channel_mute(ch, int(orig_mute) if orig_mute is not None else 0)
            time.sleep(SETTLE_S)
            print(f"Safety-мьют снят: /ch/{ch}/mute -> {readback(client, f'{C}/mute')!r}")
        client.disconnect()
        applied = sum(1 for *_, ok in results if ok)
        print("\n================ ИТОГ ================")
        print(f"Канал: {ch}. Применено и подтверждено readback'ом: {applied}/{len(results)}")
        print("Коррекции ОСТАВЛЕНЫ на пульте.")
        for name, addr, value, actual, ok in results:
            if not ok:
                print(f"  НЕ ПОДТВЕРЖДЕНО: {name} ({addr}): послано {value!r}, readback {actual!r}")
        return 0 if applied == len(results) else 1

    print("Восстановление исходного состояния...")
    for name, addr, _, kind, setter in reversed(plan):
        orig = original.get(addr)
        if addr.endswith("/mute") and args.keep_mute:
            continue
        if orig is None:
            restore_fail.append((name, addr, "исходное значение не было прочитано"))
            continue
        try:
            setter(orig if not isinstance(orig, str) else orig)
        except Exception as exc:  # ratio может вернуться числом
            client.send(addr, orig)
        time.sleep(SETTLE_S)
        actual = readback(client, addr)
        if not close_enough(orig, actual, kind):
            restore_fail.append((name, addr, f"ожидалось {orig!r}, readback {actual!r}"))

    client.disconnect() if hasattr(client, "disconnect") else None

    # 4. Итог
    applied = sum(1 for *_, ok in results if ok)
    print("\n================ ИТОГ ================")
    print(f"Канал: {ch}. Применено и подтверждено readback'ом: {applied}/{len(results)}")
    for name, addr, value, actual, ok in results:
        if not ok:
            print(f"  НЕ ПОДТВЕРЖДЕНО: {name} ({addr}): послано {value!r}, readback {actual!r}")
    if restore_fail:
        print("ПРОБЛЕМЫ ВОССТАНОВЛЕНИЯ:")
        for name, addr, msg in restore_fail:
            print(f"  {name} ({addr}): {msg}")
    else:
        print("Исходное состояние канала полностью восстановлено.")
    return 0 if applied == len(results) and not restore_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
