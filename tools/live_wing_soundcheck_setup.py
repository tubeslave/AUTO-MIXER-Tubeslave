#!/usr/bin/env python3
"""Живой саундчек: настройка всех работающих каналов реального WING под концерт.

Полный цикл, без заглушек:
1. Захват реального аудио со всех каналов USB-карты WING (~8 с).
2. Детекция каналов с сигналом (RMS-порог, отсев bleed).
3. Классификация источника: имя канала на пульте + спектральная эвристика.
4. Применение коррекций по инструментальным профилям (enhanced_gain_agent)
   и плоскостям Ayaic (CLAUDE.md): trim (gain staging), HPF, гейт (барабаны),
   компрессор, фейдер к целевой плоскости LUFS.
5. Физический readback каждой записи. Исходное состояние ВСЕХ затронутых
   параметров сохраняется в reports/ (rollback JSON). Настройки ОСТАЮТСЯ.

Правила безопасности (CLAUDE.md):
- fader никогда не поднимается выше 0 dBFS;
- boost трима ограничен +6 dB и только при явном сигнале (не bleed);
- параметрические полосы EQ НЕ трогаем (там могут жить feedback-notch);
- mute-состояния каналов не меняются;
- каналы master/reference/talkback исключаются.

Запуск:
    python3 tools/live_wing_soundcheck_setup.py --dry-run
    python3 tools/live_wing_soundcheck_setup.py \
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
sys.path.insert(0, str(REPO_ROOT / "backend"))

import yaml  # noqa: E402
import sounddevice as sd  # noqa: E402

from wing_client import WingClient  # noqa: E402
from enhanced_gain_agent import INSTRUMENT_PROFILES, InstrumentType  # noqa: E402

LIVE_WING_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING"

NUM_CH = 40
CAPTURE_S = 8.0
SR = 48000
ACTIVE_RMS_DBFS = -50.0     # выше — рабочий канал
BLEED_RMS_DBFS = -60.0      # между BLEED и ACTIVE — просачивание, не настраиваем
TRIM_BOOST_CAP = 6.0        # live-правило: boost только ограниченный
TRIM_TARGET_PEAK = -18.0    # целевой пик после трима
FADER_FLOOR = -40.0
FADER_CEIL = 0.0            # НИКОГДА не выше 0 dBFS
READBACK_TIMEOUT = 1.2

WING_RATIOS = ["1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0",
               "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"]

EXCLUDE_NAME_RE = re.compile(r"master|mstr|\bref\b|refer|talk|\btb\b|mtx|matrix|\bbus\b|\bmon\b", re.I)

NAME_RULES = [
    (re.compile(r"kick|\bbd\b|бочк", re.I), InstrumentType.KICK),
    (re.compile(r"snare|\bsn\b|малый", re.I), InstrumentType.SNARE),
    (re.compile(r"\btom\b|том", re.I), InstrumentType.TOM),
    (re.compile(r"hat|\bhh\b|\boh\b|over|ride|crash|cymb|тарел", re.I), InstrumentType.HI_HAT),
    (re.compile(r"bass|\bbs\b|бас", re.I), InstrumentType.BASS),
    (re.compile(r"g(ui)?t|guitar|гитар|ac\.?\s?g", re.I), InstrumentType.GUITAR),
    (re.compile(r"keys?|piano|synth|pad|органн|клав", re.I), InstrumentType.KEYS),
    (re.compile(r"voc|vox|vcl|lead|sing|голос|вок", re.I), InstrumentType.VOCALS),
    (re.compile(r"play|track|\bpb\b|минус|фонограм", re.I), InstrumentType.PLAYBACK),
    (re.compile(r"accord|аккорд|акорд", re.I), InstrumentType.LINE),
]

# Переопределения HPF по имени: оверхэды — не хай-хэт-микрофон, 400 Гц слишком агрессивно
HPF_NAME_OVERRIDE = [
    (re.compile(r"\boh\b|over", re.I), 150.0),
]

# Плоскости Ayaic (CLAUDE.md): цель pre-master LUFS для канала
AYAIC_PLANE = {
    InstrumentType.VOCALS: -22.0,    # lead/main element
    InstrumentType.KICK: -25.0,
    InstrumentType.SNARE: -25.0,
    InstrumentType.TOM: -28.0,
    InstrumentType.HI_HAT: -28.0,
    InstrumentType.BASS: -25.0,
    InstrumentType.GUITAR: -25.0,
    InstrumentType.KEYS: -30.0,
    InstrumentType.PLAYBACK: -25.0,
    InstrumentType.LINE: -27.0,
    InstrumentType.UNKNOWN: -27.0,
}

GATE_INSTRUMENTS = {InstrumentType.KICK, InstrumentType.SNARE, InstrumentType.TOM}


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))


def nearest_ratio(r: float) -> str:
    return min(WING_RATIOS, key=lambda s: abs(float(s) - r))


def classify(name: str, feats: dict) -> tuple[InstrumentType, str]:
    for rx, itype in NAME_RULES:
        if name and rx.search(name):
            return itype, f"по имени {name!r}"
    # спектральная эвристика
    c = feats["centroid"]
    low = feats["low_ratio"]
    crest = feats["crest_db"]
    if low > 0.75 and c < 180:
        return (InstrumentType.KICK if crest > 14 else InstrumentType.BASS), "по спектру (НЧ)"
    if c > 4500:
        return InstrumentType.HI_HAT, "по спектру (ВЧ)"
    if 250 <= c <= 2500 and crest > 10:
        return InstrumentType.VOCALS, "по спектру (СЧ, динамичный)"
    return InstrumentType.LINE, "по спектру (нейтральный)"


def capture_audio() -> np.ndarray:
    devs = sd.query_devices()
    idx = next(i for i, d in enumerate(devs) if d["name"] == "WING" and d["max_input_channels"] >= NUM_CH)
    print(f"Захват {CAPTURE_S:.0f} с аудио с устройства 'WING' ({NUM_CH} каналов)...")
    audio = sd.rec(int(CAPTURE_S * SR), samplerate=SR, channels=NUM_CH, device=idx, dtype="float32")
    sd.wait()
    return audio  # shape (n, 40)


def analyze(audio: np.ndarray) -> dict[int, dict]:
    out = {}
    n = audio.shape[0]
    freqs = np.fft.rfftfreq(min(n, SR * 2), 1 / SR)
    frame = 4800  # 100 мс
    for ch in range(NUM_CH):
        x = audio[:, ch].astype(np.float64)
        peak = float(np.max(np.abs(x)))
        nfr = len(x) // frame
        fr = x[: nfr * frame].reshape(nfr, frame)
        fr_rms = np.sqrt(np.mean(fr ** 2, axis=1))
        gate_lin = peak / (10 ** (30 / 20))  # активные фреймы: выше peak-30 dB
        active = fr_rms[fr_rms > max(gate_lin, 1e-9)]
        rms = float(np.sqrt(np.mean(active ** 2))) if len(active) else float(np.sqrt(np.mean(x ** 2)))
        seg = x[: SR * 2]
        spec = np.abs(np.fft.rfft(seg))
        p = spec ** 2
        tot = float(np.sum(p)) + 1e-20
        centroid = float(np.sum(freqs * p) / tot)
        low_ratio = float(np.sum(p[freqs < 200]) / tot)
        out[ch + 1] = {
            "rms_dbfs": db(rms),
            "peak_dbfs": db(peak),
            "crest_db": db(peak) - db(rms),
            "centroid": centroid,
            "low_ratio": low_ratio,
        }
    return out


def readback(client: WingClient, addr: str):
    r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    if r["status"] != "received":
        r = client.query_parameter(addr, timeout=READBACK_TIMEOUT)
    return r["value"] if r["status"] == "received" else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="только измерить и показать план, ничего не писать")
    ap.add_argument("--confirm-live-wing-write", default="")
    ap.add_argument("--capture-seconds", type=float, default=CAPTURE_S)
    args = ap.parse_args()

    if not args.dry_run and args.confirm_live_wing_write != LIVE_WING_CONFIRMATION:
        print(f"FAIL: без --dry-run обязателен --confirm-live-wing-write {LIVE_WING_CONFIRMATION}", file=sys.stderr)
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

    # Захват и анализ реального сигнала
    audio = capture_audio()
    feats = analyze(audio)

    # Отбор рабочих каналов
    plan: list[dict] = []
    skipped: list[str] = []
    for ch in range(1, NUM_CH + 1):
        f = feats[ch]
        name = (names.get(ch) or "").strip()
        if EXCLUDE_NAME_RE.search(name):
            skipped.append(f"ch{ch} {name!r}: служебный (master/ref/tb) — исключён")
            continue
        if f["rms_dbfs"] < BLEED_RMS_DBFS:
            continue
        if f["rms_dbfs"] < ACTIVE_RMS_DBFS:
            skipped.append(f"ch{ch} {name!r}: {f['rms_dbfs']:.1f} dBFS — похоже на bleed, не настраиваю")
            continue
        itype, how = classify(name, f)
        prof = dict(INSTRUMENT_PROFILES.get(itype, INSTRUMENT_PROFILES[InstrumentType.LINE]))
        for rx, hz in HPF_NAME_OVERRIDE:
            if name and rx.search(name):
                prof["hpf_freq"] = hz
        plan.append({"ch": ch, "name": name, "type": itype, "how": how, "prof": prof, "feats": f})

    if not plan:
        print("Рабочих каналов с сигналом не найдено. Пропущено:")
        for s in skipped:
            print("  " + s)
        client.disconnect()
        return 1

    print(f"\nРабочие каналы ({len(plan)}):")
    for p in plan:
        f = p["feats"]
        print(f"  ch{p['ch']:2d} {p['name'] or '(без имени)':14s} -> {p['type'].value:9s} ({p['how']}); "
              f"RMS {f['rms_dbfs']:6.1f}, peak {f['peak_dbfs']:6.1f} dBFS, centroid {f['centroid']:5.0f} Hz")
    for s in skipped:
        print("  SKIP: " + s)

    # Расчёт коррекций
    for p in plan:
        ch, f, prof, itype = p["ch"], p["feats"], p["prof"], p["type"]
        cur_trim = readback(client, f"/ch/{ch}/in/set/trim") or 0.0
        cur_fader = readback(client, f"/ch/{ch}/fdr")
        trim_delta = TRIM_TARGET_PEAK - f["peak_dbfs"]
        trim_delta = min(trim_delta, TRIM_BOOST_CAP)          # boost ограничен
        new_trim = float(np.clip(float(cur_trim) + trim_delta, -18.0, 18.0))
        est_lufs_post_trim = f["rms_dbfs"] + (new_trim - float(cur_trim))
        fader = AYAIC_PLANE[itype] - est_lufs_post_trim  # сырое значение, нормализуем ниже
        p["actions"] = {
            "trim": round(new_trim, 1),
            "lc_on": 1, "lc_freq": float(prof["hpf_freq"]),
            "comp_on": 1, "comp_thr": float(prof["comp_threshold"]),
            "comp_ratio": nearest_ratio(float(prof["comp_ratio"])),
            "comp_att": float(prof["comp_attack_ms"]), "comp_rel": float(prof["comp_release_ms"]),
            "gate": itype in GATE_INSTRUMENTS,
            "gate_thr": float(prof["gate_threshold_lufs"]),
            "fader": round(fader, 1),
        }
        p["cur_fader"] = cur_fader
        p["cur_trim"] = float(cur_trim)

    # Нормализация фейдеров: сохранить относительный Ayaic-баланс, макс = потолок
    raw_max = max(p["actions"]["fader"] for p in plan)
    shift = raw_max - FADER_CEIL
    for p in plan:
        p["actions"]["fader"] = round(float(np.clip(p["actions"]["fader"] - shift, FADER_FLOOR, FADER_CEIL)), 1)
    print(f"\nНормализация фейдеров: общий сдвиг {-shift:+.1f} dB (относительный баланс Ayaic сохранён)")
    for p in plan:
        a, prof = p["actions"], p["prof"]
        print(f"  план ch{p['ch']:2d}: trim {p['cur_trim']:+.1f}->{a['trim']:+.1f}, HPF {prof['hpf_freq']:.0f} Hz, "
              f"comp {a['comp_thr']}/{a['comp_ratio']}:1, gate={'on' if a['gate'] else 'off'}, "
              f"fader {p['cur_fader']}->{a['fader']}")

    if args.dry_run:
        print("\nDRY-RUN: на пульт ничего не записано.")
        client.disconnect()
        return 0

    # Снять исходное состояние всех затрагиваемых адресов
    touched = {}
    for p in plan:
        ch = p["ch"]
        addrs = [f"/ch/{ch}/in/set/trim", f"/ch/{ch}/fdr", f"/ch/{ch}/flt/lc", f"/ch/{ch}/flt/lcf",
                 f"/ch/{ch}/dyn/on", f"/ch/{ch}/dyn/thr", f"/ch/{ch}/dyn/thr26", f"/ch/{ch}/dyn/ratio",
                 f"/ch/{ch}/dyn/att", f"/ch/{ch}/dyn/rel"]
        if p["actions"]["gate"]:
            addrs += [f"/ch/{ch}/gate/on", f"/ch/{ch}/gate/thr", f"/ch/{ch}/gate/range"]
        for a in addrs:
            touched[a] = readback(client, a)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    rollback = reports_dir / f"live_wing_soundcheck_original_{stamp}.json"
    rollback.write_text(json.dumps({"mixer_ip": mixer["ip"], "saved_at": stamp, "original": touched},
                                   ensure_ascii=False, indent=2))
    print(f"\nОткат сохранён: {rollback}")

    # Применение + readback
    report = []
    for p in plan:
        ch, a = p["ch"], p["actions"]
        ok_all = True

        def apply(addr, setter, value, kind="db"):
            nonlocal ok_all
            setter(value)
            time.sleep(0.08)
            actual = readback(client, addr)
            ok = actual is not None and (
                (isinstance(value, str) and (str(actual).strip() == value or
                                             abs(float(actual) - float(value)) < 1e-6))
                or (not isinstance(value, str) and abs(float(actual) - float(value)) <=
                    (0.2 if kind == "db" else max(2.0, 0.06 * abs(float(value)))))
            )
            ok_all &= ok
            return ok

        apply(f"/ch/{ch}/in/set/trim", lambda v: client.set_channel_gain(ch, v), a["trim"])
        apply(f"/ch/{ch}/flt/lc", lambda v: client.send(f"/ch/{ch}/flt/lc", v), 1, "int")
        apply(f"/ch/{ch}/flt/lcf", lambda v: client.send(f"/ch/{ch}/flt/lcf", v), a["lc_freq"], "freq")
        apply(f"/ch/{ch}/dyn/on", lambda v: client.set_compressor_on(ch, v), 1, "int")
        apply(f"/ch/{ch}/dyn/thr", lambda v: client.set_compressor_threshold(ch, v), a["comp_thr"])
        apply(f"/ch/{ch}/dyn/ratio", lambda v: client.set_compressor_ratio(ch, v), a["comp_ratio"])
        apply(f"/ch/{ch}/dyn/att", lambda v: client.set_compressor_attack(ch, v), a["comp_att"], "ms")
        apply(f"/ch/{ch}/dyn/rel", lambda v: client.set_compressor_release(ch, v), a["comp_rel"], "ms")
        if a["gate"]:
            client.set_gate(ch, threshold=a["gate_thr"], range_db=24.0, attack=1.0, hold=20.0, release=150.0)
            time.sleep(0.08)
            apply(f"/ch/{ch}/gate/on", lambda v: client.set_gate_on(ch, v), 1, "int")
        apply(f"/ch/{ch}/fdr", lambda v: client.set_channel_fader(ch, v), a["fader"])
        report.append((ch, p["name"], p["type"].value, ok_all))
        print(f"  {'PASS' if ok_all else 'PARTIAL'} ch{ch:2d} {p['name'] or '(без имени)':14s} настроен как {p['type'].value}")

    applied_json = reports_dir / f"live_wing_soundcheck_applied_{stamp}.json"
    applied_json.write_text(json.dumps(
        {"applied_at": stamp,
         "channels": [{"ch": p["ch"], "name": p["name"], "type": p["type"].value,
                       "actions": p["actions"], "features": p["feats"]} for p in plan]},
        ensure_ascii=False, indent=2, default=float))

    client.disconnect()
    ok = sum(1 for *_, s in report if s)
    print("\n================ ИТОГ САУНДЧЕКА ================")
    print(f"Настроено каналов: {ok}/{len(report)} (настройки ОСТАВЛЕНЫ на пульте)")
    print(f"Применённые параметры: {applied_json}")
    print(f"Откат: {rollback}")
    for ch, name, t, s in report:
        if not s:
            print(f"  PARTIAL ch{ch} {name!r} ({t}) — часть параметров не подтвердилась readback'ом")
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
