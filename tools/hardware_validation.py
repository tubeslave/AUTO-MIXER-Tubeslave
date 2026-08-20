#!/usr/bin/env python3
"""
Валидация автомикшера на реальном железе (read-only, безопасно для концерта).

Проверяет по слоям, ничего не записывая на пульт:
  1. Discovery  — виден ли WING/dLive в сети (broadcast).
  2. OSC reach  — открыт ли порт управления (с понятным диагнозом,
                  если пульт найден, но OSC-remote выключен).
  3. Audio      — открывается ли аудиоустройство пульта и идёт ли сигнал.
  4. DSP        — точность измерителей (LUFS / True Peak / GCC-PHAT) на
                  реальном аудиотракте через loopback известного сигнала.

Запуск:
    PYTHONPATH=backend python tools/hardware_validation.py
    PYTHONPATH=backend python tools/hardware_validation.py --wing-ip 192.168.1.102

Управляющие записи (fader/EQ/…) НЕ выполняются — это диагностика.
"""
import argparse
import ipaddress
import socket
import sys
import time

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg):
    print(f"  {RED}✗{RESET} {msg}")


def _local_ipv4_addresses():
    """Локальные IPv4-адреса (для directed broadcast по всем интерфейсам)."""
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except Exception:
        pass
    # Резервный путь: UDP-сокет к внешнему адресу (пакет не отправляется)
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        addrs.add(probe.getsockname()[0])
        probe.close()
    except Exception:
        pass
    return addrs


def _osc_message(address: str) -> bytes:
    b = address.encode()
    b += b"\x00" * (4 - len(b) % 4)
    b += b",\x00\x00\x00"
    return b


def _valid_wing_discovery_response(data: bytes, addr, local_addresses) -> bool:
    """Accept only WING-labelled replies from the queried local /24 boundary."""
    try:
        source_ip = ipaddress.ip_address(str(addr[0]))
        source_port = int(addr[1])
    except (IndexError, TypeError, ValueError):
        return False
    if source_port != 2222 or not source_ip.is_private or source_ip.is_loopback:
        return False
    local_networks = {
        ipaddress.ip_network(f"{value}/24", strict=False)
        for value in local_addresses
        if ipaddress.ip_address(value).is_private
    }
    if not any(source_ip in network for network in local_networks):
        return False
    text = data.decode("utf-8", "ignore").strip()
    return 0 < len(text) <= 4096 and "wing" in text.casefold()


def check_discovery(timeout=3.0):
    """WING broadcast discovery на порт 2222. Возвращает список (ip, info)."""
    print(f"\n{DIM}[1/4] Discovery{RESET}")
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 0))
        # Directed broadcast по каждому локальному интерфейсу. Глобальный
        # 255.255.255.255 на macOS часто даёт EHOSTUNREACH и не должен
        # ронять весь скан — каждый sendto изолирован
        local_addresses = _local_ipv4_addresses()
        targets = ["255.255.255.255"]
        for local_ip in local_addresses:
            prefix = ".".join(local_ip.split(".")[:3])
            targets.append(f"{prefix}.255")
        for tgt in targets:
            try:
                s.sendto(b"WING?", (tgt, 2222))
            except OSError:
                pass
        s.settimeout(timeout)
        start = time.time()
        seen = set()
        while time.time() - start < timeout:
            try:
                data, addr = s.recvfrom(4096)
                if not _valid_wing_discovery_response(data, addr, local_addresses):
                    continue
                if addr[0] in seen:
                    continue
                seen.add(addr[0])
                info = data.decode("utf-8", "ignore").strip()
                found.append((addr[0], info))
                _ok(f"WING @ {addr[0]}: {info[:70]}")
            except socket.timeout:
                break
    except Exception as e:
        _fail(f"discovery error: {e}")
    if not found:
        _warn("Ни один WING не ответил на broadcast (проверьте сеть/питание пульта)")
    return found


def check_osc_reachable(ip, port=2223, timeout=2.0):
    """Достижим ли OSC-порт управления. Ключевой диагноз для оператора:
    пульт виден в discovery, но connect() падает — почти всегда OSC-remote
    выключен в меню пульта (Setup → Remote), а не сетевая проблема."""
    print(f"\n{DIM}[2/4] OSC reachability ({ip}:{port}){RESET}")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(_osc_message("/?"), (ip, port))
        s.sendto(_osc_message("/xinfo"), (ip, port))
        data, addr = s.recvfrom(65535)
        if addr[0] != ip or int(addr[1]) != int(port):
            _fail("OSC-ответ пришёл не от запрошенного WING endpoint")
            return False
        if not (data.startswith(b"/xinfo") or data.startswith(b"/?")):
            _fail("OSC-ответ не подтверждает запрошенную WING identity")
            return False
        _ok(f"OSC отвечает: {data[:60]!r}")
        return True
    except socket.timeout:
        _warn("OSC-порт молчит (открыт, но без ответа на /? — возможно нормально)")
        return None
    except OSError as e:
        if getattr(e, "errno", None) in (65, 61, 113):  # host/conn unreachable
            _fail("OSC-порт недоступен (ICMP unreachable).")
            print(f"      {YELLOW}→ Пульт найден по discovery, но OSC-remote не запущен.{RESET}")
            print(f"      {YELLOW}→ Включите на пульте: Setup → Remote → OSC (порт {port}).{RESET}")
        else:
            _fail(f"OSC error: {e}")
        return False
    finally:
        s.close()


def check_audio(name_filter="wing"):
    """Открывается ли аудиоустройство пульта и идёт ли сигнал."""
    print(f"\n{DIM}[3/4] Audio capture{RESET}")
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:
        _fail(f"sounddevice/numpy недоступны: {e}")
        return None
    idx = None
    for i, d in enumerate(sd.query_devices()):
        if name_filter in d["name"].lower() and d["max_input_channels"] > 0:
            idx = i
            break
    if idx is None:
        _warn(f"Устройство '{name_filter}' не найдено среди аудиовходов")
        return None
    dev = sd.query_devices(idx)
    nch = min(16, dev["max_input_channels"])
    _ok(f"Устройство [{idx}] {dev['name']!r}: {dev['max_input_channels']} входов")
    try:
        rec = sd.rec(int(1.0 * 48000), samplerate=48000, channels=nch,
                     device=idx, dtype="float32")
        sd.wait()
        active = 0
        for c in range(nch):
            rms = float(np.sqrt(np.mean(rec[:, c] ** 2)))
            if 20 * np.log10(rms + 1e-12) > -70:
                active += 1
        if active:
            _ok(f"Сигнал на {active}/{nch} каналах")
        else:
            _warn(f"Захват работает, но все {nch} каналов тихие "
                  f"(проверьте USB/Dante-роутинг на пульте)")
        return True
    except Exception as e:
        _fail(f"Ошибка захвата: {e}")
        return False


def check_dsp_accuracy():
    """Точность измерителей на реальном аудиотракте (loopback через
    системное устройство). Подаём известный сигнал и сверяем измерения."""
    print(f"\n{DIM}[4/4] DSP accuracy (real audio path){RESET}")
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:
        _fail(f"sounddevice/numpy недоступны: {e}")
        return None

    sr = 48000
    # loopback-устройство (BlackHole/agregate) — иначе считаем прямо на массиве
    loop_idx = None
    for i, d in enumerate(sd.query_devices()):
        if "blackhole" in d["name"].lower() and d["max_input_channels"] >= 2:
            loop_idx = i
            break

    t = np.arange(sr) / sr
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(sr).astype(np.float32)) * 0.1
    delay = 48  # сэмплов = 1.0 мс @ 48k
    noise2 = np.roll(noise, delay)
    # 1 kHz -20 dBFS RMS для проверки уровней
    tone = (10 ** (-20 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    if loop_idx is not None:
        try:
            stereo = np.column_stack([tone, tone])
            rec = sd.playrec(stereo, samplerate=sr, channels=2, device=loop_idx, dtype="float32")
            sd.wait()
            seg = rec[sr // 4: sr // 4 + sr // 2]
            if np.sqrt(np.mean(seg ** 2)) > 1e-6:
                tone = np.ascontiguousarray(seg[:, 0])
                _ok(f"Реальный loopback через [{loop_idx}] BlackHole")
            else:
                _warn("BlackHole тихий — измерения на прямом сигнале")
        except Exception as e:
            _warn(f"loopback недоступен ({e}) — измерения на прямом сигнале")

    passed = True
    try:
        from lufs_gain_staging import LUFSMeter, TruePeakMeter
        from auto_phase_gcc_phat import GCCPHATAnalyzer

        lufs = LUFSMeter(sample_rate=sr).process(tone)
        if -24 <= lufs <= -16:
            _ok(f"LUFS = {lufs:.2f} (1kHz @ -20 dBFS RMS, ожидаемо ~-20)")
        else:
            _fail(f"LUFS = {lufs:.2f} вне ожидаемого диапазона [-24, -16]")
            passed = False

        tp = TruePeakMeter(sample_rate=sr).process(tone)
        samp = 20 * np.log10(np.max(np.abs(tone)) + 1e-12)
        if tp >= samp - 0.2:
            _ok(f"True Peak = {tp:.2f} dBTP >= sample peak {samp:.2f} dBFS")
        else:
            _fail(f"True Peak {tp:.2f} < sample peak {samp:.2f} (недоизмерение)")
            passed = False

        an = GCCPHATAnalyzer(sample_rate=sr, fft_size=4096, max_delay_ms=20.0)
        r = an.compute_delay(noise, noise2)
        expect_ms = delay * 1000.0 / sr
        if abs(r.delay_ms - expect_ms) < 0.1:
            _ok(f"GCC-PHAT delay = {r.delay_ms:.3f} ms (эталон {expect_ms:.3f}), "
                f"coherence={r.coherence:.3f}")
        else:
            _fail(f"GCC-PHAT delay {r.delay_ms:.3f} != эталон {expect_ms:.3f} ms")
            passed = False
    except Exception as e:
        import traceback
        traceback.print_exc()
        _fail(f"DSP error: {e}")
        passed = False
    return passed


def main():
    ap = argparse.ArgumentParser(description="AUTO-MIXER hardware validation (read-only)")
    ap.add_argument("--wing-ip", default=None, help="явный IP WING (иначе автопоиск)")
    ap.add_argument("--osc-port", type=int, default=2223)
    ap.add_argument("--skip-audio", action="store_true")
    args = ap.parse_args()
    if args.wing_ip:
        address = ipaddress.ip_address(args.wing_ip)
        if not address.is_private or address.is_loopback:
            raise SystemExit("--wing-ip must be a private non-loopback IPv4 address")
    if args.osc_port != 2223:
        raise SystemExit("--osc-port must be the canonical WING OSC port 2223")

    print("=" * 62)
    print(" AUTO-MIXER — валидация на реальном железе (read-only)")
    print("=" * 62)

    discovered = check_discovery()
    wing_ip = args.wing_ip or (discovered[0][0] if discovered else None)

    osc = None
    if wing_ip:
        osc = check_osc_reachable(wing_ip, args.osc_port)
    else:
        print(f"\n{DIM}[2/4] OSC reachability{RESET}")
        _warn("Пропущено — WING не найден и --wing-ip не задан")

    audio = dsp = None
    if not args.skip_audio:
        audio = check_audio()
        dsp = check_dsp_accuracy()

    print("\n" + "=" * 62)
    print(" ИТОГ")
    print("=" * 62)
    print(f"  Discovery : {'OK' if discovered else 'не найден'}")
    print(f"  OSC ctrl  : {'OK' if osc else ('OSC-remote выключен на пульте' if osc is False else 'н/д')}")
    print(f"  Audio     : {'OK' if audio else 'н/д'}")
    print(f"  DSP       : {'OK' if dsp else ('FAIL' if dsp is False else 'н/д')}")
    # exit=0 если DSP-валидация прошла (или пропущена); OSC/аудио зависят от
    # конфигурации пульта и не считаются фейлом кода
    return 0 if dsp is not False else 1


if __name__ == "__main__":
    sys.exit(main())
