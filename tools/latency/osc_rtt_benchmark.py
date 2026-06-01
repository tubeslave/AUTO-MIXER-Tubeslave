#!/usr/bin/env python3
"""UDP OSC round-trip latency benchmark.

The default mode starts a local OSC echo responder on loopback. That keeps the
smoke test off the WING write path while still measuring UDP/OSC encode, send,
receive and decode timing on this host.
"""

from __future__ import annotations

import argparse
import csv
import platform
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from pythonosc.osc_message import OscMessage
    from pythonosc.osc_message_builder import OscMessageBuilder
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"python-osc is required for OSC RTT benchmark: {exc}") from exc


DEFAULT_ADDRESS = "/automixer/latency/ping"
DEFAULT_OUTPUT = Path("reports/latency/rtts.csv")
DEFAULT_SUMMARY = Path("reports/latency/summary.md")


@dataclass(frozen=True)
class RttSample:
    seq: int
    t_send_ns: int
    t_recv_ns: Optional[int]
    rtt_ms: Optional[float]


class LocalOscEchoResponder:
    """Tiny UDP OSC echo server for local ping-pong benchmarking."""

    def __init__(self, host: str, port: int, address: str):
        self.host = host
        self.port = port
        self.address = address
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.1)
        self.actual_port = int(self._sock.getsockname()[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="osc-rtt-echo", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.sendto(b"\0", (self.host, self.actual_port))
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                message = OscMessage(data)
            except Exception:
                continue
            if message.address != self.address:
                continue
            try:
                self._sock.sendto(data, addr)
            except OSError:
                break


def build_osc_ping(address: str, seq: int) -> bytes:
    builder = OscMessageBuilder(address=address)
    builder.add_arg(int(seq), arg_type="i")
    return builder.build().dgram


def _response_matches(data: bytes, address: str, seq: int) -> bool:
    try:
        message = OscMessage(data)
    except Exception:
        return False
    if message.address != address:
        return False
    if not message.params:
        return False
    try:
        return int(message.params[0]) == int(seq)
    except (TypeError, ValueError):
        return False


def run_one(
    sock: socket.socket,
    target: tuple[str, int],
    address: str,
    seq: int,
    timeout_s: float,
) -> RttSample:
    payload = build_osc_ping(address, seq)
    t_send_ns = time.perf_counter_ns()
    sock.sendto(payload, target)

    deadline = time.perf_counter() + timeout_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return RttSample(seq=seq, t_send_ns=t_send_ns, t_recv_ns=None, rtt_ms=None)
        sock.settimeout(remaining)
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            return RttSample(seq=seq, t_send_ns=t_send_ns, t_recv_ns=None, rtt_ms=None)
        t_recv_ns = time.perf_counter_ns()
        if _response_matches(data, address, seq):
            return RttSample(
                seq=seq,
                t_send_ns=t_send_ns,
                t_recv_ns=t_recv_ns,
                rtt_ms=(t_recv_ns - t_send_ns) / 1_000_000.0,
            )


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(samples: list[RttSample]) -> dict[str, object]:
    rtts = [sample.rtt_ms for sample in samples if sample.rtt_ms is not None]
    sent = len(samples)
    received = len(rtts)
    timeouts = sent - received
    if not rtts:
        return {
            "sent": sent,
            "received": received,
            "timeouts": timeouts,
            "packet_loss_pct": 100.0 if sent else 0.0,
            "min_ms": None,
            "median_ms": None,
            "mean_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "jitter_p95_minus_median_ms": None,
            "stdev_ms": None,
        }

    median = statistics.median(rtts)
    p95 = percentile(rtts, 95.0)
    p99 = percentile(rtts, 99.0)
    return {
        "sent": sent,
        "received": received,
        "timeouts": timeouts,
        "packet_loss_pct": ((sent - received) / sent * 100.0) if sent else 0.0,
        "min_ms": min(rtts),
        "median_ms": median,
        "mean_ms": statistics.fmean(rtts),
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max(rtts),
        "jitter_p95_minus_median_ms": (p95 - median) if p95 is not None else None,
        "stdev_ms": statistics.pstdev(rtts) if len(rtts) > 1 else 0.0,
    }


def write_csv(path: Path, samples: list[RttSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "t_send_ns", "t_recv_ns", "rtt_ms"])
        for sample in samples:
            writer.writerow(
                [
                    sample.seq,
                    sample.t_send_ns,
                    "" if sample.t_recv_ns is None else sample.t_recv_ns,
                    "" if sample.rtt_ms is None else f"{sample.rtt_ms:.6f}",
                ]
            )


def fmt_ms(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_summary(path: Path, stats: dict[str, object], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "local loopback echo responder" if args.local_responder else "external OSC responder"
    text = f"""# Latency Benchmark Report

## Environment

- Host: {platform.node() or 'unknown'}
- OS: {platform.platform()}
- Python: {platform.python_version()}
- OSC mode: {mode}
- OSC target: `{args.host}:{args.port}`
- OSC address: `{args.address}`
- Safety: no WING/live OSC write path was used by the default benchmark mode.

## OSC RTT

| sent | received | timeouts | packet loss % | min ms | median ms | mean ms | p95 ms | p99 ms | max ms | jitter p95-median ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {stats['sent']} | {stats['received']} | {stats['timeouts']} | {float(stats['packet_loss_pct']):.3f} | {fmt_ms(stats['min_ms'])} | {fmt_ms(stats['median_ms'])} | {fmt_ms(stats['mean_ms'])} | {fmt_ms(stats['p95_ms'])} | {fmt_ms(stats['p99_ms'])} | {fmt_ms(stats['max_ms'])} | {fmt_ms(stats['jitter_p95_minus_median_ms'])} |

## Audio E2E

Not run yet.

## Packet Capture

Not run yet.

## Buffer Analysis

Not run yet.

## Conclusion

Pending audio E2E and packet-capture analysis.

## Recommendations

- Run audio E2E against a real loopback path before treating these smoke
  results as production latency numbers.
"""
    path.write_text(text, encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> tuple[list[RttSample], dict[str, object]]:
    responder: Optional[LocalOscEchoResponder] = None
    host = args.host
    port = int(args.port)
    if args.local_responder:
        responder = LocalOscEchoResponder(args.host, port, args.address)
        responder.start()
        host = args.host
        port = responder.actual_port
        args.port = port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, 0))
    try:
        target = (host, port)
        for seq in range(-int(args.warmup), 0):
            run_one(sock, target, args.address, seq, args.timeout)

        samples = [
            run_one(sock, target, args.address, seq, args.timeout)
            for seq in range(int(args.messages))
        ]
    finally:
        sock.close()
        if responder is not None:
            responder.stop()

    return samples, summarize(samples)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UDP OSC RTT benchmark.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30223)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--messages", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--no-local-responder",
        dest="local_responder",
        action="store_false",
        help="Do not start the safe local echo responder; target must echo OSC packets.",
    )
    parser.set_defaults(local_responder=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    samples, stats = run_benchmark(args)
    write_csv(args.output, samples)
    write_summary(args.summary, stats, args)

    print(
        "OSC RTT: "
        f"sent={stats['sent']} received={stats['received']} "
        f"timeouts={stats['timeouts']} median={fmt_ms(stats['median_ms'])}ms "
        f"p95={fmt_ms(stats['p95_ms'])}ms p99={fmt_ms(stats['p99_ms'])}ms"
    )
    return 0 if stats["timeouts"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
