#!/usr/bin/env python3
"""Audio end-to-end latency smoke runner.

Default mode is a deterministic simulated loopback. Real capture mode uses the
project's AudioCapture service so this tool stays outside production mixer
logic and does not open independent PyAudio streams.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from pythonosc.osc_message_builder import OscMessageBuilder
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"python-osc is required for audio E2E benchmark: {exc}") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from osc_rtt_benchmark import percentile  # noqa: E402


DEFAULT_OUTPUT = Path("reports/latency/e2e.json")
DEFAULT_SUMMARY = Path("reports/latency/summary.md")
DEFAULT_RTT_CSV = Path("reports/latency/rtts.csv")
DEFAULT_PCAP = Path("reports/latency/bench_smoke.pcap")


@dataclass(frozen=True)
class OnsetResult:
    onset_sample: Optional[int]
    threshold: Optional[float]
    peak: float
    status: str


@dataclass(frozen=True)
class E2EResult:
    buffer_samples: int
    sample_rate_hz: int
    mode: str
    t_control_send_ns: int
    t_audio_onset_ns: Optional[int]
    e2e_ms: Optional[float]
    onset_sample: Optional[int]
    onset_threshold: Optional[float]
    audio_peak: float
    status: str
    notes: list[str]


def parse_buffers(value: str) -> list[int]:
    buffers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        buffers.append(int(item))
    if not buffers:
        raise argparse.ArgumentTypeError("at least one buffer size is required")
    return buffers


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def send_osc_trigger(host: str, port: int, address: str, buffer_samples: int, sample_rate: int) -> int:
    builder = OscMessageBuilder(address=address)
    builder.add_arg(int(buffer_samples), arg_type="i")
    builder.add_arg(int(sample_rate), arg_type="i")
    payload = builder.build().dgram

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        t_control_send_ns = time.perf_counter_ns()
        sock.sendto(payload, (host, int(port)))
        return t_control_send_ns
    finally:
        sock.close()


def detect_onset(samples: np.ndarray, threshold_ratio: float = 0.2) -> OnsetResult:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if audio.size == 0:
        return OnsetResult(None, None, 0.0, "no_audio")

    envelope = np.abs(audio)
    peak = float(np.max(envelope))
    if peak <= 1e-9:
        return OnsetResult(None, None, peak, "no_onset")

    noise_window = max(1, min(audio.size // 10, 2400))
    noise_floor = float(np.median(envelope[:noise_window]))
    threshold = max(noise_floor * 8.0, peak * float(threshold_ratio), 1e-6)
    indices = np.flatnonzero(envelope >= threshold)
    if indices.size == 0:
        return OnsetResult(None, threshold, peak, "no_onset")
    return OnsetResult(int(indices[0]), threshold, peak, "ok")


def synthetic_loopback(
    *,
    buffer_samples: int,
    sample_rate: int,
    duration_ms: float,
    control_offset_ms: float,
) -> tuple[np.ndarray, int]:
    duration_samples = max(1, int(math.ceil(sample_rate * duration_ms / 1000.0)))
    control_offset_samples = max(1, int(math.ceil(sample_rate * control_offset_ms / 1000.0)))
    onset_sample = min(duration_samples - 1, int(buffer_samples) + control_offset_samples)
    total_samples = max(duration_samples, onset_sample + max(32, buffer_samples))
    audio = np.zeros(total_samples, dtype=np.float32)

    pulse_len = min(max(16, buffer_samples // 8), total_samples - onset_sample)
    if pulse_len > 0:
        window = np.hanning(pulse_len * 2)[:pulse_len]
        audio[onset_sample:onset_sample + pulse_len] = (0.85 * window).astype(np.float32)
    return audio, onset_sample


def run_simulated(args: argparse.Namespace, buffer_samples: int, control_offset_ms: float) -> E2EResult:
    t_control_send_ns = send_osc_trigger(
        args.osc_host,
        args.osc_port,
        args.osc_address,
        buffer_samples,
        args.sample_rate,
    )
    audio, _expected_onset = synthetic_loopback(
        buffer_samples=buffer_samples,
        sample_rate=args.sample_rate,
        duration_ms=args.duration_ms,
        control_offset_ms=control_offset_ms,
    )
    onset = detect_onset(audio, args.threshold_ratio)
    t_audio_onset_ns = None
    e2e_ms = None
    if onset.onset_sample is not None:
        t_audio_onset_ns = t_control_send_ns + int(
            round(onset.onset_sample * 1_000_000_000 / args.sample_rate)
        )
        e2e_ms = (t_audio_onset_ns - t_control_send_ns) / 1_000_000.0

    return E2EResult(
        buffer_samples=buffer_samples,
        sample_rate_hz=args.sample_rate,
        mode="simulated_loopback",
        t_control_send_ns=t_control_send_ns,
        t_audio_onset_ns=t_audio_onset_ns,
        e2e_ms=e2e_ms,
        onset_sample=onset.onset_sample,
        onset_threshold=onset.threshold,
        audio_peak=onset.peak,
        status=onset.status,
        notes=[
            "Simulated loopback uses one audio buffer plus half RTT as control offset.",
            "No live WING/OSC write path was used.",
        ],
    )


def run_capture(args: argparse.Namespace, buffer_samples: int) -> E2EResult:
    try:
        from audio_capture import AudioCapture, AudioSourceType, HAS_SOUNDDEVICE
    except Exception as exc:
        return run_capture_unavailable(args, buffer_samples, f"AudioCapture import failed: {exc}")

    if not HAS_SOUNDDEVICE:
        return run_capture_unavailable(args, buffer_samples, "sounddevice is not available")

    capture = AudioCapture(
        num_channels=1,
        sample_rate=args.sample_rate,
        buffer_seconds=max(1.0, args.duration_ms / 1000.0 + 0.5),
        block_size=buffer_samples,
        source_type=AudioSourceType.SOUNDDEVICE,
        device_name=args.device,
    )

    capture_start_ns = time.perf_counter_ns()
    capture.start()
    try:
        time.sleep(max(0.0, args.preroll_ms / 1000.0))
        t_control_send_ns = send_osc_trigger(
            args.osc_host,
            args.osc_port,
            args.osc_address,
            buffer_samples,
            args.sample_rate,
        )
        time.sleep(max(0.0, args.duration_ms / 1000.0))
        sample_count = int(args.sample_rate * (args.duration_ms / 1000.0 + args.preroll_ms / 1000.0))
        audio = capture.get_buffer(channel=1, num_samples=sample_count)
    finally:
        capture.stop()

    onset = detect_onset(audio, args.threshold_ratio)
    t_audio_onset_ns = None
    e2e_ms = None
    if onset.onset_sample is not None:
        t_audio_onset_ns = capture_start_ns + int(
            round(onset.onset_sample * 1_000_000_000 / args.sample_rate)
        )
        e2e_ms = (t_audio_onset_ns - t_control_send_ns) / 1_000_000.0

    return E2EResult(
        buffer_samples=buffer_samples,
        sample_rate_hz=args.sample_rate,
        mode="audio_capture_loopback",
        t_control_send_ns=t_control_send_ns,
        t_audio_onset_ns=t_audio_onset_ns,
        e2e_ms=e2e_ms,
        onset_sample=onset.onset_sample,
        onset_threshold=onset.threshold,
        audio_peak=onset.peak,
        status=onset.status,
        notes=[
            "Real capture mode uses backend.audio_capture.AudioCapture.",
            "Hardware buffer size may still need manual configuration in the audio driver.",
        ],
    )


def run_capture_unavailable(args: argparse.Namespace, buffer_samples: int, reason: str) -> E2EResult:
    result = run_simulated(args, buffer_samples, control_offset_ms=0.0)
    return E2EResult(
        **{
            **asdict(result),
            "mode": "simulated_loopback",
            "notes": result.notes + [f"Capture fallback reason: {reason}"],
        }
    )


def load_rtt_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "sent": 0,
            "received": 0,
            "timeouts": 0,
            "packet_loss_pct": None,
            "min_ms": None,
            "median_ms": None,
            "mean_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "jitter_p95_minus_median_ms": None,
            "stdev_ms": None,
        }

    rtts: list[float] = []
    sent = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sent += 1
            value = (row.get("rtt_ms") or "").strip()
            if value:
                rtts.append(float(value))

    received = len(rtts)
    timeouts = sent - received
    median = percentile(rtts, 50.0)
    p95 = percentile(rtts, 95.0)
    p99 = percentile(rtts, 99.0)
    return {
        "sent": sent,
        "received": received,
        "timeouts": timeouts,
        "packet_loss_pct": ((sent - received) / sent * 100.0) if sent else None,
        "min_ms": min(rtts) if rtts else None,
        "median_ms": median,
        "mean_ms": float(np.mean(rtts)) if rtts else None,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max(rtts) if rtts else None,
        "jitter_p95_minus_median_ms": (p95 - median) if p95 is not None and median is not None else None,
        "stdev_ms": float(np.std(rtts)) if rtts else None,
    }


def pcap_packet_count(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size < 24:
        return None
    data = path.read_bytes()
    offset = 24
    count = 0
    while offset + 16 <= len(data):
        incl_len = int.from_bytes(data[offset + 8:offset + 12], "little")
        if incl_len < 0 or offset + 16 + incl_len > len(data):
            return None
        count += 1
        offset += 16 + incl_len
    return count


def load_pcap_status(path: Path) -> dict[str, Any]:
    status_path = Path(f"{path}.status.json")
    payload: dict[str, Any] = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"status": "status_json_invalid"}
    payload.setdefault("status", "unknown")
    payload.setdefault("output", str(path))
    payload.setdefault("output_exists", path.exists())
    payload.setdefault("output_size_bytes", path.stat().st_size if path.exists() else 0)
    payload["packet_count_estimate"] = pcap_packet_count(path)
    return payload


def verdict_for(result: E2EResult, rtt: dict[str, Any]) -> tuple[str, str]:
    if result.status != "ok" or result.e2e_ms is None:
        return "NO-GO", "audio onset was not detected"
    if rtt.get("timeouts") not in (0, None):
        return "NO-GO", "OSC timeouts were observed"
    if rtt.get("packet_loss_pct") not in (0, 0.0, None):
        return "NO-GO", "OSC packet loss was observed"

    median = rtt.get("median_ms")
    p95 = rtt.get("p95_ms")
    p99 = rtt.get("p99_ms")
    max_ms = rtt.get("max_ms")
    if median is None or p95 is None or p99 is None or max_ms is None:
        return "NO-GO", "OSC RTT stats are missing"

    jitter = float(p95) - float(median)
    stable_p95 = jitter <= max(1.0, float(median) * 3.0)
    no_large_outlier = (float(max_ms) - float(p99)) <= max(5.0, float(p99) * 10.0)
    if not stable_p95:
        return "NO-GO", "RTT p95 is unstable relative to median"
    if not no_large_outlier:
        return "NO-GO", "RTT max shows a significant outlier"
    return "GO", "no loss/timeouts and RTT tail is stable for this smoke run"


def fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def write_e2e_json(path: Path, results: list[E2EResult], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = results[0]
    payload = {
        "t_control_send_ns": primary.t_control_send_ns,
        "t_audio_onset_ns": primary.t_audio_onset_ns,
        "e2e_ms": primary.e2e_ms,
        "mode": primary.mode,
        "sample_rate_hz": args.sample_rate,
        "buffer_results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    results: list[E2EResult],
    rtt: dict[str, Any],
    pcap: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    overall_go = True
    for result in results:
        verdict, reason = verdict_for(result, rtt)
        overall_go = overall_go and verdict == "GO"
        rows.append((result, verdict, reason))

    hardware_note = ""
    if all(result.mode == "simulated_loopback" for result, _verdict, _reason in rows):
        hardware_note = (
            "\nThe audio E2E numbers are simulated loopback smoke results, not "
            "hardware-validated WING/Dante/SoundGrid measurements."
        )

    text_lines = [
        "# Latency Benchmark Report",
        "",
        "## Environment",
        "",
        f"- Host: {platform.node() or 'unknown'}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- Git commit: `{git_commit()}`",
        f"- Sample rate: {args.sample_rate} Hz",
        f"- Audio mode: `{args.mode}`",
        f"- OSC trigger target: `{args.osc_host}:{args.osc_port}` `{args.osc_address}`",
        "- Existing project tooling reused: `backend/wing_client.py`, "
        "`virtual_mixer/`, `backend/audio_capture.py`, `backend/telemetry/`.",
        "- Safety: no production Automixer logic was changed; no live WING/OSC "
        "write path was used by the default run.",
        "",
        "## OSC RTT",
        "",
        "| sent | received | timeouts | loss % | min ms | median ms | mean ms | p95 ms | p99 ms | max ms | jitter ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {rtt.get('sent')} | {rtt.get('received')} | {rtt.get('timeouts')} | "
            f"{fmt_ms(rtt.get('packet_loss_pct'))} | {fmt_ms(rtt.get('min_ms'))} | "
            f"{fmt_ms(rtt.get('median_ms'))} | {fmt_ms(rtt.get('mean_ms'))} | "
            f"{fmt_ms(rtt.get('p95_ms'))} | {fmt_ms(rtt.get('p99_ms'))} | "
            f"{fmt_ms(rtt.get('max_ms'))} | "
            f"{fmt_ms(rtt.get('jitter_p95_minus_median_ms'))} |"
        ),
        "",
        "## Audio E2E",
        "",
        "| buffer @48k | mode | onset sample | e2e ms | status |",
        "|---:|---|---:|---:|---|",
    ]
    for result, _verdict, _reason in rows:
        text_lines.append(
            f"| {result.buffer_samples} | {result.mode} | "
            f"{result.onset_sample if result.onset_sample is not None else 'n/a'} | "
            f"{fmt_ms(result.e2e_ms)} | {result.status} |"
        )

    text_lines.extend(
        [
            "",
            "## Packet Capture",
            "",
            f"- Path: `{pcap.get('output')}`",
            f"- Status: `{pcap.get('status')}`",
            f"- Size: {pcap.get('output_size_bytes')} bytes",
            f"- Packet count estimate: {pcap.get('packet_count_estimate')}",
            f"- Message: {pcap.get('message', 'n/a')}",
            "",
            "## Buffer Analysis",
            "",
            "| buffer | verdict | reason |",
            "|---:|---|---|",
        ]
    )
    for result, verdict, reason in rows:
        text_lines.append(f"| {result.buffer_samples} | {verdict} | {reason} |")

    text_lines.extend(
        [
            "",
            "Automatic hardware buffer switching was not performed. For real hardware, "
            "set 128, 256, and 512 samples in the audio interface/DAW control panel, "
            "then rerun `tools/latency/audio_e2e.py --mode capture --buffers <size>` "
            "for each setting.",
            "",
            "## Conclusion",
            "",
            f"- Local smoke verdict: {'GO' if overall_go else 'NO-GO'}",
            f"- Hardware production verdict: {'NO-GO' if hardware_note else ('GO' if overall_go else 'NO-GO')}",
        ]
    )
    if hardware_note:
        text_lines.append(hardware_note)

    text_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Run a real loopback capture through the intended Dante/SoundGrid/WING path.",
            "- Prefer 128 samples at 48 kHz if CPU headroom and dropouts remain clean; "
            "use 256 samples as the safer live fallback.",
            "- Keep OSC control on wired Ethernet, isolated from congested Wi-Fi/client traffic.",
            "- Watch p95/p99 rather than only median; investigate any repeatable p99 spikes.",
            "- Keep WING write paths behind the existing supervised safety/readback gates.",
        ]
    )
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio E2E latency smoke benchmark.")
    parser.add_argument("--mode", choices=["simulated", "capture"], default="simulated")
    parser.add_argument("--buffers", type=parse_buffers, default=[128, 256, 512])
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration-ms", type=float, default=250.0)
    parser.add_argument("--preroll-ms", type=float, default=50.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=30224)
    parser.add_argument("--osc-address", default="/automixer/latency/trigger")
    parser.add_argument("--rtt-csv", type=Path, default=DEFAULT_RTT_CSV)
    parser.add_argument("--pcap", type=Path, default=DEFAULT_PCAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    rtt = load_rtt_stats(args.rtt_csv)
    control_offset_ms = (float(rtt["median_ms"]) / 2.0) if rtt.get("median_ms") is not None else 0.0

    results: list[E2EResult] = []
    for buffer_samples in args.buffers:
        if args.mode == "capture":
            results.append(run_capture(args, buffer_samples))
        else:
            results.append(run_simulated(args, buffer_samples, control_offset_ms))

    write_e2e_json(args.output, results, args)
    pcap = load_pcap_status(args.pcap)
    write_summary(args.summary, results=results, rtt=rtt, pcap=pcap, args=args)

    print("Audio E2E:")
    for result in results:
        verdict, reason = verdict_for(result, rtt)
        print(
            f"  {result.buffer_samples} @ {result.sample_rate_hz}: "
            f"{fmt_ms(result.e2e_ms)} ms, {verdict} ({reason})"
        )
    return 0 if all(verdict_for(result, rtt)[0] == "GO" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
