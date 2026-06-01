# Latency Benchmark Report

## Environment

- Host: MacBook-Pro-Dmitrij.local
- OS: macOS-26.4-arm64-arm-64bit-Mach-O
- Python: 3.14.3
- Git commit: `52af5a2`
- Sample rate: 48000 Hz
- Audio mode: `simulated`
- OSC trigger target: `127.0.0.1:30224` `/automixer/latency/trigger`
- Existing project tooling reused: `backend/wing_client.py`, `virtual_mixer/`, `backend/audio_capture.py`, `backend/telemetry/`.
- Safety: no production Automixer logic was changed; no live WING/OSC write path was used by the default run.

## OSC RTT

| sent | received | timeouts | loss % | min ms | median ms | mean ms | p95 ms | p99 ms | max ms | jitter ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 2000 | 0 | 0.000 | 0.022 | 0.034 | 0.049 | 0.099 | 0.168 | 2.210 | 0.065 |

## Audio E2E

| buffer @48k | mode | onset sample | e2e ms | status |
|---:|---|---:|---:|---|
| 128 | simulated_loopback | 134 | 2.792 | ok |
| 256 | simulated_loopback | 267 | 5.562 | ok |
| 512 | simulated_loopback | 532 | 11.083 | ok |

## Packet Capture

- Path: `reports/latency/bench_smoke.pcap`
- Status: `tcpdump_failed`
- Size: 24 bytes
- Packet count estimate: 0
- Message: tcpdump did not produce a pcap; wrote an empty placeholder.

## Buffer Analysis

| buffer | verdict | reason |
|---:|---|---|
| 128 | GO | no loss/timeouts and RTT tail is stable for this smoke run |
| 256 | GO | no loss/timeouts and RTT tail is stable for this smoke run |
| 512 | GO | no loss/timeouts and RTT tail is stable for this smoke run |

Automatic hardware buffer switching was not performed. For real hardware, set 128, 256, and 512 samples in the audio interface/DAW control panel, then rerun `tools/latency/audio_e2e.py --mode capture --buffers <size>` for each setting.

## Conclusion

- Local smoke verdict: GO
- Hardware production verdict: NO-GO

The audio E2E numbers are simulated loopback smoke results, not hardware-validated WING/Dante/SoundGrid measurements.

## Recommendations

- Run a real loopback capture through the intended Dante/SoundGrid/WING path.
- Prefer 128 samples at 48 kHz if CPU headroom and dropouts remain clean; use 256 samples as the safer live fallback.
- Keep OSC control on wired Ethernet, isolated from congested Wi-Fi/client traffic.
- Watch p95/p99 rather than only median; investigate any repeatable p99 spikes.
- Keep WING write paths behind the existing supervised safety/readback gates.
