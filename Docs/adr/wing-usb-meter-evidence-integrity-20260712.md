# WING USB meter evidence integrity

Status: accepted implementation decision; does not grant live authority

## Context

The real WING Rack (`0100Q1V0604AAE`, firmware `3.0.5`) was reachable and
receiving signal, but a cold `soundcheck_runtime_observation_runner` run emitted
only empty meter frames because it never started the local USB capture. A second
audit found two evidence defects:

- dashboard master/analyzer selected channels 49/50 and silently fell back to
  channel 1 on the 48-channel WING USB device, while the accepted project route
  is Main LR on USB 47/48;
- `running=true` and non-empty channel rows could pass the runtime contract even
  when every sample was synthetic silence or below the signal floor.

These defects could produce a misleading green telemetry artifact. They did not
send OSC writes or change console state.

## Options considered

1. Treat a running capture as sufficient and document the prerequisite.
2. Fall back to approximate native WING OSC meters.
3. Make the proof runner own a bounded real USB capture and fail closed on
   device, callback, signal, or Main-route mismatch.

## Decision

Use option 3.

- `AudioCapture` exposes device identity, callback freshness and `real_input`.
- Live preview created through `AutoMixerServer._ensure_audio_capture` refuses a
  synthetic-silence fallback when the real device cannot start.
- Dashboard master and analyzer use the canonical stereo USB channels 47/48.
  Analyzer power is combined from both channels and reports
  `source_channels: [47, 48]`.
- Meter frames separately report capture availability and actual signal
  presence above `-90 dBFS`.
- `soundcheck_runtime_observation_runner` can start an explicitly selected
  local input, waits for a fresh real callback, stops only the capture it owns,
  and requires exact device identity plus signal on the observed 47/48 capture
  channels. It also fails closed unless connection/dashboard truth agrees on a
  connected WING in `analyze` mode with `wing_observation_only=true`, no
  proposal creation and empty live-write kinds/surfaces. Local capture control
  is read-only with respect to WING and is not a live mixer write.

Observed audio on USB 47/48 does not by itself prove the internal WING routing.
Readiness still requires the separate SHA-256-bound route artifact and operator
confirmation enforced by the supervised audio-recheck contract.

## Why this won

It proves the same high-fidelity USB signal used by analysis and supervised
audio recheck, preserves stereo Main identity, and prevents silence or channel-1
fallback from being promoted as physical soundcheck evidence.

## Rejected alternatives

- Documentation-only handling leaves the runner non-self-contained and permits
  another false-green artifact.
- Native WING meters are useful diagnostics but do not provide the full-band
  stereo audio needed by the analyzer and use an estimated RMS value.

## Implementation plan

1. Correct master/analyzer capture routing.
2. Add real-input freshness and signal fields.
3. Add bounded capture bootstrap/cleanup to the observation runner.
4. Add regression tests for 47/48, silence rejection and owned capture cleanup.
5. Repeat the proof against the physical WING in observation-only mode.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_meter_stream.py tests/test_soundcheck_runtime_observation_runner.py tests/test_audio_capture.py tests/test_audio_handlers.py -q --tb=short`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q --tb=short`
- Physical read-only artifact:
  `artifacts/soundcheck_runtime_observation/20260712T093447Z_soundcheck_runtime_observation/summary.json`

The superseding physical run passed with 285/285 available, read-only,
real-input, signal-present and master-signal frames, 48 observed channels, up
to 47 active channels, Main/analyzer source 47/48, maximum observed channel
sample peak `-3.7 dBFS`, and zero forbidden/live-write requests. Its
`artifact_manifest.json` binds the 7,567,217-byte events stream at SHA-256
`9d528d98550c1e7d81d53a411458c134373278af28e13cb5abf9a37006fa0e45`
and the summary at SHA-256
`012eacf67e7ed38949ce096148b6a15fca1525f93f6e59d8e7650f7649051f04`.
All four canonical manual-write/trace/memory/WAL hashes were identical before
and after the run. This is runtime telemetry evidence only; it does not close
M005/M006, prove internal WING routing, measure true peak, or grant authority.

## Risks and rollback

Some non-WING two-channel preview devices use channels 1/2; the helper preserves
that bounded fallback only when fewer than 48 channels exist. If real-input
startup regressions appear, revert this checkpoint and keep the new artifact
invalid; never restore synthetic silence as accepted physical evidence.
