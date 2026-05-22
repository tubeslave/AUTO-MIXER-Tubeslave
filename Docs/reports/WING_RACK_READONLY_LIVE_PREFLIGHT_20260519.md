# WING Rack Read-Only Live Preflight - 2026-05-19

## Scope

This pass used the real WING Rack and live Ableton multitrack audio, but did not
write any console value. No fader, gain, EQ, dynamics, routing, snapshot, or
scene change was sent to the mixer.

## Runtime State

- Backend WebSocket: `ws://127.0.0.1:8765`
- Frontend: `http://localhost:3000`
- Mixer: WING Rack `192.168.1.102`
- OSC connection: connected
- Mixer client: `EnhancedOSCClient`
- Cached WING state: 404 parameters
- Known channel names from WING cache:
  - Ch 1: `KICK`
  - Ch 2: `SNARE T`
- Operator mode: `assist`
- Live write policy: `suggestions_only`
- Console executor: blocked
- Manual write gate: deployment boundary active

## Audio Capture

- Audio device used: `WING` device id `1`
- Capture mode: read-only preview
- Channels: 48
- Sample rate: 48000 Hz
- Active channels detected: 14
- Active channel ids: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16`

## Short Meter Window

Max values observed across a short live meter window:

| Channel | RMS dB | Peak dB | LUFS |
| --- | ---: | ---: | ---: |
| 1 | -18.5 | -11.7 | -26.9 |
| 2 | -22.9 | -15.1 | -32.0 |
| 3 | -19.4 | -10.2 | -26.9 |
| 4 | -39.9 | -31.5 | -46.9 |
| 5 | -38.3 | -31.0 | -45.4 |
| 6 | -39.3 | -29.4 | -45.2 |
| 7 | -36.1 | -24.8 | -40.1 |
| 8 | -26.4 | -23.4 | -26.5 |
| 9 | -19.5 | -11.2 | -23.4 |
| 10 | -18.3 | -9.1 | -23.0 |
| 11 | -25.8 | -12.3 | -24.2 |
| 12 | -28.5 | -20.0 | -26.1 |
| 13 | -28.9 | -19.5 | -29.4 |
| 16 | -14.8 | -9.5 | -16.9 |

## Operator Analysis

Command: `run_operator_analysis` with `create_proposals=true`.

Result:

- Status: `clear`
- Reason: `no_operator_analysis_observations`
- Observation count: 0
- Created proposals: 0
- Pending queue: 0

No peak risk exceeded the current proposal threshold.

## Supervised Smoke Dry-Run

Dry-run runner:

```bash
python3 tools/wing_supervised_smoke_runner.py \
  --url ws://127.0.0.1:8765 \
  --channel 16 \
  --value-type fader \
  --delta-db -0.5 \
  --timeout 8
```

Result:

- Status: `planned`
- Execute: `false`
- Address: `/ch/16/fdr`
- Old value: `0.0`
- Target value: `-0.5`
- Reason: `dry_run_only_add_execute_to_write`
- Run id: `20260519T085703Z_wing_supervised_smoke_ch16_fader`

## Verdict

The project is ready for a narrow supervised write smoke-test only after the
operator explicitly chooses the target channel and value type. The safe first
candidate from this pass is a tiny fader reduction on channel 16, followed by
rollback. Broad automation remains blocked.
