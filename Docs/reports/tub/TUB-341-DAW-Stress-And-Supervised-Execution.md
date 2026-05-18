# TUB-341: DAW stress-test and supervised WING execution

Date: 2026-05-17

## Purpose

Run the full operator-facing `TUB-341` sequence on the real Behringer WING Rack:

1. extended DAW stress capture in observation mode;
2. first supervised `set_fader` / `set_gain` writes with rollback;
3. final replay/safety bundle generation.

This file is the single execution path for the physical session. It links the
existing safe tools rather than inventing a parallel workflow.

## Preconditions

- Backend is running on `ws://localhost:8765`.
- The WING Rack is reachable and the operator has selected a safe test channel.
- DAW material is prepared to cover:
  - speech;
  - vocal;
  - music;
  - multiple simultaneous channels;
  - silence;
  - sudden peaks;
  - rapid level changes.
- No uncontrolled write-capable scripts or automation loops are active.

## Safety boundary

Codex and Paperclip agents may prepare this runbook, dry-run reports, replay
artifacts, and review notes. They must not execute the Step 2 command, pass
`--confirm-real-write`, or control the physical WING Rack automatically.

Step 2 is a human-operated physical-session command. It requires a separate
explicit live-console approval after the operator confirms the channel, DAW
material, rollback plan, and emergency stop path.

## Step 1: Extended DAW stress capture

Start the backend in observation mode and capture the DAW stress pass using the
existing telemetry workflow from `TUB-337` / `TUB-338`.

Suggested session labels:

- live stress: `tub341_daw_stress_live_<timestamp>`
- replay stress: `tub341_daw_stress_replay_<timestamp>`

After each pass, keep:

- `events.jsonl`
- `metadata.json`
- one `wing_telemetry_compare.py --pretty` summary for the session
- one live-vs-replay compare JSON output

Commands:

```bash
python3 backend/wing_telemetry_compare.py \
  artifacts/wing_telemetry/<live_session_dir> \
  --pretty > artifacts/tub341/<live_session_dir>_summary.json

python3 backend/wing_telemetry_compare.py \
  artifacts/wing_telemetry/<live_session_dir> \
  artifacts/wing_telemetry/<replay_session_dir> \
  --pretty > artifacts/tub341/<live_session_dir>_vs_<replay_session_dir>.json
```

Operator review checklist for the stress pass:

- OSC stayed connected for the full DAW pass.
- Telemetry contains expected channel activity and is not fallback-only.
- No `write_sent` events appear in observation-only sessions.
- Event gaps, drift, or timing anomalies are written down before moving on.

## Step 2: First supervised live writes

Run the existing supervised harness:

```bash
python3 backend/tub345_supervised_write_runner.py \
  --mixer-ip 192.168.1.102 \
  --channel 32 \
  --fader-target-db -60.0 \
  --gain-target-db 0.5 \
  --max-true-peak-dbtp -8.4 \
  --no-clipping-threshold-dbtp -3.0 \
  --no-clipping-sample-count 24 \
  --confirm-real-write
```

Notes:

- `--max-true-peak-dbtp` is the measured peak from the real supervised session.
- If it is omitted, the saved summary stays explicit that no measured
  no-clipping evidence was captured.
- The harness already records:
  - approval-required negative control;
  - applied fader/gain writes with readback;
  - cooldown block;
  - rollback-last and rollback-all;
  - emergency-stop block after stop;
  - telemetry artifact paths.

Keep the saved summary JSON from:

- `artifacts/tub345_supervised_live/<session_label>_summary.json`

## Step 3: Build final readiness bundle

Combine:

- the shadow/stress report directory;
  use the existing shadow output directory that corresponds to the DAW stress run
- the live telemetry session
- the replay telemetry session
- the supervised write summary

Command:

```bash
python3 backend/wing_replay_readiness_bundle.py \
  --output-dir artifacts/tub341/final_bundle \
  --shadow-report-dir artifacts/<shadow_report_dir> \
  --live-telemetry-session artifacts/wing_telemetry/<live_session_dir> \
  --replay-telemetry-session artifacts/wing_telemetry/<replay_session_dir> \
  --supervised-summary-path artifacts/tub345_supervised_live/<session_label>_summary.json
```

Outputs:

- `artifacts/tub341/final_bundle/wing_replay_readiness_bundle.json`
- `artifacts/tub341/final_bundle/wing_replay_readiness_bundle.md`

## Evidence to attach back to the issue

- live stress `events.jsonl`
- live stress `metadata.json`
- replay stress `events.jsonl`
- replay stress `metadata.json`
- live summary JSON
- live vs replay compare JSON
- supervised write summary JSON
- final readiness bundle JSON
- final readiness bundle markdown

## Stop conditions

Do not proceed from Step 1 to Step 2 if any of these are true:

- observation-only capture contains `write_sent`;
- telemetry is fallback-only;
- OSC disconnects or timing drift is unexplained;
- the chosen safe channel is no longer safe.

Do not mark the session successful after Step 2 if:

- readback does not confirm the applied value;
- rollback does not restore;
- emergency stop does not block the next write;
- measured max true peak exceeds the chosen threshold.
