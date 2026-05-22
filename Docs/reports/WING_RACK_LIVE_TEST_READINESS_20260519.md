# Wing Rack Live Test Readiness - 2026-05-19

## Current Stage

The project is in the supervised live-write hardening stage. The goal is not
full autonomous mixing yet. The next real-console test must prove one narrow
path:

1. operator mode is `supervised`;
2. one approved fader or input trim write is sent through the manual gate;
3. old value is captured;
4. readback is reported;
5. rollback can immediately restore the old value;
6. emergency stop blocks later writes.

## Allowed For First Wing Rack Test

- `set_fader` through `_apply_manual_console_write`
- `set_gain` / input trim through `_apply_manual_console_write`
- `rollback_last_manual_write`
- `get_operator_mode`, `set_operator_mode`, `get_decision_queue`

## Still Blocked Before First Live Test

These surfaces must remain quarantined for WING deployment until migrated to
proposal queue plus supervised apply:

- EQ writes and resets
- compressor writes
- module bypass/reset flows
- snapshot/scene load/save
- voice-command writes
- AutoEQ direct apply
- AutoFader direct apply
- AutoSoundcheck non-fader/non-gain writes
- legacy reset scripts and restore scripts

## Verified In This Pass

- WebSocket mode handlers are present:
  - `get_operator_mode`
  - `set_operator_mode`
- AutoSoundcheck recommendation bundles can be imported into the operator proposal queue.
- Soundcheck state transition `running`/`stopped` triggers one recommendation import.
- Manual rollback bypasses cooldown, so an immediate restore after a live write is possible.
- Ordinary repeated writes still honor cooldown.
- The supervised smoke-test runner is covered by offline fake-backend tests:
  - unsafe arguments are blocked before any backend action;
  - dry-run planning sends no console write;
  - execute flow sends approved write, rollback, emergency stop, and proves later writes are blocked.

## Test Evidence

Command:

```bash
PYTHONPATH=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/backend pytest \
  tests/test_soundcheck_handlers.py \
  tests/test_operator_product_state.py \
  tests/test_server.py \
  tests/test_auto_soundcheck_engine.py \
  tests/test_wing_client.py \
  tests/test_wing_supervised_smoke_runner.py -q
```

Result:

```text
139 passed in 60.76s
```

## Live Readiness Verdict

Not ready for broad automation.

Ready for a narrow supervised smoke-test only when Wing Rack is connected and
the operator explicitly chooses the test channel/value. The runner is now
prepared and offline-tested with:

- explicit target channel;
- tiny delta;
- pre-read;
- approved write;
- readback capture;
- rollback;
- rollback readback;
- emergency stop proof;
- artifact preservation under `artifacts/wing_telemetry/`.

No real Wing Rack write has been attempted in this pass.

## Prepared Runner

Runner:

```bash
python3 tools/wing_supervised_smoke_runner.py \
  --channel 32 \
  --value-type fader \
  --delta-db -0.5
```

Default behavior is dry-run planning. It writes no console value unless
`--execute` is present.

Live smoke-test form:

```bash
python3 tools/wing_supervised_smoke_runner.py \
  --channel 32 \
  --value-type fader \
  --delta-db -0.5 \
  --execute \
  --prove-emergency-stop
```

The emergency-stop proof intentionally leaves backend emergency stop active.
Restart the backend before any later supervised write attempt.
