# show-assist-auto-v1 gated live assistant

## Context

The project needs a soundcheck/live mode that can act as a sound engineer
assistant and, when explicitly enabled, apply safe corrections to a console. The
existing AutoFOH safety controller bounds and rate-limits typed actions, but it
does not make every write prove pre-read, readback, and rollback readiness.

## Options considered

1. Keep all AutoFOH actions write-capable through the existing safety
   controller.
2. Disable live writes entirely and keep only operator proposals.
3. Add a gated executor in front of live writes, with a small promoted set and
   proposal fallback for all other methods.

## Decision

Use option 3. Add `show_assist_auto_v1` as the default live action profile.
Only channel input trim, channel fader, and emergency feedback notch actions are
promoted by default. EQ, HPF, compressor, gate, pan, send, delay, polarity,
bus/DCA/master, routing, snapshots, reset, and naming remain non-promoted until
a physical proof run promotes them.

## Why this won

This keeps the assistant useful on soundcheck while preventing broad,
unproven automatic console edits. It also preserves the current typed safety
controller instead of replacing it, so existing bounds, rate limits, runtime
state policy, and tests continue to apply.

## Rejected alternatives

- Full autonomous application for every method was rejected because the repo has
  proposal/shadow/quarantine paths for many method families and no physical
  readback proof for them.
- Read-only assistant only was rejected because the user needs a path to send
  proven gain/fader corrections to the console.

## Implementation plan

1. Add `GatedCorrectionActionExecutor` with promotion policy, pre-read,
   readback, rollback metadata, artifact logging, and failure demotion.
2. Instantiate the executor from `AutoSoundcheckEngine._connect_mixer()`.
3. Convert blocked or observe-only gated actions into soundcheck recommendation
   bundles for the operator queue.
4. Add a proof runner that defaults to fake-mixer proof and requires explicit
   flags for physical WING writes.
5. Mark frontend mock fallback as not live-ready.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_correction_action_executor.py tests/test_auto_soundcheck_engine.py tests/test_soundcheck_handlers.py tests/test_server.py -q`
- Standard backend suite before live use:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- Frontend tests and build.
- Physical WING proof remains a separate operator-confirmed step using
  `tools/wing_method_proof_runner.py --execute --confirm-real-write ...`.

## Risks and rollback

The main risk is a promoted method failing readback after a write. The executor
marks the action unverified, attempts rollback when a pre-read value exists, and
demotes that action type after the first proof failure. Rollback is to disable
`autofoh.live_action_executor.enabled` or set all
`promoted_action_types` values to `false`.
