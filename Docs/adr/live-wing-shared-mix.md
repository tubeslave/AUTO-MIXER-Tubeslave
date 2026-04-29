# Live WING Shared Mix Soundcheck

## Context

The project already had an offline multitrack mixing workflow in
`tools/chat_only_shared_mix.py`: use a +4.5 dB/oct compensated LTAS over the
densest section, build the mix around anchors, treat the master spectrum as a
balance meter, and prefer source/stem fixes over master processing.

The live WING Rack path needed the same reasoning while respecting existing
console state and live-sound safety.

## Options Considered

- Port the offline rules into a pure live planner and apply through safety.
- Embed the offline script directly into AutoSoundcheckEngine.
- Let the engine write routing, names, and master processing immediately.

## Decision

Add `backend/live_shared_mix.py` as a pure planner and call it from
AutoSoundcheckEngine after per-channel correction. The engine reads current WING
state before the pass, excludes channels 23/24 as master-reference inputs, and
routes all actions through `AutoFOHSafetyController`.

## Why This Won

The design preserves the offline mixing behavior while keeping live OSC writes
bounded, logged, and reversible where possible. It also keeps routing and naming
visible without making destructive assumptions about the operator's patch.

## Rejected Alternatives

Directly writing routing or names was rejected for now. The WING OSC addresses
are known and read back, but a wrong routing write can silence or mis-patch a
show. Master EQ correction was also rejected because the source workflow fixes
culprit stems first.

## Implementation Plan

- Add a live shared-mix planner with anchor phases:
  kick/bass, lead, rhythm, music, cymbals/air.
- Add WING readback for input routing and Main 1 send state.
- Add a `MasterFaderMove` that can only reduce Main 1, with a 1 dB max step.
- Feed source actions through control permissions and phase target guards.
- Keep routing/naming writes disabled by config until a patch/name map exists.

## Test Plan

- `PYTHONPATH=backend ./backend/venv/bin/python -m pytest tests/test_live_shared_mix.py tests/test_autofoh_safety.py tests/test_auto_soundcheck_engine.py -q`
- `PYTHONPATH=backend ./backend/venv/bin/python -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: the new pass can add more OSC traffic after the normal correction chain.
Mitigation: action count is capped, rate-limited, and routed through safety.

Rollback: set `autofoh.shared_chat_mix.enabled: false` in
`config/automixer.yaml` and restart the backend.
