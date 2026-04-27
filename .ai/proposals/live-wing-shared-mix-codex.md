# Live WING Shared Mix Soundcheck - Codex Proposal

## Short Thesis

Add a dedicated live shared-mix planner to AutoSoundcheckEngine that reuses the
offline chat-only mix principles while routing all outgoing changes through the
existing safety controller.

## What I Understood

The operator wants the same style of analysis used for offline multitrack
folders: compensated LTAS, source contribution reasoning, anchor-order balance,
small source moves, MERT shadow evaluation, and explainable rule references.
The target is now a live Behringer WING Rack over OSC.

## Proposed Solution

- Add `backend/live_shared_mix.py` as a pure planner.
- Read current WING channel state immediately before the shared-mix pass.
- Convert live channels into planner inputs with role, fader, mute, HPF, EQ,
  routing, and main-send state.
- Use channels 23/24 only as master-reference audio, not source channels.
- Apply planner actions through `AutoFOHSafetyController`.
- Add a `MasterFaderMove` action that can only cut Main 1, never raise it.
- Keep routing/naming as report-only until a patch map is available.

## Likely Files To Touch

- `backend/live_shared_mix.py`
- `backend/auto_soundcheck_engine.py`
- `backend/autofoh_safety.py`
- `backend/wing_client.py`
- `backend/config_manager.py`
- `config/automixer.yaml`
- tests for the planner, safety layer, and engine integration

## Alternatives Considered

- Fold the offline script directly into AutoSoundcheckEngine: rejected because
  it would mix pure analysis and live OSC side effects.
- Correct routing/names automatically now: rejected because no expected patch
  map was provided and a wrong patch can cut the show.
- Use master EQ for master spectrum mismatch: rejected because the offline
  workflow used master spectrum as a balance meter, not a target for master EQ.

## Risks

- Live routing writes are intentionally incomplete until a patch map exists.
- Full MERT is shadow/evaluation, not the primary low-latency decision engine.
- First MERT model load can be slow.

## Test Plan

- Planner tests for low-end anchor, lead masking, and master overload.
- Safety tests for bounded/no-raise master fader moves.
- Engine test proving channels 23/24 are not source-corrected.
- Full project pytest.

## Where The Other Agent May Disagree

Kimi may argue that routing/name correction should be implemented immediately.
My recommendation is to keep routing and naming in audit mode until the expected
map is explicit, because live patch writes are higher risk than EQ/fader moves.
