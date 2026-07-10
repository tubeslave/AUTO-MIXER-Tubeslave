# Codex proposal: soundcheck live safety kernel

## 1. Short thesis

Make the existing supervised manual gate the only physical WING write boundary,
force all soundcheck/agent reasoning into proposal mode, and fix transport proof
semantics before running further real-console validation.

## 2. What I understood

The user wants an actually usable soundcheck pipeline on the currently running
WING, not another report-only milestone. The safe product is therefore analysis
and proposals plus narrow operator-approved fader/gain-down apply, not broad
AutoSoundcheck mutation.

## 3. Proposed solution

- Bind locally by default and require token-digest authentication for non-loopback
  WebSocket binds.
- Reject client `force_auto_apply`; real mixer agents remain suggest-only.
- Start AutoSoundcheck with `auto_apply=False`; import its recommendations into
  the operator queue.
- Make emergency stop a system-wide generation latch checked before every send
  and retry.
- Separate WING commanded state from observed state; physical GET responses are
  the only post-write verification source.
- Report `applied` only for correlated physical readback.
- Bind rollback to a transaction and compare current physical state with the
  transaction's applied value before restoring.
- Replace same-address throttle drops with monotonic serialized delivery.

## 4. Likely files to touch

- `backend/server.py`
- `backend/handlers/agent_handlers.py`
- `backend/handlers/soundcheck_handlers.py`
- `backend/handlers/automation_handlers.py`
- `backend/wing_client.py`
- `backend/correction_action_executor.py`
- `automixer/safety/safety_gate.py`
- focused tests under `tests/`
- `frontend/public/electron.js` for reliable backend startup
- project goal, ADR and autopilot evidence surfaces

## 5. Alternatives considered

- Preserve broad AutoSoundcheck apply behind config: rejected because config and
  canonical readiness are currently inconsistent.
- Add safety checks independently to every writer: rejected because it preserves
  multiple authorities and emergency-stop races.
- Immediately promote EQ/dynamics: rejected until separate hardware proof exists.

## 6. Risks

- Existing tests may encode cache-as-proof or partial-as-applied behavior.
- Removing command-cache mutation can expose missing physical readback paths.
- Authentication must not break the local Electron path.
- Real hardware validation must not modify unrelated live channels.

## 7. Test plan

- Unit tests for auth, auto-apply rejection, system stop, physical-only proof,
  transaction rollback and throttle delivery.
- Existing WING rails and no-WING quick gate.
- Full backend suite, frontend tests/build and Playwright E2E.
- Read-only WING/audio smoke followed by existing operator-run M005/M006 steps.

## 8. Where the other agent may disagree

Kimi may prefer a new standalone broker immediately. My recommendation is to
first make the existing manual gate the single authority, then extract it into a
dedicated broker after behavior is proven, minimizing risk in the live checkout.
