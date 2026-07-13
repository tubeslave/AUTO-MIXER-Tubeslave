# Operator emergency-stop control

Status: accepted implementation decision; does not grant live authority

## Context

The active Automixer cockpit had no always-visible system emergency-stop
control. The backend already exposed `emergency_stop`, but an operator could
not reliably reach it from either the desktop or compact/mobile surface, and
the UI had no acknowledgement or unknown-state handling.

## Options considered

1. Keep emergency stop in a secondary settings/runtime panel.
2. Add a one-click stop with no accidental-activation guard.
3. Keep the stop visible in both layouts, require an immediate two-step
   confirmation, and fail closed until backend latch truth is reconciled.

## Decision

Use option 3.

- Desktop and compact/mobile headers always render the stop control.
- Confirmation has no time delay; safe `Cancel`, not destructive `Confirm`,
  receives initial focus.
- The exact request is `{"type":"emergency_stop","reason":"frontend_operator_emergency_stop"}`.
- A direct acknowledgement is accepted only when `success` is literal `true`,
  `status` is `stopped`, and manual-write supervision confirms the emergency
  latch.
- A fresh `emergency_stopped=true` supervision snapshot also confirms the
  latch if the direct acknowledgement is lost or reordered.
- Disconnect after acknowledgement changes the UI to unknown/error. Only a
  fresh supervision snapshot can restore `STOP ACTIVE` or `ready` truth.
- An acknowledged stop remains reassertable; duplicate requests are safe for
  the backend priority lane to coalesce.
- Backend success requires every active AutoSoundcheck task, engine, agent and
  controller teardown step to finish. Legacy asyncio tasks are cancelled and
  drained before their reference is cleared; failure returns
  `stop_incomplete`, retains the latch and exposes bounded teardown errors.
- Emergency and graceful shutdown share a synchronous phase-zero rail before
  their first await: manual-write and ingress generations advance, new normal
  ingress is rejected, gain/fader controllers freeze, agent auto-apply is
  disabled, SafeGain loses its callback/mixer reference and voice-command
  tasks are cancelled. Graceful shutdown then waits for any already-running
  emergency fast-lane task and for the console-write lock handoff before mixer
  disconnect.
- SafeGain and voice runtimes must prove worker/task/stream termination. A
  stuck SafeGain thread, incomplete audio stop, voice task failure or
  controller stop failure keeps retryable references and makes teardown
  incomplete. A separate system-teardown-complete flag prevents a WAL-only
  emergency latch from being mistaken for a completed global stop.
- The legacy `emergency_stop_agent` response forwards that exact result. Older
  iPhone/agent panels may show a confirmed manual stop only for literal
  `success=true`, `status=stopped`, `teardown_complete=true` and an active latch.
- UI wording claims only that the backend latch is engaged. It does not claim
  per-controller stop proof that the wire contract does not provide.

## Why this won

The operator can reach the safety action from every active layout while the UI
remains conservative about transport loss, malformed acknowledgements and
backend restart. The confirmation protects against an accidental pointer or
keyboard activation without imposing a timed hold during an emergency.

## Rejected alternatives

- A secondary-panel action is too slow and can disappear behind navigation.
- A one-click virtual stop is vulnerable to accidental show interruption.
- Treating socket send success as stop success would confuse delivery with an
  engaged backend latch.

## Implementation plan

1. Add the typed UI state and action.
2. Send the canonical backend emergency request.
3. Reconcile direct ACK, manual supervision, timeout and disconnect states.
4. Render the control in desktop and compact/mobile headers.
5. Cover offline, direct-ACK and supervision-before/lost-ACK behavior in E2E.

## Test plan

- `cd frontend && npm test`
- `cd frontend && npm run build`
- `cd frontend && npx playwright test e2e/automixer-prototype.spec.ts --grep 'emergency stop|supervision truth'`
- Exercise WAL-only latch versus completed system teardown, phase-zero freeze
  before a blocked client close, emergency fast-lane drain and the
  console-write barrier.

## Risks and rollback

The two-step confirmation adds one operator action. The confirm dialog opens
immediately and contains no delay. If frontend state reconciliation regresses,
revert this UI slice while retaining the backend emergency endpoint and its
priority ingress lane; never replace a missing acknowledgement with a green
local assumption.
