# Soundcheck Runtime Observation Runner

## Context

The supervised soundcheck MVP needs reproducible evidence that the operator
runtime can stay connected long enough to show connection state, channel
inventory, meters, analyzer and master-bus status without widening live
authority. The existing frontend and backend meter stream work can be unit
tested, but the project also needs an artifact from a real backend WebSocket
session that proves the telemetry loop stays read-only.

M005 and M006 still require operator and owner action. A telemetry runner must
not imply disconnect proof, readiness closure or permission to apply
corrections.

## Options considered

1. Use ad hoc WebSocket snippets for each runtime check.
2. Reuse broad live soundcheck/daemon tools in dry-run mode.
3. Add a dedicated read-only runtime observation runner.

## Decision

Use option 3. Add `tools/soundcheck_runtime_observation_runner.py` to collect
dashboard, channel inventory, decision queue, safety monitor and meter stream
events through the backend WebSocket. The runner sends only read/subscription
requests, records every send/receive event, and fails if it sends a write
request or observes live apply/rollback/emergency events during the observation
window.

## Why this won

The runner creates a bounded artifact for the exact runtime slice needed before
the next soundcheck cycle: dashboard DTOs and meter frames are alive, while
live-write surfaces remain quiet. It avoids the broad soundcheck daemon's
write-capable code paths and keeps evidence separate from M005 physical
disconnect proof.

## Rejected alternatives

- Ad hoc snippets were rejected because they do not preserve request/response
  evidence or a stable summary schema.
- Broad live soundcheck tools were rejected for this proof because even dry-run
  variants carry write-capable context and are not the smallest artifact for
  telemetry-only validation.

## Implementation plan

- Add a WebSocket runner with duration, sample interval and artifact-root flags.
- Subscribe to meters and periodically request dashboard, channel inventory,
  decision queue, safety monitor and operator mode snapshots.
- Record `events.jsonl` and `summary.json`.
- Mark the run failed/manual-review-required on forbidden sent request types or
  unexpected live events.
- Keep summary authority fields denied: `live_write=false`,
  `live_authority_granted=false`, `readiness_closure_allowed=false`,
  `m005_closure_allowed=false`, `m006_closure_allowed=false`.

## Test plan

- Fake-WebSocket success test for read-only dashboard and meter frames.
- Fake-WebSocket manual-review test for observed live apply event.
- Targeted meter/M005/runtime observation tests.
- Full backend pytest before landing.

## Risks and rollback

Risk: the runner can observe live activity from another client and mark the run
manual-review-required. That is intentional; an observation artifact should not
claim a quiet runtime if other live activity occurred.

Rollback is a normal git revert of the runner, tests, ADR and state/doc
references. M005 and M006 remain externally blocked either way.

## 2026-07-04 Operator Runtime Contract Update

The runtime observation runner now records an `operator_runtime_contract` and
blocks a `passed` result unless the observation sees the full operator-facing
runtime slice: operator mode status, connection state, dashboard snapshot,
channel inventory, decision queue, safety monitor, meter subscription, read-only
meter frames with channel levels, analyzer status and master-bus status.

This prevents a partial telemetry session with only dashboard and meter frames
from being treated as reproducible soundcheck runtime evidence. The update is
read-only artifact hardening only; it creates no proposal, applies no proposal,
contacts no WING directly and grants no readiness closure or live authority.
