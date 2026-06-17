# ADR: WING Read-Only Pre-WebSocket Contact Denial

## Context

The first WING MVP blocker remains real read-only connection proof. The runner
records `websocket_opened: false` before transport opens, but early
expected-plan failures did not expose the same top-level `contacts_backend`,
`queries_console`, and `live_authority_granted` denial fields as later blocked
or validation-only surfaces.

## Options considered

1. Rely on `websocket_opened: false` and empty `sent_request_types`.
2. Add base no-contact and no-authority fields before the WebSocket path.
3. Add the fields only to each expected-plan failure branch.

## Decision

Use option 2. The base real-run summary now starts with
`contacts_backend: false`, `queries_console: false`, and
`live_authority_granted: false`; later WebSocket paths overwrite contact fields
with observed runtime facts.

## Why this won

Early expected-plan failures are part of the safety gate before any backend/WING
contact. Carrying explicit no-contact/no-authority fields makes those artifacts
machine-checkable without inferring safety from missing keys.

## Rejected alternatives

- Option 1 leaves downstream checks to infer from indirect fields.
- Option 3 fixes only one branch and risks future pre-WebSocket branches
  drifting again.

## Implementation plan

- Add base pre-WebSocket contact and live-authority denial fields to the
  real-run summary.
- Assert the fields for missing, invalid, and mismatched expected-plan failures.

## Test plan

- Targeted expected-plan pre-WebSocket runner regressions.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The main risk is stale contact flags on later runtime paths. The runner already
overwrites contact fields after WebSocket requests and in transport exception
handlers. Rollback is to remove the base fields and assertions.
