# ADR: WING Read-Only Preflight Release Summary Prestart Detail Fields

## Context

The no-WING JSON summary carries WING preflight prestart busy-port failure detail
fields for the evidence, plan artifact, and handoff packet artifact surfaces.
The markdown release summary did not render those fields, so a human operator or
reviewer could miss the exact fail-closed evidence expected before the real
WING read-only preflight.

## Options considered

1. Keep markdown unchanged and rely on JSON.
2. Render the existing field lists in the markdown summary.
3. Add new duplicate summary fields for markdown only.

## Decision

Use option 2. The markdown summary now renders the existing prestart failure
detail fields from the WING evidence, plan artifact, and handoff packet artifact
summary payloads.

## Why this won

It keeps human-readable release evidence aligned with the already validated JSON
contract without introducing a second source of truth.

## Rejected alternatives

Option 1 keeps operator-facing drift. Option 3 adds duplication that could drift
from the contract.

## Implementation plan

- Update `tools/no_wing_release_check.py` markdown rendering.
- Extend `tests/test_no_wing_release_check.py` markdown summary assertions.
- Validate focused and adjacent WING/no-WING tests plus the full backend suite.

## Test plan

- Focused no-WING markdown summary tests.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: the markdown summary is more verbose. Rollback: remove the added markdown
lines and assertions. No hardware access or live write behavior is changed.
