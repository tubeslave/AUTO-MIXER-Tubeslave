# Brief: WING MVP handoff prestart summary SHA binding

## Task

Tighten the WING read-only handoff packet so the operator-visible packet binds
the no-WING backend prestart summary by both path and SHA256 when the summary
file already exists.

## Context

The supervised WING MVP remains on the first blocker: WING read-only connection
proof. Recent work added accepted-candidate prestart trace checks in the
decision review path. The handoff packet already carries the prestart summary
path, but not the digest that downstream accepted-candidate review requires.

## Constraints

- Do not contact WING or the backend.
- Do not grant live authority.
- Keep the packet read-only and operator-gated.
- Preserve placeholder behavior when the prestart summary file is not present.

## Done

- Handoff packet exposes a prestart summary SHA256 when the file exists.
- no-WING validation fails closed if the handoff packet SHA does not match the
  plan artifact SHA.
- Summary Markdown exposes the packet SHA, plan SHA, and match flag.
- Focused tests cover valid, missing-placeholder, and mismatch cases.
