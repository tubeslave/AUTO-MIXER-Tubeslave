# ADR: WING MVP Selected Blocker Markdown Surface

## Context

The current WING MVP blocker is still `WING read-only connection proof`. The
no-WING JSON summary exposes the selected blocker, but the human-readable
markdown artifact should also surface it before the large handoff packet detail
section.

## Options considered

1. Keep selected blocker details only in JSON and the nested packet markdown
   section.
2. Add a compact top-level markdown section for the selected blocker.
3. Create a new artifact file just for blocker status.

## Decision

Add a compact `WING Read-Only MVP Blocker Summary` section to no-WING
`summary.md`.

## Why this won

This keeps the operator review path direct and preserves the safety boundary:
the markdown says which blocker is active, confirms the packet does not close
it, and states that readiness closure and live authority remain false.

## Rejected alternatives

Nested-only markdown is too easy to miss during a handoff. A separate artifact
would add file churn without stronger evidence.

## Implementation plan

- Render the markdown section from existing top-level summary fields.
- Assert the section and safety-denial lines in the no-WING summary test.
- Validate with focused tests, no-WING quick, manifest verifier, memory lint,
  and full pytest.

## Test plan

- `py_compile` for touched files.
- Focused no-WING/WING guard pytest.
- no-WING quick artifact inspection.
- Artifact manifest verifier.
- Full pytest.

## Risks and rollback

This is artifact rendering only. Rollback removes the markdown section and the
corresponding assertions.
