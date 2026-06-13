# ADR: WING MVP Review Checklist Contract Source

## Context

The supervised WING Rack MVP remains blocked on real `WING read-only connection
proof`. Before any readiness update or supervised proof, a real read-only
preflight candidate must be manually reviewed. The checklist ids for that review
were duplicated across review decision and no-WING release-check code, while the
handoff packet builder imported them from the release checker.

## Options Considered

1. Keep the checklist ids in `tools/no_wing_release_check.py`.
2. Keep duplicate constants and rely on tests to catch drift.
3. Move checklist ids into `backend/wing_readonly_preflight_contract.py` and keep
   tool-level aliases for compatibility.

## Decision

Use option 3. The shared preflight contract now owns the manual-review checklist
ids required for WING read-only preflight candidate review.

## Why This Won

The checklist is operator safety contract data, not an implementation detail of
the no-WING release checker. A shared source keeps review decision validation,
review tooling, handoff packets, and no-WING validation aligned.

## Rejected Alternatives

Tool-owned checklist ids were rejected because handoff packet semantics should
not depend on release-check internals. Duplicated constants were rejected
because this WING MVP rail has already seen adjacent contract drift.

## Implementation Plan

- Add the shared review checklist id tuple to
  `backend/wing_readonly_preflight_contract.py`.
- Alias review-decision and no-WING constants to the shared tuple.
- Import the shared tuple directly in the handoff packet builder.
- Add focused tests for the shared checklist aliases and generated packet
  values.

## Test Plan

- Compile touched Python modules.
- Run focused review/handoff/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is accidentally changing checklist order or dropping a required
operator confirmation. Rollback would restore local constants, but that would
reintroduce ownership drift in the first WING review path.
