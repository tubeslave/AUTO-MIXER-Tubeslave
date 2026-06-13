# ADR: WING MVP Review Decision Template Contract Source

## Context

The supervised WING Rack MVP remains blocked on real `WING read-only connection
proof`. A clean read-only preflight candidate must be followed by an
operator-edited review decision JSON before readiness or supervised proof can
proceed. The required fields for that template were duplicated in the review
tool and the no-WING release checker.

## Options Considered

1. Keep the template required fields duplicated.
2. Move the fields into the review tool and make validation import that tool.
3. Move the fields into `backend/wing_readonly_preflight_contract.py` and keep
   tool-level aliases for compatibility.

## Decision

Use option 3. The shared preflight contract now owns the required fields for the
WING read-only review decision template.

## Why This Won

These fields define the operator decision artifact contract. A shared source
keeps review reports, generated decision templates, and no-WING validation
aligned without making one tool own another tool's semantics.

## Rejected Alternatives

Duplicated fields were rejected because the first WING review path already has
many hard safety invariants. Tool-owned fields were rejected because the
no-WING validator should not derive artifact semantics from the review tool.

## Implementation Plan

- Add the shared template required-field tuple to
  `backend/wing_readonly_preflight_contract.py`.
- Alias review-tool and no-WING constants to the shared tuple.
- Add focused tests for review reports and no-WING validation details.

## Test Plan

- Compile touched Python modules.
- Run focused review/no-WING/director tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is accidentally changing field order or dropping a required field.
Rollback would restore local constants, but that would reintroduce drift in the
operator decision artifact path.
