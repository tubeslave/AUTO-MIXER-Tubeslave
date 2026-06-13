# ADR: WING MVP Handoff Safety Contract Source

## Context

The supervised WING Rack MVP remains blocked on real `WING read-only connection
proof`. The no-WING handoff packet must explicitly forbid live request types and
state that key MVP blockers remain open. Those safety lists were owned by the
no-WING release checker while the handoff builder imported them from that tool.

## Options Considered

1. Keep the safety lists in `tools/no_wing_release_check.py`.
2. Duplicate the lists in the handoff packet builder and rely on tests.
3. Move the lists into `backend/wing_readonly_preflight_contract.py` and keep
   tool-level aliases for compatibility.

## Decision

Use option 3. The shared preflight contract now owns the handoff forbidden
request types and not-closed blocker ids. Because that contract intentionally
contains forbidden live-write request names as audit strings, it is covered by
the lab-only `proof_and_release_check_tools` write-surface inventory entry.

## Why This Won

These lists are safety contract data for the operator handoff, not an
implementation detail of the no-WING release checker. A shared source keeps the
packet generator and validator aligned while preserving existing public names.

## Rejected Alternatives

Tool-owned safety lists were rejected because the handoff builder should not
depend on the validator for packet semantics. Duplicated lists were rejected
because this WING MVP rail has already seen adjacent contract drift.

## Implementation Plan

- Add shared forbidden-request and not-closed-blocker tuples to
  `backend/wing_readonly_preflight_contract.py`.
- Alias the no-WING validator constants to the shared tuples.
- Import the shared tuples directly in the handoff packet builder.
- Register the shared contract file under the existing lab-only proof/release
  check write-surface inventory entry.
- Add focused tests for the shared contract aliases and generated packet values.

## Test Plan

- Compile touched Python modules.
- Run focused static-scan, handoff, no-WING, director, and rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is accidentally changing list order or dropping a forbidden live
request. Rollback would restore the tool-local list, but that would reintroduce
ownership drift in the first WING handoff path.
