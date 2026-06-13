# ADR: WING MVP Decision Review Prestart Ready Gate

## Context

The WING MVP requires a real read-only preflight artifact, an accepted operator
decision, and a valid accepted-candidate backend prestart trace before
supervised fader/gain proof planning. The proof plan already enforces this, but
the standalone decision-review report could still present the accepted decision
as ready without making the prestart trace gate authoritative.

## Options considered

1. Keep the decision-review CLI as a pure accepted/rejected decision report.
2. Make the decision-review ready status require the accepted-candidate
   prestart trace.
3. Add a separate post-decision CLI for the prestart trace gate.

## Decision

Use option 2.

## Why this won

Operators and release checks already consume the decision-review output as part
of the WING handoff path. Returning `accepted_decision_ready` without the
accepted-candidate prestart trace leaves room to misread an accepted JSON as
the next manual proof gate. Keeping the check in the ledger makes the CLI,
no-WING release summary, and proof plan agree on the same source ids.

## Rejected alternatives

- Option 1 leaves a misleading ready status on an important operator surface.
- Option 3 adds another handoff artifact without improving the core contract.

## Implementation plan

- Move the accepted-candidate prestart blocker id into the shared WING preflight
  contract.
- Compute accepted prestart source ids in the review decision ledger.
- Keep manual readiness unavailable when accepted source ids lack the prestart
  trace.
- Make the decision-review CLI block instead of returning exit code 0 in that
  case.
- Render the prestart validity fields in no-WING release summaries.

## Test plan

- Focused pytest for review decision ledger/tool, no-WING release check, and
  supervised proof plan.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback restores the older decision-review semantics. It does not change live
WING behavior, but it would reintroduce an ambiguous ready status for accepted
decisions without valid prestart trace evidence.
