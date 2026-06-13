# ADR: WING handoff command artifact binding

## Context

The WING read-only handoff is the bridge from no-WING validation to a real
read-only hardware capture. It already records expected artifact globs and
checks command equality, but the no-WING summary should explicitly show command
bindings for capture and review steps.

## Options considered

1. Keep command binding implicit in whole-command equality.
2. Expose command-level plan, artifact-root, and decision-root bindings in the
   summary and fail closed on mismatches.
3. Defer binding checks to the real WING run.

## Decision

Expose and validate command-level artifact bindings for capture, artifact
review, and decision review in the no-WING handoff summary.

## Why this won

The handoff must be operator-auditable before hardware is touched. Explicit
fields make it clear that capture and review commands point at the packet's
expected plan, artifact root, and decision root.

## Rejected alternatives

Implicit command equality was rejected because it hides the safety property in a
long command list. Deferring to the real run was rejected because stale packets
should fail before the hardware session.

## Implementation plan

- Extract command flag values in `tools/no_wing_release_check.py`.
- Add mismatch violations and summary fields.
- Add focused positive and negative tests.
- Update WING MVP rail text and docs.

## Test plan

- Focused no-WING handoff tests.
- no-WING quick release check.
- Project Director/WING rail tests.
- Full pytest before commit.

## Risks and rollback

Older hand-edited packets may fail until regenerated. Rollback is to remove the
new summary fields and mismatch checks, but that weakens operator visibility.
