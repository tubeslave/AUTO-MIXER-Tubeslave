# ADR: WING MVP proof-plan expected artifact-root violations step gate

## 1. Context

The supervised WING MVP requires accepted read-only preflight candidates to bind
their artifact root before any supervised fader/gain proof is planned. The
validator already rejects physical proof steps when the reported candidate
artifact-root violations map is non-empty. The recomputed expected map also
needs the same direct fail-closed treatment.

## 2. Options considered

- Keep mismatch-only validation for the recomputed expected map.
- Add a direct proof-step gate for non-empty recomputed expected artifact-root
  violations.

## 3. Decision

Add `next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
when the recomputed expected artifact-root violations map is non-empty and proof
steps are present.

## 4. Why this won

It keeps the no-WING release rail fail-closed if a summary tries to hide or stale
out accepted-candidate artifact-root problems while still planning physical WING
proof steps.

## 5. Rejected alternatives

Mismatch-only validation was rejected because it is indirect and less useful as
an operator-facing stop reason.

## 6. Implementation plan

- Extend `tools/no_wing_release_check.py`.
- Add a targeted regression test.
- Update WING MVP docs, runbook, and Codex autoloop guidance.

## 7. Test plan

- Targeted regression test.
- Focused no-WING/handoff suite.
- Rails/director suite.
- Syntax, memory, and diff checks.

## 8. Risks and rollback

If the new gate over-triggers, revert the single validator branch and regression
test. No live mixer operation is added by this change.
