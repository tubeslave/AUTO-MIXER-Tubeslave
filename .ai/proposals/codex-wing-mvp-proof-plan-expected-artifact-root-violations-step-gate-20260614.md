# Proposal: WING MVP proof-plan expected artifact-root violations step gate

## 1. Short thesis

The proof-plan validator should reject physical WING proof steps whenever the
recomputed accepted-candidate artifact-root contract violations map is non-empty,
not only when the reported summary map is non-empty.

## 2. What I understood

The current MVP boundary is still a supervised WING Rack read-only proof rail.
Before supervised fader/gain proof, accepted read-only candidates must prove both
prestart binding and artifact-root binding. A stale or incomplete summary must
not let planned physical steps continue.

## 3. Proposed solution

Add `next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
inside `validate_supervised_proof_report_next_wing_plan` when the recomputed
expected artifact-root violations map is non-empty and proof steps exist. Keep
the existing candidate artifact-root blocker in force.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING rail docs and runbook
- Council ADR/report files

## 5. Alternatives considered

Relying only on mismatch violations would catch stale summaries, but it would
not produce a direct stop reason tied to physical proof steps. That is weaker
for operator-facing handoff and Project Director guidance.

## 6. Risks

The main risk is over-triggering the new violation. It is scoped to non-empty
recomputed expected artifact-root violations and existing proof steps, so valid
no-WING summaries remain unaffected.

## 7. Test plan

Add a regression that removes the accepted candidate `artifact_root` and expects
the new step-gate violation, then run focused no-WING and director suites.

## 8. Where the other agent may disagree

Kimi may argue the existing mismatch violations were enough. The counterpoint is
that proof-step admission needs a direct fail-closed reason, not only a diagnostic
staleness signal.
