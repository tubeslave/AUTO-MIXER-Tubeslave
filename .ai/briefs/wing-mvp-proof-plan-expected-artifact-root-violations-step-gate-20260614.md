# Brief: WING MVP proof-plan expected artifact-root violations step gate

## Task

Close the fail-closed gap where recomputed accepted-candidate artifact-root
contract violations could be visible only as diagnostics mismatches while a
next WING supervised proof plan still contained physical proof steps.

## Scope

- Keep the active rail on the supervised WING Rack MVP.
- Do not grant live authority.
- Add a direct no-WING release violation when the recomputed
  `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`
  map is non-empty and proof steps are present.
- Document the marker in the WING read-only runbook and Goal-mode rails.

## Test Plan

- Target the new no-WING regression test.
- Run the focused no-WING/handoff suite.
- Run rails/director tests.
- Run syntax, memory, and diff hygiene checks.
