# ADR: WING MVP Proof Plan Prestart Violations Map Summary

## Context

The supervised WING MVP cannot move to supervised fader/gain proof while an
accepted read-only preflight candidate has unresolved prestart-contract
violations. The decision-review ledger exposes
`accepted_candidate_prestart_contract_violations_by_source`; the supervised
proof-plan summary exposes
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.

The no-WING release summary did not carry the proof-plan violations map into
its `next_wing_proof_plan` surface, and its schema validator did not compare
that map against the decision-review ledger.

## Options considered

1. Keep the map only in `backend/supervised_wing_proof_plan.py`.
2. Render the map in no-WING output without validating it.
3. Render the map and fail validation when it is missing or mismatched.

## Decision

Render
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`
in no-WING JSON/Markdown summaries and make the validator compare it against
the decision-review ledger map.

## Why this won

The no-WING summary is the practical handoff artifact before the real WING
session. It must show `{}` explicitly before supervised proof planning. A
missing or stale map should fail closed, not rely on the operator to infer the
state from another nested object.

## Rejected alternatives

Keeping the map only in the backend object leaves the handoff surface weaker
than the blocker contract. Rendering without validation creates a stale-report
gap.

## Implementation plan

- Parse the proof-plan map using the existing `_string_list_map` helper.
- Add mismatch and missing-map violations in the next proof-plan validator.
- Preserve the map in `next_wing_proof_plan_summary`.
- Render the map in `summary.md`.
- Add positive and negative regression tests.
- Add this report to the WING blocker evidence list and Docs index.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py -k "next_wing_proof_plan or candidate_prestart" --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to older offline proof-plan summaries failing validation until
regenerated. Rollback removes the no-WING summary/validator field and the
associated tests/docs. No live WING, backend listener, or readiness state is
changed.
