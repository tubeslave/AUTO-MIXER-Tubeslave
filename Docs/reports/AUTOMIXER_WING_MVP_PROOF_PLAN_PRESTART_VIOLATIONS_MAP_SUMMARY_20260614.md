# Automixer WING MVP Proof Plan Prestart Violations Map Summary

## Summary

The no-WING `next_wing_proof_plan` summary now preserves and renders
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.
The next proof-plan validator fails closed when this map is missing or differs
from the decision-review ledger's
`accepted_candidate_prestart_contract_violations_by_source`.

## Boundary

This is an offline artifact-contract change only. It does not start, stop, or
reconfigure the backend listener, connect to WING, query the console, mutate
console state, grant live authority, or close readiness.

## Validation

- PASS: focused no-WING release-check tests around `next_wing_proof_plan` and
  `candidate_prestart`: `7 passed, 255 deselected in 0.90s`.
- PASS: focused no-WING, supervised proof-plan, and WING rail tests:
  `281 passed in 3.03s`.
- PASS: regenerated `Docs/reports/MVP_PROGRESS_LATEST.md`.
- PASS: Project Director stayed on `WING read-only connection proof` with
  guard status `on_target`, score `90`, `stop_broad_work: true`, and the new
  report in selected evidence files.
- PASS: Direction guard stayed `on_target`, score `90`, with no drift items.
- PASS: actual `tools/no_wing_release_check.py --quick` run failed closed on
  the existing backend listener before any backend/WING session.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile` for the
  changed backend/tools Python modules.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3300 passed, 1 skipped, 1 warning in 159.62s`.
