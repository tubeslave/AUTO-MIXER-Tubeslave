# WING MVP Proof Plan Expected Artifact Trace Blocker

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff and proof-plan summaries already expose accepted-candidate
expected artifact trace fields, and the handoff text requires an
expected-artifact-bound accepted source id before supervised fader/gain proof.

## Task

Make `backend/supervised_wing_proof_plan.py` fail closed before producing
physical proof steps when the accepted read-only preflight candidate does not
have a complete expected artifact trace.

## Constraints

- Offline plan logic only.
- Do not contact WING, start a backend, or query a console.
- Do not grant live authority or readiness closure.
- Preserve the existing manual review and prestart/artifact-root blockers.
- Keep physical proof planning limited to existing supervised fader and
  negative gain paths.

## Test Plan

- Focused proof-plan/no-WING/director tests.
- Full `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.
- `tools/no_wing_release_check.py --quick --skip-frontend`.
- `tools/memory/memory_lint.py`.
- `/opt/homebrew/bin/git diff --check`.
