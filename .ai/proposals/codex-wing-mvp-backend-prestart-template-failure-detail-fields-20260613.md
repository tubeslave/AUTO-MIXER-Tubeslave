# Proposal: WING MVP Backend Prestart Template Failure Detail Fields

## Short thesis

Promote the pending backend-listener template markers into the shared
backend-port prestart failure-detail field list so every handoff/review/proof
surface expects the same fail-closed evidence.

## What I understood

The first blocker remains WING read-only connection proof. The no-WING gate now
surfaces template pending and authority-denial fields, but the shared prestart
failure-detail contract still names only the older listener-review fields.

## Proposed solution

- Add `listener_confirmation_template_status` to the shared failure-detail
  tuple.
- Add `listener_confirmation_template_accepted_by_template`.
- Add `listener_confirmation_template_execution_allowed_by_template`.
- Add `listener_confirmation_template_readiness_closure_allowed_by_template`.
- Regenerate latest progress from the blocker table and update docs/tests.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/`
- `Docs/reports/`

## Alternatives considered

Leaving the newer fields as separate prose keeps compatibility but weakens the
machine-readable contract. Duplicating a second template-specific field list
would add drift risk.

## Risks

Existing stale handoff or plan artifacts with the old field list will fail
contract validation. That is intended because the first hardware step should
use the current fail-closed evidence contract.

## Test plan

Run no-WING release check tests, WING rail tests, Project Director/direction
guard, memory lint, no-WING quick fail-closed validation, and the standard full
pytest command.

## Where the other agent may disagree

Kimi may prefer less churn in the shared tuple. The stricter contract is worth
it because these fields already describe whether a busy-port template can be
used as execution or readiness authority.
