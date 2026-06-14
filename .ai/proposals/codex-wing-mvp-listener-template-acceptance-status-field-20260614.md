# Codex Proposal: WING MVP Listener Template Acceptance Status Field

## Short thesis

Make the generated listener-confirmation template acceptance status an
explicit required failure-detail field for the backend-port prestart gate.

## What I understood

The gate already writes `listener_confirmation_template_acceptance_status` to
busy-port no-WING summaries. Because it is not in the required
`backend_port_prestart_check_failure_detail_fields` list, downstream reviewers
can see the field but are not forced to preserve it in stale plans, packets, or
reports.

## Proposed solution

Add `listener_confirmation_template_acceptance_status` to the shared prestart
failure-detail field constant. Update WING MVP rails and tests so Project
Director, generated progress, and the blocker table all require the field.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`

## Alternatives considered

Leave the status as an optional detail. Rejected because an optional acceptance
status weakens the fail-closed operator-review trail.

Add a new validator-only check without changing the list. Rejected because the
field list is already the shared contract that stale plans and packets compare.

## Risks

The main risk is stale generated expected-plan or handoff packet artifacts that
still use the older field list. Existing mismatch checks should fail closed and
force regeneration.

## Test plan

Run focused no-WING/WING rail tests, regenerate latest MVP progress, run
Project Director and direction guard, then run memory lint, `git diff --check`,
and the standard full test suite.

## Where Kimi may disagree

Kimi may prefer keeping acceptance state outside the failure-detail field list.
The field is already emitted in the failure detail and directly controls
whether the generated template can be accepted, so making it required is the
clearer safety contract.
