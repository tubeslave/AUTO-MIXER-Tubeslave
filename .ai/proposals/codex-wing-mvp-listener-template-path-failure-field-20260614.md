# Codex Proposal: WING MVP Listener Template Path Failure Field

## Short thesis

Require the generated listener-confirmation template path in the backend-port
prestart failure-detail contract.

## What I understood

When a backend listener already occupies the expected port, the no-WING gate
must fail closed and tell the operator where the generated confirmation
template lives. The summary already emits `listener_confirmation_template_path`,
but stale plan/packet checks can still pass the failure-detail field list
without requiring that path.

## Proposed solution

Add `listener_confirmation_template_path` to
`WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
Update WING rails and tests so Project Director, generated progress, the
blocker table, and docs require the path inside the mandatory field list.

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

Leave the path as a separate prose-only requirement. Rejected because the
shared field list is what stale summaries and packets can compare
mechanically.

Require only the acceptance status. Rejected because acceptance status without
the path does not tell the operator which template artifact to review and
rerun.

## Risks

The change can make stale expected plans or handoff packets fail validation
until regenerated. That is desired fail-closed behavior.

## Test plan

Run focused no-WING/WING rail tests, regenerate latest MVP progress, run
Project Director and direction guard, run memory lint, `git diff --check`, and
the standard full test suite.

## Where Kimi may disagree

Kimi may argue the path should remain a separate top-level detail rather than
part of the failure-detail list. Since the field is required to continue the
operator confirmation workflow, putting it in the shared list makes stale
artifacts fail in the right place.
