# Proposal: WING MVP Backend Prestart Template Review Fields

## Short thesis

Promote the remaining generated listener-template review markers into the
shared backend-port prestart failure-detail contract, so every no-WING and
handoff surface requires the same fail-closed template evidence.

## What I understood

The current MVP must stay on the `WING read-only connection proof` blocker.
Offline and no-WING artifacts can prepare the operator path, but they cannot
close readiness. The busy-port no-WING summary already emits enough template
evidence to prove a generated confirmation template is still pending operator
review, but the shared contract is incomplete.

## Proposed solution

- Add these fields to
  `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`:
  - `listener_confirmation_template_safe_to_stop_listener_by_gate`
  - `listener_confirmation_template_live_authority_granted`
  - `listener_confirmation_template_operator_placeholder_fields`
  - `listener_confirmation_template_required_edits`
- Update docs and autoloop rails that quote the exact field list.
- Add focused tests proving the shared constant carries the fields.
- Add a report and ADR that record the safety boundary and validation.

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
- `Docs/INDEX.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Leave the fields only as separate prose in Project Director and docs. That is
weaker because future tools can compare the shared field list while missing
operator-critical template review details.

## Risks

The main risk is a brittle doc/test mismatch because the exact field list is
quoted in several places. Keeping the tuple order consistent across surfaces
reduces that risk.

## Test plan

- Compile the touched Python files.
- Run focused WING/no-WING rail tests.
- Run the no-WING quick check and inspect the saved busy-port summary.
- Run Project Director, direction guard, memory lint, and full pytest.

## Where the other agent may disagree

Kimi may argue that these fields are already enforced by separate summary
checks. The counterpoint is that the shared contract should list the complete
failure detail evidence before a real WING read-only preflight can proceed.
