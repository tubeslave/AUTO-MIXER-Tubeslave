# Proposal: WING Read-Only Preflight Prestart Plan Evidence Gate

## Short thesis

The expected-plan evidence gate should fail closed when the plan does not preserve the mandatory prestart no-WING release check contract.

## What I understood

The current MVP remains blocked on real WING read-only connection proof. Offline work may only harden the tooling around that proof path. The runbook requires a no-WING quick check before the hardware session; a captured artifact should not be reviewable if its linked expected plan weakens that boundary.

## Proposed solution

- Reuse `wing_readonly_preflight_contract.py` constants in `backend/wing_readonly_preflight_evidence.py`.
- Validate these expected-plan fields during file-backed artifact review:
  - `backend_port_prestart_check_required`
  - `backend_port_prestart_check_must_pass_before_starting_backend`
  - `backend_port_prestart_check_command`
  - `backend_port_prestart_check_failure_detail_fields`
- Extend the expected-plan policy metadata regression to prove tampering is rejected.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `Docs/adr/wing-readonly-preflight-prestart-plan-evidence-gate-20260611.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PRESTART_PLAN_EVIDENCE_GATE_20260611.md`

## Alternatives considered

- Trust the runner's real-run `validate_expected_plan`: rejected because artifact review should also fail closed for archived or hand-edited expected-plan files.
- Require a completed no-WING artifact inside the preflight summary: rejected for this slice because the runbook's current boundary is a plan-level prestart requirement, not a real-WING readiness proof.

## Risks

- Older local sample artifacts with incomplete expected-plan metadata may fail review until regenerated.
- This does not prove real WING connectivity and must not close the MVP blocker.

## Test plan

- Focused evidence validator tests.
- Adjacent WING preflight/no-WING tests.
- Full backend suite if the focused stack passes.

## Where the other agent may disagree

Kimi may argue for requiring the actual no-WING summary hash in the plan. That may be useful later, but it changes the runbook contract more broadly. This slice keeps the current prestart boundary explicit and fail-closed without adding a new artifact dependency.
