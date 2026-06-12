# Proposal: WING Read-Only Preflight Summary Prestart Contract

## Short thesis

Add the backend-port prestart contract to real WING read-only preflight summary
artifacts and make evidence intake validate it.

## What I understood

The next meaningful progress toward the hardware proof is strengthening the
artifact that will come back from the real WING read-only run. It should be
self-describing about the prestart gate, not rely only on the expected plan.

## Proposed solution

- Add `backend_port_prestart_check_required`,
  `backend_port_prestart_check_must_pass_before_starting_backend`,
  `backend_port_prestart_check_command`, and
  `backend_port_prestart_check_failure_detail_fields` to runner summaries.
- Validate those fields in `backend/wing_readonly_preflight_evidence.py`.
- Update tests and fixtures so missing or mismatched summary fields fail closed.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Keep the contract only in the expected plan.
- Validate only the JSON plan and not the real-run summary.

## Risks

Older captured preflight artifacts without these fields will need replacement or
explicit migration. That is acceptable because the current blocker still waits
for the first real WING proof.

## Test plan

- Focused runner and evidence tests.
- Adjacent WING/no-WING validation stack.
- `git diff --check`, memory lint, and full backend suite.

## Where the other agent may disagree

Kimi may argue the expected plan is enough. The real summary is the durable proof
artifact, so it should independently state the prestart contract it depended on.
