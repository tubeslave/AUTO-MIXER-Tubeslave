# Proposal: WING Read-Only Preflight Runbook Prestart Detail Fields

## Short thesis

Update the operator runbook to name the same busy-port prestart detail fields as
the shared contract, and add a static test that keeps those surfaces aligned.

## What I understood

The next real WING step is still read-only connection proof. The system must make
operator handoff safer without claiming hardware readiness or adding any live
write behavior.

## Proposed solution

- Expand the runbook precondition for `backend_port_free` failures.
- List the stable shared detail fields the operator must inspect.
- Keep `command_line` as optional diagnostics because OS permissions can prevent
  collecting it.
- Add a no-WING static test that reads the runbook and asserts the contract
  fields are present.

## Likely files to touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Leave the runbook unchanged and rely on generated summaries.
- Add `command_line` to the shared contract.

## Risks

The main risk is over-constraining optional diagnostics. Keeping `command_line`
outside the shared required field list avoids that.

## Test plan

- Run the focused runbook sync test.
- Run the adjacent WING/no-WING validation stack.
- Run `git diff --check`, memory lint, and the full backend suite.

## Where the other agent may disagree

Kimi may argue that generated handoff packets are enough. The operator runbook is
still a durable manual procedure, so it should not lag behind the generated
contract.
