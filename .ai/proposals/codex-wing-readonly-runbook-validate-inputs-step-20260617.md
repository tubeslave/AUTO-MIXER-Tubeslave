# Codex Proposal: WING Read-Only Runbook Validate Inputs Step

## Short thesis

Update the operator-facing WING read-only runbook so it matches the committed 11-step handoff contract.

## What I understood

The code now inserts `validate_readonly_preflight_inputs` before `capture_readonly_preflight`. The runbook still tells operators to go directly from plan preview to capture and still names the old operator sequence count and indices.

## Proposed solution

- Add the `--validate-run-inputs-only` command between preview and capture.
- Update the handoff summary expectations from 10 to 11 steps.
- Shift capture to index 5, MVP blocker digest review to index 9, and the supervised prestart gate to index 10.
- Require the new validate-inputs command binding fields in the runbook.

## Likely files to touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/briefs/wing-readonly-runbook-validate-inputs-step-20260617.md`
- `Docs/adr/wing-readonly-runbook-validate-inputs-step-20260617.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_VALIDATE_INPUTS_STEP_20260617.md`

## Alternatives considered

Leave the runbook unchanged and rely on generated packet output. Rejected because operators use the runbook before hardware capture, so stale guidance can bypass the validation-only dry run.

## Risks

The main risk is another stale surface still referencing the old handoff order. This patch keeps scope narrow and relies on no-WING checks plus focused tests to catch contract drift.

## Test plan

- Stale-string search over the runbook.
- Focused WING handoff/no-WING tests.
- no-WING quick dry run.
- Full pytest.

## Where the other agent may disagree

Kimi may prefer updating every downstream prompt/readme in the same change. I would keep this package operator-runbook scoped, then let the guard pick the next stale surface if it remains target-aligned.
