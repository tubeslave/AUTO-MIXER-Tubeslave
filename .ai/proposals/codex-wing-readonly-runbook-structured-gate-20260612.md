# Codex Proposal: WING Read-Only Runbook Structured Gate

## Short thesis

The read-only preflight runbook should name the structured supervised prestart
gate contract, not only the derived operator-sequence fields.

## What I understood

The handoff packet and no-WING release summaries now expose
`supervised_prestart_gate` and `expected_supervised_prestart_gate`. The runbook
still instructed reviewers to check only `operator_sequence_*` fields before
moving to supervised fader/gain proof.

## Proposed solution

- Add structured gate checks to `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Require exact match between `supervised_prestart_gate` and
  `expected_supervised_prestart_gate`.
- Explicitly keep `grants_live_authority: false` and
  `requires_accepted_candidate_prestart_trace: true`.
- Extend the existing WING MVP rails test.

## Likely files to touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-readonly-runbook-structured-gate-20260612.md`
- `.ai/proposals/codex-wing-readonly-runbook-structured-gate-20260612.md`
- `Docs/adr/wing-readonly-runbook-structured-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md`

## Alternatives considered

- Leave the runbook on `operator_sequence_*` fields. Rejected because the
  release artifact now has stronger machine-readable evidence.
- Replace operator-sequence checks with only structured gate checks. Rejected
  because human-readable final-step proof remains useful.

## Risks

The runbook now references summary field names directly. This is acceptable
because field-name drift is exactly what the rails test should catch.

## Test plan

- Focused WING MVP runbook test.
- Full WING MVP rails test file.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer to move these fields into a schema doc. That can follow; the
operator runbook still needs the active gate names today.
