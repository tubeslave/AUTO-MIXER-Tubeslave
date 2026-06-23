# Codex Proposal: WING MVP Operator Markdown Value-Gate Authority Denial

## Short thesis

The no-WING operator Markdown contract should fail closed unless the
accepted-candidate value-gate authority-denial fields are listed as required
Markdown lines.

## What I understood

The Project Director-selected action already requires the value-gate
summary-surface denial fields. The generated `summary.md` renders those fields,
but the `operator_markdown_summary_required_field_lines` list only required the
value-gate mismatch and match fields, leaving the authority-denial lines
outside the explicit Markdown contract.

## Proposed solution

Add
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed`
and
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted`
to `NEXT_WING_PROOF_PLAN_OPERATOR_MARKDOWN_REQUIRED_FIELD_LINES`.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

Leaving the fields as rendered-only was rejected because it lets the operator
surface pass the Markdown contract without requiring the denial fields.

## Risks

This may require updating Project Director baseline marker expectations. The
risk is acceptable because the intended result is stricter fail-closed behavior.

## Test plan

Run focused no-WING and Project Director tests, then no-WING quick validation.
Run the full test suite if focused validation passes.

## Where the other agent may disagree

Kimi may argue this is another contract-surface hardening step rather than a
direct hardware proof. That is true, but the current environment still lacks a
real WING read-only artifact; this change prevents paper readiness from
advancing without explicit authority denial.
