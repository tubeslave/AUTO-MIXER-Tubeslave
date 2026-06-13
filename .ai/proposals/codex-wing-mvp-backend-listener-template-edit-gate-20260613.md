# Proposal: WING MVP Backend Listener Template Edit Gate

## Short thesis

Make backend-listener confirmation validation reject half-edited generated
templates while still accepting a fully reviewed operator confirmation.

## What I understood

The first MVP blocker remains the WING read-only connection proof. The current
template helps operators start from exact listener diagnostics, but a template
with placeholders or pending markers should not clear the prestart gate.

## Proposed solution

- Add validation for placeholder `operator_reviewed_at` and `operator_name`.
- Reject `template_status: pending_operator_edit` on accepted confirmations.
- If `accepted_by_template` is present, require it to be `true`.
- Add tests for partial template edit rejection and full template edit
  acceptance.
- Update the runbook and evidence report.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/`

## Alternatives considered

- Reject all confirmations that contain template metadata. That is safe but
  wastes the generated template path.
- Keep current validation. That leaves placeholder review fields as a possible
  false acceptance path.

## Risks

Operators editing the generated template must change one extra marker field.
The runbook and generated required-edits list make that explicit.

## Test plan

Run focused no-WING confirmation tests, MVP rail tests, Project Director, and
the full standard pytest command before commit.

## Where the other agent may disagree

Kimi may prefer removing template-origin fields before acceptance. The chosen
gate keeps those fields as auditable provenance while requiring an explicit
operator-reviewed marker.
