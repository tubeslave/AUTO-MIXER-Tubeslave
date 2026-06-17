# Codex Proposal: WING MVP no-WING Proof Plan Required Field Markdown Summary

## Short thesis

Render the required-field contract in `summary.md` so no-WING proof-plan failures
are visible in the operator handoff surface, not only in JSON.

## What I understood

The supervised WING MVP is still waiting on real read-only hardware evidence.
Offline work should therefore improve fail-closed diagnosis and operator
handoff clarity without granting any execution authority. The aggregated
`next_wing_proof_plan` already propagates the required and missing summary field
lists into `summary.json`; the markdown output should match.

## Proposed solution

- Add markdown lines for `required_supervised_proof_plan_summary_fields`.
- Add markdown lines for `missing_required_supervised_proof_plan_summary_fields`.
- Extend the existing next-plan markdown test for the empty missing-list case.
- Extend the missing-field regression so it writes a no-WING summary and checks
  the concrete missing field appears in markdown.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Leave the contract visible only in JSON. Rejected because the operator
  handoff path is markdown-first in many local workflows.
- Add a new section instead of lines under `## Next WING Proof Plan`. Rejected
  because these fields describe that exact proof-plan summary and should stay
  near the other next-plan diagnostics.

## Risks

- The markdown section grows longer. This is acceptable because these fields are
  only emitted when a next WING proof plan result exists.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field`
- Run focused no-WING/WING rail tests before handoff.

## Where the other agent may disagree

Kimi may prefer keeping markdown terse and relying on JSON for full detail. I
kept the fields visible because this is a fail-closed operator gate and hidden
missing-field causes slow down real WING proof prep.
