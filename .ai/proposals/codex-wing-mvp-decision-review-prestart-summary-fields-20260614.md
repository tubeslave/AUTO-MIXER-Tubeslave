# Proposal: Show prestart summary fields in decision review reports

## Short thesis

The decision review Markdown should show the same accepted-candidate prestart
summary fields that the JSON ledger already validates.

## What I understood

Project Director requires accepted review source ids to have a valid
accepted-candidate prestart trace before supervised proof planning. The JSON
ledger already includes the trace fields, but the operator-facing Markdown
report did not render the no-WING prestart summary binding and safety flags.

## Proposed solution

- Extend `tools/review_wing_readonly_preflight_decisions.py` to render
  `backend_port_prestart_summary_required`, path, SHA256, type, status,
  connection policy, no-contact flags, and `backend_port_prestart_check_status`
  for each candidate review row.
- Add focused Markdown assertions in the decision review CLI test.
- Add this report to WING MVP evidence.

## Likely files to touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Leaving the fields JSON-only was rejected because the operator review artifact
is the Markdown report. Duplicating validation logic was rejected because the
ledger already validates the contract; this change is presentation-only.

## Risks

The report becomes slightly longer. That is acceptable because the extra rows
are exactly the manual review evidence required before supervised proof.

## Test plan

- Focused decision review CLI test.
- Focused no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may argue that the JSON ledger is enough. I recommend rendering the fields
because the operator-facing report is the practical review surface.
