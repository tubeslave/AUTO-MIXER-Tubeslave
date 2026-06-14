# Codex Proposal: WING MVP Current Evidence Latest Match Reports

## Short thesis

Refresh the first WING blocker evidence list so Project Director sees the
latest handoff/prestart match reports during Goal-mode cycles.

## What I understood

The system is still blocked on real WING read-only proof. The newest validated
offline reports already strengthen gate equality and no-WING prestart summary
match semantics, but the first blocker evidence row does not yet cite them.

## Proposed solution

Add these reports to `.ai/project-layer/wing_mvp_blockers.md`:

- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md`

Then assert them through blocker-table parsing and Project Director
`selected_evidence_files`.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`
- Council trail files for this task.

## Alternatives considered

- Leave the evidence row unchanged and rely on `Docs/INDEX.md`.
- Teach Project Director to infer latest reports from the docs index.

## Risks

The main risk is evidence-list churn making tests brittle. The mitigation is to
assert only the three reports that matter for this gap, not a full exact list.

## Test plan

Run focused rails/Director tests, Project Director JSON, direction guard JSON,
memory lint, diff check, no-WING quick check, and full pytest.

## Where the other agent may disagree

Kimi may prefer deriving latest reports automatically. I would keep this
explicit because the blocker row is already the audited source of Director
evidence and should remain readable by humans.
