# Codex Proposal: WING MVP Operator Sequence Count Next Action

## Short Thesis

Expose the existing no-WING handoff `operator_sequence_count: 10` requirement in
the first blocker selected action and the matching operator-facing rails.

## What I Understood

The handoff packet and release checker already treat the exact operator
sequence length as a safety invariant. The selected action names individual
steps and indexes, but does not yet require the total sequence count in the
operator-facing text.

## Proposed Solution

- Add `operator_sequence_count: 10` to the first WING blocker next action and
  current command.
- Mirror the requirement in `Docs/runbooks/WING_READONLY_PREFLIGHT.md`,
  `Docs/MVP_WING_SUPERVISED.md`, and autoloop docs.
- Assert the field in Project Director and WING MVP rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Leave the count as release-check-only evidence. Rejected because operator
  reviewers should see the same invariant before supervised proof planning.
- Change live proof behavior. Rejected because this blocker still requires real
  read-only hardware evidence first.

## Risks

The main risk is doc/test drift. No runtime WING behavior or live authority is
changed.

## Test Plan

- Focused WING rail tests.
- Project Director and direction guard JSON checks.
- Full pytest.

## Where The Other Agent May Disagree

Kimi may consider this a small docs rail. The counterargument is that the field
already exists as release-check evidence; making it selected-action-visible
reduces chance of accepting a truncated or expanded handoff sequence.
