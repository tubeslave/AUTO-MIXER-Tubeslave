# Codex Proposal: Project Goal Supervised AutoMix Structured Gate

## Short thesis

The project goal capability matrix should mention the structured WING handoff
gate before widening Supervised AutoMix, because it is the current proof
boundary for physical WING steps.

## What I understood

The WING runbooks, blocker rows, handoff packet, release summaries, and MVP doc
now require `supervised_prestart_gate` to match
`expected_supervised_prestart_gate`. `mode_supervised_automix.next_step` still
mentioned only accepted-candidate prestart trace from read-only preflight
review.

## Proposed solution

- Update `backend/project_goal_capability_matrix.py` for
  `mode_supervised_automix`.
- Extend `tests/test_project_goal_capability_matrix.py` to assert both
  structured gate field names.
- Record ADR/report evidence.

## Likely files to touch

- `backend/project_goal_capability_matrix.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/project-goal-supervised-automix-structured-gate-20260612.md`
- `.ai/proposals/codex-project-goal-supervised-automix-structured-gate-20260612.md`
- `Docs/adr/project-goal-supervised-automix-structured-gate-20260612.md`
- `Docs/reports/AUTOMIXER_PROJECT_GOAL_SUPERVISED_AUTOMIX_STRUCTURED_GATE_20260612.md`

## Alternatives considered

- Leave the matrix unchanged because lower-level WING docs have the gate.
  Rejected because the project goal matrix is the long-term objective map.
- Replace accepted-candidate trace wording with structured gate wording.
  Rejected because both gates are part of the current proof boundary.

## Risks

The row text becomes more specific to WING. This is acceptable because the row
already calls out WING physical proof before widening.

## Test plan

- Focused project goal capability matrix test.
- Affected project goal/WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer adding structured fields to the dataclass instead of wording.
That is worth considering if more goal rows need machine-readable gate metadata.
