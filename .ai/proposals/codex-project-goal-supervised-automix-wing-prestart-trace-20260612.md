# Codex Proposal: Project Goal Supervised AutoMix WING Prestart Trace

## Short thesis

The project goal capability matrix should state that Supervised AutoMix widening depends on accepted-candidate prestart trace from WING read-only preflight review.

## What I understood

The proof plan, release gate, runbook, and registry now enforce the candidate-prestart boundary. The high-level goal matrix still said only to add physical proof before widening, which is true but incomplete for the current WING path.

## Proposed solution

- Update `mode_supervised_automix.next_step`.
- Add a matrix regression test that requires `accepted-candidate prestart trace`, `read-only preflight review`, and `before widening` wording.
- Keep current live-write kinds and autonomy ceiling unchanged.

## Likely files to touch

- `backend/project_goal_capability_matrix.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Leave the matrix generic. That makes long-term status reporting less precise than the WING proof gates.
- Add a separate capability row. Not needed; this is a prerequisite on the existing Supervised AutoMix mode.

## Risks

- Minimal. This is declarative status wording and a regression test.

## Test plan

- Focused matrix pytest.
- Affected matrix/registry/WING proof tests.
- Full standard pytest suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer to add WING-specific fields to the matrix dataclass. I would keep this as wording until multiple mode rows need the same structured gate.
