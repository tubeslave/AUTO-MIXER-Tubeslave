# Proposal: WING MVP Operator Sequence List Markdown Surface

## Short thesis

Add a fail-closed Markdown-surface check so the operator-readable no-WING
`summary.md` cannot omit the promoted operator sequence list fields while JSON
still passes.

## What I understood

The WING read-only proof rail still requires a real physical
`wing_readonly_preflight.v1 summary.json`. Offline no-WING summaries can make
the handoff safer, but the operator-readable Markdown artifact must stay aligned
with the JSON review fields because it is part of the pre-WebSocket inspection
flow.

## Proposed solution

- Add `wing_readonly_operator_sequence_list_markdown_surface`.
- Require Markdown lines for `operator_sequence`, `expected_operator_sequence`,
  and the list summary self-check verdicts.
- Run the check only when the WING handoff artifact is present.
- Emit missing field lines, rendered verdict, and readiness/live denial fields.
- Fail the no-WING summary if required Markdown lines are missing.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/adr/wing-mvp-operator-sequence-list-markdown-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_20260615.md`

## Alternatives considered

- Add the WING-specific fields to the global artifact-manifest required field
  list. Rejected because summaries without a handoff artifact should not be
  forced to contain WING handoff fields.
- Rely on unit-test string assertions only. Rejected because runtime no-WING
  summaries should fail closed on Markdown drift.

## Risks

- The check could run on summaries without WING handoff data. Mitigation: the
  check returns without action unless `wing_readonly_preflight_handoff_packet_artifact`
  is present.

## Test plan

- Focused py_compile for touched Python files.
- Focused no-WING release tests and Project Director tests.
- Direct missing-Markdown-lines regression test.
- no-WING quick with frontend skipped.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Where the other agent may disagree

Kimi may prefer putting the fields in the global artifact-manifest required list.
The conditional check is safer because the WING handoff fields exist only when
the handoff artifact is present.
