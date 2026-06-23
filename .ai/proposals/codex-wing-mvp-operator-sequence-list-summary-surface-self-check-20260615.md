# Proposal: WING MVP Operator Sequence List Summary Surface Self-Check

## Short thesis

Add a fail-closed self-check so the flat `operator_sequence` and
`expected_operator_sequence` review fields cannot silently drift from the handoff
artifact or expected checklist contract.

## What I understood

The current WING read-only rail still waits for a real physical
`wing_readonly_preflight.v1 summary.json`. Offline no-WING summaries can improve
the operator handoff, but they must not create implied readiness. The promoted
operator sequence lists are review aids, so they need their own consistency
verdict and denial flags.

## Proposed solution

- Add a `wing_readonly_operator_sequence_list_summary_surface` check.
- Compare promoted top-level lists with the nested handoff artifact lists.
- Compare both promoted lists with the shared expected handoff operator sequence.
- Compare promoted list lengths with the existing count fields.
- Emit mismatches, a match verdict, and readiness/live-authority denial fields.
- Fail the no-WING summary when the match verdict is false.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/adr/wing-mvp-operator-sequence-list-summary-surface-self-check-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_SUMMARY_SURFACE_SELF_CHECK_20260615.md`

## Alternatives considered

- Rely only on the existing sequence count and digest checks. Rejected because
  they prove the handoff artifact, not that the promoted flat review fields are
  still present and aligned.
- Add documentation only. Rejected because the summary must fail closed on drift.

## Risks

- The self-check could duplicate existing contract gates. Mitigation: keep it
  narrowly focused on the promoted summary surface and retain the existing
  count/hash checks as authoritative identity checks.

## Test plan

- Focused py_compile for touched Python files.
- Focused no-WING release tests.
- Direct mismatch regression test for the self-check.
- no-WING quick with frontend skipped.
- Project Director marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Where the other agent may disagree

Kimi may argue that the existing digest contract already proves the sequence.
The counterpoint is that this change guards the flat operator review surface,
not the nested handoff packet identity.
