# Proposal: WING MVP Operator Sequence List Markdown Verdict Self-Check

## Short thesis

Extend the existing operator-sequence-list Markdown-surface contract so it also
requires the check's own visible verdict fields.

## What I understood

The live WING blocker is still waiting on a real read-only preflight artifact.
Offline no-WING hardening is useful only when it prevents operator-visible
drift without implying live authority.

## Proposed solution

- Add the `wing_readonly_operator_sequence_list_markdown_surface_*` verdict
  fields to `WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES`.
- Keep the check conditional on a WING handoff artifact.
- Add a regression test that omits one verdict line while all other required
  lines are present.
- Keep all denial fields false.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-mvp-operator-sequence-list-markdown-verdict-self-check-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_MARKDOWN_VERDICT_SELF_CHECK_20260615.md`
- `Docs/INDEX.md`

## Alternatives considered

Leaving the rendered verdict fields covered only by string assertions was
rejected because runtime no-WING summaries should fail closed on artifact drift.
Adding a second independent check was rejected because the existing conditional
Markdown-surface check already owns this field family.

## Risks

The required field list now contains fields that describe the check itself.
This is acceptable because defaults are seeded before Markdown generation and
the check runs after the Markdown text is assembled.

## Test plan

Focused py_compile, focused no-WING/WING tests, direct missing-verdict-line
regression test, no-WING quick inspection, Project Director marker check, full
pytest, memory lint, and diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a separate `markdown_verdict_surface` check. I would avoid it
for now because the behavior is narrow and belongs to the existing conditional
Markdown-surface contract.
