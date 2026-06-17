# Proposal: WING MVP Operator Sequence List Markdown Verdict Render

## Short thesis

Make the operator-readable `summary.md` show the Markdown-surface check's own
verdict fields, not only the operator sequence list fields that the check
inspects.

## What I understood

The active WING blocker is still the real read-only connection proof. Offline
no-WING summaries should reduce operator error before that proof, but must not
grant live authority or imply readiness closure.

## Proposed solution

- Add a defaults builder for
  `wing_readonly_operator_sequence_list_markdown_surface_*` fields.
- Seed the defaults before Markdown generation when a WING handoff artifact is
  present.
- Render the fields near the existing operator sequence list self-check fields.
- After the runtime Markdown check runs, replace the rendered lines with the
  actual missing-lines/rendered verdict.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-mvp-operator-sequence-list-markdown-verdict-render-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_MARKDOWN_VERDICT_RENDER_20260615.md`
- `Docs/INDEX.md`

## Alternatives considered

Relying on `summary.json` was rejected because the operator runbook reviews
Markdown directly. Adding a second post-write Markdown mutation was rejected
because the existing replacement helper can keep a single final Markdown render.

## Risks

The change could drift if the check field names change. Keeping the defaults,
rendering, and replacement loops in one tool limits that risk.

## Test plan

Focused py_compile, focused no-WING/WING tests, temporary-artifact no-WING quick
inspection, Project Director marker check, full pytest, memory lint, and diff
whitespace check.

## Where the other agent may disagree

Kimi may prefer adding these fields to the Markdown-surface required-line
contract itself. I would keep that contract focused on the fields the operator
must review and treat these verdict fields as the check's visible result.
