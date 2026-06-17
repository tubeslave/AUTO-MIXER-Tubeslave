# Codex Proposal: WING MVP Selected Blocker Markdown Surface

## Short thesis

Add a compact markdown section so the no-WING human-readable artifact shows the
current WING blocker and its non-closure status without requiring a deep scan of
the handoff packet section.

## What I understood

The project is still blocked on real `wing_readonly_preflight.v1` evidence. The
current summary JSON now carries `selected_mvp_blocker`, but the markdown
artifact should also show that the packet remains tied to `WING read-only
connection proof` and does not grant readiness or live authority.

## Proposed solution

Render `WING Read-Only MVP Blocker Summary` before the handoff packet artifact
section in `summary.md`, using values already present in the top-level summary.
Add assertions to the existing no-WING summary test.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-selected-blocker-markdown-surface-20260617.md`
- `Docs/adr/wing-mvp-selected-blocker-markdown-surface-20260617.md`
- `Docs/reports/AUTOMIXER_WING_MVP_SELECTED_BLOCKER_MARKDOWN_SURFACE_20260617.md`

## Alternatives considered

- Leave blocker details only in JSON and the nested handoff packet markdown
  section. That preserves evidence but makes operator review slower.
- Add another JSON field contract. That is unnecessary because this is a
  markdown rendering gap, not a data model gap.

## Risks

The main risk is overclaiming readiness in a human-readable artifact. The
section explicitly renders readiness closure and live authority as false.

## Test plan

- Compile touched files.
- Run focused no-WING/WING guard tests.
- Run no-WING quick and check `summary.md` for the new section.
- Run artifact manifest verifier, memory lint, and full pytest.

## Where the other agent may disagree

Kimi may prefer avoiding more report surface fields while waiting for real WING
hardware. The counterpoint is that this reduces operator-review ambiguity before
the real capture.
