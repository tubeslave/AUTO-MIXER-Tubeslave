# Proposal: WING MVP Operator Sequence List Markdown Required Lines Value Gate

## Short thesis

Make the Markdown-surface check verify the value of its required-field-lines
contract, not only the presence of the line.

## What I understood

The active work is still the supervised WING read-only proof rail. Offline
no-WING summaries help prevent operator mistakes before a real WING session, but
must not imply live authority or readiness closure.

## Proposed solution

- Add a small helper to extract a Markdown summary field value.
- Compare
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`
  in `summary.md` with `json.dumps(WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES)`.
- Add expected, actual, and match verdict fields to the summary.
- Mark the existing
  `wing_readonly_operator_sequence_list_markdown_surface` check failed when the
  values differ.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-mvp-operator-sequence-list-markdown-required-lines-value-gate-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_MARKDOWN_REQUIRED_LINES_VALUE_GATE_20260615.md`

## Alternatives considered

Presence-only validation was rejected because it permits stale or empty required
line lists. Parsing the Markdown value as JSON was rejected for now because an
exact rendered-value comparison is simpler and matches the writer output.

## Risks

The check depends on the writer's `json.dumps` formatting. The same function is
used for expected and rendered values, so valid summaries stay stable.

## Test plan

Focused py_compile, focused no-WING/WING tests, direct value-drift regression
test, no-WING quick inspection, Project Director marker check, full pytest,
memory lint, and diff whitespace check.

## Where the other agent may disagree

Kimi may prefer JSON parsing for tolerance. I would keep exact comparison here
because this is an artifact contract, and drift should be visible.
