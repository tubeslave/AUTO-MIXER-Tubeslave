# Proposal: WING MVP Operator Sequence List Markdown Value Gate Self-Surface

## Short thesis

Make the Markdown required-lines value gate self-visible and fail-closed: the
expected, actual, and match verdict lines must exist and match the computed
verdict.

## What I understood

The WING read-only proof blocker is still open. Offline no-WING checks should
reduce operator error before a real WING session without granting readiness or
live authority.

## Proposed solution

- Add the value-gate verdict fields to
  `WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES`.
- Extract their rendered Markdown values during the existing check.
- Record value mismatches in JSON.
- Fail the existing Markdown-surface check when any value-gate line is missing
  or mismatched.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-mvp-operator-sequence-list-markdown-value-gate-self-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_MARKDOWN_VALUE_GATE_SELF_SURFACE_20260615.md`

## Alternatives considered

Relying on JSON-only diagnostics was rejected because the operator review path
uses `summary.md`. Adding a separate checker was rejected because the existing
Markdown-surface check already owns this surface.

## Risks

The value comparison is intentionally exact. If the Markdown rendering format
changes, this check should fail and force the contract update to be explicit.

## Test plan

Focused py_compile, focused no-WING/WING tests, direct missing-line and
value-drift regressions, no-WING quick inspection, Project Director marker
check, full pytest, memory lint, and diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a looser JSON parse. I would keep exact comparison because this
is an operator artifact contract, not a free-form document.
