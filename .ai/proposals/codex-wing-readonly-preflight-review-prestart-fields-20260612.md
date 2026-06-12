# Proposal: WING Read-Only Preflight Review Prestart Fields

## Short thesis

Propagate prestart contract fields into artifact review rows and render them in
the manual review report.

## What I understood

The real summary now proves the backend-port prestart boundary. The next review
surface should make that proof visible to the operator before any readiness
decision is recorded.

## Proposed solution

- Add the prestart fields to `WingReadonlyPreflightArtifactRecord`.
- Include them in `tools/review_wing_readonly_preflight_artifacts.py` artifact
  review rows.
- Render them in the markdown report and cover with tests.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Leave operators to inspect raw `summary.json`.
- Re-read the summary file directly in the review tool.

## Risks

The review report becomes slightly more verbose. This is acceptable because the
fields are now part of the real proof contract.

## Test plan

- Focused review report test.
- Adjacent WING/no-WING validation stack.
- `git diff --check`, memory lint, and full backend suite.

## Where the other agent may disagree

Kimi may prefer keeping review rows compact. For a real hardware handoff, the
review report should expose the fail-closed prestart evidence inline.
