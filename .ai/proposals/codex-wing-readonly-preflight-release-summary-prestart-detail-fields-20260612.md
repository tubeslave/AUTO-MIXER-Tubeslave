# Proposal: WING Read-Only Preflight Release Summary Prestart Detail Fields

## Short thesis

Expose the existing prestart busy-port failure detail fields in the no-WING
release markdown summary so the operator-facing report matches the JSON release
contract.

## What I understood

The WING read-only proof is still blocked on real hardware. Offline release
artifacts should be precise enough that the operator can see why the prestart
port gate is fail-closed and which fields to inspect before starting the
intended backend/WING session.

## Proposed solution

- Add markdown lines for prestart failure detail fields in WING preflight
  evidence, plan artifact, and handoff packet artifact sections.
- Reuse existing summary payload fields.
- Extend no-WING markdown tests to assert the field names render.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Leave markdown unchanged because JSON summary already contains the fields.
- Add new summary fields instead of rendering existing ones.

## Risks

The markdown report becomes slightly longer. This is acceptable because it keeps
operator-facing release evidence aligned with the safety contract.

## Test plan

- Focused no-WING markdown summary tests.
- Adjacent WING/no-WING validation stack.
- `git diff --check`, memory lint, and full backend suite.

## Where the other agent may disagree

Kimi may prefer only machine-readable JSON. The markdown summary is still a
human review surface, so it should not hide required prestart detail evidence.
