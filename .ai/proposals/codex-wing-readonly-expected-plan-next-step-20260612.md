# Codex Proposal: WING Read-Only Expected Plan Next Step

## Short thesis

Make expected-plan fail-closed summaries actionable by adding
`operator_next_step` before the runner can open a backend/WING WebSocket.

## What I understood

The runner already blocks missing, invalid, and mismatched expected plans before
network contact. The summary tells operators the reason and violations, but not
the next safe step to recover before the hardware run.

## Proposed solution

- Add a small helper that maps expected-plan failure reasons to operator next
  steps.
- Include `operator_next_step` in missing, invalid, and mismatch summaries.
- Add focused tests that assert the guidance and no sent requests.
- Update the read-only preflight runbook to mention the field.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives considered

- Leave guidance only in the runbook. Rejected because operators inspect
  generated summaries during a blocked hardware setup.
- Add new CLI output only. Rejected because `summary.json` is the durable proof
  artifact.

## Risks

Low. The change adds explanatory fields only to already-blocked paths.

## Test plan

- Focused runner expected-plan tests.
- Full runner test file.
- Adjacent WING/no-WING tests, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer to keep summaries minimal. I would keep this because blocked
hardware-preflight artifacts must be operationally actionable.
