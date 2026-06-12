# Codex Proposal: WING Read-Only Runner Stale Prestart Plan Regression

## Short thesis

Add a runner-level regression proving stale expected plans cannot reach the
backend/WING WebSocket when the backend-port prestart detail contract changes.

## What I understood

The current hardware step depends on running an offline expected plan first,
then binding the real read-only preflight run to that plan. After adding
`listener_confirmation_checklist`, a stale plan missing that field must be
rejected before any network contact.

## Proposed solution

- Create a valid offline expected plan in the test.
- Remove `listener_confirmation_checklist` from
  `backend_port_prestart_check_failure_detail_fields`.
- Run the real preflight path with WebSocket patched to fail if opened.
- Assert the runner returns `expected_plan_mismatch` and sends no requests.

## Likely files to touch

- `tests/test_wing_readonly_preflight_runner.py`

## Alternatives considered

- Add production code. Rejected because the runner already compares the field;
  the missing piece was regression coverage for this exact stale-plan case.
- Rely on evidence-layer tests only. Rejected because the runner is the thing
  that must stop before WebSocket contact.

## Risks

Low. This is test-only coverage around an existing fail-closed path.

## Test plan

- Focused runner expected-plan tests.
- Adjacent WING/no-WING suite.
- Full repo validation before commit.

## Where the other agent may disagree

Kimi may call this redundant with generic expected-plan mismatch coverage. I
would keep it because backend-port prestart drift is the current operator gate
for the first WING MVP blocker.
