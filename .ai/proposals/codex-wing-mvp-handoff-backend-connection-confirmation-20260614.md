# Proposal: Require backend confirmation before WING capture

## Short Thesis

The WING read-only handoff should fail closed unless the operator sequence
confirms the existing backend connection is the intended WING Rack before
capture starts.

## What I Understood

The first MVP blocker is still `WING read-only connection proof`. Offline
handoff packets must guide the real hardware run without granting authority or
claiming readiness. The current packet has the boundary flag, but the
operator-visible sequence should make the backend/WING connection confirmation
explicit and validated.

## Proposed Solution

- Add a shared backend connection confirmation step and marker list.
- Use that step in `build_wing_readonly_preflight_handoff_packet.py`.
- Validate that the step exists, is at index `2`, and appears before
  `Run capture_readonly_preflight.`.
- Surface the new fields in the no-WING handoff summary and docs.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`

## Alternatives Considered

Leaving the connection confirmation as prose was rejected because stale packets
could still pass validation while omitting the dedicated operator step. Checking
only `physical_preflight_requires_existing_backend_connection: true` was
rejected because it does not prove the sequence orders the confirmation before
capture.

## Risks

Older handoff packets that do not include the dedicated confirmation step will
fail validation. That is intended; they should be regenerated before a real
WING session.

## Test Plan

- Compile touched Python.
- Focused handoff/no-WING tests.
- no-WING quick release check.
- Project Director/WING rail tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue this is redundant with `physical_preflight_requires_existing_backend_connection`.
I recommend keeping the explicit sequence guard because it is the human-facing
boundary immediately before the real capture step.
