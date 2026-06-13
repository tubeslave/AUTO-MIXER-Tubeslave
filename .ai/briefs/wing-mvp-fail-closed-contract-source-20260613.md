# WING MVP Fail-Closed Contract Source

Task id: `wing-mvp-fail-closed-contract-source-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The WING readonly preflight plan fail-closed conditions were duplicated across:

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`

The previous prestart fail-closed condition mismatch proved that duplicated
lists can drift and make the no-WING release gate reject a valid offline plan.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not loosen fail-closed validation.
- Preserve existing exported names used by tests and tools.

## Required Outcome

Move the canonical fail-closed condition list to
`backend/wing_readonly_preflight_contract.py` and have runner, evidence, and
release-check validation consume that shared source.
