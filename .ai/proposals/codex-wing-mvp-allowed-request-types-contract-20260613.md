# Proposal: WING MVP Allowed Request Types Contract

## Short thesis

Centralize the WING read-only preflight allowed request types in
`backend/wing_readonly_preflight_contract.py` so runner plan output, evidence
intake, no-WING validation, and tests cannot disagree about the read-only proof
surface.

## What I understood

The current proof rail must remain read-only. The preflight runner may send only
`get_operator_mode`, `get_state`, and `query_mixer_parameter` through the
already-running backend. The list is part of the safety boundary because adding
or removing request types changes what the physical proof can demonstrate.

## Proposed solution

- Add `WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES` to the shared preflight
  contract.
- Have the runner write `allowed_request_types` from that constant.
- Have evidence intake derive both sent-request and expected-plan allow-lists
  from that constant.
- Have no-WING validate generated plans against the shared constant.
- Update focused tests and fixtures to assert contract binding.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- Focused tests for runner, evidence, and no-WING.

## Alternatives considered

- Leave the three-value list duplicated. Rejected because it creates a drift
  path in the physical proof surface.
- Only centralize test fixtures. Rejected because runtime validators would still
  own separate lists.

## Risks

- Future request-surface changes now require an explicit contract edit. That is
  desirable because any new request type must be reviewed before a real WING
  preflight.
- Old expected-plan artifacts with a stale allow-list should fail closed until
  regenerated.

## Test plan

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director, direction guard, and WING rails checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may argue that the list is stable enough to duplicate. The counterpoint is
that it defines the allowed request surface for real physical readback proof, so
it should be contract-owned.
