# Proposal: WING MVP Response Types Contract

## Short thesis

Centralize the WING read-only preflight expected response types in
`backend/wing_readonly_preflight_contract.py` so runner fail-closed handling and
evidence event-log validation cannot drift.

## What I understood

The current proof rail is read-only and must reject unexpected backend response
types. The runner expects `operator_mode_status`, `state_update`, and readback
responses for `query_mixer_parameter`; evidence intake checks the captured event
log against the same expectations. These expected response sets are part of the
physical proof boundary.

## Proposed solution

- Add shared response-type tuples for operator-mode, state, and physical
  readback responses.
- Derive runner `EXPECTED_*_EVENTS` sets from the shared tuples.
- Derive evidence expected event sets from the same tuples.
- Add focused tests that assert runner and evidence use the shared contract.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- Focused runner and evidence tests.

## Alternatives considered

- Keep local response sets. Rejected because it allows runner and evidence to
  disagree about `unexpected_response_type`.
- Centralize only readback responses. Rejected because operator-mode and state
  responses are also part of the event-log proof sequence.

## Risks

- Future response schema changes will require an explicit contract edit. That
  is desirable because accepting a new response type changes the proof surface.
- Stale runner/evidence code should fail tests or fail closed rather than
  silently accepting mismatched event logs.

## Test plan

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director, direction guard, and WING rails checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer to leave the small sets local. The counterpoint is that these
sets control when the proof fails closed on unexpected backend responses, so
they should be contract-owned.
