# Proposal: WING MVP Request Response Map Contract

## Short thesis

Centralize the request-to-response mapping for WING read-only preflight in
`backend/wing_readonly_preflight_contract.py` so runtime fail-closed behavior
and event-log evidence validation use the same proof surface.

## What I understood

The runner sends only read-only backend requests and fails closed if the backend
returns a response type that does not match the current request. Evidence intake
checks the captured event log after the run. Those two checks must agree exactly
for `get_operator_mode`, `get_state`, and `query_mixer_parameter`.

## Proposed solution

- Add `WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE` to the
  shared contract.
- Derive runner expected response sets from that mapping.
- Derive evidence event-log response checks from that mapping.
- Add focused tests that assert runner and evidence use the shared mapping.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- Focused runner and evidence tests.

## Alternatives considered

- Keep local dictionaries in runner and evidence. Rejected because it allows
  runtime checks and artifact review to diverge.
- Only centralize the raw response type tuples. Rejected because the mapping is
  the actual safety invariant for `unexpected_response_type`.

## Risks

- Future response mapping changes now require an explicit contract edit. That is
  desirable because the accepted backend response for a request is part of the
  real WING proof boundary.
- Stale expected-plan or evidence paths should fail closed rather than silently
  accepting mismatched responses.

## Test plan

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director, direction guard, and WING rails checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may consider response type tuples sufficient. The counterpoint is that the
request-to-response mapping is what determines whether an event log is valid or
blocked as an unexpected response.
