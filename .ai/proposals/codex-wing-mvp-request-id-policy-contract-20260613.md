# Proposal: WING MVP Request ID Policy Contract

## Short thesis

Centralize the WING read-only preflight request-id policy in
`backend/wing_readonly_preflight_contract.py` so runner summaries, evidence
intake, no-WING validation, and tests cannot diverge.

## What I understood

The current proof rail depends on correlating every physical readback response
with the exact `query_mixer_parameter` request that produced it. The same policy
is currently encoded as a repeated string in several surfaces. That leaves a
small but real drift path in the WING read-only proof contract.

## Proposed solution

- Add `WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY` to the shared preflight
  contract.
- Have the runner write `request_id_policy` from that constant in offline plans
  and real-run summaries.
- Have evidence intake validate summaries against the shared constant.
- Have no-WING validate generated plans against the shared constant.
- Update focused tests and fixtures to assert contract binding.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- Focused tests for runner, evidence, and no-WING.

## Alternatives considered

- Keep the duplicate string literals. Rejected because request-id policy is part
  of the physical proof boundary.
- Only add a regression test around the literals. Rejected because that would
  still leave multiple sources of truth.

## Risks

- A future policy rename now requires updating the shared contract first. That
  is acceptable because changing identity-correlation policy changes the proof
  contract.
- If an old plan uses a stale policy string, no-WING and evidence should fail
  closed until the plan is regenerated.

## Test plan

- Compile touched modules.
- Run focused runner, evidence, and no-WING tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director, direction guard, and WING rails checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may consider this too small for a shared constant. The counterpoint is that
request-id identity matching is a safety contract for real physical readbacks,
not cosmetic summary text.
