# Proposal: WING MVP Value Types Contract

## Short thesis

Centralize the WING read-only preflight supported value types in
`backend/wing_readonly_preflight_contract.py` so runner validation, handoff
validation, defaults, and no-WING evidence cannot diverge.

## What I understood

The current proof rail must stay bounded to read-only `fader`, `gain`, and
`pan` queries. The default value-types string already comes from the shared
contract, but the runner and handoff allow-lists still had local sets. That
creates a drift path where generated command text and actual tool validation
could disagree before the real WING preflight.

## Proposed solution

- Add `WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES` to the shared
  preflight contract.
- Derive `WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES` from that tuple.
- Use the shared allow-list in the runner and handoff builder validators.
- Have no-WING introspect the runner allow-list and fail closed if it differs
  from the contract.
- Add JSON/Markdown summary fields for expected and runner-supported value
  types.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- Focused tests for those tools.

## Alternatives considered

- Leave local sets in place. Rejected because command defaults and validation
  could drift.
- Only test the existing local sets. Rejected because no-WING should expose and
  block drift before an operator attempts hardware preflight.

## Risks

- Future expansion of value types will now require an explicit contract change.
  That is desirable because adding read targets changes the real proof scope.
- The no-WING runner introspection can fail if the runner cannot be imported;
  that remains a safe prestart failure.

## Test plan

- Compile touched modules.
- Run focused runner, handoff, no-WING, Project Director, direction guard, and
  WING rails tests.
- Run no-WING quick with no hardware and no backend writes.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may see `{"fader", "gain", "pan"}` as stable enough to duplicate. The
counterpoint is that this set defines the physical proof scope, so it should be
contract-owned and release-checked.
