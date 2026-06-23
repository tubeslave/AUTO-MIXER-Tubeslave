# ADR: Correction Method Operator Contract

Date: 2026-06-21
Status: accepted

## Context

Automixer is developing correction methods while real WING proof is still
missing. Existing method modules are mostly transport-free and proposal-only,
but the shared Product Layer contract did not yet require every method family to
publish source evidence, measured or audible problem, proposed correction,
confidence, bounded delta, safety blockers, applicable headroom/feedback/true
peak checks, operator explanation, rollback expectation, and future WING
readback requirements.

## Options considered

1. Leave the existing per-method proposal contracts unchanged.
2. Extend the shared proposal-only contract surface with the operator correction
   fields.
3. Promote selected method families toward live apply.

## Decision

Use option 2. `build_proposal_only_method_contracts()` will expose a shared
operator correction contract for every registered proposal-only method contract,
including method-family applicability for bounded deltas, safety checks,
rollback, and future WING readback evidence. Product Layer preview payloads must
preserve this contract so operator-facing UI/API readers see the same read-only
authority boundary that tests enforce.

## Why this won

It advances the user-requested method layer while preserving the current WING
safety boundary. The change is visible to Product Layer previews and tests, but
it does not create queue entries, live executor requests, backend/WING contact,
or any new live write authority.

## Rejected alternatives

Leaving contracts unchanged keeps the method surface inconsistent with the
requested operator explanation contract. Live promotion was rejected because
physical readback, rollback, emergency-stop, hardware proof, and explicit
operator approval are still missing.

## Implementation plan

1. Add shared correction contract field constants.
2. Add method-family contract metadata for the user-named method families.
3. Include the shared contract in `build_proposal_only_method_contracts()`.
4. Add concrete profiles for quarantined/direct-write placeholders that remain
   visible as contracts, such as FX parameter writes, so they do not fall back to
   generic live-apply language.
5. Add tests proving all named method families are covered and still
   non-live-applicable.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_processing_proposal_contracts.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_operator_product_state.py::test_main_pipeline_preview_exposes_non_fader_gain_proposal_contracts -q`
- Targeted existing proposal-contract tests if needed.

## Risks and rollback

Risk: the shared contract may expose a method as schema-covered before its
algorithm is fully mature. The contract therefore stays explicit that live apply
is denied and future WING readback is only a requirement, not proof. Rollback is
to remove the added fields, focused tests, and this ADR. No WING, OSC, queue, or
live executor behavior is introduced.
