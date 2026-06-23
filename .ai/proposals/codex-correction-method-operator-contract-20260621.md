# Codex Proposal: Correction Method Operator Contract

## Short thesis

Expose one shared proposal-only correction contract so every Automixer method
family can tell the operator what evidence it used, what problem it found, what
bounded correction it recommends, why it is blocked or safe, and what future
WING readback would be required before any live promotion.

## What I understood

The repository already has proposal modules for EQ, compressor, gate/expander,
de-esser, FX sends, pan, bus/group balance, routing, scenes, master/output, and
gain/fader preview paths. The missing piece is a common contract shape that
matches the user-required fields for all correction families without widening
live apply.

## Proposed solution

Extend `backend/processing_proposal_adapters.py` so
`build_proposal_only_method_contracts()` emits a shared
`operator_correction_contract` and expanded `required_fields` list. The contract
will keep live apply denied, state the family-specific bounded delta/readback
scope, and require headroom, feedback, true-peak, rollback, and operator
explanation evidence where applicable.

## Likely files to touch

- `backend/processing_proposal_adapters.py`
- `tests/test_processing_proposal_contracts.py`
- `.ai/briefs/correction-method-operator-contract-20260621.md`
- `.ai/proposals/codex-correction-method-operator-contract-20260621.md`
- `Docs/adr/correction-method-operator-contract-20260621.md`

## Alternatives considered

Changing each proposal serializer separately was rejected for this slice because
the broad goal needs a uniform schema first, and the existing modules already
emit method-specific evidence. Adding live executors was rejected because WING
proof, readback, rollback, emergency-stop, and operator approval are missing.

## Risks

The shared contract is declarative, so it does not by itself prove every
algorithm is finished. It does give tests a stable contract to enforce as method
families continue moving from analysis-only to operator proposals.

## Test plan

Run focused contract tests, then targeted existing method-family tests if the
adapter contract changes interact with them.

## Where the other agent may disagree

Kimi may argue that the contract should live in a new module instead of the
existing adapter. The adapter already owns Product Layer proposal-only contracts,
so extending it keeps the change visible without adding another registry surface.
