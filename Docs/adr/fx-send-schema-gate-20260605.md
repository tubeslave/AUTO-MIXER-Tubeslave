# ADR: fx-send-schema-gate-20260605

## 1. Context

Automixer must eventually tune FX sends automatically, but FX rack, insert, send,
and FX parameter writes remain outside the approved live executor. The Product
Layer exposes read-only FX parameter schema hints for cached FX slots. The FX send
proposal layer should use that evidence before producing proposal-only
recommendations.

## 2. Options considered

1. Block proposal-only FX-send recommendations when explicit schema evidence is
   generic, missing, unscoped, or mismatched.
2. Require schema evidence for every FX-send recommendation immediately.
3. Ignore schema evidence until physical WING FX fixtures exist.

## 3. Decision

Use option 1. Add optional `current_fx_parameter_schema` evidence to
`FXSendAnalysisInput`. When supplied, it must be mapped and scoped to the
selected FX family (`reverb`, `delay`, or `modulation`) or the result remains
`analysis_only`.

## 4. Why this won

This improves assisted FX recommendations without breaking legacy callers that
do not yet pass Product Layer schema payloads. It also prevents the proposal
layer from treating generic numeric FX parameters as known semantics.

## 5. Rejected alternatives

Mandatory schema evidence is the cleaner final contract, but callers need a
transition period. Ignoring schema evidence would waste the new read-only FX
snapshot work and keep a known ambiguity in FX decisions.

## 6. Implementation plan

1. Add `current_fx_parameter_schema` to `FXSendAnalysisInput`.
2. Normalize schema evidence into recommendation/proposal evidence.
3. Add blockers for generic, missing, unscoped, and mismatched schema payloads.
4. Update proposal-only adapter contract and registry metadata.
5. Add focused tests for mapped, generic, and mismatched schemas.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/fx_send_proposal.py backend/processing_proposal_adapters.py backend/processing_registry.py tests/test_fx_send_proposal.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

Risk: schema hints may be confused with physical model proof. The proposal
evidence remains non-queueable, `can_apply_live=False`, and promotion still
requires physical readback, rollback, feedback checks, and hardware proof.

Rollback is removing the schema field, blockers, evidence fields,
registry/adapter metadata, tests, and this ADR. No live write path is introduced.
