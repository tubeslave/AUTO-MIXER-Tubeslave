# Matrix State Physical Readback Contract

## Task

Define the next read-only proof contract for `matrix_state` so matrix controls,
EQ, dynamics, inserts, and metering can be validated by a future physical
readback artifact before any matrix, routing, FX-send, or master-bus promotion
trusts cached state.

## Safety constraints

- Do not query WING or dLive.
- Do not mutate console state.
- Do not grant live authority or Full Autonomous readiness.
- Keep the contract schema-only and transport-free.
- Preserve `physical_fixture_gap_remains=true` until an operator-run physical
  artifact exists.

## Expected outputs

- Backend matrix-state physical readback contract and validator.
- Processing registry, goal matrix, remediation, fixture evidence, no-WING
  summary, Product Layer, and frontend visibility wiring.
- Focused tests proving contract shape, drift detection, safety normalization,
  and no live authority.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_matrix_state_physical_readback_contract.py -q`
- Focused read-surface/no-WING/Product Layer/frontend contract tests.
- Full backend and frontend validation before committing.
