# ADR: manual-write-cache-freshness-gate-20260605

## Context

Supervised manual writes perform pre-read, write, readback, and rollback
metadata for the only currently approved live value types: `fader` and `gain`.
Before this decision, cache-backed pre-read did not verify cache age. A stale
cached value could therefore become the rollback baseline.

## Options considered

1. Keep cache-backed pre-read behavior unchanged.
2. Require fresh cache timestamps when pre-read uses mixer state cache.
3. Require physical readback for every supervised write.

## Decision

Use option 2. Physical readback remains sufficient. Cache-backed pre-read is
allowed only when the exact fader/gain address has fresh `state_timestamps`
evidence.

## Why this won

It strengthens the real apply boundary without widening write authority or
introducing hidden transport calls. It also gives future autonomous producers a
single fail-closed apply-time contract.

## Rejected alternatives

Option 1 leaves stale rollback baselines possible. Option 3 is safer but too
broad for current client parity because not all supported clients expose a
physical query API.

## Implementation plan

- Add cache freshness helpers to `AutoMixerServer`.
- Include freshness evidence on cache-backed `_manual_write_read_value()` rows.
- Block `_apply_manual_console_write()` when cache-backed pre-read is not fresh.
- Block `apply_operator_proposal` validation on the same evidence before it
  calls the manual gate.
- Add `state_timestamps` to Mixing Station cache updates.
- Add stale-cache regression tests.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_server.py::TestWingDeploymentWriteBoundary::test_websocket_fader_blocks_when_cache_pre_read_is_stale tests/test_operator_product_state.py::test_supervised_operator_proposal_apply_blocks_stale_cache_pre_read -q`
- Broader operator/manual gate tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: clients without timestamp support will block cache-backed supervised
writes. Mitigation: WING/dLive already timestamp cache updates, Mixing Station
now does too, and physical readback bypasses the cache timestamp requirement.
Rollback is removing the freshness helper, pre-read blocks, Mixing Station
timestamp field, tests, and this ADR.
