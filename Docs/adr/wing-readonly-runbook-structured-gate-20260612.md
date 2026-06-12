# ADR: WING Read-Only Runbook Structured Gate

## Context

The WING read-only handoff packet and no-WING release summaries now expose a
structured `supervised_prestart_gate` contract. The read-only preflight runbook
still named only the derived `operator_sequence_*` fields before allowing a
transition to supervised fader/gain proof.

## Options considered

1. Keep the runbook unchanged.
2. Add the structured gate fields while keeping the operator-sequence checks.
3. Replace the operator-sequence checks with only the structured gate.

## Decision

Use option 2. The runbook now requires `supervised_prestart_gate` to match
`expected_supervised_prestart_gate`, with `grants_live_authority: false` and
`requires_accepted_candidate_prestart_trace: true`.

## Why this won

The structured gate is the stronger machine-readable contract, while the final
operator-sequence step remains the clearest human-facing transition proof. Both
must remain visible.

## Rejected alternatives

Leaving the runbook unchanged was rejected because it would trail the release
artifact contract. Removing operator-sequence checks was rejected because they
are still useful for manual review.

## Implementation plan

- Patch `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Extend `tests/test_wing_mvp_rails.py`.
- Record verification in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks_keep_readonly`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is wording drift if the release summary field names change. Roll
back by updating the runbook and rails assertions in the same commit that
renames the structured gate fields.
