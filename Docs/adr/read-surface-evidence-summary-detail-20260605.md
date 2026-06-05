# ADR: read-surface-evidence-summary-detail-20260605

## Context

The no-WING read-surface fixture schema now validates declaration and
contract/gap consistency. However, `summary.json` and `summary.md` only expose
basic counts. Release proof consumers need top-level visibility into whether the
summary was recomputed from rows and whether declarations/contracts were
validated.

## Options considered

1. Add proof-quality fields to the existing read-surface fixture evidence
   summary.
2. Leave details in the verbose `results` array.
3. Create a separate proof-quality artifact.

## Decision

Use option 1. The existing summary section will include declaration counts,
physical gap count, and boolean enforcement flags.

## Why this won

It gives operators and Product Layer readers direct evidence without adding a new
artifact or changing validator behavior.

## Rejected alternatives

Deep `results` parsing is too indirect for release review. A separate artifact is
unnecessary while the facts belong to the current read-surface evidence summary.

## Implementation plan

1. Extend `read_surface_fixture_evidence_summary`.
2. Add matching Markdown lines.
3. Update summary embedding tests.
4. Run focused no-WING and backend gates.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: enforcement flags could be mistaken for physical readiness. The summary
continues to expose `full_autonomous_ready=false` and the physical-missing IDs.

Rollback is removing the new summary fields and test assertions.
