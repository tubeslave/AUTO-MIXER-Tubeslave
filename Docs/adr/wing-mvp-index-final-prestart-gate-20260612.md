# ADR: WING MVP Index Final Prestart Gate

## Context

The read-only WING preflight runbook now requires the final handoff
operator-sequence step to carry the accepted-candidate prestart gate before
supervised fader/gain proof. The main docs index did not yet link the new
report that records this boundary.

## Options considered

1. Leave `Docs/INDEX.md` unchanged.
2. Add the final prestart gate report to the reports list only.
3. Add the report to both the reports list and the Current WING MVP rails row.

## Decision

Use option 3. The final prestart gate report is part of the current WING MVP
operator rail and should be visible from the canonical docs entrypoint.

## Why this won

The index is the repo's main documentation entrypoint. Linking the active gate
from the WING MVP rails row reduces the chance that a future agent or operator
follows older accepted-decision wording without the final prestart handoff
requirement.

## Rejected alternatives

Leaving the index unchanged was rejected because it would hide the newest
safety boundary. A reports-only link was rejected because the report is an
active rail, not just historical evidence.

## Implementation plan

- Patch `Docs/INDEX.md`.
- Add a focused assertion in `tests/test_wing_mvp_rails.py`.
- Record a report with verification results.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k docs_index_points`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The only material risk is stale index wording if the active WING gate changes.
Rollback by replacing the linked report and test strings with the new current
gate in the same commit that changes the WING MVP boundary.
