# ADR: WING decision expected-artifacts binding

## Context

The WING read-only evidence layer now binds real candidate files to the
offline plan's `expected_artifacts`. The decision ledger uses candidate review
rows to decide whether an accepted operator record is ready for the supervised
prestart gate. Without carrying the same binding through that decision path, a
candidate could have a valid no-WING prestart trace but still lack proof that
the accepted source id is the expected real WING artifact.

## Options Considered

1. Keep expected-artifact binding only in the evidence review.
2. Render the binding in decision reports but leave readiness unchanged.
3. Require expected-artifact binding in the accepted-candidate prestart
   contract.

## Decision

Require accepted candidate review rows to include `expected_artifacts`,
expanded expected summary/event artifact paths, and true summary/event path
match flags before they count as accepted-candidate prestart-ready.

## Why This Won

The supervised prestart gate depends on the accepted candidate source id. That
source id should represent both a valid no-WING prestart trace and the exact
real artifact path declared in the operator-approved plan. Enforcing both in the
shared contract keeps the rail fail-closed without contacting WING or granting
authority.

## Rejected Alternatives

Evidence-only binding was rejected because decision reports can be generated
from candidate rows and should not silently drop the identity check. Report-only
rendering was rejected because it would still allow a weak candidate to reach
`accepted_decision_ready`.

## Implementation Plan

- Add expected-artifact binding checks to the shared contract helper.
- Propagate binding fields into decision candidate review rows.
- Render the fields in the decision review Markdown.
- Add ledger and tool regressions for missing binding.
- Record this report in the first WING MVP blocker row as tooling evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py`
- Adjacent WING/no-WING rail tests.
- Project Director and direction guard tests.
- Full test suite before merge.

## Risks And Rollback

The stricter decision contract can block stale candidate rows produced before
expected-artifact fields existed. Rollback is to remove the binding requirement
from the shared contract helper while keeping the report fields visible.
