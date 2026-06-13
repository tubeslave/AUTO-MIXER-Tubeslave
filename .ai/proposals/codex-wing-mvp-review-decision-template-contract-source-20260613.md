# Codex Proposal: WING MVP Review Decision Template Contract Source

## Short Thesis

Move WING read-only review decision template required fields into the shared
preflight contract so the operator decision artifact shape cannot drift between
review tooling and no-WING validation.

## What I Understood

The current MVP is still the supervised WING Rack proof, with
`WING read-only connection proof` first. A real read-only candidate is not enough
by itself: an operator-edited decision JSON must exist before readiness or
supervised proof can proceed. Its required fields are therefore a safety
contract, not a review-tool implementation detail.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS` to
`backend/wing_readonly_preflight_contract.py`. Update
`tools/review_wing_readonly_preflight_artifacts.py` and
`tools/no_wing_release_check.py` to keep compatibility aliases from the shared
source. Add focused tests that compare the review report and no-WING validator
against the shared tuple.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- focused review/no-WING tests
- WING MVP report/ADR/index docs

## Alternatives Considered

Leaving the fields duplicated preserves current behavior but keeps a drift point
in the real operator decision path. Moving them only into the review-decision
module would still make no-WING validation depend on a narrower implementation
module. The shared preflight contract is the right owner because these fields
define artifact safety semantics.

## Risks

The main risk is changing field order or dropping a required operator-edit field.
Focused tests should compare exact field lists in review reports, no-WING
details, and the shared contract.

## Test Plan

- Compile touched modules.
- Run focused review/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue this is still preparatory. The counterpoint is that this exact
template is the required manual review artifact before any real WING readiness
update or supervised proof, so reducing drift here directly protects the first
hardware gate.
