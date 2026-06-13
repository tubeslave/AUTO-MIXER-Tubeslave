# Codex Proposal: WING MVP Prestart Summary Contact Boundary

## Short Thesis

Make the no-WING prestart summary contract self-contained by requiring explicit
summary-level `contacts_backend`, `queries_console`, and `executes_tests`
booleans to be `false` before any accepted-candidate trace can feed review or
supervised proof planning.

## What I Understood

The current first blocker is still real WING read-only proof. Offline packet and
prestart artifacts are allowed to prepare operator gates, but they must not
contact hardware, query the console, run tests, grant authority, or close the
blocker.

## Proposed Solution

Update the no-WING prestart summary writer to emit:

- `contacts_backend: false`
- `queries_console: false`
- `executes_tests: false`

Then propagate those values through the WING readonly preflight runner, evidence
builder, review artifact summary, review-decision gate, supervised proof-plan
gate, and progress rails.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`
- `backend/autonomy_promotion_readiness.py`
- WING MVP docs, reports, rails, and focused tests

## Alternatives Considered

Only documenting the new fields would be cheaper, but accepted candidates could
still pass without machine validation. Requiring the fields only in docs would
not protect the supervised proof plan.

## Risks

Older local artifacts without the new fields will be rejected and must be
regenerated. That is intentional because accepted candidate evidence should
come from the current prestart contract.

## Test Plan

- Python compile for touched modules.
- Focused WING readonly/prestart/review/proof/no-WING rail tests.
- no-WING quick check.
- Project Director and direction guard.
- Full pytest.

## Where The Other Agent May Disagree

Kimi may argue that `contacts_backend`, `queries_console`, and `executes_tests`
already exist at packet level. The response is that the accepted-candidate
prestart trace cites a specific summary path and SHA256; that artifact should
carry its own no-contact boundary.
