# Codex Proposal: WING MVP Review Template Status Contract

## Short Thesis

Make the pending WING read-only review decision template status a shared
contract and have no-WING validation reject drift before the operator handoff can
look accepted.

## What I Understood

The supervised WING Rack MVP still waits for a real read-only proof. Offline
review tooling can prepare a decision template, but that template must remain a
pending operator-edit artifact until a human replaces placeholders and records an
actual decision.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS` to
`backend/wing_readonly_preflight_contract.py`. Alias
`REVIEW_DECISION_TEMPLATE_STATUS` in the review tool to that source and expose a
small `review_decision_template_status()` helper. Update
`tools/no_wing_release_check.py` to load the review tool status, compare it to
the shared contract, include the result in detail/summary Markdown, and fail
closed on mismatch.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- focused review/no-WING tests
- WING MVP brief, ADR, report, and docs index entry

## Alternatives Considered

Leaving the status local to the review tool keeps current behavior but leaves a
drift point in the manual-review artifact. Checking only for the
`--decision-template-output` command flag proves a template is written, not that
the template still identifies itself as pending operator work.

## Risks

The main risk is adding a stricter no-WING invariant that old synthetic test
fixtures do not include. This is contained by focused tests and by keeping the
runtime path report-only and no-hardware.

## Test Plan

- Compile touched Python modules.
- Run focused review/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may say this is another preparatory contract change. The counterpoint is
that the status distinguishes a generated pending template from an accepted
operator decision, which directly protects the first WING readiness gate.
