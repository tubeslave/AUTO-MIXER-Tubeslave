# WING Read-Only Canonical Root Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095 blocked reserved
no-WING/control-plane roots and M096 blocked stale summary/events path shapes.
A remaining direct-payload gap allowed a candidate to use a non-canonical
`artifact_root`, such as `artifacts/stale_wing_readonly_preflight`, as long as
the summary and events paths matched that non-canonical root.

## Options Considered

1. Leave canonical artifact-root enforcement to later review-decision gates.
2. Require the canonical WING read-only capture root at first evidence intake.
3. Accept any root that has plausible read-only evidence fields.

## Decision

Use option 2. Candidate WING read-only summaries now fail intake unless
`artifact_root` is the canonical capture root, or an absolute/temp path ending
with `artifacts/wing_readonly_preflight`.

## Why This Won

The first accepted-candidate boundary should not allow stale or alternate
artifact roots to appear as real WING proof. Failing early keeps the operator
review focused on the exact handoff location.

## Rejected Alternatives

Review-only enforcement was rejected because it permits a bad candidate to look
accepted before manual review. Accepting any plausible root was rejected because
it weakens M001's artifact-location contract.

## Implementation Plan

- Add canonical capture-root matching for candidate `artifact_root`.
- Preserve absolute temp/test paths that end in the canonical capture root.
- Add focused and rail coverage.

## Test Plan

- Focused WING read-only evidence tests.
- M097 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting legacy artifacts captured outside the canonical
handoff root. That is intended for M001: real proof must use the documented
handoff location. Roll back by reverting this ADR, the M097 plan/report/state
entries, the test and the canonical-root guard. No live transport or console
state is affected.
