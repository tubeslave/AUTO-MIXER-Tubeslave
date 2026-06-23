# Codex Proposal: WING MVP Review Decision Artifact Root Gate

## Short thesis

Manual review decisions should pin the artifact root, not only the summary and
event-log paths and hashes.

## What I understood

The active blocker remains `WING read-only connection proof`. Runtime read-only
summaries now declare `artifact_root`, and evidence intake checks that the
offline plan root matches the captured summary. The next manual boundary is the
operator review decision: an accepted record should prove it belongs to the
same artifact tree as the candidate.

## Proposed solution

Carry `artifact_root` through read-only evidence records, review report rows,
and pending decision templates. In
`backend/wing_readonly_preflight_review_decision.py`, reject decision records
when the field is missing or does not match the candidate root.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- Focused review/evidence tests.
- Council/report docs for this slice.

## Alternatives considered

- Infer root from `summary_path`. Rejected because the runtime summary already
  declares the explicit root and the manual record should preserve it.
- Keep only path/hash binding. Rejected because stale hand-edited root context
  could still pass if paths are copied without the root declaration.

## Risks

Existing hand-authored review decisions without `artifact_root` will now fail
closed and must be regenerated or edited. That is intended for the manual WING
readiness boundary.

## Test plan

- Focused review-decision and review-tool tests.
- Evidence/runner compatibility tests.
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer deriving the root from path parents. The explicit field is less
ambiguous for operator review and keeps the runtime summary, evidence row, and
manual decision record aligned.
