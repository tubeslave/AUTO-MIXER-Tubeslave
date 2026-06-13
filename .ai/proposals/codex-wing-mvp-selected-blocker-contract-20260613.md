# Codex Proposal: WING MVP Selected Blocker Contract

## Short Thesis

Move the selected MVP blocker name and status into the shared WING preflight
contract so runner, evidence intake, and no-WING validation agree on the same
next hardware proof target.

## What I Understood

The current MVP remains supervised WING Rack proof. Offline work should keep
the first open blocker explicit while preserving the boundary that only real
WING artifacts plus manual review can update readiness.

## Proposed Solution

Add shared selected-blocker constants, replace local literals in runner,
evidence intake, no-WING validation, and update tests/fixtures to reference the
contract.

## Likely Files to Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- WING preflight and no-WING tests

## Alternatives Considered

Leaving literals local is smaller, but allows the selected proof target to
drift silently. Parsing the blocker table live inside the runner would be more
complex and less deterministic for a capture tool.

## Risks

The main risk is stale contract values if the selected blocker changes. The
release gates already keep the current first open blocker visible and fail
closed on mismatches.

## Test Plan

Run compile checks, focused WING/no-WING tests, no-WING quick, rails, direction
guard, project director, memory lint, diff check, and the standard pytest suite.

## Where Kimi May Disagree

Kimi may prefer deriving this directly from `.ai/project-layer/wing_mvp_blockers.md`.
For the runner path, an explicit contract constant keeps the capture artifact
stable and testable.
