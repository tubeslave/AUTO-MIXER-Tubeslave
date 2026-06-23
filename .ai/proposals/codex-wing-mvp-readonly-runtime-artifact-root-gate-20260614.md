# Codex Proposal: WING MVP read-only Runtime Artifact Root Gate

## Short thesis

The real read-only preflight artifact should state its runtime artifact root,
and evidence review should reject any candidate whose root is missing or
differs from the offline expected plan.

## What I understood

The active blocker remains `WING read-only connection proof`. The runner has
strong expected-plan and prestart-summary gates before WebSocket, but the
captured summary did not carry `artifact_root`. Review tooling could verify
expected artifact paths from the plan, yet the summary itself did not expose the
runtime root as a comparable field.

## Proposed solution

Add `artifact_root` to real read-only runner summaries. Add `artifact_root` to
the evidence expected-plan summary field comparison and reject summaries where
the field is missing. Cover both the runner summary output and a file-level
plan/root mismatch regression.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- Council/report docs for this slice.

## Alternatives considered

- Keep deriving the root from `summary_path`. Rejected because inferred path
  structure is weaker than a declared runtime argument bound to the plan.
- Check only `expected_artifacts`. Rejected because that validates plan-internal
  consistency, not that the runtime summary exposed the same root.

## Risks

Old captured summaries without `artifact_root` will be rejected by the stricter
evidence intake. That is intended for readiness gating; such artifacts can be
recaptured with the current runner.

## Test plan

- Focused runner/evidence tests.
- Full no-WING release-check module.
- Project Director rails, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer preserving backwards compatibility for older read-only
artifacts. For live-readiness gating, fail-closed recapture is safer than
accepting an artifact whose runtime root is only inferred.
