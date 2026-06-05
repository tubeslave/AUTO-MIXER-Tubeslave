# Codex Proposal: Raw Mixer State Physical Readback Artifact Intake

## Short thesis

Add a shadow-only artifact intake layer that validates operator-captured raw
mixer-state physical readback artifacts without closing fixture gaps or granting
live authority.

## What I understood

The project is moving from read-surface plans/contracts toward real physical
proof. The raw mixer-state contract can validate a payload, but release and
operator surfaces do not yet show whether a candidate artifact exists.

## Proposed solution

Create `raw_mixer_state_physical_readback_artifact_intake` as a pure read-only
builder. It scans configured JSON artifact roots or accepts inline payloads,
uses the existing raw mixer-state artifact validator, and emits a normalized
summary: missing, valid candidate available, or invalid candidate present.

The intake should be visible in the registry, capability matrix, no-WING
summary, Product Layer proof quality, and frontend Module Status. A valid
candidate stays behind manual review and never closes the fixture gap by itself.

## Likely files to touch

- `backend/raw_mixer_state_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `backend/operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- Focused backend/frontend tests

## Alternatives considered

The simplest alternative was to keep intake inside the raw contract module. That
would make the contract less clear by mixing schema definition with current
artifact inventory state.

Another option was to fail the release gate whenever no artifact exists. That is
too strict for the current no-WING phase, where missing physical proof is an
explicit visible blocker, not a release-check implementation failure.

## Risks

- Operators could treat a valid candidate as proof closure.
- Malformed artifact folders could hide behind a passing missing-artifact state.
- Source summaries could claim live authority or fixture closure.

## Test plan

- Unit-test missing, valid, invalid, file-discovery, registry, and capability
  behavior.
- Extend no-WING validator and summary tests.
- Extend Product Layer normalization tests for good and malicious source
  summaries.
- Extend frontend contract tests for the new Module Status row.

## Where the other agent may disagree

Kimi may argue that intake should fail when no artifact exists. I would keep
missing artifacts as a passing schema state for now because the release check is
validating safe reporting, while readiness remains visibly blocked by the
physical fixture gap.
