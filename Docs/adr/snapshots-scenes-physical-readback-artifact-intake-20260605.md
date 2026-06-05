# ADR: Snapshots Scenes Physical Readback Artifact Intake

## Context

Snapshots/scenes physical readback has a schema contract, but future
operator-captured JSON artifacts need a review surface. Scene load, save, and
recall can overwrite console state, so artifact intake must not become an
executor or promotion signal.

## Options considered

1. Keep only the schema contract.
2. Add a read-only artifact-intake layer.
3. Build a hardware capture or scene recall/save runner.

## Decision

Add a read-only artifact-intake layer for snapshots/scenes physical readback
artifacts.

The intake may discover JSON artifacts, validate them against the contract, and
surface a valid artifact as a manual-review candidate. It must preserve
`fixture_closure_allowed_by_intake=false`,
`physical_fixture_gap_remains=true`, `scene_load_allowed=false`,
`scene_save_allowed=false`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

## Why this won

The read-only layer lets operators see whether a future artifact is structurally
usable without granting live control over destructive scene/preset operations.
It also gives no-WING release checks and frontend status rows a consistent
place to show missing, invalid, or review-required artifact state.

## Rejected alternatives

Keeping only the schema contract was too passive once artifacts exist: reviewers
would not have a normalized status row.

Building a hardware capture or scene runner was rejected for this slice. That
requires separate operator approval, hardware identity checks, rollback proof,
freshness policy, and emergency-stop handling.

Treating a valid artifact as fixture closure was rejected. The intake is only a
candidate gate; fixture closure remains a separate manual and evidence-driven
decision.

## Implementation plan

- Keep the intake module transport-free and file/inline-payload based.
- Register the method as `shadow_only` and outside executor profiles.
- Add Product Layer proof-quality normalization.
- Add no-WING validation and summary output.
- Add frontend read-only module status visibility.
- Add tests that invalid or mixed artifacts fail closed.

## Test plan

- Focused intake tests for missing, valid, invalid, discovery, registry, and
  capability behavior.
- no-WING release-check tests for safety flags, mixed-candidate rejection, JSON
  summary, and Markdown summary.
- Frontend adapter contract tests for the read-only module row.
- Memory lint and diff hygiene before relying on the slice.

## Risks and rollback

The main risk is misreading artifact intake as permission to load or save
scenes. The implementation mitigates this by preserving scene load/save denial,
fixture gap state, and live authority denial in backend, no-WING summary, and
frontend rows.

Rollback is to remove the intake method from registry/Product Layer/frontend
surfaces and leave the existing snapshots/scenes schema contract in place. No
console state is changed by this decision.
