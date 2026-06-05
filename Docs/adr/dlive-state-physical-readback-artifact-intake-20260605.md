# ADR: dLive State Physical Readback Artifact Intake

## Context

dLive-state physical readback has a schema contract, but future
operator-captured JSON artifacts need a review surface. dLive MIDI/TCP sends,
dLive writes, and dLive apply operations are live-console authority, so artifact
intake must not become an executor or promotion signal.

## Options considered

1. Keep only the schema contract.
2. Add a read-only artifact-intake layer.
3. Build a hardware capture or dLive MIDI/TCP runner.

## Decision

Add a read-only artifact-intake layer for dLive-state physical readback
artifacts.

The intake may discover JSON artifacts, validate them against the contract, and
surface a valid artifact as a manual-review candidate. It must preserve
`fixture_closure_allowed_by_intake=false`,
`physical_fixture_gap_remains=true`, `midi_send_allowed=false`,
`tcp_send_allowed=false`, `dlive_write_allowed=false`,
`dlive_apply_allowed=false`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

## Why this won

The read-only layer lets operators see whether a future dLive state and WING
parity artifact is structurally usable without granting dLive control. It also
gives no-WING release checks, Product Layer proof quality, and frontend module
status rows a consistent place to show missing, invalid, or review-required
artifact state.

## Rejected alternatives

Keeping only the schema contract was too passive once artifacts exist: reviewers
would not have a normalized status row.

Building a hardware capture or dLive runner was rejected for this slice. That
requires separate operator approval, passive capture identity checks, WING
parity review, rollback proof, freshness policy, and emergency-stop handling.

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
- no-WING release-check tests for safety flags, authority normalization, JSON
  summary, and Markdown summary.
- Product Layer and frontend adapter contract tests for read-only rows.
- Memory lint and diff hygiene before relying on the slice.

## Risks and rollback

The main risk is misreading artifact intake as permission to send MIDI/TCP or
apply dLive state. The implementation mitigates this by preserving dLive
authority denial, fixture gap state, and live authority denial in backend,
no-WING summary, Product Layer, and frontend rows.

Rollback is to remove the intake method from registry/Product Layer/frontend
surfaces and leave the existing dLive schema contract in place. No console
state is changed by this decision.
