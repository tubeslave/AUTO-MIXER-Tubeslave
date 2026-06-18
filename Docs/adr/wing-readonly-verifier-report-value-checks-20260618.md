# WING Read-Only Verifier Report Value Checks

## Context

The WING read-only runner blocks WebSocket access until the no-WING artifact
manifest verifier passes. The verifier report already publishes
`artifact_manifest_self_consistency_match`,
`artifact_manifest_contract_cross_surface_match`, and
`artifact_manifest_self_consistency_summary_surface_match`, but those top-level
report values were not mirrored as explicit `checks` entries required by the
runner.

## Options considered

1. Leave the runner to rely on `artifact_manifest_verifier_report_value_mismatches`.
2. Add explicit verifier `checks` entries and require them before WebSocket.
3. Refactor the full verifier/runner check list into a shared module.

## Decision

Use option 2. The verifier now emits explicit expected-value checks for the
three top-level manifest report booleans, and the runner requires those checks
before any backend/WING WebSocket can open.

## Why this won

It closes the drift risk with a small safety-contract change. A stale verifier
report can no longer pass just because the top-level fields happen to exist;
the runner requires named check entries that prove the values were evaluated.

## Rejected alternatives

Option 1 keeps weaker diagnostics and leaves the pre-WebSocket gate easier to
drift. Option 3 may be worthwhile later, but it is broader than the immediate
safety gap.

## Implementation plan

Add the verifier check entries in `tools/verify_no_wing_artifact_manifest.py`,
require them in `tools/wing_readonly_preflight_runner.py`, and cover the
contract with a focused runner regression test.

## Test plan

Run focused WING runner tests, focused no-WING artifact manifest tests,
`py_compile`, `git diff --check`, no-WING quick validation, and direct sidecar
manifest verification on the fresh no-WING artifact directory.

## Risks and rollback

Risk is limited to older verifier bundles failing closed before WebSocket. Roll
back by removing the three required check entries from runner/verifier and the
regression test.
