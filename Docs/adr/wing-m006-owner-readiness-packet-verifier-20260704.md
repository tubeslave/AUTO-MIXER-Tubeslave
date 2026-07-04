# WING M006 Owner Readiness Packet Verifier

## Context

M006 is an owner readiness review step. It must not update readiness rows or
grant live authority until M005 has complete operator-present disconnect,
timeout and stale-state evidence. The report-only M006 packet now records the
required M005 summary contract, but a markdown packet alone is too easy to
misread during handoff.

## Options considered

1. Keep M006 verification as manual markdown review only.
2. Fold M006 packet checks into the M005 runner.
3. Add an offline verifier for the M006 owner packet and M005 summary pair.

## Decision

Use option 3. Add `tools/verify_m006_owner_readiness_packet.py` as an offline
fail-closed verifier. It reads the M006 packet and, when supplied, the M005
summary artifact. Without a complete M005 summary it reports `blocked`.

## Why this won

M005 and M006 have different responsibilities. The M005 runner proves
connection-loss safety. The M006 verifier proves that owner review inputs are
complete and still deny live/readiness authority. Keeping them separate makes
the handoff auditable without contacting backend or WING.

## Rejected alternatives

- Manual markdown review only was rejected because it does not machine-check
  `m005_complete=true`, empty missing evidence, phase completion or authority
  denial.
- Extending the M005 runner was rejected because M006 packet review must remain
  usable after the M005 artifact already exists and must not rerun any proof.

## Implementation plan

- Verify the tracked M006 packet schema, scope, authority denial flags and
  fader/gain proof rows.
- Require a supplied M005 summary before owner-review readiness can be true.
- Verify M005 schema, `status=passed`, `m005_complete=true`, empty missing and
  remaining evidence, all required evidence keys, physical cable-loss mode,
  key proof phases and denied authority flags.
- Return a JSON verifier report with `owner_review_input_ready=false` unless
  all checks pass.
- Support `--write-artifact` so the verifier can persist `summary.json` under
  `artifacts/wing_owner_readiness_review_verifier/<run_id>/` for owner-review
  audit without contacting backend or WING.

## Test plan

- Unit test that the current packet without M005 summary fails closed.
- Unit test that a partial M005 summary fails closed.
- Unit test that a complete M005 summary makes owner-review input ready while
  keeping live authority and readiness updates false.
- Unit test that packet-level authority widening fails closed.
- Unit test that persisted verifier artifacts preserve blocked status and
  authority denial.

## Risks and rollback

Risk: the verifier can be confused with an owner decision. It mitigates this by
returning only `ready_for_owner_review`, never readiness closure, and by keeping
`live_authority_granted=false`, `readiness_closure_allowed=false`, and
`readiness_matrix_update_allowed=false`.

Rollback is a normal git revert of the verifier, tests and ADR. M006 remains
blocked without the verifier.
