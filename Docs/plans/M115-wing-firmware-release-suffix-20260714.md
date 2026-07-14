# M115 — WING Firmware Release Suffix

Status: VALIDATED_FOR_COMMIT

## Sequence

1. Record the zero-write real HW01 firmware-format blocker.
2. Council-review a narrow shared firmware predicate.
3. Implement and test both server and runner trust boundaries.
4. Release the development lease, validate and obtain independent review.
5. Commit source, rebind CP05 marker-only and repeat HW01 from a clean detached
   checkout.

## Exit criteria

- The real `3.0.5-4-ga4ea833d:release` string is accepted unchanged.
- Wrong versions, direct/unrecognized tags and extra separators fail closed.
- Backend issuance and runner verification use the same predicate.
- Tests and independent review pass.
- CP05 verifies a clean final tree before hardware contact.
- No reset or correction occurs before verified HW01 route/headroom evidence.

## Validation evidence

- Route-proof plus attended-runner suites: `171 passed`.
- Combined route, runner, M110 capability and M006 restore slice:
  `237 passed`.
- Python compilation and `git diff --check`: passed.
- Independent implementation review: `CLEAN`, P0/P1/P2 = 0.
- Real pre-fix HW01 result: blocked only on the firmware-format contract with
  `console_mutation_write_attempt_count=0` and no live authority.
- WING rails and full-suite validation remain deferred until the non-marker
  commit and marker-only CP05 rebind.
