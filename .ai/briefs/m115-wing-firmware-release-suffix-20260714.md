# M115 WING Firmware Release Suffix

## Goal

Allow the attended M110 HW01 gate to recognize the exact firmware family
reported by the connected WING Rack without accepting a different firmware
version or weakening any route, identity, signal, headroom or write gate.

## Real evidence

- Console serial: `0100Q1V0604AAE`.
- Product: `wing-rack`.
- Firmware readback: `3.0.5-4-ga4ea833d:release`.
- M114 route-only run reached the first fixed route pass and blocked only on
  `console_firmware_not_m110_verified_family`.
- The blocked result reported
  `console_mutation_write_attempt_count=0`, `live_authority_granted=false` and
  `readiness_closure_allowed=false`.

## Current defect

`backend/server.py` accepts the `3.0.5` family plus an alphanumeric build
suffix, but its suffix grammar excludes the colon in the observed terminal
tag `:release`.

## Constraints

- Continue requiring the exact base version `3.0.5`.
- Accept only a narrowly defined terminal release tag, not arbitrary colon
  content.
- Preserve the existing accepted base and build-suffix forms.
- Add regression and adversarial tests before another hardware attempt.
- Rebind CP05 and use a new clean detached worktree for the retry.
- No reset or correction may occur until HW01 succeeds.

