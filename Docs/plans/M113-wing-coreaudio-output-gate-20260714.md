# M113 — WING CoreAudio Output Gate

Status: COMPLETE_OFFLINE_STRICT_88_EXTERNAL

## Purpose

Require the complete CoreAudio 48x48/48 kHz identity before M110 can accept a
physical WING capture device. This milestone changes no live authority and
does not contact the console.

## Sequence

1. Record Council proposals, critiques and ADR.
2. Add the exact output-channel predicate and bind the verified count.
3. Add adversarial output-metadata tests and update the existing architecture
   description.
4. Release the writer lease, run focused and full validation, then commit all
   non-marker changes.
5. Reacquire the lease, rebind only the two CP05 markers, release it and commit
   the marker pair.
6. Verify CP05 and WING/static rails on the final committed tree.

## Exit criteria

- Only exact integer `48` input and exact integer `48` output at 48 kHz can
  verify as the selected WING CoreAudio device.
- Missing, zero, wrong, boolean and string output counts fail closed.
- Successful evidence binds both channel counts into `binding_sha256`.
- Standard tests and CP05 pass on the final clean tracked tree.
- Strict live readiness stays `88/100`; M110-HW01 remains blocked until the
  physical WING USB prerequisite is present and attended.

## Validation result

- Focused identity, route-refresh and attended-runner suite: `158 passed`.
- Independent final implementation review: `CLEAN`, no P0/P1/P2.
- WING/static truth rails: `136 passed`.
- macOS release/security/architecture contracts: `62 passed`.
- Standard full suite: `7180 passed, 3 skipped in 433.98s`.
- Detached no-WING manifest verification and memory lint: passed.
