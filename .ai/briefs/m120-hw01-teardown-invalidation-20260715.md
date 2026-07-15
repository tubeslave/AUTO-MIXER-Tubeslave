# M120 — HW01 teardown readiness-invalidation handling

## Task

Repair the proof-owned M110 route-only teardown so a successful real-WING
HW01 route proof is not reclassified as blocked merely because the backend
emits its expected `auto_soundcheck_status: readiness_invalidated` after the
confirmed mixer disconnect.

## Physical reproduction

- Source: `45e2a2ae66853dc79a8db693f8ccfe15cbc67cc8`
- Tree: `ff9c2bd91a34ea91fa2530a5de4361f13d1b6595`
- Console: WING Rack `0100Q1V0604AAE`, firmware
  `3.0.5-4-ga4ea833d:release`
- CoreAudio: `WING`, device 1, 48 inputs, 48 outputs, 48 kHz
- Main 47/48 preflight peaks: `-13.17/-13.19 dBFS`
- Accepted proof artifact:
  `/Users/dmitrijvolkov/.automixer/runtime/m110_master_route_proofs/4937d06623084a2d98e7bc3f9b8f8a17.json`
- Artifact result: `accepted=true`, serial/USB identity verified, two complete
  route passes, conservative true peak `-12.318 dBTP`, console mutation write
  attempts `0`
- Runner summary:
  `/Users/dmitrijvolkov/.automixer/runtime/m110_route_only_45e2a2ae6_20260715T051100Z/plan-summary.json`
- Runner terminal: blocked only by
  `proof_owned_master_route_teardown_failed:wing_disconnect_failed:RuntimeError`

## Required behavior

1. Active-session readiness invalidation remains fatal everywhere outside
   terminal teardown.
2. During an explicit disconnect teardown, the runner may consume the one
   expected readiness-invalidation notification associated with the confirmed
   disconnect.
3. Duplicate disconnect terminals, unexpected control terminals, malformed
   disconnect results and unrelated backend failures remain fatal.
4. Clean backend termination, stopped capture, disconnected mixer and revoked
   route authority remain mandatory before HW01 can pass.
5. No live authority, allowlist, OSC write path or dependency changes.

## Acceptance

- Focused teardown regression tests pass, including adversarial negative cases.
- Existing M110 runner, route-proof and static WING rails pass.
- Full repository and no-WING validation pass from committed source.
- Independent review accepts the diff.
- CP05 is rebound in one marker-only commit before the next real-WING run.
- A repeated attended route-only run reports verified HW01 and verified
  teardown with `live_write=false`.
