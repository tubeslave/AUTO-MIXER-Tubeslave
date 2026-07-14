# ADR — M114 M110 Capture-Ready Race

## 1. Context

The first CP05-bound real HW01 attempt connected to WING, read the expected
channel names and started its 48-channel CoreAudio stream. It then failed before
route proof because the immediate start response preceded the first audio
callback and reported `real_input=false`. No console mutation was attempted.

## 2. Options considered

Retry the whole attended run, add a blind sleep, accept `running=true`, wait in
the shared audio handler, or add a bounded attended-runner readiness poll with
strict identity and freshness checks.

## 3. Decision

The first preview terminal must prove a running preview owned by the retained
WebSocket and pin exact sounddevice identity. The runner immediately latches
preview teardown ownership. If the first callback is pending, it polls only the
read-only capture status at no more than the sustained WebSocket rate for an
absolute maximum of one second. Readiness requires unchanged identity,
`real_input=true`, callback age 0..1 seconds and positive strict-integer sample
progression. Every drift, malformed status, error, stop or timeout fails closed.

## 4. Why this won

It removes scheduler nondeterminism while preserving evidence of actual audio
progress. It changes only the attended hardware runner, creates no standing
authority and leaves the backend's independent ownership, serial, route,
headroom and sample-progression gates intact.

## 5. Rejected alternatives

Retries and sleeps are nondeterministic. `running` alone does not prove real
input. Handler-wide waiting affects unrelated consumers. A second start can
change ownership or device state. Unbounded or high-rate polling can hang or
trigger transport rate limits.

## 6. Implementation plan

Add strict initial/status validators and one bounded same-session poll helper;
set `capture_started` immediately after initial ownership; add delayed-success,
timeout, drift, malformed and teardown tests; validate and review; commit the
non-marker tree; then rebind CP05 in a marker-only commit before retrying HW01.

## 7. Test plan

Run the complete attended-runner test file plus relevant server/audio route
tests, WING/static rails, standard full suite, no-WING verification, diff and
memory checks, and independent final review. Hardware retry begins only from a
clean detached worktree whose CP05 binding verifies.

## 8. Risks and rollback

One second may be insufficient on an overloaded host; that remains an explicit
safe failure, not a reason to widen the window during a run. Roll back by
reverting the M114 non-marker commit and recomputing CP05 markers. Never reuse a
route or capability artifact from the failed pre-M114 attempt.
