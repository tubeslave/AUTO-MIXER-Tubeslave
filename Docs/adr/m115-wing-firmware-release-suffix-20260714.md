# ADR — M115 WING Firmware Release Suffix

## 1. Context

After M114, real HW01 connected to WING Rack serial `0100Q1V0604AAE`, started a
fresh 48-channel capture and reached the fixed route pass. It blocked with zero
writes because the real firmware string `3.0.5-4-ga4ea833d:release` contains a
colon excluded by duplicated server and runner regexes.

## 2. Options considered

Add colon to the generic suffix, use prefix matching, pin the full build string,
revise both inline regexes, or introduce one shared strict predicate.

## 3. Decision

Add a pure shared full-match predicate to `master_capture_route_proof.py`:
base `3.0.5`, optional existing build suffix, and only an optional exact final
`:release` tag. Both server issuance and runner semantic verification call it.
The exact raw firmware string remains unchanged in authority and evidence.

## 4. Why this won

It explains the physical readback without accepting another firmware family or
arbitrary tags, and it removes demonstrated drift between two trust boundaries.

## 5. Rejected alternatives

Generic colon acceptance and prefix matching are too broad. Exact build pinning
would reject another legitimate 3.0.5 release build. Two inline regexes would
retain maintenance drift.

## 6. Implementation plan

Add the predicate, replace both gates, add positive/negative unit cases plus
backend issuance and runner artifact integration tests, validate, commit and
perform a marker-only CP05 rebind.

## 7. Test plan

Run route-proof and runner suites, combined M110 slice, WING rails, full
standard tests, no-WING and memory lint. Repeat route-only HW01 from a new clean
detached worktree.

## 8. Risks and rollback

An unknown future terminal tag remains blocked deliberately. Rollback is the
M115 source commit plus a recomputed CP05 marker-only commit. No hardware retry
may reuse authority from either failed route attempt.

