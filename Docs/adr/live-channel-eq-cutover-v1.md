# Live Channel EQ Cutover v1

Date: 2026-09-23
Status: simulator-tested migration slice; physical WING HIL pending

## Decision

The canonical live EQ write path is now owned by `backend/live_runtime/`.
Legacy AutoEQ policy is not reused as a decision authority.

The first migrated EQ surface is channel PEQ band gain. A write must carry:

- channel target `ch:N`;
- an explicit `EqBandLocator` with WING band number, expected frequency and Q;
- a relative gain proposal `eq_gain_delta_db` from the Director;
- an explicit `max_step` bound.

`LiveControlPlane` reads the current gain first and resolves the relative delta to an absolute `eq_gain_db` write. `WingWriteAdapter` then verifies that the physical WING band's fresh frequency/Q readback still matches the locator before changing gain, performs the write, and the control plane requires fresh gain readback before accepting the action.

## Why

The previous live hypotheses emitted values such as `eq_gain_db = -0.7` while the musical intent was "reduce this area by 0.7 dB". That mixes absolute and relative semantics. If the existing console band were at +6 dB, an absolute write to -0.7 dB would make a 6.7 dB jump even though the proposal declared a 1 dB maximum step.

A second ambiguity was the missing physical band identity. "Reduce harshness" is a musical hypothesis, not a safe mixer command. Without an exact band/frequency/Q locator it is impossible to know which WING EQ slot should change.

The new contract removes both ambiguities.

## Runtime invariants

1. `eq_gain_delta_db` is never sent directly to hardware.
2. Relative EQ gain is resolved from fresh current band gain inside `LiveControlPlane`.
3. The action's declared `max_step` is enforced even in `BENCH_TEST`.
4. Missing EQ locator fails closed before any OSC write.
5. WING channel PEQ bands are limited to 1..4 for this surface.
6. WING band gain is bounded to -15..+15 dB.
7. Frequency/Q are queried from the physical WING before mutation and must match the locator within explicit adapter tolerances.
8. Successful OSC send is not success; fresh post-write readback must match the resolved gain target.
9. `BENCH_TEST` bypasses production policy gates, but not these engineering invariants, readback, audit or proposal bounds.

## Director behavior

`live_runtime.decision_engine` no longer emits hardware-actionable masking or harshness EQ moves without an upstream locator. When a locator is present, the hypothesis uses `eq_gain_delta_db`, not absolute `eq_gain_db`.

This intentionally creates a temporary capability gap: until the realtime state/evidence layer selects a physical band from current WING state, the system may describe an EQ problem but will not mutate the console for it. Failing closed is preferable to reviving an ambiguous legacy heuristic.

## Renovation classification

### KEEP_CORE

- `backend/wing_client.py`
- `backend/wing_addresses.py`
- OSC callback/query transport and fresh physical readback

The WING protocol map confirms `/ch/{ch}/eq/{band}g`, `/f` and `/q` for bands 1..4, with gain -15..+15 dB, frequency 20..20000 Hz and Q 0.44..10.

### ADAPT

- `backend/live_runtime/contracts.py`: typed `EqBandLocator` and proposal contract.
- `backend/live_runtime/control_plane.py`: relative-to-absolute resolution and verification.
- `backend/live_runtime/wing_adapter.py`: physical WING EQ transport boundary.
- `backend/cross_adaptive_eq.py`: remains only a candidate DSP/evidence primitive pending validation; no decision policy imported.

### ARCHIVE

- `backend/auto_eq.py` and its heuristic decision policy.
- legacy AutoFOH channel-EQ decision paths once runtime references are severed.

`backend/server.py` still imports `AutoEQController`, so `auto_eq.py` cannot yet be moved out of the runtime tree.

### DELETE_AFTER_PROOF

None in this slice. No legacy EQ module is deleted before import/runtime cutover and WING HIL evidence.

## Replacement evidence

Automated tests cover:

- ambiguous EQ hypothesis suppressed without a locator;
- masking/harshness hypotheses use located relative gain when a locator exists;
- relative gain resolves from current band gain;
- max-step rejection in `BENCH_TEST`;
- WING band-frequency/Q fingerprint checks;
- missing locator fail-closed behavior;
- callback-driven post-write gain readback;
- direct relative hardware write rejection;
- band and gain range validation.

Physical HIL remains a gate before legacy AutoEQ write authority is severed.

## Next step

Build the realtime EQ locator selector from current WING band state plus spectral evidence. It must choose an existing band deliberately rather than inventing a slot. After that, run one small `BENCH_TEST` EQ delta on a real WING, capture before/frequency/Q/gain -> write -> readback -> rollback evidence, then begin severing the legacy `AutoEQController` write path.
