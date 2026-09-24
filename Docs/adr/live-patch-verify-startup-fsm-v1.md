# Live PATCH_VERIFY Startup FSM v1

Date: 2026-09-24
Status: implemented in software; physical WING HIL pending

## Decision

The canonical live runtime owns a read-only startup coordinator for the Main-return PATCH_VERIFY gate in `backend/live_runtime/patch_startup.py`.

The coordinator may execute only while the soundcheck FSM is in `PATCH_VERIFY`. It proves the configured post-console Main return in this strict causal order:

1. fresh WING USB-output route readback for every reserved Main tap slot;
2. one independent native WING Main-meter observation;
3. level-coherence proof between that physical meter and the post-console USB tap evidence.

The ordering is deliberate. The native meter is not read when route identity fails, so a plausible Main meter sample can never bless a stale or incorrectly routed USB return.

## State transitions

- complete route + meter + coherence proof: `PATCH_VERIFY -> LISTEN`;
- route mismatch/readback failure: `PATCH_VERIFY -> HOLD`;
- native meter transport/protocol failure: `PATCH_VERIFY -> HOLD`;
- invalid meter evidence: `PATCH_VERIFY -> HOLD`;
- level/time/signal coherence failure: `PATCH_VERIFY -> HOLD`;
- invocation from any state other than `PATCH_VERIFY`: programming error before either transport is touched.

Expected proof failures return a structured result instead of escaping into the realtime loop. The result retains route observations, level-verification evidence when available, the physical meter source and an audit-friendly reason.

## Safety contract

This startup gate is read-only. It does not repair routing, move Main faders, change EQ/dynamics, or create a second persistent WING control authority. The WING OSC adapter remains responsible for fresh route readback, and `WingNativeMainMeterProvider` remains a bounded one-shot native TCP/UDP reader.

The coordinator emits one `live_patch_verify_complete` audit event when an attempt reaches LISTEN or HOLD. The event records the before/after FSM states, proof outcome, reason, observed routes, physical source, and level-coherence evidence when available.

Passing the software gate does **not** authorize autonomous Main mutation yet. Hardware HIL must still establish the real WING firmware meter collection, detector/ballistics relation to the USB post-console tap, and practical timestamp skew/offset bounds before the Main Director may use the proof as a production write gate.

BENCH_TEST semantics are unchanged. This module contains no write bypass and therefore does not infer or enable BENCH_TEST. Production soundcheck/show mode protections remain authoritative.

## Legacy audit

`backend/auto_soundcheck_engine.py` was re-audited before lifecycle reuse. It currently combines infrastructure with extensive heuristic policy, including fixed instrument EQ/HPF/compressor/LUFS presets and direct imports of legacy AutoFOH evaluation/rollback authorities.

Classification by responsibility:

- **KEEP_CORE**: validated mixer discovery/connection and audio-capture/device plumbing only, after extraction behind stable interfaces.
- **ADAPT**: startup/lifecycle concepts that are still needed to bridge discovery, audio capture and an active physical mixer into `live_runtime`.
- **ARCHIVE**: its heuristic soundcheck FSM, fixed musical presets, legacy AutoFOH pending-action evaluation/rollback and automatic correction policy once canonical runtime references are severed and HIL passes.
- **DELETE_AFTER_PROOF**: none in this pass. `LiveSoundcheckService` still uses `AutoSoundcheckEngine` as a temporary bridge, so deletion or physical archival would violate the migration rule.

No new feature was added to any legacy `auto_*` module.

## Tests

`tests/test_live_patch_startup.py` covers:

- exact route-readback-before-meter ordering;
- success transition to LISTEN;
- route failure short-circuit without native meter read;
- native meter failure -> HOLD;
- invalid meter provider output -> HOLD;
- level mismatch -> HOLD with physical evidence retained;
- wrong-FSM-state rejection before any WING transport read;
- no WING writes during PATCH_VERIFY.

The test is part of the focused `Stem Offline Test` workflow on `live-soundcheck-renovation`.

## Next gate

Wire `MainTapPatchStartupCoordinator` into the `LiveSoundcheckService` startup/session lifecycle so an active WING session enters `PATCH_VERIFY`, consumes a current post-console Main tap snapshot, runs this coordinator, and exposes the resulting LISTEN/HOLD state through service status and audit. That wiring must not relax the existing HIL block on autonomous Main writes.
