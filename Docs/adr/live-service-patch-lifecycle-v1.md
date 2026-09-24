# Live Service PATCH_VERIFY Lifecycle v1

Date: 2026-09-24
Status: accepted for R3 software migration; hardware HIL still required

## Context

`MainTapPatchStartupCoordinator` already proved the configured post-console Main tap with a read-only chain:

`fresh USB route readback -> independent native WING Main meter -> level coherence`.

That proof was previously callable as an isolated component, while `LiveSoundcheckService.process_feature_snapshot()` could enter the Director/Critic loop immediately after the temporary legacy engine started. This left the canonical service lifecycle weaker than the documented FSM: feature-driven autonomous work was not causally gated by `PATCH_VERIFY`.

## Decision

`LiveSoundcheckService` owns the startup soundcheck state for the new runtime.

1. A successfully started compatibility engine enters `DISCOVER`.
2. `verify_main_tap_patch()` is the only service path from `DISCOVER` into `PATCH_VERIFY`.
3. The service composes `MainTapPatchStartupCoordinator` with the same fresh `WingWriteAdapter` transport used by the live runtime and the active WING host.
4. Successful route + physical meter + level coherence proof enters `LISTEN`.
5. Any completed proof failure enters `HOLD`.
6. `process_feature_snapshot()` fails closed while the service is in `DISCOVER`, `PATCH_VERIFY` or `HOLD`; it does not construct a new iteration, query an EQ locator, or issue a console mutation from those startup states.
7. Once `LISTEN` is reached, the existing one-hypothesis Director/Critic iteration owns `LISTEN -> VERIFY -> LISTEN/HOLD` behavior.
8. Service status exposes the startup state plus the latest PATCH_VERIFY verified/reason/physical-source evidence for UI, logs and HIL inspection.
9. `stop()` clears startup proof so a later session must prove its patch again.

`verify_main_tap_patch()` itself remains read-only. It never repairs routing and never treats cached WING state as proof.

## Mode boundary

This ADR does not broaden write permissions. `BENCH_TEST` remains an explicitly declared development/test mode where the existing control-plane mode contract may expose decisions on the console while snapshots, audit and fresh readback remain active. A real soundcheck/show must use production protections.

The low-level `execute_action()` seam remains separately mode-authorized for migration/HIL tooling; it is not the autonomous feature-loop entry point. Production autonomous feature processing is gated by the service startup state described above.

## Legacy audit and migration classification

`backend/auto_soundcheck_engine.py` still mixes useful infrastructure with legacy decision authority: its module imports audio capture/device discovery and mixer discovery, but also AutoFOH detectors/evaluation/rollback and instrument-specific EQ/compressor/fader policy. Therefore it must not be reused wholesale.

- **KEEP_CORE:** mixer discovery/connection, audio capture/device access and validated transport/readback primitives.
- **ADAPT:** temporary lifecycle/connection ownership behind `LiveSoundcheckService` while those infrastructure seams move into `live_runtime`.
- **ARCHIVE:** legacy heuristic FSM, instrument presets, AutoFOH feature-to-action/evaluation/rollback policy after runtime references are severed and replacement HIL passes.
- **DELETE_AFTER_PROOF:** none added by this change.

No new behavior is added to legacy `auto_*` decision modules.

## Evidence required

Software replacement evidence must cover:

- service start reports `DISCOVER`;
- feature snapshots are inert before PATCH_VERIFY;
- successful read-only proof moves the service to `LISTEN`;
- route failure moves the service to `HOLD` without reading the independent Main meter;
- successful proof is required by service-owned iteration tests before any feature-driven WING action;
- stop clears the proof state;
- focused CI includes the service lifecycle tests.

Physical WING HIL remains mandatory before legacy Main/AutoFOH write authority can be severed or archived. In particular, HIL must confirm native meter collection identity, meter-vs-USB tap comparability, real timestamp skew and verified console readback on the target firmware.

## Consequence / next migration seam

The remaining gap is the lifecycle ingress: `LiveAudioCaptureBridge` owns coherent post-console Main tap evidence, but startup orchestration still needs to feed the first suitable fresh tap evidence into `LiveSoundcheckService.verify_main_tap_patch()` and hold feature-driven processing until that proof completes. That wiring is the next bounded R3 task; it must remain fail-closed and read-only during PATCH_VERIFY.
