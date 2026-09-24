# ADR: Capture-driven PATCH_VERIFY lifecycle v1

Status: accepted for software migration; hardware proof still required before autonomous Main mutation.

## Context

The canonical live path already had separate validated pieces for 48-channel USB capture, coherent post-console Main evidence, read-only Main PATCH_VERIFY, and the service-owned soundcheck FSM. A manual seam remained between them: a caller had to take the first Main snapshot and explicitly invoke `LiveSoundcheckService.verify_main_tap_patch()` before normal feature frames could enter LISTEN.

That seam is unsafe as a production composition point because a caller could forget the proof, feed stale Main evidence, or accidentally use the same frame both as startup proof and as a musical decision frame.

## Decision

`backend/live_runtime/capture_bridge.py` may receive an explicit `MainTapPatchContract` in addition to its existing coherent `PostConsoleMainTapEvidenceProvider`.

The bridge does not become an FSM authority. For each worker-side snapshot it first calls the service's normal `process_feature_snapshot()` entry point. While the service is in DISCOVER that call is already fail-closed and returns a `DISCOVER` result without running a Director or issuing a console action. Only for that explicit result does the bridge immediately hand the same `MainFeatureEvidence` and configured contract to the service-owned `verify_main_tap_patch()` gate.

A successful proof advances the service to LISTEN. The proof frame is consumed by startup and is not replayed through the Director. Only a later coherent snapshot can become the first musical decision frame. A failed proof advances the service to HOLD, and later snapshots stay blocked by the service FSM.

Automatic capture-driven PATCH_VERIFY is allowed only when Main evidence comes from reserved channels in the same coherent USB snapshot. An external timestamp-only Main provider cannot be used for this startup mode. The bridge validates this composition at construction time.

The ordered proof remains unchanged:

1. fresh physical WING USB route readback;
2. independent native WING Main-meter evidence;
3. level coherence with the post-console USB Main tap;
4. LISTEN on complete proof, otherwise HOLD.

No routing repair or console write is added to PATCH_VERIFY.

## Migration classification

- **KEEP_CORE:** `backend/audio_capture.py` device/callback/ring-buffer transport; WING query/callback transport and readback primitives.
- **ADAPT:** `backend/live_runtime/capture_bridge.py`, post-console Main evidence, PATCH_VERIFY orchestration, and `LiveSoundcheckService` lifecycle composition.
- **ADAPT (temporary UI seam):** `backend/handlers/soundcheck_handlers.py`. It already routes start/stop/status through `LiveSoundcheckService`; compatibility aliases (`server.auto_soundcheck_engine`, legacy event names/status fields) remain only until all callers are migrated. It is not a decision authority.
- **ARCHIVE after reference severing + HIL:** legacy `AutoSoundcheckEngine` heuristic FSM, fixed instrument presets, and AutoFOH/AutoFader feature-to-action/evaluation/rollback policy paths.
- **DELETE_AFTER_PROOF:** none introduced by this change.

## Evidence required

Software tests must prove that:

- automatic PATCH_VERIFY refuses non-coherent external Main evidence;
- the first coherent snapshot reaches PATCH_VERIFY exactly once;
- the proof frame cannot also become a musical proposal frame;
- a later snapshot is the first frame eligible for the normal Director/Critic path;
- the real service composition advances `DISCOVER -> PATCH_VERIFY -> LISTEN` from the capture bridge with read-only WING traffic.

Hardware evidence is still required before autonomous Main mutations are enabled: actual WING USB route identity, native meter point/ballistics, level offset, timestamp skew, and verified rollback behavior in an explicitly declared BENCH_TEST session.

## Consequences

The software chain is now continuous from coherent USB capture through startup safety proof to the live decision loop, without adding any new authority to legacy `auto_*` modules. The remaining gap is lifecycle ownership of bridge construction/start/stop by the production composition root and real WING HIL proof.