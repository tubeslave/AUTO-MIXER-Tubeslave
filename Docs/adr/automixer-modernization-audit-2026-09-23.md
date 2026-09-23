# Automixer modernization audit — 2026-09-23

## Purpose
Move the old Automixer onto the new Studio + Live architecture. Prefer deletion/rebuild over preserving legacy heuristics that can silently compete with the new Directors.

## Repository snapshot
- 1,821 tracked files at the start of the audit.
- backend/: 666 files.
- audio_workbench/: 111 files.
- tests/: 143 files.
- Docs/: 142 files.
- sessions/: 448 generated session/config artifacts committed to Git.
- backend/ai/knowledge/: at least 430 auto-collected article/book/video cards, including clearly unrelated material.

## KEEP / make authoritative
### Shared hardware and infrastructure
- backend/wing_client.py and WING OSC address/protocol support.
- backend/osc/, osc_manager.py.
- mixer state/readback, discovery and snapshot primitives.
- audio device discovery/capture primitives.
- Dante routing primitives and WING official protocol documentation.
- feedback detector.
- typed AutoFOH safety actions where useful as transport contracts.
- Dugan/NOM components when used as a dedicated speech/vocal automixer, not as the music-mix brain.

### New studio authority
- audio_workbench/editing/
- audio_workbench/mixing/
- audio_workbench/mastering/
- Perceptual Critic / Artifact Critic / section context / autonomous iteration.

### New live authority
- backend/live_runtime/
- Docs/adr/live-soundcheck-pipeline-v2.md
- Docs/adr/studio-to-live-transfer-matrix.md

## MIGRATE, then retire legacy implementation
These contain useful plumbing or concepts, but must not remain parallel decision authorities:
- backend/auto_soundcheck_engine.py
- backend/live_shared_mix.py
- backend/autofoh_analysis.py
- backend/autofoh_detectors.py
- backend/autofoh_evaluation.py
- backend/autofoh_profiles.py
- backend/auto_fader_v2/
- backend/auto_eq*.py
- backend/auto_compressor*.py
- backend/auto_effects.py / auto_fx.py / auto_reverb.py
- backend/auto_panner*.py
- backend/auto_phase_gcc_phat.py
- legacy gain-staging orchestration

Migration rule: extract proven I/O, state, measurement or safety code behind new live_runtime interfaces. Do not copy old threshold heuristics merely because tests exist.

## QUARANTINE
- old standalone snapshot/routing/debug scripts such as load_snap variants, find-routing, reset and ad-hoc route scripts: retain only until equivalent typed service + tests exist.
- old experimental ML/training paths: no runtime import from production Studio or Live entrypoints.
- legacy frontend controls tied to removed modules: quarantine after API inventory.

## DELETE / stop tracking now
- generated top-level sessions/ artifacts. Git history remains the archive; runtime output must not live in source control.

## High-priority architectural defect
backend/server.py is still a legacy composition root. It imports and owns many old automatic decision controllers directly: AutoEQ, AutoFader, AutoCompressor, old AutoSoundcheckEngine, bleed service and training service. Even if new live_runtime exists, those old authorities can still compete with it.

Target composition:
- backend/server.py becomes transport/UI shell only.
- Studio execution enters through audio_workbench.
- Live execution enters through backend/live_runtime.
- WING writes have exactly one live write path.
- legacy automatic controllers are not instantiated by default.

## Knowledge-base cleanup
backend/ai/knowledge contained hundreds of auto-collected cards, including unrelated books/videos/articles. It is not an authoritative runtime knowledge source.
Plan:
1. preserve only curated instrument/mixing/live/WING documents;
2. move source-grounded learning to the newer Mixing Learning store;
3. remove unrelated auto-collected cards after dependency check.

## Migration phases
1. Hygiene: generated sessions, duplicate artifacts, ignore rules.
2. Composition root: stop legacy controllers autostarting/competing.
3. Hardware adapters: WING/OSC/audio/Dante behind live_runtime contracts.
4. Decision migration: replace AutoFOH heuristics with studio-derived Directors.
5. UI/API migration: expose new Studio/Live modes and BENCH_TEST.
6. Quarantine removal: delete old controllers/scripts after replacement tests pass.
7. Knowledge cleanup.
8. Final dead-code/import audit and README rewrite.

## Deletion policy
A legacy module may be deleted when:
- no authoritative entrypoint imports it;
- its useful behavior has a replacement test;
- hardware-specific knowledge has been transferred;
- deleting it does not remove the only rollback/readback/safety implementation.

Git history is the archive. We do not keep dead code in the active tree merely as a museum.

## Cleanup pass 1 — completed
- Removed 448 generated top-level session artifacts from the active tree.
- Added sessions/ to .gitignore so runtime state does not return to source control.
- Removed 430 auto-collected article/book/video cards from backend/ai/knowledge.
- Preserved the six runtime-curated knowledge files explicitly used by KnowledgeBase.AGENT_RUNTIME_CATEGORIES:
  agent_auto_apply_protocol, instrument_profiles, live_sound_checklist, mixing_rules, troubleshooting, wing_osc_reference.
- Tracked-file count dropped from 1,821 to 944 without deleting WING/OSC/audio/Dante infrastructure or new Studio/Live code.
- Git history remains the archive for all removed material.

## Cleanup pass 2 — live composition seam
Classification and action:
- KEEP: websocket handler API names and callback/event transport, because the frontend still depends on them.
- MIGRATE: AutoSoundcheckEngine. It remains temporarily underneath a dedicated compatibility bridge because it still contains useful discovery/audio/readback/logging plumbing.
- KEEP/AUTHORITY: backend/live_runtime/service.py now owns construction of the live soundcheck engine.
- DELETE FROM NEW ENTRYPOINTS: handlers no longer import or construct AutoSoundcheckEngine directly.
- QUARANTINE-IN-PLACE: legacy engine decision heuristics remain reachable only through the bridge until their useful plumbing is split out and their decision authority is replaced.

Behavioral changes:
- missing live mode now defaults to OBSERVE, not the old implicit write-capable path;
- explicit BENCH_TEST is preserved for visible WING development testing;
- explicit legacy `observe_only: false` maps to SUPERVISED, never BENCH_TEST;
- invalid modes are rejected before engine construction;
- live mode is surfaced in start/state events for auditability.

Replacement tests:
- tests/test_soundcheck_handlers.py verifies the live_runtime construction seam, safe default, explicit BENCH_TEST and invalid-mode blocking;
- tests/test_live_runtime_service.py verifies bridge mode mapping and isolates the one remaining legacy engine import.

## Cleanup pass 3 — live lifecycle ownership
Classification and action:
- KEEP: websocket message names and event callbacks remain transport compatibility only.
- KEEP/AUTHORITY: `LiveSoundcheckService` now owns the active-engine lifecycle: construct, start, parallel-start rejection, status, stop and failed-start cleanup.
- MIGRATE: `AutoSoundcheckEngine` remains only as the temporary engine underneath the service.
- QUARANTINE: `server.auto_soundcheck_engine` is now explicitly a temporary compatibility alias for legacy server cleanup/sync code; live handlers no longer use it to decide whether an engine is active or to query/stop the engine.
- DELETE FROM HANDLER AUTHORITY: handlers no longer call `engine.start_async()`, `engine.stop()` or `engine.get_status()` directly.

Why this matters:
- there is now one lifecycle authority for the new live entrypoint;
- a failed engine start cannot leave a phantom active live runtime;
- parallel starts are rejected by `live_runtime`, not by duplicated UI heuristics;
- the next server cleanup can remove the `AutoSoundcheckEngine` type import/ownership without changing the websocket API.

Replacement tests added:
- service start/stop/status ownership;
- rejection of parallel live starts;
- cleanup after failed engine start;
- handler start/stop/status through the service;
- BENCH_TEST and production-mode mapping remain covered.

CI evidence:
- focused `Stem Offline Test` run on commit `fbe616f327056523264de8f87373f2de9cea7de3` passed, including `tests/test_live_decision_engine.py`, `tests/test_live_runtime_service.py` and `tests/test_soundcheck_handlers.py`;
- the full Python 3.10/3.11/3.12 `Tests` matrix for the same code commit was still running when this audit entry was written.

## Next cleanup target
- remove the direct AutoSoundcheckEngine import/type dependency from backend/server.py;
- inventory server-owned AutoEQ/AutoFader/AutoCompressor startup surfaces and move them behind live_runtime or disable them by default;
- then split WING/audio/readback plumbing out of AutoSoundcheckEngine so the legacy decision loop can be deleted.
