# Automixer modernization audit — 2026-09-23

## Purpose
Move the old Automixer onto the new Studio + Live architecture. Prefer deletion/rebuild over preserving legacy heuristics that can silently compete with the new Directors.

## Repository snapshot
- 1,821 tracked files.
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
backend/ai/knowledge contains hundreds of auto-collected cards, including unrelated books/videos/articles. Do not use it as an authoritative runtime knowledge source.
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

Next cleanup target: backend/server.py composition root and parallel legacy decision controllers.
