# Automixer Repository Renovation Plan v1

Date: 2026-09-23
Status: active migration plan

## Finding

The repository currently contains roughly 950 tracked files, including 238 under backend/, 111 under audio_workbench/, 145 tests and 143 files under Docs/. The main risk is no longer missing functionality. It is competing generations of the same functionality.

Examples:
- auto_fader.py, auto_fader_hybrid.py and auto_fader_v2/*;
- auto_compressor.py and auto_compressor_cf.py;
- auto_panner.py and auto_panner_adaptive.py;
- several voice_control implementations;
- several snapshot loaders and routing/debug scripts;
- legacy AutoFOH heuristic decision code alongside the new live_runtime decision architecture;
- old mastering/live helpers alongside the new studio Audio Workbench.

## Target architecture

### shared/
Reusable infrastructure only:
- WING/dLive transport clients and address maps;
- OSC transport;
- audio device discovery/capture;
- mixer snapshots/readback;
- routing primitives;
- metrics primitives;
- source knowledge;
- logging/config/session infrastructure.

### studio/
Canonical implementation: audio_workbench/.
- editing/model cleanup;
- Mixing Directors;
- Perceptual Critic;
- autonomous iteration;
- mastering.

### live/
Canonical implementation: backend/live_runtime/.
- USB/Dante audio adapters;
- realtime feature engine;
- live context;
- instrument/group/main directors;
- soundcheck/show FSM;
- WING control plane;
- BENCH_TEST / production safety modes.

## Migration classes

KEEP_CORE
Code that is already infrastructure or validated and should survive essentially intact.

ADAPT
Useful old functionality whose transport/DSP primitive survives, but decision policy must be replaced by the new Director/Critic architecture.

ARCHIVE
Old implementation retained temporarily for regression comparison or historical reference, but removed from runtime imports.

DELETE_AFTER_PROOF
Debug scripts, duplicate loaders, obsolete variants and dead glue. Delete only after import/reference scan and replacement tests prove they are unused.

## Initial classification

### KEEP_CORE
- backend/wing_client.py
- backend/wing_addresses.py
- backend/osc/*
- backend/mixer_client_base.py
- backend/mixer_state.py
- backend/mixer_discovery.py
- backend/audio_capture.py
- backend/audio_device_scanner.py
- backend/audio_devices.py
- backend/dante_routing_config.py
- backend/routing.py
- backend/feedback_detector.py (detector primitive only)
- backend/phase_alignment.py (analysis primitive only)
- backend/autofoh_safety.py until live_runtime safety reaches parity
- backend/source_knowledge/*
- audio_workbench/* as canonical studio branch

### ADAPT INTO live_runtime
- backend/live_shared_mix.py: retain useful measurements, replace heuristic decisions.
- backend/auto_soundcheck_engine.py: retain orchestration concepts, replace policy/FSM.
- backend/autofoh_analysis.py / detectors.py / evaluation.py: extract evidence functions.
- backend/auto_fader_v2/core/acoustic_analyzer.py
- backend/auto_fader_v2/core/activity_detector.py
- backend/auto_fader_v2/core/bleed_detector.py as confidence/evidence only.
- backend/auto_fader_v2/core/channel_classifier.py
- backend/auto_fader_v2/core/integrated_lufs.py
- backend/cross_adaptive_eq.py: retain DSP/evidence ideas only if new masking tests validate them; do not reuse its hard-coded decision policy.
- backend/compressor_adaptation.py: retain parameter/DSP helpers, not old policy.
- backend/auto_effects.py / auto_fx.py / auto_reverb.py: consolidate into live Space/FX Director.

### ARCHIVE FIRST
- backend/agents/*
- backend/ai/rule_engine.py and old auto-apply decision rules
- backend/auto_eq.py
- backend/auto_fader.py
- backend/auto_fader_hybrid.py
- backend/auto_fader_v2/balance/fuzzy_controller.py
- backend/auto_fader_v2/balance/pid_controller.py
- backend/auto_fader_v2/balance/static_balancer.py
- backend/auto_compressor.py / auto_compressor_cf.py after dynamics migration
- backend/auto_panner.py / auto_panner_adaptive.py after spatial migration
- backend/auto_mastering.py: studio mastering is canonical
- backend/ml/mix_quality.py and heuristic/legacy quality decision paths pending audit
- duplicate voice stacks after one local command path is selected

### DELETE_AFTER_PROOF candidates
- backend/check_channel_10.py
- backend/check_output_routing.py
- backend/check_test_ready.py
- backend/find_and_load_snap.py
- backend/find_routing_addresses.py
- backend/find_routing_outputs_addresses.py
- backend/get_channel_node.py
- backend/list_all_snapshots.py
- backend/list_snapshots.py
- backend/load_snap.py
- backend/load_snap_final.py
- backend/load_snap_v2.py
- backend/monitor_osc_routing.py
- backend/query_channel1_routing.py
- backend/query_snap_info.py
- backend/reset_all_channels.py
- backend/reset_mixer.sh
- backend/reset_modules_trim_faders.py
- backend/route_channel_inputs.py
- backend/route_channels.py
- backend/route_custom_outputs.py
- backend/route_dante_outputs.py
- backend/scan_and_load_snap.py
These are not deleted until reference/import scans and replacement CLI/tests are complete.

## Renovation sequence

R0 INVENTORY
Generate module/import/reference graph, entrypoints, duplicate families and test ownership.

R1 FREEZE LEGACY
No new features in legacy auto_* decision modules. Bug fixes only when needed to extract/validate behavior.

R2 EXTRACT CORE
Move reusable transports, routing, snapshots, metrics and DSP primitives behind stable interfaces.

R3 LIVE CUTOVER
Replace old heuristic AutoFOH decisions one subsystem at a time with live_runtime Directors and verify on simulator/BENCH_TEST.

R4 STUDIO CUTOVER
Make audio_workbench the only offline mixing/mastering implementation and remove legacy backend mastering/mix decision paths.

R5 ARCHIVE
Move comparison-worthy legacy code to legacy/ with a manifest and no runtime imports.

R6 DELETE
Delete dead scripts/duplicates after two conditions:
1. zero runtime/import references;
2. replacement tests or HIL evidence pass.

R7 ROOT CLEANUP
Collapse stale READMEs/test reports/start scripts/config duplicates; rewrite top-level README around Studio + Live products.

## Current migration state — 2026-09-23

R1 is active: legacy `auto_*` decision modules are frozen for feature work.

R2/R3 are active for the WING control path:
- `backend/live_runtime/control_plane.py` is the canonical authorization/write/readback/verification boundary;
- `backend/live_runtime/wing_adapter.py` adapts the existing WING OSC transport and requires fresh inbound readback rather than trusting the optimistic `WingClient.state` cache;
- `backend/live_runtime/service.py` composes that control plane for an active WING session and owns the control audit trail;
- the temporary `AutoSoundcheckEngine` remains only as an ADAPT bridge for discovery/audio/mixer connection plumbing while its heuristic policy is retired;
- channel and Main faders are migrated transport surfaces;
- relative fader proposals use `fader_delta_db`, are resolved against fresh console state inside `LiveControlPlane`, and cannot exceed their own declared `max_step` even in BENCH_TEST;
- the new `main_headroom_protection` hypothesis requests a relative `-0.5 dB` move rather than the unsafe ambiguous absolute value `-0.5 dB`.

R3 channel PEQ gain cutover now includes deliberate realtime band selection:
- `EqBandLocator` carries the exact WING band plus expected frequency/Q fingerprint;
- musical EQ reductions use `eq_gain_delta_db`, not ambiguous absolute `eq_gain_db` semantics;
- `LiveControlPlane` resolves the delta against fresh current band gain and applies the proposal's own `max_step` in every write-capable mode, including BENCH_TEST;
- `WingWriteAdapter` accepts channel bands 1..4 only, re-queries physical frequency/Q before mutation, writes the resolved gain, then requires fresh post-write gain readback;
- `WingWriteAdapter.read_eq_locators()` now enumerates all four PEQ frequency/Q fingerprints from fresh inbound OSC callbacks for read-only selection;
- `backend/live_runtime/eq_locator.py` selects an already-existing band from explicit realtime spectral evidence using an octave-distance corridor plus optional Q constraints; low-confidence or no-match evidence returns no locator and therefore no hardware-actionable EQ move;
- masking/harshness hypotheses still refuse hardware EQ when no explicit locator has been produced;
- the protocol/address primitive comes from KEEP_CORE `wing_addresses.py`; no legacy `auto_eq.py` decision code was imported into the new authority.

The audit of `backend/cross_adaptive_eq.py` keeps it in ADAPT only as a possible DSP/evidence reference. Its current policy hard-codes seven band centers, channel priority rules, overlap tolerance and mirror boost/cut behavior, so it is not suitable as the new live locator or decision authority and is not imported by `live_runtime`.

Automated replacement evidence covers callback-driven channel/Main fader readback, BENCH_TEST/OBSERVE control behavior, relative fader and EQ-gain resolution, max-step rejection, explicit EQ locator requirements, fresh four-band WING F/Q enumeration, evidence-driven nearest-band selection, Q eligibility, low-confidence/no-match fail-closed behavior, WING frequency/Q fingerprint checks, and post-write verification. Physical WING HIL evidence is still required before legacy AutoFader/AutoEQ/MasterFader write authority can be severed or archived.

Legacy `MasterFaderMove` references remain in `live_shared_mix.py`, `auto_soundcheck_engine.py` and `autofoh_safety.py`. `backend/server.py` still imports the legacy `AutoEQController`. Those paths remain ARCHIVE/ADAPT candidates, not deletion candidates, until the new runtime owns their required behavior and HIL proves the replacements.

No legacy module is promoted to DELETE_AFTER_PROOF by this pass. The old fader/EQ implementations still have runtime/import references and therefore remain ARCHIVE candidates only.

See also `Docs/adr/live-channel-eq-cutover-v1.md` and `Docs/adr/live-eq-locator-selector-v1.md` for the channel-EQ migration contract and HIL gate.

## Non-negotiable migration rule

Do not delete first and debug later. First sever runtime imports, add replacement tests, run CI/HIL, then archive/delete. Conversely, do not keep old code merely because it once worked: if it has no validated role in the target architecture, it leaves the runtime tree.
