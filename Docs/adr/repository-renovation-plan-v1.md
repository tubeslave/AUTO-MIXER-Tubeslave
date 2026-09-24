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

## Current migration state — 2026-09-24

R1 is active: legacy `auto_*` decision modules are frozen for feature work.

Repository continuity for the R3 work is now explicit: the validated live-runtime history ending at `9e015eea` was found to exist in the repository without any branch currently pointing at it, while default `master` was behind that renovation state. A dedicated `live-soundcheck-renovation` branch now anchors that history and is the working branch for subsequent R3 migration. The focused live workflow was updated to run on pushes to this branch so new migration commits cannot silently become untested/orphaned work.

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

R3 realtime EQ composition is now service-owned rather than caller-owned:
- `LiveSoundcheckService` reuses one `WingWriteAdapter` for both read-only PEQ fingerprint selection and verified writes on the same physical WING transport;
- `LiveSoundcheckService.select_eq_locator()` resolves explicit `EqTargetEvidence` against fresh physical bands and audits selected/unresolved outcomes;
- `LiveSoundcheckService.propose_hypothesis()` accepts evidence keyed by `(channel, intent)`, builds only evidence-backed locators, then calls the new `live_runtime` Director;
- the complete software path `spectral evidence -> fresh WING bands -> locator -> Director -> relative EQ action -> LiveControlPlane -> write -> fresh readback` is covered by focused CI;
- low-confidence evidence fails closed without querying or mutating the console.

R3 verified rollback is now owned by the canonical live control plane:
- every reversible successful/mismatched write retains the fresh pre-write value as its rollback target;
- `LiveControlPlane.rollback()` performs a fresh read, restores the captured absolute value, performs a second fresh readback and audits verified/mismatched restoration;
- relative fader/EQ proposals roll back to the captured absolute value rather than applying an inverse delta;
- PEQ rollback preserves the original frequency/Q locator fingerprint and therefore fails closed if the physical band identity has changed;
- rollback is treated as restoration, not a new musical decision: write-capable modes may restore a captured value even if that parameter would be blocked as a new AUTO_SAFE proposal, while OBSERVE/PROPOSE/FREEZE and manual freeze still prohibit the write;
- `LiveSoundcheckService.rollback_action()` exposes the same authoritative path to the runtime rather than delegating rollback to legacy AutoFOH evaluation code.

R3 one-hypothesis iteration now exists as a canonical live-runtime component:
- `backend/live_runtime/iteration.py` permits exactly one applied hypothesis to wait for verification;
- the verify window is causal and non-blocking: the realtime loop feeds a later feature snapshot instead of the coordinator sleeping or blocking audio/control processing;
- the default Critic is `backend/live_runtime/decision_engine.verify`, preserving the studio-derived `one hypothesis -> verify -> keep/rollback` architecture without importing studio editing/mastering;
- an immediate mixer readback mismatch triggers verified restoration before perceptual verification begins;
- a Critic regression triggers the same canonical `rollback_action()` path; failed restoration enters HOLD and prevents more autonomous hypotheses;
- operator touch has priority: `operator_took_control` enters HOLD without issuing rollback, so automation cannot undo an operator's corrective move;
- OBSERVE/PROPOSE/policy-blocked actions never open a verification window because no mutation occurred.

R3 authoritative Main evidence now has a concrete post-console path:
- `backend/live_runtime/main_evidence.py` measures RMS/peak/crest only from explicitly routed post-console Main USB tap channels; it never synthesizes Main by summing input stems;
- `LiveAudioCaptureBridge` accepts exactly one authoritative Main source: either an external meter provider or a snapshot provider;
- snapshot Main evidence is derived from the same coherent 48-channel copy as channel features, so the Main and feature timestamps are identical;
- capture slots reserved for the Main tap are excluded from channel-level features so a returned Main cannot be mistaken for a controllable source channel.

R3 Main PATCH_VERIFY now has two explicit proof layers:
- `MainTapPatchVerifier` fresh-reads every declared WING USB output route and requires exact `MAIN N` identity; it does not trust optimistic routing cache and never mutates routing;
- `PhysicalMainMeterEvidence` is the typed ingress for an independent Main-meter observation, with no guessed mixer address embedded in the proof layer;
- `MainTapLevelCoherenceVerifier` rejects non-finite evidence, stale/time-skewed measurements, near-silence, excessive peak mismatch and optional/required RMS mismatch;
- `MainTapPatchGateVerifier` authorizes the tap only when both physical route identity and independent level coherence pass, and short-circuits level proof when routing is wrong;
- repository search did not reveal a validated physical WING Main-meter OSC endpoint, so the concrete physical meter provider remains deliberately unimplemented rather than guessing an address;
- autonomous Main mutation therefore remains blocked until that provider is validated on real hardware and the complete route + level gate passes HIL.

The audit of `backend/autofoh_evaluation.py` splits its migration class by responsibility. Its explicit observability warning and any validated detector/metric evidence remain **ADAPT** candidates. Its `PendingActionEvaluation`, legacy typed-action rollback construction, proxy evaluation thresholds and proxy evaluation/rollback orchestration are **ARCHIVE** candidates once runtime imports are severed, because those authorities are now owned by `live_runtime` control/iteration/PATCH_VERIFY layers. No legacy AutoFOH rollback or decision code is imported by the new coordinator.

The audit of `backend/cross_adaptive_eq.py` keeps it in ADAPT only as a possible DSP/evidence reference. Its current policy hard-codes seven band centers, channel priority rules, overlap tolerance and mirror boost/cut behavior, so it is not suitable as the new live locator or decision authority and is not imported by `live_runtime`.

The audit of `backend/signal_analysis.py` classifies its level/envelope/transient/spectral measurement ideas as **ADAPT** evidence references only. The file is coupled to compressor-specific state, LUFS/TruePeak objects and WING ratio helpers, so it is not imported wholesale into `live_runtime`; reusable primitives should be re-expressed behind canonical feature contracts instead of reviving its old compressor policy.

Automated replacement evidence covers callback-driven channel/Main fader readback, BENCH_TEST/OBSERVE control behavior, relative fader and EQ-gain resolution, max-step rejection, explicit EQ locator requirements, fresh four-band WING F/Q enumeration, evidence-driven nearest-band selection, Q eligibility, low-confidence/no-match fail-closed behavior, service-owned evidence/locator/Director composition, WING frequency/Q fingerprint checks, post-write verification, software write -> rollback -> second-readback restoration, single in-flight hypothesis enforcement, verify-window gating, Critic KEEP/rollback outcomes, operator-touch HOLD, rollback-failure HOLD, coherent 48-channel capture snapshots, post-console Main tap extraction/exclusion, fresh Main routing identity proof and fail-closed Main route + level coherence logic. The focused `Stem Offline Test` workflow runs these live-runtime tests on `live-soundcheck-renovation`. Physical WING HIL evidence is still required before legacy AutoFader/AutoEQ/MasterFader/AutoFOH evaluation write or rollback authority can be severed or archived.

Legacy `MasterFaderMove` references remain in `live_shared_mix.py`, `auto_soundcheck_engine.py` and `autofoh_safety.py`. `backend/server.py` still imports the legacy `AutoEQController`, `AutoFaderController`, `AutoCompressorController` and `AutoSoundcheckEngine`. Those paths remain ARCHIVE/ADAPT candidates, not deletion candidates, until the new runtime owns their required behavior and HIL proves the replacements.

No legacy module is promoted to DELETE_AFTER_PROOF by this pass. The old fader/EQ/evaluation implementations still have runtime/import references and therefore remain ARCHIVE candidates only.

See also `Docs/adr/live-channel-eq-cutover-v1.md`, `Docs/adr/live-eq-locator-selector-v1.md`, `Docs/adr/live-eq-evidence-service-composition-v1.md`, `Docs/adr/live-verified-rollback-v1.md`, `Docs/adr/live-one-hypothesis-iteration-v1.md`, `Docs/adr/live-main-evidence-post-console-tap-v1.md` and `Docs/adr/live-main-level-coherence-v1.md` for the current WING control migration contracts and HIL gates.

## Non-negotiable migration rule

Do not delete first and debug later. First sever runtime imports, add replacement tests, run CI/HIL, then archive/delete. Conversely, do not keep old code merely because it once worked: if it has no validated role in the target architecture, it leaves the runtime tree.
