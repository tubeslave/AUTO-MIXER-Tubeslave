# Current Architecture Map

## Purpose

This document describes the current repository state.

- It does not approve new product behavior.
- It does not replace policy review.
- It is derived from `docs/runtime_ownership_matrix.md` and from current repo entrypoints and module wiring.

## Executive Summary

- The repository contains several overlapping architectures rather than one clean runtime.
- The canonical live coordinator is `backend/server.py`.
- The canonical live input trim path is `backend/server.py -> backend/arrangement_automation/live_input_trim_controller.py`.
- `automixer/production_mix_v1` is an offline-first pipeline, not an approved live console path.
- `backend/auto_soundcheck_engine.py` is active but overlapping, not the canonical live runtime.
- Panic stop is not system-wide. Current documentation scope is manual `LiveApplyService` paths only.

## Runtime Classification

### Active Live Runtime

- `frontend/src/services/websocket.js`: frontend WebSocket entry to the backend runtime.
- `backend/server.py`: main live runtime coordinator and dispatch owner.
- `backend/handlers/`: WebSocket handler layer registered through `register_all_handlers()`.
- `backend/audio_capture.py`: shared live audio ingestion service.
- `backend/live_apply.py`: current gated manual live-write service for supported operations.
- `backend/arrangement_automation/live_input_trim_controller.py`: canonical live trim path.
- `backend/wing_client.py` and `backend/osc/enhanced_osc_client.py`: live Wing console transport layer.

### Active Offline Runtime

- `automixer/production_mix_v1/__main__.py`: CLI entry for the current offline-first production mix pipeline.
- `automixer/production_mix_v1/pipeline.py`: offline analyze -> candidate -> evaluate -> safety -> queue flow.
- `automixer/production_mix_v1/osc_queue.py`: offline queue that can hold console commands but is not an approved live path.

### Active But Overlapping

- `start_soundcheck.py -> backend/auto_soundcheck_engine.py`: headless soundcheck flow that can still reach real-write surfaces.
- `backend/auto_fader.py`: live automation path with direct fader writes outside `LiveApplyService`.
- `backend/arrangement_automation/controller.py`: live arrangement automation with direct fader application behavior.

### Safety-Critical Exception Candidate

- `backend/feedback_detector.py` together with `backend/auto_soundcheck_engine.py::_handle_feedback_event`: emergency feedback response candidate only.
- This path is not an approved exception yet and requires future policy review before any live expansion.
- It must be kept policy-separated from `backend/auto_soundcheck_engine.py` general live-apply expansion work. See `Docs/TUB-194_AUTOSOUNDCHECK_FEEDBACK_POLICY_SPLIT.md`.

### Operator-Only Legacy Tools

- `backend/restore_channels.py`
- `backend/load_snap_final.py`
- `backend/find_and_load_snap.py`
- `backend/load_snap.py`
- `backend/load_snap_v2.py`
- `backend/scan_and_load_snap.py`
- `backend/reset_all_channels.py`
- `backend/reset_modules_trim_faders.py`
- `backend/reset_mixer.sh`

### Research / Experimental

- `ai_mixing_pipeline/`: offline-only AI mixing pipeline with render/evaluate/report intent.
- `backend/auto_fader_v2/`: partial secondary auto-fader surface with shared bleed-analysis components and freeze hooks, but no documented canonical ownership.
- `backend/ai/` and `backend/ml/`: separate ML/agent-era surfaces that do not define the current canonical live path.

### Unclear Ownership

- `mix_agent/`: directory tree exists, but the repo currently exposes only `__pycache__` artifacts and indirect imports from `ai_mixing_pipeline`.
- `backend/handlers/agent_handlers.py`: handler surface exists for `server.mixing_agent`, but current `backend/server.py` evidence does not establish it as an active canonical runtime.
- `backend/auto_fader_v2/`: partially referenced from freeze/status flows and shared bleed utilities, but its runtime ownership remains unclear.

### Archive Candidates

- Operator-only restore/snapshot/reset utilities are archive candidates for future cleanup, but not approved for removal.
- `mix_agent/` is an archive-or-rebuild candidate if no source-backed runtime is restored.

## Live Runtime Map

### Frontend WebSocket UI

- `frontend/src/services/websocket.js` is the visible control surface for the browser UI.
- It connects to `ws://localhost:8765` and sends message-type commands for mixer control, automation, voice control, and snapshot operations.

### Main Coordinator

- `backend/server.py` is the central live runtime owner.
- It creates the mixer client, audio capture, controller instances, WebSocket dispatch table, and cleanup lifecycle.
- It also contains both gated and non-gated write paths, which is why it remains a high-risk integration surface.

### Handler Layer

- `backend/handlers/__init__.py` registers the message dispatch table.
- `backend/handlers/mixer_handlers.py` is the current manual write surface for `set_fader`, `set_gain`, `set_eq`, and `set_compressor`.
- `backend/handlers/voice_handlers.py` is the frontend-facing entrypoint for voice-control lifecycle.
- `backend/handlers/snapshot_handlers.py` exposes direct snapshot load/save operations through the current mixer client.
- `backend/handlers/arrangement_automation_handlers.py` exposes arrangement automation start/stop/tick/live-apply controls.
- `backend/handlers/automation_handlers.py` exposes freeze and emergency-stop style controls, but those semantics do not equal a system-wide console-write panic stop.

### Audio Input Path

- `backend/audio_capture.py` is the shared capture service for live analyzers and automation.
- It owns ring buffers, device detection, subscriber fan-out, and test fallback generation.
- Live modules are expected to reuse this service rather than create independent capture loops.

### Canonical Live Trim Path

- `backend/arrangement_automation/live_input_trim_controller.py` is the canonical live input trim path confirmed by the board.
- It uses role-aware, bleed-aware, meter-aware logic and applies supported writes through `LiveApplyService`.

### Current Gated Write Service

- `backend/live_apply.py` is the current gated live-write service.
- The documented approved manual path is `set_fader` / `set_gain`.
- Manual `set_eq` / `set_compressor` are currently blocked at this gate as unsupported operations.
- Panic-stop wording must remain limited to manual `LiveApplyService` paths only.

### Console Client Layer

- `backend/wing_client.py` is the primary concrete Wing transport implementation.
- `backend/osc/enhanced_osc_client.py` wraps `WingClient` with reconnect/state behavior and acts as a drop-in transport surface.
- `backend/server.py` can also mention other client types, but current canonical live-console focus is the Wing path.

### Automation Paths

- `backend/auto_fader.py` contains an active auto-fader path with direct fader writes and only local freeze protections.
- `backend/arrangement_automation/controller.py` contains arrangement-aware fader automation with controller-local safety logic, but it is still a direct bypass path rather than a `LiveApplyService`-gated path.
- `backend/auto_soundcheck_engine.py` can analyze and apply multiple correction types, but it remains overlapping rather than canonical.
- `backend/auto_fader_v2/` exists as a secondary auto-fader surface and shared bleed-analysis dependency, but current ownership and deployment status are unclear.

### Feedback Path

- `backend/feedback_detector.py` is the analysis component for rapid feedback detection.
- `backend/auto_soundcheck_engine.py::_handle_feedback_event` is the current reaction path for EQ notch or fader-reduction behavior.
- For documentation, this path is a safety-critical emergency exception candidate only, not an approved exception.

### Voice-Command Path

- `backend/handlers/voice_handlers.py` starts and stops voice-control runtime behavior.
- `backend/server.py` routes voice `set_fader` / `set_gain` through `LiveApplyService`.
- The repo also contains multiple voice-control variants:
  - `backend/voice_control.py`
  - `backend/voice_control_v2.py`
  - `backend/voice_control_vosk.py`
  - `backend/voice_control_sherpa.py`
- This means the voice surface is active but internally duplicated.

### Utility Script Surface

- Snapshot, restore, and reset scripts remain outside the canonical live runtime.
- They can still reach real console writes through direct client usage.
- For documentation, they are operator-only legacy tools and archive candidates.

## Offline Runtime Map

### `automixer/production_mix_v1`

- `automixer/production_mix_v1/__main__.py` is the current offline-first CLI entry.
- `automixer/production_mix_v1/pipeline.py` is the main pipeline for file loading, analysis, candidate generation, rendering, evaluation, safety checks, and reporting.
- `automixer/production_mix_v1/osc_queue.py` can hold OSC commands and can send them only when a real sender is attached and dry-run is disabled.
- This path is not an approved live console path without separate future policy approval and a unified console-write gate.

### `ai_mixing_pipeline`

- `ai_mixing_pipeline/__init__.py` declares the package as offline-only and intentionally separate from live OSC/MIDI control.
- The package is aimed at local candidate rendering, evaluation, and reporting.
- Current repo evidence does not support treating it as a live console path.
- Any future attachment of a real sender is not approved without a unified console-write gate and separate policy review.

### Offline Boundary

- Offline pipelines may produce candidate actions, reports, or queued OSC commands.
- They are not canonical live-runtime owners.
- They are not approved for live console use simply because sender plumbing exists.

## Console-Write Boundaries

### Already Gated

- Manual UI `set_fader` / `set_gain`
- Voice `set_fader` / `set_gain`
- Canonical live input trim via `LiveInputTrimController`

### Blocked At Gate

- Manual UI `set_eq`
- Manual UI `set_compressor`

### Direct Bypass

- `backend/auto_fader.py`
- `backend/arrangement_automation/controller.py`
- parts of `backend/auto_soundcheck_engine.py`
- snapshot/load/save utility and handler surfaces that call client methods directly

### Emergency Exception Candidate

- `backend/feedback_detector.py` + `backend/auto_soundcheck_engine.py::_handle_feedback_event`

### Operator-Only Legacy

- restore/snapshot/reset scripts
- direct snapshot handler surface in `backend/handlers/snapshot_handlers.py`

## Stale / Duplicate / Legacy Surface

### Stale Docs

- `README.md` still presents a much smaller project shape centered on a simple backend/frontend split.
- `Docs/ARCHITECTURE.md` describes an older multi-layer architecture that does not match the current canonical runtime or current ownership boundaries.

### Duplicate Auto-Fader Surfaces

- `backend/auto_fader.py`
- `backend/auto_fader_v2/`
- `backend/auto_fader_hybrid.py`

Current ownership and intended runtime status are not unified across these paths.

### Voice-Control Variants

- `backend/voice_control.py`
- `backend/voice_control_v2.py`
- `backend/voice_control_vosk.py`
- `backend/voice_control_sherpa.py`

These variants indicate an active but duplicated voice-control surface.

### Snapshot / Load / Reset Variants

- `backend/load_snap.py`
- `backend/load_snap_v2.py`
- `backend/load_snap_final.py`
- `backend/find_and_load_snap.py`
- `backend/scan_and_load_snap.py`
- `backend/reset_all_channels.py`
- `backend/reset_modules_trim_faders.py`
- `backend/reset_mixer.sh`

These tools overlap in operator workflows and direct console access.

### Duplicate ML / DSP Surfaces

- `backend/ai/`
- `backend/ml/`
- `automixer/production_mix_v1/`
- `ai_mixing_pipeline/`
- `mix_agent/`

The repository contains multiple analysis, decision, and automation surfaces that do not currently resolve into one documented canonical architecture.

### `mix_agent` Status

- `mix_agent/` has directory structure but no visible source files in the current workspace snapshot, only cached bytecode.
- `ai_mixing_pipeline` still imports `mix_agent.*` symbols in several places.
- For current documentation, `mix_agent` should be treated as unclear/incomplete.

## Open Decisions

- Whether the repo will continue to document `AutoSoundcheck` live apply and feedback emergency reaction as separate policy tracks instead of one shared live-apply scope.
- Whether the feedback reaction path should later become an approved emergency exception or be migrated to a different gated model.
- Whether operator-only restore/snapshot/reset tools should remain as-is for operators or later move into archival/removal planning.
- Whether any offline pipeline should ever be allowed to attach to a real sender outside a unified console-write gate.
- Whether `backend/auto_fader_v2/` should be documented later as active, experimental, or legacy after a dedicated ownership audit.
- Whether `mix_agent/` should be restored as source-backed functionality, archived, or removed from downstream imports.
