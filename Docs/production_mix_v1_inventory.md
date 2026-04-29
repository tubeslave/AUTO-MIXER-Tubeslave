# production_mix_v1 Inventory

## Scope

This audit maps existing AUTO-MIXER-Tubeslave modules that can be reused by a
new opt-in `production_mix_v1` pipeline. No existing default pipeline should be
changed.

## Found Modules

### Offline render and render engine

- `tools/offline_agent_mix.py`: current large no-reference offline renderer. It
  includes `ChannelPlan`, virtual console adapters, channel rendering, render
  cache, source/stem/bus balancing, AutoFOH analyzer passes, MuQ stem drift
  logging, and report helpers.
- `tools/online_soundcheck_mix.py`: soundcheck-style offline simulation.
- `tools/channel_triggered_soundcheck_mix.py`: channel-triggered soundcheck
  simulation.
- `automixer/experiments/runner.py`: metric/action-plan offline experiment
  harness for Decision Engine v2.

### Candidate generation

- `tools/offline_agent_mix.py`: staged candidate-like logic exists inside a
  monolithic script: source rules, spectral windows, layer balancing, phase
  candidates, and post-render groups.
- `backend/heuristics/spectral_ceiling_eq.py`: safe role-aware EQ proposal
  generator for the default no-reference spectral guard.
- `backend/autofoh_detectors.py`: lead masking, low-end, mud, harshness, and
  sibilance action recommenders.
- `automixer/decision/decision_engine.py`: converts analyzer/critic/knowledge
  evidence into explainable `ActionPlan` decisions.

### Critic and scoring

- `backend/evaluation/muq_eval_service.py`: MuQ-Eval A/B validation with
  fallback heuristic and stem EWMA drift support.
- `backend/perceptual/`: optional perceptual/MERT shadow evaluator with a
  lightweight fallback backend.
- `automixer/critics/adapter.py`: normalizes critic payloads for Decision
  Engine v2.
- `backend/autofoh_evaluation.py`: current AutoFOH evaluation/rollback helpers.
- `backend/source_knowledge/`: source-grounded rules and decision logging.

### MuQ-Eval integration

- `backend/evaluation/muq_eval_service.py`
- `config/muq_eval.yaml`
- `config/automixer.yaml` `muq_eval` section
- Tests: `tests/test_muq_eval_service.py`, `tests/test_muq_eval_director.py`

### MERT integration

- `backend/perceptual/embedding_backend.py`: `MERTEmbeddingBackend` with
  optional HuggingFace `torch`/`transformers`.
- `backend/perceptual/perceptual_evaluator.py`
- `config/perceptual.yaml` and `config/automixer.yaml` `perceptual` section.
- Tests: `tests/test_perceptual_evaluation.py`

### Essentia integration

- `backend/auto_eq.py` has comments and paths around Essentia/scipy spectral
  analysis, but there is no small standalone Essentia analyzer facade.
- No required Essentia dependency should be added for `production_mix_v1`; use an
  optional adapter with warning and deterministic numpy fallback.

### Safety Governor / safety gate

- `automixer/safety/safety_gate.py`: v2 `SafetyGate` for `ActionPlan` objects.
- `backend/autofoh_safety.py`: current typed correction action safety controller
  used by live/soundcheck flows.
- `backend/feedback_detector.py`: emergency feedback guard with priority over
  regular automation.
- `Docs/SAFETY_GATE.md` and `Docs/ARCHITECTURE_V2_AUDIT.md`: existing safety
  architecture decisions.

### OSC sender / OSC client

- `backend/wing_client.py`: Behringer WING OSC client.
- `backend/osc_manager.py` and `backend/osc/enhanced_osc_client.py`: OSC
  management/throttling.
- `backend/mixer_client_base.py`: shared mixer client interface.
- `Docs/TOOLS.md`: canonical WING OSC address/rate-limit reference.

### Config loading

- `automixer/config.py`: v2 config loader for `config/automixer.yaml`.
- `backend/config_manager.py`: runtime YAML loader/hot reload for backend
  services.
- `config/automixer.yaml`: current production runtime config.
- New pipeline should use a separate `config/pipelines/production_mix_v1.yaml`
  and not mutate `config/automixer.yaml`.

### Report/log generation

- `automixer/logs/human_logger.py`: JSONL human-readable decision logs.
- `backend/autofoh_logging.py`: AutoFOH async logs/session summary.
- `backend/output_paths.py`: common AI log path helper.
- `tools/offline_agent_mix.py`: JSON/Markdown offline render reports.

### Channel analysis and stem rendering

- `backend/signal_metrics.py`: LUFS, true peak, dynamics, spectral metrics.
- `backend/autofoh_analysis.py`: named frequency bands, stem contribution
  matrix, analysis features.
- `backend/autofoh_models.py`: source/stem roles, bands, analysis dataclasses.
- `backend/mix_bus_balance.py`: bus/stem balance helpers.
- `tools/offline_agent_mix.py`: stem/channel render functions and summed layer
  group logic.

### Existing CLI / entrypoints

- `backend/server.py`: live backend server, must not receive new routes unless
  explicitly needed.
- `backend/auto_soundcheck_engine.py`: soundcheck/live entry with v2 flags.
- `tools/offline_agent_mix.py`: current offline mix CLI.
- `automixer/experiments/runner.py`: offline Decision Engine v2 experiments.

### Tests

- Existing safety/decision tests:
  - `tests/test_decision_engine_v2.py`
  - `tests/test_decision_engine_v2_config.py`
  - `tests/test_safety_gate_v2.py`
  - `tests/test_autofoh_safety.py`
  - `tests/test_osc_manager.py`
- Existing offline/evaluation tests:
  - `tests/test_offline_experiment_v2.py`
  - `tests/test_muq_eval_service.py`
  - `tests/test_perceptual_evaluation.py`
  - `tests/test_spectral_ceiling_eq.py`
  - `tests/test_mix_bus_balance.py`

## Reusable Functions And Patterns

- Use `automixer.decision.models.ActionPlan`/`ActionDecision` concepts for
  explainable decisions, but keep `production_mix_v1` final candidate selection
  separate from v2 live default behavior.
- Reuse `automixer.safety.SafetyGate` semantics through a production safety
  adapter for final action validation.
- Reuse MuQ and perceptual/MERT services as optional critics. If model loading
  fails, record a warning and mark the score as fallback/proxy.
- Reuse AutoFOH frequency bands and named problem concepts, but keep numpy
  fallback analyzers in the new package to avoid hard dependencies.
- Reuse existing WING OSC address conventions from `Docs/TOOLS.md`, but route
  all output through a new `OSCCommandQueue`.

## Missing Modules

- A compact, independent `production_mix_v1` package.
- A staged candidate generator with an explicit baseline candidate.
- A final candidate Decision Engine that compares candidate scores with
  threshold/rationale and proxy-only rejection.
- A production Safety Governor adapter that runs after the Decision Engine and
  before any action queueing/apply step.
- An OSC command queue with dry-run default.
- A small standalone CLI for offline/dry-run smoke execution.
- Dedicated tests for this pipeline.

## Dependencies That May Be Missing

- `yaml`/PyYAML: needed for config; fallback should use defaults if missing.
- `soundfile`: useful for WAV/AIFF I/O; new smoke path should also work without
  it by using generated audio or stdlib `wave`.
- `torch`, `transformers`, `huggingface_hub`: needed for real MERT/MuQ model
  loading; fall back to lightweight/proxy scoring with warnings.
- `essentia`: optional only; use numpy technical analysis when unavailable.
- `scipy`: present in many current DSP paths but not required for the new
  smoke path.

## Breakage Risks

- Importing heavy model services in live mode could add startup latency or fail
  when model caches are absent.
- Accidentally wiring the new Decision Engine directly to OSC would violate the
  safety boundary.
- Treating MuQ fallback or MERT lightweight scores as authoritative could let a
  proxy choose a winner.
- Applying offline fader/EQ limits to live mode would be too permissive unless
  mode-specific limits are enforced.
- Reusing `tools/offline_agent_mix.py` directly would couple a new production
  pipeline to a large monolithic script and make rollback harder.

## Implementation Plan

1. Add `config/pipelines/production_mix_v1.yaml` without touching the default
   `config/automixer.yaml`.
2. Add `automixer/production_mix_v1/` with isolated modules:
   config, models, audio I/O, analyzers, renderer, candidate generator, critics,
   decision engine, safety adapter, OSC command queue, reporting, and CLI.
3. Keep all real OSC disabled by default. Live mode stays dry-run unless an
   explicit injected sender is used and dry-run is false.
4. Make every optional external evaluator warn and fall back instead of
   crashing.
5. Add focused pytest coverage and run an offline dry-run smoke command.

## Rollback Plan

- Delete `automixer/production_mix_v1/`.
- Delete `config/pipelines/production_mix_v1.yaml`.
- Delete new `tests/test_production_mix_v1.py` and related docs/ADR/proposal
  files.
- No legacy pipeline, default config, server route, or mixer client is modified,
  so rollback does not require mixer-state migration.
