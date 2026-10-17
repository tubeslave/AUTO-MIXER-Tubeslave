# Architecture v2 Audit

## Scope

This audit maps the current project to the requested v2 architecture:

Analyzer -> Knowledge/Rules -> Critic -> Decision Engine -> Safety Gate -> Executor -> Logger/Test Harness.

The repository already has most pieces, but they are spread across backend live control, offline render scripts, mix-agent, AI/RAG, ML experiments, and external research repos. The safe path is to add v2 as a role layer, not to merge every module into one live pipeline.

## What Is Really Used

### Audio analysis

- `backend/audio_capture.py`: canonical live audio capture service. This must remain the shared live input.
- `backend/signal_metrics.py`: per-channel LUFS, true peak, dynamics, spectral metrics.
- `backend/autofoh_analysis.py`: AutoFOH feature extraction, named frequency bands, stem contribution helpers.
- `backend/autofoh_detectors.py`: lead masking, mud, harshness, sibilance, low-end detectors.
- `backend/channel_recognizer.py`: channel name and spectral fallback classification.
- `mix_agent/analysis/`: offline loudness, spectral, dynamics, stereo, masking, artifact analysis.
- `backend/heuristics/spectral_ceiling_eq.py`: default spectral guard for offline/no-reference work.

### Rules and heuristics

- `mix_agent/rules/`: explainable offline rules for gain staging, balance, tonal, dynamics, masking, stereo, space, artifacts, translation.
- `backend/ai/rule_engine.py`: legacy/live AI fallback rules used by `MixingAgent`.
- `backend/auto_soundcheck_engine.py`: contains current live/soundcheck heuristics and presets.
- `backend/autofoh_runtime.py`: runtime-state permission policy for action families.
- `backend/source_knowledge/`: source-grounded candidate logging and rule storage.

### Critic and evaluation

- `backend/autofoh_evaluation.py`: action evaluation and rollback helpers.
- `backend/perceptual/`: perceptual shadow scoring.
- `backend/evaluation/muq_eval_service.py`: MuQ-Eval quality/reward service.
- `mix_agent/agent/evaluator.py`: offline before/after metric deltas.
- Tests under `tests/` are part of the evaluation memory.

### OSC/control/executor

- `backend/wing_client.py`: Behringer WING OSC client.
- `backend/dlive_client.py`: Allen & Heath dLive MIDI/TCP client.
- `backend/mixer_client_base.py`: common mixer client interface.
- `backend/osc_manager.py` and `backend/osc/enhanced_osc_client.py`: OSC management and batching.
- `backend/autofoh_safety.py`: current typed action safety controller and mixer translator.
- `mix_agent/backend_bridge.py`: translates offline recommendations into existing AutoFOH typed actions.

### Offline rendering and testing

- `tools/offline_agent_mix.py`: main offline no-reference render path with AutoFOH analyzer passes.
- `tools/online_soundcheck_mix.py`: staged soundcheck-style offline render.
- `tools/channel_triggered_soundcheck_mix.py`: channel-triggered soundcheck simulation.
- `mix_agent/__main__.py`: smaller offline analysis/suggest/apply facade.
- `external/automix-toolkit`: useful for offline ML evaluation/training, not live mode.
- `external/FxNorm-automix`: useful offline baseline for processed/wet multitracks, not live mode.
- `external/AutomaticMixingPapers`: local research index for knowledge categories.

### Logging

- `backend/autofoh_logging.py`: non-blocking AutoFOH JSONL logging and session summaries.
- `backend/output_paths.py`: common AI output/log paths.
- `mix_agent/reporting/`: JSON and Markdown reports.
- New v2 logs live in `automixer/logs/human_logger.py`.

## What Is Duplicated

- Safety exists in two forms now:
  - legacy live safety: `backend/autofoh_safety.py`
  - v2 ActionPlan safety: `automixer/safety/safety_gate.py`
  This is intentional for now. The v2 gate protects v2 plans before the executor. It does not replace the legacy safety controller.
- Decision-like behavior exists in:
  - `backend/agents/coordinator.py`
  - `backend/ai/agent.py`
  - `mix_agent/agent/planner.py`
  - `backend/auto_soundcheck_engine.py`
  None of these was removed. v2 introduces an explicit `ActionPlan` contract.
- Knowledge exists in:
  - `backend/ai/knowledge/`
  - `backend/source_knowledge/`
  - `mix_agent/config/`
  - new `automixer/knowledge/mixing_knowledge_base.json`
  The new file is a compact decision reference, not a replacement for RAG/source logs.
- Offline logic is duplicated across `tools/offline_agent_mix.py`, `mix_agent`, online soundcheck simulation, and channel-triggered simulation. This is legacy reality; v2 experiment harness reads metrics/plans instead of forcing a rewrite.

## What Looks Legacy Or Transitional

- `tools/offline_agent_mix.py` is large and does many passes in one file. It remains important and should not be deleted during v2.
- `backend/auto_soundcheck_engine.py` still contains presets, heuristics, analyzer calls, action execution, logging, and live loop orchestration in one module.
- `backend/ai/agent.py` and `backend/ai/rule_engine.py` are useful but should not become a mandatory live decision path.
- External repos under `external/` are research and offline experiment references. They should not be imported into live mode by default.
- Some role names are mixed style, for example `leadVocal`, `backVocal`, `electricGuitar`, `overhead`, `toms`. v2 normalizes roles before knowledge lookup.

## What Cannot Be Removed

- `backend/audio_capture.py`: canonical live audio capture.
- `backend/feedback_detector.py`: absolute safety priority for feedback.
- `backend/wing_client.py`, `backend/dlive_client.py`, `backend/mixer_client_base.py`: real mixer IO.
- `backend/autofoh_safety.py`: existing live safety and typed action translator.
- `backend/auto_soundcheck_engine.py`: current live/soundcheck entry point.
- `tools/offline_agent_mix.py`: current offline render pipeline.
- `mix_agent/`: existing offline analysis, rules, and reporting facade.
- `Docs/WING Remote Protocols v3.0.5.pdf` and protocol docs.
- Tests and ADRs.

## Role Mapping

| v2 role | Existing modules | New v2 modules |
|---|---|---|
| Analyzer | `audio_capture.py`, `signal_metrics.py`, `autofoh_analysis.py`, `autofoh_detectors.py`, `mix_agent/analysis/` | `automixer/analyzer/snapshot.py` |
| Knowledge/Rules | `mix_agent/rules/`, `backend/ai/rule_engine.py`, `backend/source_knowledge/`, `backend/ai/knowledge/` | `automixer/knowledge/` |
| Critic | `autofoh_evaluation.py`, `perceptual/`, `evaluation/muq_eval_service.py`, `mix_agent/agent/evaluator.py` | `automixer/critics/adapter.py` |
| Decision Engine | partial: `mix_agent/agent/planner.py`, `backend/agents/coordinator.py` | `automixer/decision/` |
| Safety Gate | `backend/autofoh_safety.py`, `backend/autofoh_runtime.py` | `automixer/safety/` |
| Executor | `wing_client.py`, `dlive_client.py`, `osc_manager.py`, `mix_agent/backend_bridge.py` | `automixer/executor/` |
| Logger/Test Harness | `autofoh_logging.py`, `mix_agent/reporting/`, tests | `automixer/logs/`, `automixer/experiments/` |

## Integration Conflicts

1. Existing live safety already executes typed actions directly. v2 solves this by producing `ActionPlan` first, then using a separate Safety Gate before any v2 Executor call.
2. Existing AutoFOH can auto-apply corrections in `backend/auto_soundcheck_engine.py`. v2 is only used when `--use-decision-engine-v2` is passed or config explicitly enables it.
3. Config currently has perceptual/MuQ shadow systems enabled. v2 does not import or require their heavy models in the live path.
4. Offline research repos are present locally. v2 references their ideas in docs and harness design but does not add them as live dependencies.

## Safe Integration Chosen

- Keep legacy default behavior unchanged.
- Add `decision_engine_v2.enabled: false` and `dry_run: true` in `config/automixer.yaml`.
- Add CLI flags:
  - `--use-decision-engine-v2`
  - `--dry-run`
  - `--offline-experiment`
- Decision Engine returns only `ActionPlan`.
- Safety Gate evaluates all v2 actions before Executor.
- Executor dry-run performs no mixer writes.
