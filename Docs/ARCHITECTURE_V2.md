# Architecture v2

## The New Flow

Architecture v2 is a role layer:

```text
Analyzer -> Knowledge/Rules -> Critic -> Decision Engine -> Safety Gate -> Executor -> Logger/Test Harness
```

It is opt-in. The old live pipeline still works by default.

## Where Each Role Lives

- Analyzer:
  - existing: `backend/audio_capture.py`, `backend/signal_metrics.py`, `backend/autofoh_analysis.py`
  - v2 adapter: `automixer/analyzer/snapshot.py`
- Knowledge/Rules:
  - existing: `mix_agent/rules/`, `backend/ai/rule_engine.py`
  - v2 reference: `automixer/knowledge/mixing_knowledge_base.json`
- Critic:
  - existing: `backend/autofoh_evaluation.py`, `backend/perceptual/`, `backend/evaluation/muq_eval_service.py`
  - v2 adapter: `automixer/critics/adapter.py`
- Decision Engine:
  - v2: `automixer/decision/decision_engine.py`
- Safety Gate:
  - v2: `automixer/safety/safety_gate.py`
- Executor:
  - v2: `automixer/executor/plan_executor.py`
  - mixer clients remain `backend/wing_client.py` and `backend/dlive_client.py`
- Logs/Test Harness:
  - v2 logs: `automixer/logs/human_logger.py`
  - v2 experiments: `automixer/experiments/runner.py`

## What Changed

The Decision Engine now creates an `ActionPlan`. Each decision includes:

- `reason`
- `confidence`
- `risk_level`
- `source_modules`
- `expected_audio_effect`
- `safe_to_apply`

The Decision Engine does not send OSC.

## Live Safety

Any v2 live write must pass:

1. `DecisionEngine.create_action_plan(...)`
2. `SafetyGate.evaluate_plan(...)`
3. `ActionPlanExecutor.execute(...)`

The config defaults are conservative:

- `decision_engine_v2.enabled: false`
- `decision_engine_v2.dry_run: true`

## Commands

Live dry-run:

```bash
python start_soundcheck.py --use-decision-engine-v2 --dry-run
```

Offline experiment:

```bash
python -m automixer.experiments.runner \
  --input-metrics ./metrics.json \
  --output-dir ./decision_engine_v2_experiment
```

Or through the soundcheck entry point:

```bash
python start_soundcheck.py \
  --offline-experiment \
  --experiment-input ./metrics.json \
  --experiment-output-dir ./decision_engine_v2_experiment
```

## Why This Shape

AutomaticMixingPapers informs the knowledge categories: level, EQ/spectral shaping, compression/dynamics, panning, spatial choices, knowledge-based mixing, and ML-based mixing.

`automix-toolkit` remains an offline ML evaluation/training reference.

`FxNorm-Automix` remains an offline baseline for wet/processed multitracks.

None of those external projects is made a required dependency for live mode.
