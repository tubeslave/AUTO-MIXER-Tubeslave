# Task training-experiments-pipeline

## Context

`TrainerAgent` already wraps local training commands and a dataset-discovery task. The new
need is to run a small suite of configured experiments, persist their outputs, select the
best successful run by a metric, and hand that run to `EvaluatorAgent`.

## Options considered

1. Replace the current trainer command runner with the YAML experiment runner.
2. Add a separate experiment-runner class and make `TrainerAgent` delegate to it.
3. Add an opt-in `experiments_config` mode to `TrainerAgent`.

## Decision

Use option 3. `TrainerAgent.run()` keeps the current single-command behavior unless callers
provide `experiments_config`.

## Why this won

This is the smallest compatible change. Existing agent operations can continue to use
`command`, `train_config`, and `discover_datasets`, while the new training pipeline gets a
clear config-driven entrypoint and persisted run artifacts.

## Rejected alternatives

Replacing the old runner was rejected because it would change response shapes for existing
coordinator calls. A separate class was rejected for now because the workflow is still small
and would add indirection before there is a stable experiment schema.

## Implementation plan

- Add `configs/training/experiments.yaml`.
- Extend `TrainerAgent` with YAML loading, experiment execution, metric parsing, best-run
  selection, and `training_summary.json`.
- Extend `EvaluatorAgent` to accept `best_run` and write `evaluation.json`.
- Add `scripts/run_training_pipeline.py`.
- Teach the sample `scripts/train.py` to accept `--epochs`.
- Add focused tests for trainer selection and evaluator persistence.

## Test plan

- Run `PYTHONPATH=backend python -m pytest tests/test_training_pipeline_agents.py tests/test_trainer_agent_dataset_discovery.py -q`.
- Run `python scripts/run_training_pipeline.py`.
- Run `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## Risks and rollback

The metric parser only supports simple `key=value` tokens; rollback is to remove the new
YAML mode and runner script. Since outputs are confined to `artifacts/runs/`, generated
artifacts can be deleted without touching live mixer state or production configuration.
