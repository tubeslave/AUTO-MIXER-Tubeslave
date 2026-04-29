# Task training-experiments-pipeline: YAML training experiments

## Problem

TrainerAgent should run several training experiments from a YAML config, persist per-run logs
and metadata, select the best successful run by a configured metric, and pass that run to
EvaluatorAgent for a simple persisted evaluation.

## Constraints

- Preserve the existing single-command TrainerAgent behavior and dataset-discovery task.
- Do not touch mixer control paths or live-sound safety logic.
- Avoid adding a new production dependency; PyYAML is already present in requirements.
- Keep outputs local to `artifacts/runs/`.

## Definition of Done

- `python scripts/run_training_pipeline.py` runs the configured experiments.
- `artifacts/runs/training_summary.json` is written.
- The selected run contains `train.log`, `metadata.json`, and `evaluation.json`.
- Focused tests cover YAML selection and evaluator output.

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
