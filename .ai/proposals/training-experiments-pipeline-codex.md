1. Short thesis

Extend the existing lightweight `src/agent_ops` training wrapper with an opt-in YAML
experiment mode, preserving current command and dataset-discovery behavior.

2. What I understood

The requested pipeline is offline orchestration only: load `configs/training/experiments.yaml`,
run each command, parse simple `key=value` metrics, choose the best successful run by
`selection_metric` and `selection_mode`, then let `EvaluatorAgent` evaluate that run.

3. Proposed solution

Add `runs_dir` to `TrainerAgent`, branch `run()` when `experiments_config` is supplied,
write one directory per experiment under `artifacts/runs/`, and write an aggregate
`training_summary.json`. Add an evaluator path for `best_run` while keeping the old
`model_dir` placeholder path for existing coordinator usage.

4. Likely files to touch

- `src/agent_ops/trainer_agent.py`
- `src/agent_ops/evaluator_agent.py`
- `configs/training/experiments.yaml`
- `scripts/run_training_pipeline.py`
- `scripts/train.py`
- `tests/test_training_pipeline_agents.py`
- `.ai/briefs/training-experiments-pipeline.md`
- `Docs/adr/training-experiments-pipeline.md`

5. Alternatives considered

One option was to replace the current trainer response shape with the new summary shape.
That is riskier because `scripts/run_agents.py` and existing tests rely on the current
single-run command wrapper.

6. Risks

The metric parser is intentionally simple and only reads `key=value` tokens, so richer
training logs will need a structured metrics file later. Commands come from local YAML and
are run with `shell=False`, which keeps command execution explicit.

7. Test plan

Add focused pytest coverage for YAML experiment selection and evaluator output, run those
tests, then run the standard repository test command.

8. Where the other agent may disagree

Kimi may push for a stronger schema validator or a separate experiment runner class. That
is reasonable later, but the smallest safe diff is to keep this inside `TrainerAgent`
until the training workflow grows beyond simple local commands.
