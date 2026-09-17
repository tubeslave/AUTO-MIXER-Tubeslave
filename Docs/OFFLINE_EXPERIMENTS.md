# Offline Experiments v2

## Purpose

The offline harness compares decision strategies without touching a live mixer.

It is metric/action-plan based, so it can consume outputs from the current offline renderer, rules-only reports, or simple test fixtures.

## Compared Pipelines

- `rules_only`
- `rules_critic`
- `rules_critic_decision_engine`
- `decision_engine_dry_run`

Each experiment saves:

- input metrics
- action plans
- Safety Gate result
- applied corrections
- final scores if supplied
- diff between variants
- JSON report
- Markdown report

## Input Metrics

Minimal example:

```json
{
  "analyzer_output": {
    "channels": [
      {
        "channel_id": 1,
        "role": "lead_vocal",
        "metrics": {
          "lufs": -24.0,
          "target_lufs": -20.0,
          "true_peak_dbtp": -10.0
        },
        "confidence": 0.9
      }
    ]
  },
  "current_state": {
    "channel:1": {
      "fader_db": -12.0,
      "true_peak_dbtp": -10.0
    }
  }
}
```

## Run

Direct harness:

```bash
python -m automixer.experiments.runner \
  --input-metrics ./metrics.json \
  --output-dir ./decision_engine_v2_experiment
```

Through the soundcheck CLI:

```bash
python start_soundcheck.py \
  --offline-experiment \
  --experiment-input ./metrics.json \
  --experiment-output-dir ./decision_engine_v2_experiment
```

Reports:

```text
decision_engine_v2_experiment/experiment_report.json
decision_engine_v2_experiment/experiment_report.md
```

## Relation To automix-toolkit And FxNorm-Automix

`automix-toolkit` is useful for offline ML evaluation and training experiments.

`FxNorm-Automix` is useful as an offline baseline for wet or processed multitracks.

Neither is required in the live path.
