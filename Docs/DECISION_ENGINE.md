# Decision Engine v2

## Purpose

The Decision Engine turns analyzer evidence, critic feedback, and knowledge rules into an `ActionPlan`.

It does not send OSC and does not touch the mixer.

## Input

The engine accepts flexible analyzer payloads. The preferred shape is:

```json
{
  "source_module": "auto_soundcheck_engine",
  "channels": [
    {
      "channel_id": 1,
      "name": "Lead",
      "role": "lead_vocal",
      "metrics": {
        "lufs": -24.0,
        "target_lufs": -20.0,
        "true_peak_dbtp": -8.0,
        "SibilanceIndex": 3.0
      },
      "confidence": 0.9
    }
  ]
}
```

Critic payloads can be global or per-channel:

```json
{
  "channels": [
    {
      "channel_id": 1,
      "confidence": 0.85,
      "risk_level": "low"
    }
  ]
}
```

## Output

The output is an `ActionPlan` with one or more `ActionDecision` objects.

Supported v2 action types:

- `gain_correction`
- `eq_correction`
- `compression_correction`
- `pan_correction`
- `no_action`

Every decision includes:

- `reason`: why it was proposed
- `confidence`: 0..1
- `risk_level`: `low`, `medium`, `high`, or `critical`
- `source_modules`: analyzer/critic/knowledge sources
- `expected_audio_effect`: plain-language expected result
- `safe_to_apply`: Decision Engine pre-safety flag

## Important Boundary

`safe_to_apply` is not permission to send OSC. It only says the Decision Engine did not see an obvious reason to block the action.

The actual permission boundary is Safety Gate.

## Python Example

```python
from automixer.decision import DecisionEngine

engine = DecisionEngine()
plan = engine.create_action_plan(analyzer_output, critic_evaluations={})
```

Then pass the plan to Safety Gate and Executor.
