# Safety Gate v2

## Purpose

Safety Gate sits between Decision Engine and Executor.

It blocks or allows v2 `ActionPlan` decisions before any mixer call can happen.

## What It Checks

- maximum gain change per step
- maximum live gain increase
- minimum gain hysteresis
- EQ boost/cut limits
- EQ frequency and Q limits
- compression ratio and threshold-change limits
- pan range and live pan step
- true-peak headroom before gain boosts
- rate limit
- cooldown
- dry-run mode
- emergency bypass flag

## Config

Limits live in `config/automixer.yaml`:

```yaml
decision_engine_v2:
  enabled: false
  dry_run: true
  safety:
    max_gain_change_db: 1.0
    max_live_gain_increase_db: 0.5
    max_eq_boost_db: 1.0
    max_eq_cut_db: 3.0
    min_interval_sec: 3.0
    cooldown_sec: 2.0
    hysteresis_db: 0.25
```

## Dry-Run

When dry-run is active, Safety Gate can allow recommendations but marks them as not sendable:

```json
{
  "safety_gate": {
    "allowed": true,
    "dry_run": true,
    "send_to_executor": false
  }
}
```

Executor then records the recommendation and sends nothing.

## Python Example

```python
from automixer.safety import SafetyGate

gate = SafetyGate()
result = gate.evaluate_plan(plan, current_state=current_state, live_mode=True)
```

Use `result.blocked` to see why actions were blocked.
