# Phase-Aware EQ Guard

`backend/phase_aware_eq_guard.py` is an original AutoMixer plug-in module for
group phase analysis before EQ decisions. It is inspired by the observable
control surface of Sound Radix Pi, but it does not copy proprietary code,
models, PACE logic, or hidden implementation.

## What It Does

The module answers one question before AutoEQ makes a move:

> Is this really an EQ problem, or is the source group cancelling because of
> phase, polarity, delay, or bleed?

If phase is the likely cause, the module emits a dry-run delay/polarity
proposal and tells AutoEQ to hold aggressive boosts.

## Covered Function Surface

| Function | AutoMixer implementation |
| --- | --- |
| Group | `PhaseAwareEQGuardConfig.group_id` and `PhaseChannelInput.group_id` |
| Group Enable Mode | `GROUP_FULL`, `GROUP_INTERNAL_ONLY`, `GROUP_PHASE_LOCK`, `GROUP_OFF` |
| Group Gain | summed group RMS report in `group_gain_db` |
| Gain | per-channel `gain_db` used for analysis weighting |
| Weight | per-channel `weight` used to choose and protect the anchor |
| Global Low Freq Focus | `low_freq_focus` with configurable low-frequency range |
| Global Enabled | `global_enabled` fail-closed switch |
| Lookahead Buffer | `lookahead_buffer_ms` metadata |
| Report Latency | `report_latency` and `logic_latency_ms` |
| Group Graph / Frequency Graph | `is_group_graph` and `is_freq_graph` reporting flags |

## Output Policies

- `phase_ok`: phase is stable enough for normal EQ proposal flow.
- `phase_risk`: EQ should be conservative.
- `phase_candidate`: review delay/polarity/low-frequency phase before EQ.
- `phase_unreliable`: collect cleaner or longer soundcheck evidence.
- `group_disabled`: no group phase action.

## Safety Boundary

The module is transport-free:

- no WING client import
- no OSC writes
- no `set_delay`
- no `set_polarity`
- no `set_eq`
- no fader/gain writes

Live mode remains proposal-only unless a separate supervised hardware proof is
completed. The processing registry marks this method as proposal-only and keeps
the only live executor as supervised fader/gain.

## Example

```python
from phase_aware_eq_guard import (
    GroupEnableMode,
    PhaseAwareEQGuardConfig,
    PhaseAwareEQGuardPlugin,
    PhaseChannelInput,
)

plugin = PhaseAwareEQGuardPlugin(
    PhaseAwareEQGuardConfig(
        group_id="DRUMS",
        group_enable_mode=GroupEnableMode.GROUP_PHASE_LOCK,
        low_freq_focus=True,
        lookahead_buffer_ms=5.0,
    )
)

result = plugin.process(
    [
        PhaseChannelInput(1, kick_in_audio, group_id="DRUMS", weight=2.0),
        PhaseChannelInput(2, kick_out_audio, group_id="DRUMS", weight=0.8),
    ]
)

if result.eq_action_policy == "phase_candidate":
    # Show result.suggested_actions to the operator before any EQ boost.
    ...
```
