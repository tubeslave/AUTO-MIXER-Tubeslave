# Decision engine contract

The decision engine converts observations into typed action proposals. It must not directly control a mixer.

## Inputs

The decision engine may read:

- channel audio features;
- instrument classification;
- mixer state;
- session state;
- user profile/config;
- safety mode;
- critic feedback;
- previous action history.

## Outputs

The decision engine outputs typed action proposals.

```text
Observation -> Decision -> TypedActionProposal
```

Example proposal shape:

```json
{
  "action_type": "ChannelEQMove",
  "source": "auto_eq",
  "target": {"channel_id": 4, "instrument": "lead_vocal"},
  "parameters": {
    "band": 2,
    "frequency_hz": 2800,
    "gain_db_delta": -1.0,
    "q": 1.2
  },
  "reason": "harshness detected in 2-5 kHz band",
  "confidence": 0.72,
  "risk": "medium"
}
```

## Decision layers

Recommended internal layers:

1. `ObservationNormalizer` — converts raw analyzer/controller data into a stable observation model.
2. `RuleEngine` — deterministic rules and heuristics.
3. `PolicyEngine` — chooses between competing actions.
4. `ConflictResolver` — prevents contradictory actions.
5. `ActionEmitter` — returns typed actions to safety controller.

## Conflict examples

The engine should avoid:

- boosting and cutting the same band from different modules at the same time;
- raising vocal fader while a feedback guard warns;
- applying EQ based on bleed-dominated channel analysis;
- applying MuQ-driven global changes that conflict with live mode limits;
- applying repeated small moves that accumulate into large unsafe changes.

## Relation to critics

Critics may influence decisions by providing scores or warnings. They should not own the final actuator path.

```text
critic score -> decision context -> typed action proposal -> safety controller
```

## Relation to mixer clients

The decision engine should not know OSC address details.

Bad:

```python
send_message('/ch/01/eq/1/g', -1.0)
```

Good:

```python
ChannelEQMove(channel_id=1, band=1, gain_db_delta=-1.0)
```

The mixer mapper converts typed actions to mixer-specific commands after safety approval.

## Testing expectations

Decision tests should validate:

- same input produces same action;
- actions stay within configured max deltas;
- no action is emitted when confidence is too low;
- bleed/phase/headroom guards can suppress risky decisions;
- critics cannot bypass the action model;
- offline policies do not enter live pipeline directly.
