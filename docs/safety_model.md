# Safety model

The safety model is the central contract of the automixer. It protects the physical console, the PA, the musicians, and the audience from unsafe automated corrections.

## Main rule

No module may directly apply live corrections unless the action has passed through the safety controller.

```text
proposal -> typed action -> safety decision -> mixer command
```

## Safety controller responsibilities

The safety controller should validate:

- channel exists and is controllable;
- action type is allowed in the current mode;
- delta is within limits;
- absolute value is within limits;
- rate of change is safe;
- accumulated movement is safe;
- headroom is preserved;
- phase/mono guard is respected;
- feedback detector is not warning;
- live/emergency state allows write access;
- rollback snapshot exists when needed.

## Modes

Recommended modes:

| mode | write access | purpose |
|---|---:|---|
| `observe` | no | show what would be changed |
| `soundcheck_safe` | limited | safe soundcheck corrections |
| `live_assist` | very limited | small bounded live corrections |
| `live_locked` | no | analysis only during show |
| `emergency_stop` | no | stop all automation |
| `offline` | no physical mixer | render/test only |

## Action classes

Suggested typed action classes:

```text
ChannelGainMove
ChannelFaderMove
ChannelEQMove
ChannelCompressorMove
ChannelGateMove
MuteAction
BusSendMove
SceneSnapshot
RollbackAction
NoOpAction
```

Every action should carry:

- action id
- timestamp
- source module
- target channel/bus
- old value if available
- proposed new value or delta
- reason
- confidence
- risk level
- mode

## Rejection reasons

Standard rejection reasons should be explicit and machine-readable:

```text
limit_exceeded
rate_limit_exceeded
unsafe_mode
missing_snapshot
phase_guard
feedback_guard
headroom_guard
unknown_channel
unsupported_mixer_capability
offline_module_live_write_attempt
critic_direct_osc_attempt
```

## Logging

Every approved, rejected, simulated, or rolled-back action should be logged.

Minimum log fields:

```json
{
  "timestamp": 0,
  "session_id": "",
  "mode": "soundcheck_safe",
  "source": "auto_eq",
  "action_type": "ChannelEQMove",
  "target": {"channel": 1},
  "proposal": {},
  "decision": "approved",
  "reason": "",
  "safety_checks": {},
  "osc_commands": [],
  "rollback_snapshot": ""
}
```

## Rollback

Risky action classes should have rollback data before execution:

- EQ changes
- dynamics changes
- fader moves
- gain moves
- routing changes
- bus send changes

Rollback should restore known previous mixer state, not guess a reverse delta.

## Boundary enforcement test

A test should scan offline/critic/experiment paths for forbidden direct OSC usage.

Forbidden direct usage outside mixer transport modules:

```text
pythonosc.udp_client.SimpleUDPClient
SimpleUDPClient
send_message(
send_osc(
osc_client.send
WingClient(
EnhancedOSCClient(
```

Allowed locations:

```text
backend/mixers/
backend/osc/
legacy compatibility wrappers during migration
```

Any exception must be documented with a reason and a migration plan.
