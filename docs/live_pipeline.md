# Live pipeline

This document defines the live soundcheck/concert control path. It is a safety contract for future refactors.

## Live pipeline contract

The live pipeline may analyze audio and mixer state continuously, but it must only apply mixer changes through the safety gate.

```text
Audio capture / mixer state
  -> analysis modules
  -> decision modules
  -> typed action
  -> safety controller
  -> mixer client
  -> OSC/MIDI/vendor transport
  -> event log + rollback state
```

## Inputs

Live modules may read:

- raw channel audio from Dante/SoundGrid/sounddevice/file simulation
- channel names and expected instrument labels from mixer state
- preamp/trim/fader/EQ/dynamics/routing state where available
- master bus observations
- safety config and live mode state

## Outputs

Live modules must output typed actions before reaching the mixer layer.

Examples:

```text
ChannelGainMove(channel_id, delta_db, reason, confidence)
ChannelEQMove(channel_id, band, freq, gain_db, q, reason, confidence)
ChannelCompressorMove(channel_id, threshold_db, ratio, attack_ms, release_ms, reason)
ChannelFaderMove(channel_id, delta_db, reason, confidence)
RollbackAction(snapshot_id, reason)
```

Raw OSC addresses should be generated only by mixer-specific mappers after safety approval.

## Safety approval requirements

Before any live command is sent:

1. Action must be typed.
2. Action must include channel/instrument context where possible.
3. Action must include reason and source module.
4. Action must pass numerical limits.
5. Action must pass mode limits: soundcheck/live/concert/emergency.
6. Action must pass phase/headroom/feedback guards where relevant.
7. Action must be logged.
8. Rollback snapshot must exist for risky action classes.

## Live modules

The following module categories are live-capable:

- mixer clients
- WebSocket handlers
- audio capture
- real-time analyzers
- AutoSoundcheckEngine
- auto gain/fader/EQ/compressor controllers
- feedback detector
- safety controller
- backup/restore/rollback

## Non-live modules

The following must not apply live corrections directly:

- MuQ-Eval critic
- offline director
- candidate renderer
- benchmark scripts
- training/study/discovery service
- old experiments
- archived prototypes

## Direct OSC bypass rule

The only accepted path to OSC is:

```text
approved typed action -> mixer client -> OSC transport
```

Forbidden examples:

```python
# Forbidden in critics/experiments/rendering:
client.send_message('/ch/01/mix/fader', value)
osc_client.send_osc(...)
SimpleUDPClient(...).send_message(...)
```

If such code exists in offline or critic modules, it must be documented as a safety violation and wrapped behind the safety controller before live use.

## Observation/dry-run mode

Observation mode may simulate changes and report proposed operations. It must not alter the physical console.

Dry-run output should include:

- action type
- target channel
- proposed value/delta
- safety decision
- reason
- estimated risk

## Emergency stop

Live mode must preserve a hard stop path that:

1. stops all controllers;
2. cancels running soundcheck tasks;
3. stops audio capture;
4. disconnects or disables mixer write access;
5. keeps logs for post-mortem analysis.
