# Behringer Wing Rack OSC Reference

Complete OSC (Open Sound Control) protocol reference for the Behringer Wing Rack
digital mixer. Used by AUTO-MIXER for real-time mixer control.

## Connection Protocol

- **Transport**: UDP
- **Port**: 2222 (default, configurable on the mixer)
- **Keepalive Packet**: Send `WING\0` (5 bytes: 0x57 0x49 0x4E 0x47 0x00)
  every 5 seconds to maintain connection. If the mixer does not receive
  a keepalive within 10 seconds, it drops the client.
- **Discovery**: The mixer responds to `WING\0` with its firmware version
  and model information.
- **Max Packet Size**: 1500 bytes (standard UDP MTU)
- **Byte Order**: Big-endian for OSC, mixed-endian for some WING-specific messages
- **Subscription**: Send `/$subscribe` messages to receive parameter change
  notifications. Re-subscribe every 10 seconds.

## Channel Address Format

The Wing Rack uses these base addresses for different channel types:

| Channel Type   | OSC Address Base | Count | Description                  |
|---------------|------------------|-------|------------------------------|
| Input Channel  | `/$ch/`          | 1-40  | Mono input channels          |
| Aux Input      | `/$aux/`         | 1-8   | Stereo aux inputs (USB, BT)  |
| Bus            | `/$bus/`         | 1-16  | Mix buses (groups/subgroups)  |
| Main           | `/$main/`        | 1-4   | Main LR outputs              |
| Matrix         | `/$mtx/`         | 1-8   | Matrix outputs               |
| DCA            | `/$dca/`         | 1-8   | DCA groups                   |
| FX Send        | `/$fxsend/`      | 1-4   | Effects send buses           |
| FX Return      | `/$fxreturn/`    | 1-4   | Effects return channels       |
| Monitor        | `/$mon/`         | 1-16  | Monitor outputs              |

## Fader Control

### Channel Fader
- **Path**: `/$ch/{N}/fader`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (linear)
- **Mapping**:
  - 0.0 = -infinity (mute)
  - 0.5 = approximately -10 dB
  - 0.75 = 0 dB (unity)
  - 0.875 = +5 dB
  - 1.0 = +10 dB
- **Conversion formulas**:
  - dB to linear: `linear = 10^(dB/20)` then scale to 0-1 range
  - For the Wing specifically: Use the built-in scale where fader position
    maps logarithmically. The exact mapping is:
    - 0.0 = -inf dB
    - 0.1 = -50 dB
    - 0.2 = -30 dB
    - 0.3 = -20 dB
    - 0.4 = -15 dB
    - 0.5 = -10 dB
    - 0.6 = -5 dB
    - 0.7 = -2 dB
    - 0.75 = 0 dB
    - 0.8 = +2 dB
    - 0.9 = +5 dB
    - 1.0 = +10 dB

### Bus/Main Fader
- **Path**: `/$bus/{N}/fader` or `/$main/{N}/fader`
- **Type**: Float (f)
- **Range**: Same 0.0-1.0 mapping as channel faders

### DCA Fader
- **Path**: `/$dca/{N}/fader`
- **Type**: Float (f)
- **Range**: 0.0-1.0 (same mapping)

## Mute and Solo

### Channel Mute
- **Path**: `/$ch/{N}/mute`
- **Type**: Integer (i)
- **Values**: 0 = unmuted, 1 = muted

### Channel Solo
- **Path**: `/$ch/{N}/solo`
- **Type**: Integer (i)
- **Values**: 0 = solo off, 1 = solo on

### Bus Mute
- **Path**: `/$bus/{N}/mute`
- **Type**: Integer (i)
- **Values**: 0 = unmuted, 1 = muted

### Main Mute
- **Path**: `/$main/{N}/mute`
- **Type**: Integer (i)
- **Values**: 0 = unmuted, 1 = muted

### DCA Mute
- **Path**: `/$dca/{N}/mute`
- **Type**: Integer (i)
- **Values**: 0 = unmuted, 1 = muted

## Preamp / Input Gain

### Analog Preamp Gain
- **Path**: `/$ch/{N}/preamp/gain`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to +10 dB to +60 dB)

### Digital Trim
- **Path**: `/$ch/{N}/preamp/trim`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to -18 dB to +18 dB, 0.5 = 0 dB)

### Phantom Power (48V)
- **Path**: `/$ch/{N}/preamp/48v`
- **Type**: Integer (i)
- **Values**: 0 = off, 1 = on

### Phase Invert
- **Path**: `/$ch/{N}/preamp/invert`
- **Type**: Integer (i)
- **Values**: 0 = normal, 1 = inverted (180 degrees)

## Parametric EQ (6 Bands)

Each channel has a 6-band fully parametric EQ. Band numbers are 1-6.

### EQ Enable
- **Path**: `/$ch/{N}/eq/on`
- **Type**: Integer (i)
- **Values**: 0 = EQ bypassed, 1 = EQ active

### EQ Band Type
- **Path**: `/$ch/{N}/eq/{band}/type`
- **Type**: Integer (i)
- **Values**:
  - 0 = Low Cut (HPF)
  - 1 = Low Shelf
  - 2 = Bell (parametric)
  - 3 = High Shelf
  - 4 = High Cut (LPF)
  - 5 = Notch
  - 6 = Band Pass

### EQ Band Frequency
- **Path**: `/$ch/{N}/eq/{band}/freq`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 20 Hz to 20,000 Hz, logarithmic)
- **Mapping formula**: `freq = 20 * (1000 ^ value)` Hz
  - 0.0 = 20 Hz
  - 0.25 = ~112 Hz
  - 0.5 = ~632 Hz
  - 0.75 = ~3,556 Hz
  - 1.0 = 20,000 Hz

### EQ Band Gain
- **Path**: `/$ch/{N}/eq/{band}/gain`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to -15 dB to +15 dB, 0.5 = 0 dB)
- **Mapping**: `gain_dB = (value - 0.5) * 30`

### EQ Band Q (Bandwidth)
- **Path**: `/$ch/{N}/eq/{band}/q`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0.3 to 12.0, logarithmic)
- **Mapping**: Higher value = narrower bandwidth
  - 0.0 = Q 0.3 (very wide)
  - 0.25 = Q 0.7
  - 0.5 = Q 1.5
  - 0.75 = Q 4.0
  - 1.0 = Q 12.0 (very narrow / notch)

### HPF Slope (Band 1 when type=0)
- **Path**: `/$ch/{N}/eq/1/slope`
- **Type**: Integer (i)
- **Values**: 0 = 6 dB/oct, 1 = 12 dB/oct, 2 = 18 dB/oct, 3 = 24 dB/oct

## Compressor / Dynamics

### Compressor Enable
- **Path**: `/$ch/{N}/dyn/comp/on`
- **Type**: Integer (i)
- **Values**: 0 = bypassed, 1 = active

### Compressor Threshold
- **Path**: `/$ch/{N}/dyn/comp/thr`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to -60 dB to 0 dB)
- **Mapping**: `threshold_dB = value * 60 - 60`

### Compressor Ratio
- **Path**: `/$ch/{N}/dyn/comp/ratio`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 1:1 to 20:1, logarithmic)
- **Mapping**:
  - 0.0 = 1:1 (no compression)
  - 0.25 = 2:1
  - 0.5 = 4:1
  - 0.75 = 10:1
  - 1.0 = 20:1 (hard limiting)

### Compressor Attack
- **Path**: `/$ch/{N}/dyn/comp/attack`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0.1 ms to 200 ms, logarithmic)
- **Mapping**:
  - 0.0 = 0.1 ms
  - 0.25 = 1 ms
  - 0.5 = 10 ms
  - 0.75 = 50 ms
  - 1.0 = 200 ms

### Compressor Release
- **Path**: `/$ch/{N}/dyn/comp/release`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 5 ms to 2000 ms, logarithmic)
- **Mapping**:
  - 0.0 = 5 ms
  - 0.25 = 30 ms
  - 0.5 = 100 ms
  - 0.75 = 500 ms
  - 1.0 = 2000 ms

### Compressor Makeup Gain
- **Path**: `/$ch/{N}/dyn/comp/gain`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0 dB to +24 dB)

### Compressor Mode
- **Path**: `/$ch/{N}/dyn/comp/mode`
- **Type**: Integer (i)
- **Values**: 0 = COMP (compressor), 1 = EXP (expander)

### Compressor Knee
- **Path**: `/$ch/{N}/dyn/comp/knee`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (0 = hard knee, 1 = soft knee)

### Compressor Detection
- **Path**: `/$ch/{N}/dyn/comp/det`
- **Type**: Integer (i)
- **Values**: 0 = peak, 1 = RMS

### Compressor Auto Makeup
- **Path**: `/$ch/{N}/dyn/comp/automakeup`
- **Type**: Integer (i)
- **Values**: 0 = manual, 1 = auto

## Gate / Expander

### Gate Enable
- **Path**: `/$ch/{N}/dyn/gate/on`
- **Type**: Integer (i)
- **Values**: 0 = bypassed, 1 = active

### Gate Threshold
- **Path**: `/$ch/{N}/dyn/gate/thr`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to -80 dB to 0 dB)

### Gate Range (Attenuation depth)
- **Path**: `/$ch/{N}/dyn/gate/range`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0 dB to -80 dB attenuation)

### Gate Attack
- **Path**: `/$ch/{N}/dyn/gate/attack`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0.05 ms to 100 ms)

### Gate Hold
- **Path**: `/$ch/{N}/dyn/gate/hold`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0 ms to 2000 ms)

### Gate Release
- **Path**: `/$ch/{N}/dyn/gate/release`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 5 ms to 2000 ms)

### Gate Mode
- **Path**: `/$ch/{N}/dyn/gate/mode`
- **Type**: Integer (i)
- **Values**: 0 = GATE (hard gate), 1 = EXP2 (2:1 expander), 2 = EXP3 (3:1), 3 = EXP4 (4:1), 4 = DUCK

## Delay / Phase Alignment

### Channel Delay Enable
- **Path**: `/$ch/{N}/delay/on`
- **Type**: Integer (i)
- **Values**: 0 = off, 1 = on

### Channel Delay Time
- **Path**: `/$ch/{N}/delay/time`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (maps to 0.0 ms to 500.0 ms)

## Bus Send Levels

### Send Level
- **Path**: `/$ch/{N}/send/{bus}/level`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (same mapping as fader)

### Send Enable
- **Path**: `/$ch/{N}/send/{bus}/on`
- **Type**: Integer (i)
- **Values**: 0 = off, 1 = on

### Send Pan
- **Path**: `/$ch/{N}/send/{bus}/pan`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (0.0 = full left, 0.5 = center, 1.0 = full right)

### Send Pre/Post
- **Path**: `/$ch/{N}/send/{bus}/pre`
- **Type**: Integer (i)
- **Values**: 0 = post-fader, 1 = pre-fader

## DCA Group Assignment

### DCA Assignment
- **Path**: `/$ch/{N}/dca/{dca_num}`
- **Type**: Integer (i)
- **Values**: 0 = not assigned, 1 = assigned

## Channel Configuration

### Channel Name
- **Path**: `/$ch/{N}/config/name`
- **Type**: String (s)
- **Max Length**: 12 characters

### Channel Color
- **Path**: `/$ch/{N}/config/color`
- **Type**: Integer (i)
- **Values**: 0-15 (color palette index)

### Channel Icon
- **Path**: `/$ch/{N}/config/icon`
- **Type**: Integer (i)
- **Values**: 0-74 (icon index)

### Channel Stereo Link
- **Path**: `/$ch/{N}/config/link`
- **Type**: Integer (i)
- **Values**: 0 = mono, 1 = linked stereo with next channel

## Pan

### Channel Pan
- **Path**: `/$ch/{N}/pan`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (0.0 = full left, 0.5 = center, 1.0 = full right)

## Snapshot / Scene Controls

### Save Snapshot
- **Path**: `/$snippet/save`
- **Type**: Integer (i) — snapshot number (1-100)

### Recall Snapshot
- **Path**: `/$snippet/recall`
- **Type**: Integer (i) — snapshot number (1-100)

### Scene Recall
- **Path**: `/$show/recall`
- **Type**: Integer (i) — scene number

## Metering

### Channel Input Meter
- **Path**: `/$meters/ch/{N}/in`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (linear amplitude, peak hold)
- **Note**: Metering data is sent by the mixer as a continuous stream
  when subscribed. Subscribe with `/$meters/subscribe`.

### Channel Output Meter
- **Path**: `/$meters/ch/{N}/out`
- **Type**: Float (f)

### Main Bus Meter
- **Path**: `/$meters/main/{N}/out`
- **Type**: Float (f)

### Gain Reduction Meter
- **Path**: `/$meters/ch/{N}/gr`
- **Type**: Float (f)
- **Range**: 0.0 to 1.0 (amount of gain reduction)

## Subscription Commands

To receive real-time parameter changes and metering:

- **Subscribe to all channel changes**: `/$subscribe/ch/{N}` (integer: renewal interval in seconds, typically 10)
- **Subscribe to metering**: `/$meters/subscribe` (integer: update interval in ms, typically 50)
- **Unsubscribe**: `/$unsubscribe/ch/{N}` or `/$meters/unsubscribe`

## Important Implementation Notes

1. **Rate Limiting**: Do not send more than 50 OSC messages per second to avoid
   overloading the mixer CPU.
2. **Atomic Updates**: Group related parameter changes (e.g., EQ freq + gain + Q)
   within a 10ms window for glitch-free transitions.
3. **Float Precision**: The Wing uses 32-bit float internally. Round OSC values
   to 4 decimal places.
4. **String Encoding**: UTF-8, null-terminated, padded to 4-byte boundary per
   the OSC specification.
5. **Subscription Renewal**: Re-subscribe every 10 seconds or subscriptions expire.
6. **Thread Safety**: OSC send/receive should use a dedicated thread or async loop
   to avoid blocking the main mixing logic.
7. **Error Handling**: If a parameter set fails (no acknowledgment within 100ms),
   retry once, then log and skip.
