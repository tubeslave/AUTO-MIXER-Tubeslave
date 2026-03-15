# Behringer Wing OSC Protocol Reference

Complete OSC (Open Sound Control) protocol reference for the Behringer Wing and Wing Rack digital mixers. All addresses and value ranges are based on the Wing Remote Protocols v3.0.5 documentation and verified against the AUTO-MIXER Tubeslave codebase.

## Connection Overview

### Network Configuration

- **Discovery port**: UDP 2222 (for initial handshake only)
- **OSC command port**: UDP 2223 (all parameter control and queries)
- **Protocol**: OSC over UDP (no TCP support)
- **Default mixer IP**: 192.168.1.102 (configurable on the mixer)
- **Client IP**: Any IP on the same subnet; the mixer responds to the source IP/port

### Connection Handshake

The Wing requires a two-step connection process:

1. **Discovery**: Send the raw ASCII string `WING?` (not OSC-formatted) to UDP port 2222. The mixer responds with its identity string containing the model name, firmware version, and mixer name.
2. **OSC verification**: Send an OSC query (e.g., `/ch/1/fdr` with no arguments) to UDP port 2223. The mixer responds with the current value, confirming OSC communication is working.

### Keeping the Connection Alive

The Wing uses a subscription model. To receive parameter updates:

- Send `/xremote` (no arguments) to port 2223 to subscribe to parameter changes.
- The subscription expires after approximately 10 seconds of inactivity.
- Send `/xremote` every 8 seconds to maintain the subscription.
- If the subscription lapses, the mixer stops sending updates but still accepts commands.

### Querying Values

To query the current value of any parameter, send the OSC address with no arguments. The mixer responds with the address and its current value(s).

Example: Send `/ch/1/fdr` (no args) -> Mixer responds `/ch/1/fdr` with a float value.

---

## Channel Numbering and Address Format

### General Format

All channel addresses use the format `/ch/{ch}/...` where `{ch}` is the channel number (1-40) **without zero padding**. The documented format is `/ch/1`, not `/ch/01`, though some firmware versions accept both.

### Channel Count

- **Input channels**: 1-40
- **Aux inputs**: 1-8
- **Buses (mix buses)**: 1-16
- **Main outputs**: 1-4 (Main 1 = Main LR stereo pair)
- **Matrix outputs**: 1-8
- **DCA groups**: 1-16
- **FX slots**: 1-16

### Read-Only Parameters

Parameters prefixed with `$` in the address are read-only and cannot be set. They can only be queried. Examples: `/ch/1/$fdr` (DCA-affected fader level), `/ch/1/$solo` (solo state).

---

## Input Channel Addresses (/ch/{ch}/...)

### Channel Identity

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/name` | string | up to 12 chars | Channel name (scribble strip) |
| `/ch/{ch}/icon` | int | icon index | Channel icon |
| `/ch/{ch}/col` | int | color index | Channel color (scribble strip LED) |
| `/ch/{ch}/led` | int | 0-1 | Channel LED indicator |
| `/ch/{ch}/tags` | string | text | Channel tags for organization |
| `/ch/{ch}/clink` | int | 0/1 | Custom channel link |

### Fader, Mute, Pan, Solo

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/fdr` | float | -144.0 to +10.0 | Channel fader level in dB. -144 = off, 0 = unity, +10 = max |
| `/ch/{ch}/mute` | int | 0/1 | Channel mute. 0 = unmuted, 1 = muted |
| `/ch/{ch}/pan` | float | -100 to +100 | Channel pan. -100 = full left, 0 = center, +100 = full right |
| `/ch/{ch}/wid` | float | -150 to +150 | Stereo width in percent |
| `/ch/{ch}/$solo` | int | 0/1 | [READ-ONLY] Solo switch state |
| `/ch/{ch}/$sololed` | int | 0/1 | [READ-ONLY] Solo LED state |
| `/ch/{ch}/solosafe` | int | 0/1 | Solo safe — channel excluded from solo clear |
| `/ch/{ch}/mon` | string | A, B, A+B | Monitor mode selection |
| `/ch/{ch}/proc` | int | varies | Processing order on the channel strip |
| `/ch/{ch}/ptap` | int | varies | Pre-tap point selection |
| `/ch/{ch}/$fdr` | float | -144.0 to +10.0 | [READ-ONLY] Effective fader level (affected by DCA assignments) |
| `/ch/{ch}/$mute` | int | 0/1 | [READ-ONLY] Effective mute state |
| `/ch/{ch}/$muteovr` | int | 0/1 | Mute override |

### Input Configuration

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/in/set/$mode` | string | M, ST, M/S | [READ-ONLY] Input mode |
| `/ch/{ch}/in/set/srcauto` | int | 0/1 | Auto source switching |
| `/ch/{ch}/in/set/altsrc` | int | 0/1 | Main/alt source toggle |
| `/ch/{ch}/in/set/inv` | int | 0/1 | Phase invert (polarity flip) |
| `/ch/{ch}/in/set/trim` | float | -18.0 to +18.0 | Digital trim in dB |
| `/ch/{ch}/in/set/bal` | float | -9.0 to +9.0 | Input balance in dB |
| `/ch/{ch}/in/set/$g` | float | varies | [READ-ONLY] Current gain (depends on source) |
| `/ch/{ch}/in/set/$vph` | int | 0/1 | [READ-ONLY] Phantom power state |
| `/ch/{ch}/in/set/dlymode` | string | M, FT, MS, SMP | Delay mode: meters, feet, ms, samples |
| `/ch/{ch}/in/set/dly` | float | varies | Delay value (unit depends on dlymode) |
| `/ch/{ch}/in/set/dlyon` | int | 0/1 | Channel delay on/off |

### Input Connection / Routing

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/in/conn/grp` | int | varies | Input connection group (local, AES50, USB, Dante, etc.) |
| `/ch/{ch}/in/conn/in` | int | varies | Input connection index within the group |
| `/ch/{ch}/in/conn/altgrp` | int | varies | Alternate input connection group |
| `/ch/{ch}/in/conn/altin` | int | varies | Alternate input connection index |

---

## Filters (High-Pass / Low-Cut)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/flt/lc` | int | 0/1 | Low cut (high-pass filter) on/off |
| `/ch/{ch}/flt/lcf` | float | 20 to 2000 | Low cut frequency in Hz |
| `/ch/{ch}/flt/lcs` | int | 6, 12, 18, 24 | Low cut slope in dB/octave |
| `/ch/{ch}/flt/hc` | int | 0/1 | High cut (low-pass filter) on/off |
| `/ch/{ch}/flt/hcf` | float | 50 to 20000 | High cut frequency in Hz |
| `/ch/{ch}/flt/hcs` | int | 6, 12 | High cut slope in dB/octave |
| `/ch/{ch}/flt/tf` | int | 0/1 | Tool filter on/off |
| `/ch/{ch}/flt/mdl` | string | TILT, MAX, AP1, AP2 | Filter model selection |
| `/ch/{ch}/flt/tilt` | float | -6.0 to +6.0 | Tilt EQ level in dB |

---

## Parametric EQ (/ch/{ch}/eq/...)

The Wing provides a full parametric equalizer per channel with a low shelf, 4 parametric mid bands (1-4), and a high shelf.

### EQ Global Controls

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/eq/on` | int | 0/1 | EQ on/off |
| `/ch/{ch}/eq/mdl` | string | STD, SOUL, E88, E84, F110, PULSAR, MACH4 | EQ model emulation |
| `/ch/{ch}/eq/mix` | float | 0 to 125 | EQ wet/dry mix in percent |
| `/ch/{ch}/eq/$solo` | int | 0/1 | [READ-ONLY] EQ solo |
| `/ch/{ch}/eq/$solobd` | int | varies | [READ-ONLY] EQ solo band |

### Low Shelf Band

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/eq/lg` | float | -15.0 to +15.0 | Low shelf gain in dB |
| `/ch/{ch}/eq/lf` | float | 20 to 2000 | Low shelf frequency in Hz |
| `/ch/{ch}/eq/lq` | float | 0.44 to 10.0 | Low shelf Q factor |
| `/ch/{ch}/eq/leq` | string | PEQ, SHV | Low band type: parametric or shelving |

### Parametric Bands 1-4

Each band (1-4) follows the same pattern. Replace `{band}` with 1, 2, 3, or 4:

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/eq/{band}g` | float | -15.0 to +15.0 | Band gain in dB |
| `/ch/{ch}/eq/{band}f` | float | 20 to 20000 | Band center frequency in Hz |
| `/ch/{ch}/eq/{band}q` | float | 0.44 to 10.0 | Band Q factor (bandwidth) |

### High Shelf Band

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/eq/hg` | float | -15.0 to +15.0 | High shelf gain in dB |
| `/ch/{ch}/eq/hf` | float | 50 to 20000 | High shelf frequency in Hz |
| `/ch/{ch}/eq/hq` | float | 0.44 to 10.0 | High shelf Q factor |
| `/ch/{ch}/eq/heq` | string | SHV, PEQ | High band type: shelving or parametric |

### Pre-Send EQ (3-Band)

A secondary 3-band parametric EQ applied before sends. Useful for monitor mix adjustments independent of the main EQ.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/peq/on` | int | 0/1 | Pre-EQ on/off |
| `/ch/{ch}/peq/{band}g` | float | -15.0 to +15.0 | Band gain in dB (bands 1-3) |
| `/ch/{ch}/peq/{band}f` | float | 20 to 20000 | Band frequency in Hz (bands 1-3) |
| `/ch/{ch}/peq/{band}q` | float | 0.44 to 10.0 | Band Q (bands 1-3) |

---

## Gate / Expander (/ch/{ch}/gate/...)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/gate/on` | int | 0/1 | Gate on/off |
| `/ch/{ch}/gate/mdl` | string | varies | Gate model emulation |
| `/ch/{ch}/gate/thr` | float | -80.0 to 0.0 | Gate threshold in dB |
| `/ch/{ch}/gate/range` | float | 3.0 to 60.0 | Gate range (depth of attenuation) in dB |
| `/ch/{ch}/gate/att` | float | 0.0 to 120.0 | Gate attack time in ms |
| `/ch/{ch}/gate/hld` | float | 0.0 to 200.0 | Gate hold time in ms |
| `/ch/{ch}/gate/rel` | float | 4.0 to 4000.0 | Gate release time in ms |
| `/ch/{ch}/gate/acc` | float | 0 to 100 | Accent sensitivity (percent) |
| `/ch/{ch}/gate/ratio` | string | 1:1.5, 1:2, 1:3, 1:4, GATE | Gate/expander ratio |

### Gate Sidechain

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/gatesc/type` | string | Off, LP12, HP12, BP | Sidechain filter type |
| `/ch/{ch}/gatesc/f` | float | 20 to 20000 | Sidechain filter frequency in Hz |
| `/ch/{ch}/gatesc/q` | float | 0.44 to 10.0 | Sidechain filter Q |
| `/ch/{ch}/gatesc/src` | string | varies | Sidechain source (channel or external) |
| `/ch/{ch}/gatesc/tap` | string | varies | Sidechain tap point |
| `/ch/{ch}/gatesc/$solo` | int | 0/1 | [READ-ONLY] Sidechain solo |

---

## Dynamics / Compressor (/ch/{ch}/dyn/...)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/dyn/on` | int | 0/1 | Compressor on/off |
| `/ch/{ch}/dyn/mdl` | string | varies | Compressor model emulation |
| `/ch/{ch}/dyn/mix` | float | 0 to 100 | Parallel compression mix (percent). 100 = full compression, 50 = 50/50 |
| `/ch/{ch}/dyn/gain` | float | -6.0 to +12.0 | Makeup gain in dB |
| `/ch/{ch}/dyn/thr` | float | -60.0 to 0.0 | Compressor threshold in dB |
| `/ch/{ch}/dyn/ratio` | string | 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10, 20, 50, 100 | Compression ratio (discrete steps) |
| `/ch/{ch}/dyn/knee` | float | 0 to 5 | Knee curve. 0 = hard knee, 5 = soft knee |
| `/ch/{ch}/dyn/det` | string | PEAK, RMS | Detector type |
| `/ch/{ch}/dyn/att` | float | 0.0 to 120.0 | Attack time in ms |
| `/ch/{ch}/dyn/hld` | float | 1.0 to 200.0 | Hold time in ms |
| `/ch/{ch}/dyn/rel` | float | 4.0 to 4000.0 | Release time in ms |
| `/ch/{ch}/dyn/env` | string | LIN, LOG | Envelope mode: linear or logarithmic |
| `/ch/{ch}/dyn/auto` | int | 0/1 | Auto release switch |

### Compressor Crossover (Frequency-Selective Compression)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/dynxo/depth` | float | 0 to 20 | Crossover depth in dB |
| `/ch/{ch}/dynxo/type` | string | OFF, LO6, LO12, HI6, HI12, PC | Crossover filter type |
| `/ch/{ch}/dynxo/f` | float | 20 to 20000 | Crossover frequency in Hz |
| `/ch/{ch}/dynxo/$solo` | int | 0/1 | [READ-ONLY] Crossover solo |

### Compressor Sidechain

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/dynsc/type` | string | Off, LP12, HP12, BP | Sidechain filter type |
| `/ch/{ch}/dynsc/f` | float | 20 to 20000 | Sidechain filter frequency in Hz |
| `/ch/{ch}/dynsc/q` | float | 0.44 to 10.0 | Sidechain filter Q |
| `/ch/{ch}/dynsc/src` | string | SELF, CH.1..CH.40 | Sidechain source |
| `/ch/{ch}/dynsc/tap` | string | varies | Sidechain tap point |
| `/ch/{ch}/dynsc/$solo` | int | 0/1 | [READ-ONLY] Sidechain solo |

---

## Inserts

### Pre-Insert (before EQ/dynamics)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/preins/on` | int | 0/1 | Pre-insert on/off |
| `/ch/{ch}/preins/ins` | string | NONE, FX1..FX16 | Insert FX slot assignment |
| `/ch/{ch}/preins/$stat` | string | -, OK, N/A | [READ-ONLY] Insert status |

### Post-Insert (after EQ/dynamics)

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/postins/on` | int | 0/1 | Post-insert on/off |
| `/ch/{ch}/postins/mode` | string | FX, AUTO_X, AUTO_Y | Insert mode |
| `/ch/{ch}/postins/ins` | string | NONE, FX1..FX16 | Insert FX slot assignment |
| `/ch/{ch}/postins/w` | float | -12 to +12 | Autogain weight |
| `/ch/{ch}/postins/$stat` | string | varies | [READ-ONLY] Insert status |

---

## Channel Sends (/ch/{ch}/send/...)

Each channel can send to up to 16 mix buses and matrix outputs.

### Send to Mix Bus

Replace `{send}` with the bus number (1-16):

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/send/{send}/on` | int | 0/1 | Send on/off |
| `/ch/{ch}/send/{send}/lvl` | float | -144.0 to +10.0 | Send level in dB |
| `/ch/{ch}/send/{send}/pon` | int | 0/1 | Pre-fader always on |
| `/ch/{ch}/send/{send}/mode` | string | PRE, POST, GRP | Send mode: pre-fader, post-fader, or group |
| `/ch/{ch}/send/{send}/plink` | int | 0/1 | Pan link to channel pan |
| `/ch/{ch}/send/{send}/pan` | float | -100 to +100 | Send pan position |

### Send to Matrix

Matrix sends use the prefix `MX`: `/ch/{ch}/send/MX{x}/on`, `/ch/{ch}/send/MX{x}/lvl`, etc.

### Main Sends

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/ch/{ch}/main/pre` | int | 0/1 | Pre-fader to main |
| `/ch/{ch}/main/1/on` | int | 0/1 | Main 1 (LR) send on/off |
| `/ch/{ch}/main/1/lvl` | float | -144.0 to +10.0 | Main 1 send level in dB |

Mains 2-4 follow the same `/ch/{ch}/main/{n}/on` and `/ch/{ch}/main/{n}/lvl` pattern.

---

## Aux Input Channels (/aux/{aux}/...)

Aux inputs (1-8) share a simplified version of the channel strip.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/aux/{aux}/in/set/trim` | float | -18.0 to +18.0 | Trim in dB |
| `/aux/{aux}/in/set/bal` | float | -9.0 to +9.0 | Balance in dB |
| `/aux/{aux}/in/set/inv` | int | 0/1 | Phase invert |
| `/aux/{aux}/mute` | int | 0/1 | Mute |
| `/aux/{aux}/fdr` | float | -144.0 to +10.0 | Fader level in dB |
| `/aux/{aux}/pan` | float | -100 to +100 | Pan |
| `/aux/{aux}/wid` | float | -150 to +150 | Width |
| `/aux/{aux}/eq/on` | int | 0/1 | EQ on/off |
| `/aux/{aux}/dyn/on` | int | 0/1 | Dynamics on/off |
| `/aux/{aux}/dyn/thr` | float | -60.0 to 0.0 | Compressor threshold |
| `/aux/{aux}/dyn/depth` | float | varies | Dynamics depth |

---

## Mix Buses (/bus/{bus}/...)

Mix buses (1-16) provide subgroup and monitor mix routing.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/bus/{bus}/in/set/trim` | float | -18.0 to +18.0 | Bus trim in dB |
| `/bus/{bus}/in/set/bal` | float | -9.0 to +9.0 | Bus balance |
| `/bus/{bus}/in/set/inv` | int | 0/1 | Phase invert |
| `/bus/{bus}/mute` | int | 0/1 | Bus mute |
| `/bus/{bus}/fdr` | float | -144.0 to +10.0 | Bus fader in dB |
| `/bus/{bus}/pan` | float | -100 to +100 | Bus pan |
| `/bus/{bus}/wid` | float | -150 to +150 | Bus width |
| `/bus/{bus}/busmono` | int | 0/1 | Mono summing switch |
| `/bus/{bus}/eq/on` | int | 0/1 | Bus EQ on/off |
| `/bus/{bus}/dyn/on` | int | 0/1 | Bus dynamics on/off |
| `/bus/{bus}/dly/on` | int | 0/1 | Bus delay on/off |
| `/bus/{bus}/dly/mode` | string | varies | Bus delay mode |
| `/bus/{bus}/dly/dly` | float | varies | Bus delay time |

Bus EQ and dynamics have the same structure as channel EQ and dynamics, substituting `/bus/{bus}/` for `/ch/{ch}/`.

---

## Main Outputs (/main/{main}/...)

Main outputs (1-4) control the main stereo buses. Main 1 is the primary LR pair.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/main/{main}/mute` | int | 0/1 | Main mute |
| `/main/{main}/fdr` | float | -144.0 to +10.0 | Main fader in dB |
| `/main/{main}/pan` | float | -100 to +100 | Main pan |
| `/main/{main}/wid` | float | -150 to +150 | Main width |
| `/main/{main}/eq/on` | int | 0/1 | Main EQ on/off |
| `/main/{main}/dyn/on` | int | 0/1 | Main dynamics on/off |

Main EQ and dynamics follow the same band structure as channel processing.

---

## Matrix Outputs (/mtx/{mtx}/...)

Matrix outputs (1-8) provide additional output buses typically used for PA zones, delays, or broadcast feeds.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/mtx/{mtx}/mute` | int | 0/1 | Matrix mute |
| `/mtx/{mtx}/fdr` | float | -144.0 to +10.0 | Matrix fader in dB |
| `/mtx/{mtx}/pan` | float | -100 to +100 | Matrix pan |
| `/mtx/{mtx}/wid` | float | -150 to +150 | Matrix width |
| `/mtx/{mtx}/eq/on` | int | 0/1 | Matrix EQ on/off |

---

## DCA Groups (/dca/{dca}/...)

DCA (Digitally Controlled Amplifier) groups (1-16) provide master-level control without altering internal gain structure.

| Address | Type | Range | Description |
|---------|------|-------|-------------|
| `/dca/{dca}/fdr` | float | -144.0 to +10.0 | DCA fader in dB |
| `/dca/{dca}/mute` | int | 0/1 | DCA mute |

---

## Snapshots and Scenes

### Snapshot Recall

Snapshots are recalled via specific OSC messages. The Wing supports recalling snapshots by index or name.

| Address | Type | Description |
|---------|------|-------------|
| `/snap/load` | int (1-based index) | Recall snapshot by index |
| `/snap/save` | int (1-based index) | Save snapshot to index |

### Snapshot Scope

When recalling snapshots, the scope (which parameters are recalled) is configured on the mixer itself. The OSC command triggers the recall; it does not control scope.

---

## System and Utility Addresses

| Address | Type | Description |
|---------|------|-------------|
| `/xremote` | (no args) | Subscribe to parameter updates (renew every 8-10 seconds) |
| `/info` | (no args) | Query mixer info (model, firmware version, name) |
| `/status` | (no args) | Query mixer status |

---

## Metering

### Channel Meters

| Address | Type | Description |
|---------|------|-------------|
| `/meters/ch/{ch}/in` | float | [READ-ONLY] Pre-fader input level in dB |
| `/meters/ch/{ch}/out` | float | [READ-ONLY] Post-fader output level in dB |

### Bus and Main Meters

| Address | Type | Description |
|---------|------|-------------|
| `/meters/bus/{bus}/out` | float | [READ-ONLY] Bus output level in dB |
| `/meters/main/out` | float | [READ-ONLY] Main output level in dB |

---

## Value Encoding and Type Tags

### OSC Type Tags

The Wing uses standard OSC type tags:

- **f** (float32): Used for fader levels, gain, frequency, Q, pan, etc.
- **i** (int32): Used for on/off switches, index selections, etc.
- **s** (string): Used for names, model selections, enum values.

### Fader Value Mapping

Fader values are transmitted in dB, not as a 0.0-1.0 linear scale:

| dB Value | Meaning |
|----------|---------|
| -144.0 | Fader fully down (off / -inf) |
| -40.0 | Very quiet |
| -20.0 | Moderate level |
| -10.0 | Nominal level region |
| 0.0 | Unity gain |
| +5.0 | Above unity |
| +10.0 | Maximum fader position |

The Wing's internal processing uses 40-bit floating point, so there is effectively unlimited headroom between the ADC and DAC stages.

### Boolean Values

All on/off switches use integer 0 or 1:
- **0** = off / unmuted / disabled
- **1** = on / muted / enabled

Note: For mute, `1 = muted` (signal blocked), which is the opposite of what some might expect.

---

## Value Ranges Summary

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Fader | -144 | +10 | dB |
| Trim | -18 | +18 | dB |
| Pan | -100 | +100 | L/R |
| Width | -150 | +150 | % |
| EQ Gain | -15 | +15 | dB |
| EQ Frequency | 20 | 20000 | Hz |
| EQ Q | 0.44 | 10 | - |
| Comp Threshold | -60 | 0 | dB |
| Comp Attack | 0 | 120 | ms |
| Comp Hold | 1 | 200 | ms |
| Comp Release | 4 | 4000 | ms |
| Comp Makeup | -6 | +12 | dB |
| Comp Knee | 0 | 5 | - |
| Comp Mix | 0 | 100 | % |
| Gate Threshold | -80 | 0 | dB |
| Gate Attack | 0 | 120 | ms |
| Gate Hold | 0 | 200 | ms |
| Gate Release | 4 | 4000 | ms |
| Gate Range | 3 | 60 | dB |
| HPF Frequency | 20 | 2000 | Hz |
| LPF Frequency | 50 | 20000 | Hz |
| Send Level | -144 | +10 | dB |
| Filter Tilt | -6 | +6 | dB |
| Balance | -9 | +9 | dB |
| Dynxo Depth | 0 | 20 | dB |
| Accent | 0 | 100 | % |
| EQ Mix | 0 | 125 | % |

---

## AUTO-MIXER Tubeslave OSC Usage Patterns

### Common Query Patterns

The AUTO-MIXER system uses these patterns frequently:

```
# Query all channel faders (one at a time)
/ch/1/fdr    (no args)
/ch/2/fdr    (no args)
...

# Query channel names for identification
/ch/1/name   (no args)
/ch/2/name   (no args)
...

# Subscribe to all channel fader changes
Pattern: /ch/*/fdr

# Subscribe to specific channel parameters
Pattern: /ch/1/*
```

### Setting Parameters

```
# Set channel 1 fader to unity (0 dB)
/ch/1/fdr  0.0

# Mute channel 5
/ch/5/mute  1

# Set channel 3 EQ band 2 to cut 4 dB at 800 Hz with Q of 2.0
/ch/3/eq/2g  -4.0
/ch/3/eq/2f  800.0
/ch/3/eq/2q  2.0

# Enable gate on channel 2
/ch/2/gate/on  1
/ch/2/gate/thr  -30.0
/ch/2/gate/range  40.0
/ch/2/gate/att  0.5
/ch/2/gate/rel  200.0

# Set compressor on channel 1
/ch/1/dyn/on  1
/ch/1/dyn/thr  -18.0
/ch/1/dyn/ratio  "3.0"
/ch/1/dyn/att  10.0
/ch/1/dyn/rel  120.0
/ch/1/dyn/knee  3.0
/ch/1/dyn/det  "RMS"
```

### Feedback Detector Notch Filter Deployment

The feedback detector uses EQ bands as notch filters. It uses main EQ bands 1-4, then falls back to pre-EQ bands 1-3, allowing up to 8 notch filters per channel:

```
# Deploy a notch filter at 2500 Hz on channel 4, using EQ band 2
/ch/4/eq/2g  -8.0      # Cut 8 dB
/ch/4/eq/2f  2500.0    # At 2500 Hz
/ch/4/eq/2q  8.0       # Narrow Q for surgical notch

# If all 4 main EQ bands are used, fall back to pre-EQ
/ch/4/peq/1g  -10.0
/ch/4/peq/1f  3150.0
/ch/4/peq/1q  8.0
```

### Rate Limiting

The OSC manager enforces rate limiting to avoid flooding the mixer:

- **Default global rate**: 50 Hz (50 messages/second maximum)
- **Per-address throttle**: Messages to the same address are deduplicated if sent faster than the rate limit allows.
- **Query messages** (no arguments) bypass the per-address throttle.
- The send queue holds up to 4096 messages; overflow messages are dropped with a warning.

### Health Monitoring

The system monitors connection health:

- If no messages are received for 15 seconds (configurable), the connection is considered unhealthy.
- On health timeout, the system automatically sends `/xremote` and `/ch/1/fdr` to probe the mixer.
- Health callbacks (`on_connected`, `on_disconnected`) notify the application layer.
