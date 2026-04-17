# Behringer WING Rack OSC Reference

This file intentionally tracks the official WING FW 3.1 protocol naming.

Primary local sources:

- `Docs/Behringer_WING_Official/text/WING_Remote_Protocols_FW_3.1_MediaValet.txt`
- `Docs/Behringer_WING_Official/text/WING_Manual_WING_WING_RACK_WING_Compact.txt`
- `Docs/Behringer_WING_Official/text/WING_Effects_Guide.txt`

## Connection

- Handshake: send `WING?` to UDP `2222`
- OSC control port used in this repo: `2223`

## Root nodes

| Target | Official root |
|---|---|
| Input channels | `/ch/{n}` |
| Aux inputs | `/aux/{n}` |
| Buses | `/bus/{n}` |
| Main | `/main/{n}` |
| Matrix | `/mtx/{n}` |
| FX rack | `/fx/{n}` |

## Common channel paths

| Function | Official path |
|---|---|
| Fader | `/ch/{n}/fdr` |
| Mute | `/ch/{n}/mute` |
| Name | `/ch/{n}/name` or `/ch/{n}/$name` |
| Polarity invert | `/ch/{n}/in/set/inv` |
| Trim | `/ch/{n}/in/set/trim` |
| Delay mode | `/ch/{n}/in/set/dlymode` |
| Delay value | `/ch/{n}/in/set/dly` |
| Delay on | `/ch/{n}/in/set/dlyon` |

## EQ and dynamics model selectors

| Slot | Official path |
|---|---|
| Gate model | `/ch/{n}/gate/mdl` |
| EQ model | `/ch/{n}/eq/mdl` |
| Dynamics model | `/ch/{n}/dyn/mdl` |

Examples of official model families from the protocol:

- Gate: `GATE`, `DUCK`, `E88`, `9000G`, `D241`, `DS902`, `WAVE`, `DEQ`, `WARM`, `76LA`, `LA`, `RIDE`, `PSE`, `CMB`
- EQ: `STD`, `SOUL`, `E88`, `E84`, `F110`, `PULSAR`, `MACH4`
- Dynamics: `COMP`, `EXP`, `B160`, `B560`, `D241`, `ECL33`, `9000C`, `SBUS`, `RED3`, `76LA`, `LA`, `F670`, `BLISS`, `NSTR`, `WAVE`, `RIDE`, `2250`, `L100`, `CMB`

## Insert addresses

### Channel

- Pre on: `/ch/{n}/preins/on`
- Pre slot: `/ch/{n}/preins/ins`
- Post on: `/ch/{n}/postins/on`
- Post mode: `/ch/{n}/postins/mode`
- Post slot: `/ch/{n}/postins/ins`

### Bus

- Pre on: `/bus/{n}/preins/on`
- Pre slot: `/bus/{n}/preins/ins`
- Post on: `/bus/{n}/postins/on`
- Post slot: `/bus/{n}/postins/ins`

### Main

- Pre on: `/main/{n}/preins/on`
- Pre slot: `/main/{n}/preins/ins`
- Post on: `/main/{n}/postins/on`
- Post slot: `/main/{n}/postins/ins`

### Matrix

- Pre on: `/mtx/{n}/preins/on`
- Pre slot: `/mtx/{n}/preins/ins`
- Post on: `/mtx/{n}/postins/on`
- Post slot: `/mtx/{n}/postins/ins`

## FX rack addresses

| Function | Official path |
|---|---|
| FX model | `/fx/{slot}/mdl` |
| FX mix | `/fx/{slot}/fxmix` |
| FX source | `/fx/{slot}/$esrc` |
| FX mode | `/fx/{slot}/$emode` |
| Assigned channel | `/fx/{slot}/$a_chn` |
| Assigned position | `/fx/{slot}/$a_pos` |
| Numbered parameters | `/fx/{slot}/1` .. `/fx/{slot}/33` |

Important:

- The official parameter is `fxmix`, not `mix`.
- FX slots `1..8` support premium time-based engines.
- FX slots `9..16` are standard-only.

## FX categories we need to support in code

### Time-based / send FX

- `HALL`, `ROOM`, `CHAMBER`, `PLATE`, `CONCERT`, `AMBI`
- `V-ROOM`, `V-REV`, `V-PLATE`
- `GATED`, `REVERSE`, `DEL/REV`, `SHIMMER`, `SPRING`
- `ST-DL`, `TAP-DL`, `TAPE-DL`, `OILCAN`, `BBD-DL`
- `VSS3`, `BPLATE`

### Insert / utility FX

- `PCORR`, `DE-S2`, `SPKMAN`, `DEQ3`, `LIMITER`, `ENHANCE`, `EXCITER`
- `P-BASS`, `ROTARY`, `PHASER`, `PANNER`, `TAPE`, `MOOD`
- `RACKAMP`, `UKROCK`, `ANGEL`, `JAZZC`, `DELUXE`
- `BODY`, `SOUL`, `E88`, `E84`, `F110`, `PULSAR`, `MACH4`

### Channel strips

- `*EVEN*`
- `*SOUL*`
- `*VINTAGE*`
- `*BUS*`
- `*MASTER*`

## Repo guidance

- Prefer helpers in `backend/wing_client.py` over hardcoding OSC strings.
- For dry-run workflows, use `backend/observation_mixer.py`.
- Do not use old unofficial patterns like `/$ch/{n}/fader` in new code.
