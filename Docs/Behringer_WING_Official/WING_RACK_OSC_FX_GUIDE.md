# WING Rack OSC and FX Guide

Working summary based on official WING FW 3.1 documents.

## Core OSC facts

- Discovery handshake: send `WING?` to UDP port `2222`
- OSC control port used by this project: `2223`
- Main official protocol source:
  `Docs/Behringer_WING_Official/text/WING_Remote_Protocols_FW_3.1_MediaValet.txt`

## Important OSC roots

- Channels: `/ch/1` .. `/ch/40`
- Aux inputs: `/aux/1` .. `/aux/8`
- Buses: `/bus/1` .. `/bus/16`
- Main buses: `/main/1` .. `/main/4`
- Matrices: `/mtx/1` .. `/mtx/8`
- FX rack: `/fx/1` .. `/fx/16`

## Official FX slot addresses

- Model: `/fx/{slot}/mdl`
- Mix: `/fx/{slot}/fxmix`
- Source: `/fx/{slot}/$esrc`
- Mode: `/fx/{slot}/$emode`
- Assigned channel: `/fx/{slot}/$a_chn`
- Assigned insert position: `/fx/{slot}/$a_pos`
- Parameters: `/fx/{slot}/1` .. `/fx/{slot}/33`

Important note:
- The official path is `fxmix`, not `mix`.

## Insert assignment addresses

### Channels

- Pre insert on: `/ch/{n}/preins/on`
- Pre insert slot: `/ch/{n}/preins/ins`
- Post insert on: `/ch/{n}/postins/on`
- Post insert mode: `/ch/{n}/postins/mode`
- Post insert slot: `/ch/{n}/postins/ins`

### Buses

- Pre insert on: `/bus/{n}/preins/on`
- Pre insert slot: `/bus/{n}/preins/ins`
- Post insert on: `/bus/{n}/postins/on`
- Post insert slot: `/bus/{n}/postins/ins`

### Main

- Pre insert on: `/main/{n}/preins/on`
- Pre insert slot: `/main/{n}/preins/ins`
- Post insert on: `/main/{n}/postins/on`
- Post insert slot: `/main/{n}/postins/ins`

### Matrix

- Pre insert on: `/mtx/{n}/preins/on`
- Pre insert slot: `/mtx/{n}/preins/ins`
- Post insert on: `/mtx/{n}/postins/on`
- Post insert slot: `/mtx/{n}/postins/ins`

## FX rack architecture

From the official manual:

- WING has 16 FX rack slots.
- Slots `FX1..FX8` are Premium FX slots.
- Slots `FX9..FX16` are Standard FX slots.
- Premium slots can host time-based FX, standard processors, and channel strips.
- Standard slots cannot host memory-heavy time-based effects.
- Channel strips combine 3 processors in one slot.
- `MASTER` combines 4 processors in one slot.

## FX model groups

### Premium-capable examples

- `HALL`, `ROOM`, `CHAMBER`, `PLATE`, `CONCERT`, `AMBI`
- `V-ROOM`, `V-REV`, `V-PLATE`
- `GATED`, `REVERSE`, `DEL/REV`, `SHIMMER`, `SPRING`
- `DIMCRS`, `CHORUS`, `FLANGER`
- `ST-DL`, `TAP-DL`, `TAPE-DL`, `OILCAN`, `BBD-DL`
- `PITCH`, `D-PITCH`, `VSS3`, `BPLATE`

### Standard FX / inserts

- `GEQ`, `PIA`, `DOUBLE`, `PCORR`, `LIMITER`, `DE-S2`
- `ENHANCE`, `EXCITER`, `P-BASS`, `ROTARY`, `PHASER`, `PANNER`
- `TAPE`, `MOOD`, `SUB`, `RACKAMP`, `UKROCK`, `ANGEL`, `JAZZC`, `DELUXE`
- `BODY`, `SOUL`, `E88`, `E84`, `F110`, `PULSAR`, `MACH4`
- `C5-CMB`, `SUB-M`, `V-IMG`, `SPKMAN`, `DEQ3`
- Channel strips: `*EVEN*`, `*SOUL*`, `*VINTAGE*`, `*BUS*`, `*MASTER*`

## Channel strip FX content

From the manual:

- `*EVEN*` = Even 88 Gate + Even 88 Formant + Even Comp/Lim
- `*SOUL*` = SOUL 9000 Gate/Expander + SOUL Analogue + SOUL 9000
- `*VINTAGE*` = 76 Limiter Amp + Pulsar EQ + LA Leveler
- `*BUS*` = SOUL Warmth + Even 84 + SOUL Bus Comp
- `*MASTER*` = Tape Machine + Mach EQ4 + Ultra Enhancer + Precision Limiter

## Practical routing guidance

- Reverbs and delays are usually used on sends.
- Amp sims, de-esser, dynamic EQ, speaker manager, phase tools are usually inserts.
- `DE-S2` is FX-rack-only and should be used as an insert.
- `PCORR` should be treated as an insert processor, not a send effect.

## Project implementation notes

- Use `backend/wing_client.py` as the source of real control methods.
- Use `backend/observation_mixer.py` when we need dry-run logging without writing to the desk.
- Old unofficial address patterns such as `/$ch/{n}/fader` are deprecated for WING work in this repo.
