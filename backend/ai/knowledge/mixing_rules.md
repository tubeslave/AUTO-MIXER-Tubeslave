# Live Sound Mixing Rules

Comprehensive reference for automated mixing decisions in live concert environments.
These rules are designed for the Behringer Wing Rack digital mixer with 40 input channels.

## Gain Staging Fundamentals

Proper gain staging is the foundation of a good mix. Every channel must maintain
sufficient headroom while keeping signal well above the noise floor.

- **Preamp Gain Target**: Set input gain so that peaks hit approximately -18 dBFS to -12 dBFS
  on the channel meter. This provides 12-18 dB of headroom before digital clipping.
- **Unity Gain Principle**: Each processing stage (EQ, compressor, insert) should have
  roughly equal input and output levels. Use makeup gain on compressors to compensate
  for gain reduction.
- **Fader Position**: Aim to keep channel faders at or near 0 dB (unity) during normal
  operation. This gives the most resolution for fine adjustments. If faders are
  consistently above +5 dB, increase preamp gain. If consistently below -20 dB,
  reduce preamp gain.
- **Bus Headroom**: Subgroup and main bus levels should peak at -6 dBFS to -3 dBFS.
  Never allow main bus to exceed -1 dBFS.
- **True Peak Limit**: Maximum allowed true peak is -1 dBTP on any output bus.
  For broadcast feeds, limit to -2 dBTP.
- **Noise Floor**: For 24-bit digital systems, the effective noise floor is around
  -120 dBFS. Any channel with signal below -60 dBFS should be considered inactive.
- **Digital Trim**: On the Wing Rack, use /$ch/N/preamp/trim (range -18 to +18 dB)
  for fine digital gain adjustment without touching the analog preamp.

## High-Pass Filter Settings

Every non-bass source should have a high-pass filter (HPF) engaged to remove
unwanted low-frequency content (rumble, handling noise, stage vibration).

| Instrument         | HPF Frequency | Slope   | Notes                         |
|--------------------|---------------|---------|-------------------------------|
| Lead Vocal         | 80 Hz         | 18 dB/oct | Prevents plosive rumble       |
| Backing Vocal      | 100 Hz        | 18 dB/oct | Slightly higher for clarity   |
| Acoustic Guitar    | 80 Hz         | 12 dB/oct | Preserves body                |
| Electric Guitar    | 80 Hz         | 18 dB/oct | Amp already filters sub       |
| Snare Top          | 80 Hz         | 18 dB/oct | Removes kick bleed sub        |
| Snare Bottom       | 100 Hz        | 18 dB/oct | Snare wire resonance focus    |
| Hi-Hat             | 200 Hz        | 24 dB/oct | Aggressive — only cymbals     |
| Overhead L/R       | 100-200 Hz    | 18 dB/oct | Depends on cymbal-vs-kit mix  |
| Rack Tom           | 80 Hz         | 18 dB/oct | Keep fundamental              |
| Floor Tom          | 60 Hz         | 12 dB/oct | Lower fundamental             |
| Piano              | 60 Hz         | 12 dB/oct | Preserve left-hand bass       |
| Organ              | 40 Hz         | 12 dB/oct | Organ pedals go very low      |
| Strings            | 80 Hz         | 12 dB/oct | Bow noise below 80 Hz         |
| Brass              | 80 Hz         | 18 dB/oct | Stand/handling noise           |
| Percussion         | 100 Hz        | 18 dB/oct | Most percussion is mid/high   |
| Kick (NO HPF)      | ---           | ---     | Never HPF a kick drum         |
| Bass Guitar (NO HPF)| ---          | ---     | Never HPF bass guitar         |

## EQ Techniques

### Subtractive EQ (Cut First)

Always cut before boosting. Subtractive EQ removes problem frequencies without
adding noise or level. Use a narrow Q (3-6) for surgical cuts and a wider Q (0.5-1.5)
for gentle shaping.

- **Mud Region (200-400 Hz)**: The most common problem area in live sound. Cut 2-4 dB
  with Q of 1.5-2.0 on most sources to add clarity. Be careful not to thin the sound.
- **Boxiness (400-800 Hz)**: Especially problematic on toms, snare, and some vocal mics.
  Sweep with a narrow boost to find the offending frequency, then cut.
- **Harshness (2-5 kHz)**: Ear-fatiguing frequencies. Cut 1-3 dB on electric guitars,
  brass, and aggressive vocals to tame harshness.
- **Sibilance (5-8 kHz)**: On vocals, a narrow cut at 5-7 kHz can tame sibilance. A
  dedicated de-esser is preferred for dynamic control.

### Additive EQ (Boost)

Use wider Q (0.5-1.5) for boosts. Keep boosts under 4 dB whenever possible.

- **Vocal Presence (2-4 kHz)**: A gentle 2-3 dB shelf or bell boost adds intelligibility.
- **Vocal Air (10-14 kHz)**: A high shelf boost of 1-2 dB adds "air" and openness.
- **Kick Attack (3-5 kHz)**: Beater click. Boost 2-4 dB for definition in dense mixes.
- **Kick Thump (60-80 Hz)**: Low shelf or bell boost 2-3 dB for weight.
- **Snare Body (200 Hz)**: Gentle 2 dB boost for fullness.
- **Snare Crack (4-6 kHz)**: 2-3 dB boost for snap and cut.
- **Bass Growl (600-1000 Hz)**: Where bass "note" definition lives. 1-2 dB boost.
- **Guitar Presence (2-3 kHz)**: Helps guitars cut through without volume increase.
- **Cymbal Shimmer (8-12 kHz)**: Delicate boost for cymbals and hi-hat presence.

## Compression Settings Per Instrument

### Lead Vocal
- **Threshold**: -20 to -16 dBFS (should trigger on every phrase)
- **Ratio**: 3:1 to 4:1 (consistent level without squashing)
- **Attack**: 5-15 ms (fast enough to catch peaks, slow enough for consonant transients)
- **Release**: 80-150 ms (should release before next phrase)
- **Knee**: Soft knee for natural feel
- **Gain Reduction Target**: 4-8 dB on peaks
- **Makeup Gain**: Match input/output levels visually

### Kick Drum
- **Threshold**: -16 to -12 dBFS
- **Ratio**: 4:1 to 6:1
- **Attack**: 10-30 ms (let transient through for click)
- **Release**: 50-80 ms (fast — must release before next hit)
- **Gain Reduction**: 4-6 dB
- **Note**: Parallel compression can add weight without killing transient

### Snare Drum
- **Threshold**: -14 to -10 dBFS
- **Ratio**: 3:1 to 4:1
- **Attack**: 5-15 ms (preserve initial crack)
- **Release**: 80-120 ms
- **Gain Reduction**: 3-6 dB

### Bass Guitar
- **Threshold**: -18 to -14 dBFS
- **Ratio**: 3:1 to 4:1
- **Attack**: 10-20 ms
- **Release**: 100-200 ms (follow the groove)
- **Gain Reduction**: 4-8 dB (bass is very dynamic in live settings)
- **Note**: Some engineers use two compressors in series — a fast one for peaks
  and a slow one for overall envelope

### Acoustic Guitar
- **Threshold**: -18 to -14 dBFS
- **Ratio**: 2:1 to 3:1 (gentle — preserve dynamics)
- **Attack**: 15-30 ms (keep pick attack)
- **Release**: 100-200 ms
- **Gain Reduction**: 2-4 dB

### Electric Guitar
- **Ratio**: 2:1 (amps already compress heavily)
- **Attack**: 20-40 ms
- **Release**: 100-200 ms
- **Note**: Distorted guitars may need no compression — they are already compressed

### Keys/Piano
- **Threshold**: -18 to -14 dBFS
- **Ratio**: 2:1 to 3:1
- **Attack**: 15-25 ms
- **Release**: 100-200 ms
- **Gain Reduction**: 3-5 dB

## Vocal Mixing Chain

The standard vocal processing chain order for live sound:

1. **Preamp Gain**: Set for -18 to -12 dBFS peaks
2. **High-Pass Filter**: 80 Hz, 18 dB/oct
3. **De-Esser**: 5-7 kHz, 4:1 ratio, threshold set to catch only sibilants
4. **EQ**:
   - Cut 250 Hz by 2-3 dB (proximity effect/mud)
   - Boost 3 kHz by 2 dB (presence/intelligibility)
   - Gentle high shelf at 10 kHz, +1.5 dB (air)
5. **Compressor**: 3:1, 10 ms attack, 100 ms release, 4-6 dB GR
6. **Effects Send**: Plate reverb 1.2-1.8 second decay, pre-delay 30-50 ms.
   Low-cut the reverb return at 200 Hz to keep low end clean.

## Drum Mixing Approach

### Kick Drum
1. Gate: Threshold -30 dB, attack 0.5 ms, hold 100 ms, release 50 ms, range -40 dB
2. EQ: Boost 60 Hz (+3 dB), cut 350 Hz (-4 dB), boost 4 kHz (+3 dB)
3. Compressor: 4:1, 20 ms attack, 60 ms release
4. Pan: Center (0.0)

### Snare Top
1. Gate: Threshold -25 dB, attack 0.5 ms, hold 80 ms, release 40 ms, range -30 dB
2. EQ: HPF 80 Hz, boost 200 Hz (+2 dB), cut 800 Hz (-2 dB), boost 5 kHz (+3 dB)
3. Compressor: 3:1, 8 ms attack, 100 ms release
4. Pan: Center or slight right (+0.05 to +0.1)

### Toms
1. Gate: Threshold -28 dB, attack 0.5 ms, hold 120 ms, release 60 ms, range -40 dB
2. EQ: HPF 60-80 Hz, boost fundamental (100-200 Hz, +2 dB), cut mud (400 Hz, -3 dB), boost attack (3-5 kHz, +2 dB)
3. Compressor: 3:1, 15 ms attack, 80 ms release
4. Pan: Rack toms left-center, floor tom right-center (audience perspective)

### Overheads
1. No gate (ambient mics)
2. EQ: HPF 150 Hz (remove low bleed), gentle cut 400 Hz (-2 dB), boost 10 kHz (+2 dB)
3. Light compression: 2:1, 20 ms attack, 150 ms release
4. Pan: Hard left and right (L=-0.8, R=+0.8) or spaced

## Bass Guitar Techniques

- **DI Signal**: Clean, full range. Apply HPF at 30 Hz to remove sub-sonic rumble.
  EQ: Cut 250 Hz (-2 dB) for clarity, boost 700 Hz (+2 dB) for note definition.
- **Amp Signal**: Mid-focused, may have grind. HPF at 80 Hz, blend with DI.
- **Blend Ratio**: Typically 60% DI / 40% amp for rock. Adjust per genre.
- **Compression**: Essential. Bass is extremely dynamic live. 3:1-4:1 ratio.
- **Low-End Management**: Bass and kick must occupy complementary frequency ranges.
  If kick is boosted at 60 Hz, cut bass at 60 Hz and boost at 80-100 Hz (or vice versa).
- **Side-chain**: Some engineers side-chain the bass compressor from the kick drum
  to create space when both hit simultaneously.

## Guitar Mixing

- **Mid-Range Focus**: Guitars live in 200 Hz-5 kHz. Keep them out of the vocal and
  bass territories.
- **HPF Always**: 80 Hz minimum, even on distorted guitars.
- **Delay-Based Effects**: Use short delays (50-120 ms) for width. Tempo-sync longer
  delays for rhythmic effects (quarter note, dotted eighth).
- **Reverb**: Short plate or room, 0.8-1.5 seconds. Low-cut the return at 200 Hz.
- **Dual Guitar Panning**: Hard pan left and right (L=-0.6, R=+0.6) for two guitars.
  Single guitar slightly off-center (-0.2 or +0.2).
- **Clean vs. Distorted**: Clean guitars need more compression (3:1). Distorted
  guitars are self-compressing and often need only subtle EQ shaping.

## Stereo Image Management

### LCR Panning Philosophy
For live sound, the LCR (Left-Center-Right) approach provides the strongest and
most consistent stereo image for large venues:

- **Center (0.0)**: Kick, snare, bass, lead vocal, main keys
- **Off-Center (0.2-0.4)**: Backing vocals, rhythm guitar, secondary keys
- **Wide (0.5-0.8)**: Overheads, stereo keys, dual guitars, percussion
- **Hard Pan (0.9-1.0)**: Only for true stereo sources (stereo keys, overhead pair)

### Width Management
- Mono-compatible mix is essential — many audience members are off-axis.
- Check mono compatibility regularly by summing L+R and listening for cancellation.
- Reverb and delay returns should be stereo to create width without panning dry sources.

## Effects Recommendations

### Reverb Types
| Source        | Reverb Type | Decay Time | Pre-Delay | Notes                     |
|---------------|-------------|------------|-----------|---------------------------|
| Lead Vocal    | Plate       | 1.2-1.8s   | 30-50 ms  | Warm, smooth, adds depth  |
| Backing Vocals| Plate/Hall  | 1.5-2.0s   | 20-40 ms  | Slightly longer than lead |
| Snare         | Plate       | 0.8-1.2s   | 0-10 ms   | Adds body and sustain     |
| Toms          | Room/Plate  | 0.6-1.0s   | 0-5 ms    | Short, supportive         |
| Guitar        | Room/Spring | 0.8-1.5s   | 10-30 ms  | Genre dependent           |
| Piano/Keys    | Hall        | 1.5-2.5s   | 20-40 ms  | Concert hall ambiance     |

### Delay Settings
- **Slapback**: 80-120 ms, single repeat, -6 dB feedback. For rockabilly/country vocals.
- **Quarter Note**: BPM-synced. 120 BPM = 500 ms. 2-3 repeats, -10 dB feedback.
- **Dotted Eighth**: BPM-synced. 120 BPM = 375 ms. Popular for guitar leads and U2-style.
- **Stereo Ping-Pong**: Different L/R times for width. Useful on keys and guitar solos.

## Live Mixing Best Practices

### Feedback Prevention
- Ring out monitors before the show using 1/3 octave or parametric EQ.
- Keep monitor levels as low as possible while meeting musician needs.
- Avoid boosting frequencies in the 1-4 kHz range on vocal monitors.
- Place monitors at 45 degrees from the front of microphone rejection zone.
- Use cardioid or hypercardioid microphones for better rejection.
- If feedback starts, reduce gain first, then identify and notch the frequency.
- Never exceed +3 dB boost on any frequency band in monitor EQ.

### Monitor Mixing
- Vocals should be the loudest element in vocal monitors.
- Drummer needs kick and snare in their monitor, plus any click track.
- Bass player needs kick drum and their own bass.
- Guitar players need their vocals and a bit of the other guitar (if applicable).
- Keys player often needs the most elements — full band at lower level.
- Keep monitor reverb minimal — it causes feedback and muddiness.
- Each monitor send is pre-fader on the Wing Rack (/$ch/N/send/1-16/level).

### Safety Rules
- **Maximum Preamp Gain**: Never exceed +50 dB of preamp gain. If more is needed,
  check microphone connection and cable.
- **True Peak Limit**: -1 dBTP on all outputs. Use the Wing limiter on main bus.
- **SPL Monitoring**: Target 95-100 dBA for general live music. Never exceed
  105 dBA sustained without hearing protection provisions.
- **Noise Gate Best Practice**: Do not set gate thresholds too high — missed
  notes are worse than a slightly higher noise floor.
- **Emergency Mute**: Have main bus mute accessible at all times. On the Wing,
  this is /$main/1/mute.
- **Limiter on Outputs**: Always engage limiters on main L/R and any monitor/IEM
  outputs. On the Wing, use /$main/1/dyn/comp as a limiter (ratio 10:1 or higher,
  threshold at -3 dBFS, fast attack 0.1 ms).
