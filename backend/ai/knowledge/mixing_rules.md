# Comprehensive Mixing Rules for Live Sound

This document contains professional mixing rules and best practices for live concert mixing, specifically tailored for the Behringer Wing Rack digital mixer platform. All values are based on established audio engineering standards and professional live sound literature.

## Gain Staging

Proper gain staging is the foundation of a clean mix. Every stage in the signal chain must operate within its optimal range to maximize signal-to-noise ratio while preventing clipping.

### K-System Metering Standards

The K-System, developed by mastering engineer Bob Katz, provides three metering scales for different applications:

- **K-20 (Music mixing)**: 0 dBFS = +20 dB above reference. Target integrated loudness: -20 LUFS. This is the primary standard for live concert mixing. The meter should hover around the 0 mark (which corresponds to -20 dBFS). Peaks should rarely exceed +12 on the K-20 scale (which is -8 dBFS). This provides 20 dB of headroom for transients and dynamics.

- **K-14 (Broadcast/Film)**: 0 dBFS = +14 dB above reference. Target: -14 LUFS. Used for broadcast work where less dynamic range is acceptable.

- **K-12 (Commercial music)**: 0 dBFS = +12 dB above reference. Target: -12 LUFS. Not recommended for live sound as it leaves insufficient headroom.

### Input Gain Procedure

1. Start with all faders at unity (0 dB) and all channel processing bypassed.
2. Have the performer play or sing at their expected performance level.
3. Adjust the preamp gain until the input meter averages around -20 to -18 dBFS.
4. Peaks should not exceed -6 dBFS at the preamp stage.
5. On the Behringer Wing, trim control range is -18 dB to +18 dB.

### Headroom Management

- Maintain at least 12 dB of headroom on the master bus at all times.
- Individual channels should average -20 LUFS with peaks no higher than -6 dBFS.
- Bus masters should operate at or below unity (0 dB).
- DCA/VCA groups should start at unity and be used for broad mix moves.

### Digital vs. Analog Gain Staging

On digital consoles like the Wing Rack:
- The analog-to-digital converter clips at 0 dBFS — this is a hard wall, not soft saturation.
- Internal processing on the Wing uses 40-bit floating point, providing virtually unlimited internal headroom.
- The critical clip points are: (1) the preamp/ADC stage, and (2) the DAC output stage.
- Always leave headroom at both conversion stages.

## EQ by Instrument

### Subtractive vs. Additive EQ

The golden rule: cut narrow, boost wide. Use subtractive EQ to remove problems before reaching for additive boosts.

- **Cuts**: Use a narrow Q (2.0-8.0) to surgically remove resonances, mud, or problem frequencies.
- **Boosts**: Use a wider Q (0.5-1.5) to gently enhance desired character.
- **Maximum recommended boost**: +6 dB for live sound. Anything beyond this suggests a mic placement or source issue.
- **Shelf EQ**: Use low and high shelves for broad tonal shaping; parametric bands for surgical work.

### Frequency Ranges by Character

Understanding frequency character helps diagnose problems quickly:

- **20-60 Hz**: Sub-bass. Felt more than heard. Important for kick and bass. Roll off on everything else.
- **60-250 Hz**: Bass/warmth. Fundamental range for most instruments. Excess here causes mud.
- **250-500 Hz**: Low-midrange. "Boxy" and "muddy" frequencies. Common problem area in live rooms.
- **500-2000 Hz**: Midrange. Body and presence of most instruments. Critical for vocal intelligibility.
- **2000-4000 Hz**: Upper midrange. Presence and attack. Human ear is most sensitive here. Harshness lives here.
- **4000-8000 Hz**: Presence/brilliance. Sibilance (4-8 kHz), cymbal definition, string attack.
- **8000-20000 Hz**: Air/sparkle. Adds openness and life. Roll off above 12 kHz on sources that don't need it.

### Kick Drum EQ

- **Sub punch** (50-80 Hz): Boost 2-4 dB, Q=1.5. The "chest thump."
- **Mud/boxiness** (300-500 Hz): Cut 3-5 dB, Q=2.0. Almost always needs a cut here.
- **Beater attack** (2-4 kHz): Boost 2-3 dB, Q=1.5. Provides click and definition.
- **Air** (8-10 kHz): Gentle shelf boost 1-2 dB for modern pop/rock kick sounds.
- **HPF**: 30 Hz, 18 dB/oct. Removes sub-rumble below the useful fundamental.

### Snare Drum EQ

- **Body/fatness** (150-250 Hz): Boost 2-3 dB, Q=1.5. Adds weight to the snare.
- **Boxiness** (600-900 Hz): Cut 2-4 dB, Q=2.0. Removes cardboard-like character.
- **Crack/attack** (3-5 kHz): Boost 2-4 dB, Q=1.5. The "snap" of the stick on the head.
- **Sizzle** (8-12 kHz): Shelf boost 1-3 dB for snare wire shimmer and brightness.
- **HPF**: 80 Hz to remove kick bleed.

### Vocal EQ (Lead)

- **Proximity effect** (100-250 Hz): Cut 2-4 dB, Q=1.5. Compensates for close-mic proximity boost.
- **Nasal/honk** (600-900 Hz): Cut 1-3 dB, Q=2.0. Most common vocal problem frequency.
- **Presence** (2-4 kHz): Boost 2-4 dB, Q=1.0-1.5. Critical for intelligibility and cut-through.
- **Air** (8-12 kHz): Shelf boost 2-3 dB for breath and openness.
- **Sibilance** (5-8 kHz): If problematic, narrow cut 2-4 dB at the specific sibilant frequency, or use de-esser.
- **HPF**: 80-100 Hz at 18 dB/oct.

### Bass Guitar/Bass DI EQ

- **Sub weight** (60-100 Hz): Boost 1-3 dB for power. Be careful not to overdo it.
- **Mud** (200-350 Hz): Cut 2-4 dB, Q=2.0. Cleans up the low-mid region.
- **Growl/definition** (600-900 Hz): Boost 1-3 dB for midrange presence and note definition.
- **String attack** (2-3 kHz): Boost 1-2 dB for finger/pick attack clarity.
- **HPF**: 30-40 Hz. Only remove sub-rumble; preserve the low fundamental.

### Electric Guitar EQ

- **Boom** (150-250 Hz): Cut 2-3 dB to reduce low-end buildup from amps.
- **Body/crunch** (600-1000 Hz): Gentle boost 1-2 dB for midrange character.
- **Presence/bite** (2-4 kHz): Boost 2-3 dB for cut-through in a dense mix.
- **Harshness** (5-7 kHz): Cut 1-2 dB if the amp is overly bright.
- **HPF**: 100 Hz. Electric guitar fundamentals start at ~82 Hz (low E string).

### Acoustic Guitar EQ

- **Boominess** (80-150 Hz): Cut 2-4 dB. Acoustic bodies amplify low frequencies excessively.
- **Mud** (200-350 Hz): Cut 1-3 dB for clarity.
- **String clarity** (2-4 kHz): Boost 2-3 dB for pick/strum definition.
- **Sparkle** (8-12 kHz): Shelf boost 2-3 dB for string shimmer and brightness.
- **HPF**: 80 Hz.

## Compression Settings

### Compression Principles for Live Sound

Live compression serves three purposes:
1. **Dynamics control**: Reduce the difference between quiet and loud passages.
2. **Tonal shaping**: Slower attack lets transients through for punch; faster attack controls them.
3. **Level consistency**: Keep sources at a predictable level in the mix.

### Attack and Release Guidelines

- **Fast attack** (0.1-5 ms): Catches transients immediately. Good for controlling dynamics but reduces punch. Use on vocals and bass for consistent level.
- **Medium attack** (5-20 ms): Lets the initial transient through, then compresses. Ideal for drums where you want punch plus control.
- **Slow attack** (20-50 ms): Preserves most transient character. Good for acoustic instruments and gentle vocal compression.
- **Auto release**: Many modern compressors, including the Wing, offer program-dependent release. This is often the best starting point for live sound.

### Per-Instrument Compression

**Kick drum**: Ratio 4:1, threshold -10 to -15 dB, attack 15-25 ms (let beater transient through), release 60-100 ms (fast enough to recover before next hit at 120 BPM).

**Snare drum**: Ratio 4:1 to 6:1, threshold -8 to -12 dB, attack 3-8 ms, release 80-120 ms. Faster attack than kick to control rimshot dynamics.

**Toms**: Ratio 3:1 to 4:1, threshold -10 to -14 dB, attack 8-15 ms, release 80-120 ms.

**Bass guitar**: Ratio 4:1, threshold -12 to -18 dB, attack 20-40 ms (preserve finger/pick attack), release 150-250 ms.

**Lead vocal**: Ratio 3:1 to 4:1, threshold -15 to -20 dB, attack 5-15 ms, release 80-150 ms. Most important channel to compress — keeps vocal present and audible.

**Backing vocals**: Ratio 3:1 to 4:1, threshold -18 to -22 dB, attack 5-10 ms, release 80-120 ms. Can be compressed harder than leads for consistent blend.

**Electric guitar**: Ratio 2:1 to 3:1, threshold -12 to -16 dB, attack 10-20 ms, release 150-250 ms. Amps already compress the signal; be gentle.

**Acoustic guitar**: Ratio 2:1 to 3:1, threshold -16 to -20 dB, attack 10-20 ms, release 120-180 ms.

**Keys/Synth**: Ratio 2:1 to 3:1, threshold -14 to -18 dB, attack 10-20 ms, release 120-200 ms.

## Panning Conventions

### Standard Live Stage Layout (Audience Perspective)

Panning should reflect the physical stage layout from the audience's perspective:

- **Center (0)**: Lead vocals, bass, kick, snare. These form the core of the mix and should be mono-compatible.
- **Slight off-center (10-30)**: Hi-hat (slightly left in drummer's perspective from audience), ride (slightly right).
- **Moderate pan (30-50)**: Guitars (typically one left, one right if two guitars), keyboards, tom fills spread L-to-R.
- **Wide (50-80)**: Backing vocals (spread across stereo field), string sections, horn sections.
- **Extreme (80-100)**: Rarely used in live sound due to PA system design. Only for special effects.

### Tom Panning

From audience perspective (drummer facing audience):
- Floor tom: 30-40 right
- Mid tom: 10-20 right
- High tom: 10-20 left
- Rack toms follow a left-to-right (high-to-low) spread

### Overhead Panning

- Left overhead: Pan 70-100 left
- Right overhead: Pan 70-100 right
- This creates the natural stereo image of the drum kit

## Reverb and Effects

### Reverb Types by Application

- **Plate reverb**: Tight, bright, smooth. Ideal for vocals and snare. Decay 1.0-2.0 seconds.
- **Hall reverb**: Spacious, natural. Good for overall ambience. Decay 1.5-3.0 seconds.
- **Room reverb**: Short, natural. Adds depth without washing out clarity. Decay 0.3-1.0 seconds.
- **Chamber reverb**: Warm, colored. Good for vocals and acoustic instruments. Decay 0.8-1.5 seconds.

### Send Levels (General Starting Points)

- Lead vocal: -10 to -6 dB send to plate reverb
- Snare: -15 to -10 dB send to plate reverb
- Toms: -18 to -12 dB send to room or hall reverb
- Acoustic guitar: -15 to -10 dB send to hall reverb
- Never send kick, bass, or sub-heavy sources to reverb (muddies the low end)

### Effects Rules

1. Use pre-fader sends for monitor mixes, post-fader for effects.
2. High-pass the reverb return at 200-300 Hz to keep the low end clean.
3. In a reverberant venue, reduce reverb sends. In a dry venue, increase them.
4. Delay (slapback, 80-120 ms) can add depth without the wash of reverb.
5. De-esser before reverb on vocal sends to prevent sibilant reverb artifacts.

## Dynamics Processing

### Gating Guidelines

Gates are essential for reducing bleed between close-miked drum sources:

- **Threshold**: Set just below the quietest intentional hit. Too high = missed hits; too low = bleed passes through.
- **Attack**: As fast as possible (0.1-1 ms) for drums to capture the full transient.
- **Hold**: Long enough to sustain the note — 30-80 ms for drums, 50-100 ms for vocals.
- **Release**: Should match the natural decay — 100-250 ms for drums, 200-400 ms for vocals.
- **Range**: 20-40 dB for drums (fully close the gate). 6-15 dB for vocals (gentle gain reduction, not full closure).

### When NOT to Gate

- Overheads and room mics (you want the full kit sound)
- Acoustic instruments (natural dynamics are part of the sound)
- Sustained synth pads
- Pre-mixed playback tracks
- Sources with very dynamic performance (gating can cut off quiet passages)

## Bus and Group Processing

### Subgroup Strategy

- **Drum bus**: Gentle compression (2:1, -20 dB threshold) for glue. Parallel compression for punch (heavy 10:1, blended 20-30% wet).
- **Vocal bus**: Light compression (2:1, -18 dB threshold) for consistent vocal blend. De-esser on the bus rather than individual channels if multiple vocals.
- **Instrument bus**: Very gentle compression if any (2:1, -24 dB threshold). Mostly for level control.
- **Master bus**: Limiter only, set at -1 dBFS. No master bus compression in live sound — it reduces your ability to make mix moves.

### DCA/VCA Groups

Use DCA groups for performance-level control without altering the internal gain structure:
- DCA 1: All drums
- DCA 2: All vocals
- DCA 3: All instruments
- DCA 4: Effects returns
- This allows quick scene-level adjustments during a show.

## Level Management

### Mix Balance Starting Points

These are typical starting fader positions for a rock/pop band (relative to lead vocal at unity):

- Lead vocal: 0 dB (reference)
- Kick: -3 to -6 dB below vocal
- Snare: -6 to -8 dB below vocal
- Bass: -3 to -6 dB below vocal
- Electric guitar: -6 to -10 dB below vocal
- Acoustic guitar: -8 to -12 dB below vocal
- Keys/synth: -8 to -12 dB below vocal
- Backing vocals: -6 to -10 dB below lead vocal
- Overheads: -12 to -18 dB below vocal
- Toms: -10 to -15 dB below vocal (unmuted only when playing)

### Loudness Target for FOH

For front-of-house concert mixing:
- Standard rock/pop show: 95-100 dBA SPL (A-weighted, slow)
- Acoustic/jazz: 85-95 dBA SPL
- Corporate/speech: 80-90 dBA SPL
- Festival main stage: 100-105 dBA SPL
- Note: Always comply with local noise ordinances and venue limits.
