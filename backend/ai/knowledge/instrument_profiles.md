# Instrument Profiles for Live Sound Mixing

Detailed per-instrument profiles with frequency ranges, EQ moves, compression settings, and gate parameters. All values are professional starting points for live concert mixing on the Behringer Wing Rack.

## Kick Drum

### Frequency Ranges
- **Fundamental**: 40-80 Hz. The deep "thud" and weight of the kick.
- **Body**: 80-200 Hz. Fullness and power.
- **Boxiness**: 300-500 Hz. Almost always needs a cut. This is where cardboard-like resonance lives.
- **Beater attack**: 2-5 kHz. The "click" of the beater hitting the head.
- **Air/presence**: 6-10 kHz. Modern pop/rock kick brightness.

### EQ Moves
- HPF: 30 Hz, 18 dB/oct
- Boost: 60 Hz, +3 dB, Q=1.5 (sub weight)
- Cut: 400 Hz, -4 dB, Q=2.0 (remove boxiness)
- Boost: 2.5 kHz, +2 dB, Q=1.5 (beater click)
- Shelf: 8 kHz, +1.5 dB (air, optional)

### Compression
- Ratio: 4:1
- Threshold: -12 dB
- Attack: 20 ms (lets beater transient through for punch)
- Release: 80 ms (recovers before next hit at 150 BPM)
- Detector: Peak
- Knee: Medium (2)

### Gate
- Threshold: -30 dB
- Attack: 0.5 ms (capture full transient)
- Hold: 50 ms
- Release: 200 ms (match kick drum decay)
- Range: 40 dB (full gate closure)

---

## Snare Drum

### Frequency Ranges
- **Fundamental**: 150-250 Hz. The body and weight of the snare.
- **Boxiness**: 600-900 Hz. Cardboard resonance to cut.
- **Crack**: 3-5 kHz. The sharp stick-on-head transient.
- **Snare wires**: 8-12 kHz. The sizzle and shimmer of the snare wires.

### EQ Moves
- HPF: 80 Hz, 18 dB/oct
- Boost: 200 Hz, +2 dB, Q=1.5 (body)
- Cut: 800 Hz, -3 dB, Q=2.0 (boxiness)
- Boost: 3.5 kHz, +2.5 dB, Q=1.5 (crack)
- Shelf: 10 kHz, +2 dB (wire sizzle)

### Compression
- Ratio: 4.5:1
- Threshold: -10 dB
- Attack: 5 ms
- Release: 100 ms
- Detector: Peak
- Knee: Medium (2)

### Gate
- Threshold: -28 dB
- Attack: 0.5 ms
- Hold: 40 ms
- Release: 150 ms
- Range: 30 dB

---

## Toms (Floor Tom, Rack Toms)

### Frequency Ranges
- **Fundamental**: 60-120 Hz (floor tom) / 100-200 Hz (rack toms)
- **Body**: 120-300 Hz
- **Boxiness**: 300-600 Hz
- **Attack**: 2-4 kHz. Stick impact definition.
- **Air**: 5-8 kHz

### EQ Moves
- HPF: 60 Hz (floor tom) / 80 Hz (rack toms)
- Boost: 100 Hz, +2.5 dB, Q=1.5 (fundamental body)
- Cut: 400 Hz, -3.5 dB, Q=2.0 (boxiness)
- Boost: 3 kHz, +2 dB, Q=1.5 (stick attack)

### Compression
- Ratio: 4:1
- Threshold: -12 dB
- Attack: 10 ms
- Release: 100 ms
- Detector: Peak
- Knee: Medium (2)

### Gate
- Threshold: -32 dB
- Attack: 1 ms
- Hold: 60 ms
- Release: 250 ms
- Range: 40 dB

---

## Hi-Hat

### Frequency Ranges
- **Fundamental ring**: 300-600 Hz (usually bleed from other drums)
- **Body**: 600-2 kHz
- **Definition**: 3-6 kHz. The character of the hi-hat sound.
- **Shimmer**: 8-16 kHz. Brightness and air.

### EQ Moves
- HPF: 200 Hz, 18 dB/oct (aggressive — remove all drum bleed)
- Cut: 400 Hz, -4 dB, Q=1.0 (bleed cleanup)
- Boost: 6 kHz, +2 dB, Q=1.5 (brightness)
- Shelf: 10 kHz, +1.5 dB (shimmer)

### Compression
- Ratio: 3:1
- Threshold: -18 dB
- Attack: 5 ms
- Release: 80 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -36 dB
- Attack: 0.3 ms
- Hold: 30 ms
- Release: 100 ms
- Range: 20 dB

---

## Overheads (Drum Kit)

### Frequency Ranges
- **Low bleed**: 80-250 Hz. Contains kick and tom bleed.
- **Kit body**: 250-2 kHz. The overall tonality of the kit.
- **Cymbal definition**: 2-6 kHz. Where cymbal character lives.
- **Air/sparkle**: 8-16 kHz. Brightness and life of cymbals.

### EQ Moves
- HPF: 100-150 Hz, 18 dB/oct
- Cut: 250 Hz, -2 dB, Q=1.0 (reduce low bleed)
- Boost: 3 kHz, +1.5 dB, Q=1.0 (cymbal clarity)
- Shelf: 12 kHz, +2 dB (air and sparkle)

### Compression
- Ratio: 3:1
- Threshold: -20 dB
- Attack: 10 ms
- Release: 150 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Not recommended. Overheads should capture the full kit ambience.

---

## Bass Guitar (DI + Amp)

### Frequency Ranges
- **Sub**: 30-60 Hz. Deep fundamental rumble.
- **Fundamental**: 60-150 Hz. The primary note weight.
- **Mud**: 200-350 Hz. Common problem area.
- **Growl**: 500-900 Hz. Midrange character and string definition.
- **Attack**: 1.5-3 kHz. Finger and pick articulation.
- **Fret noise**: 3-5 kHz. String noise (usually undesirable).

### EQ Moves
- HPF: 30 Hz, 18 dB/oct
- Boost: 80 Hz, +2 dB, Q=1.5 (weight)
- Cut: 250 Hz, -3 dB, Q=2.0 (mud cleanup)
- Boost: 700 Hz, +1.5 dB, Q=1.5 (growl)
- Boost: 2.5 kHz, +2 dB, Q=1.5 (attack clarity)

### Compression
- Ratio: 4:1
- Threshold: -15 dB
- Attack: 25 ms (preserve pick/finger transient)
- Release: 200 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -40 dB
- Attack: 2 ms
- Hold: 80 ms
- Release: 300 ms
- Range: 15 dB (gentle — don't fully close on bass)

---

## Electric Guitar

### Frequency Ranges
- **Low-end boom**: 80-200 Hz. Cabinet resonance; usually too much.
- **Body/crunch**: 400-1 kHz. The core tone and character.
- **Presence/bite**: 2-4 kHz. Cut-through in the mix.
- **Harshness**: 5-7 kHz. Can be piercing; often needs taming.
- **Air**: 8-12 kHz. Rarely useful for electric guitar.

### EQ Moves
- HPF: 100 Hz, 18 dB/oct
- Cut: 200 Hz, -2 dB, Q=1.5 (reduce cabinet boom)
- Boost: 800 Hz, +1.5 dB, Q=1.5 (body/crunch)
- Boost: 3 kHz, +2 dB, Q=1.5 (presence)
- Cut: 6 kHz, -1.5 dB, Q=2.0 (harshness taming)

### Compression
- Ratio: 3:1
- Threshold: -14 dB
- Attack: 15 ms
- Release: 180 ms
- Detector: RMS
- Knee: Medium (2)
- Note: Tube amps already compress the signal. Use gentle compression.

### Gate
- Threshold: -45 dB
- Attack: 2 ms
- Hold: 60 ms
- Release: 250 ms
- Range: 10 dB (gentle — prevent noise during pauses, preserve sustain)

---

## Acoustic Guitar

### Frequency Ranges
- **Boominess**: 80-200 Hz. Body resonance amplified by soundhole.
- **Warmth**: 200-400 Hz. Fullness of the instrument.
- **Nasal**: 500-1 kHz. Mid-range honk.
- **String clarity**: 2-4 kHz. Pick/strum articulation.
- **Sparkle**: 8-16 kHz. String shimmer and brightness.

### EQ Moves
- HPF: 80 Hz, 18 dB/oct
- Cut: 100 Hz, -2 dB, Q=1.5 (boominess from body)
- Cut: 250 Hz, -1.5 dB, Q=2.0 (mud reduction)
- Boost: 3 kHz, +2 dB, Q=1.5 (string clarity)
- Shelf: 10 kHz, +2 dB (sparkle)

### Compression
- Ratio: 3:1
- Threshold: -18 dB
- Attack: 15 ms
- Release: 150 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -42 dB (conservative — avoid cutting off quiet finger-picked passages)
- Attack: 2 ms
- Hold: 80 ms
- Release: 300 ms
- Range: 10 dB

---

## Lead Vocal

### Frequency Ranges
- **Proximity effect**: 80-200 Hz. Low-end buildup from close mic technique.
- **Chest resonance**: 200-400 Hz. Warmth and body.
- **Nasal/honk**: 600-1 kHz. The most common vocal problem frequency.
- **Presence**: 2-4 kHz. Intelligibility and cut-through. Human ear is most sensitive here.
- **Sibilance**: 4-8 kHz. "S" and "T" sounds.
- **Air**: 8-16 kHz. Breath, openness, intimacy.

### EQ Moves
- HPF: 80 Hz, 18 dB/oct
- Cut: 200 Hz, -2 dB, Q=1.5 (proximity compensation)
- Cut: 800 Hz, -1.5 dB, Q=2.0 (nasal reduction)
- Boost: 3 kHz, +2.5 dB, Q=1.5 (presence for intelligibility)
- Shelf: 10 kHz, +2 dB (air and breath)

### Compression
- Ratio: 3:1
- Threshold: -18 dB (aim for 3-6 dB of gain reduction during loud passages)
- Attack: 10 ms (fast enough to control dynamics, slow enough to preserve articulation)
- Release: 120 ms
- Detector: RMS
- Knee: Soft (3)

### Gate
- Threshold: -40 dB
- Attack: 1 ms
- Hold: 80 ms
- Release: 300 ms
- Range: 10 dB (gentle — never fully gate a vocal)

---

## Backing Vocals

### Frequency Ranges
Similar to lead vocal but processed for blend rather than prominence.

### EQ Moves
- HPF: 100 Hz, 18 dB/oct (higher than lead — less proximity warmth needed)
- Cut: 200 Hz, -3 dB, Q=1.5 (proximity cleanup)
- Cut: 800 Hz, -2 dB, Q=2.0 (nasal reduction)
- Boost: 3.5 kHz, +2 dB, Q=1.5 (blend presence)
- Shelf: 8 kHz, +1.5 dB (air)

### Compression
- Ratio: 3.5:1 (slightly harder than lead for more consistent blend)
- Threshold: -20 dB
- Attack: 8 ms
- Release: 100 ms
- Detector: RMS
- Knee: Soft (3)

### Gate
- Threshold: -36 dB
- Attack: 1 ms
- Hold: 60 ms
- Release: 250 ms
- Range: 15 dB

---

## Keyboard / Synth

### Frequency Ranges
- **Sub bass** (synth bass): 30-80 Hz
- **Low body**: 80-250 Hz. Pad warmth.
- **Mid presence**: 500-2 kHz. Melodic content.
- **Brightness**: 3-8 kHz. Attack and definition.
- **Air**: 10-16 kHz. Shimmer.

### EQ Moves
- HPF: 30-60 Hz (depends on patch — higher for pads, lower for synth bass)
- Cut: 300 Hz, -2 dB, Q=1.5 (mud reduction)
- Boost: 2 kHz, +1.5 dB, Q=1.0 (presence)
- Shelf: 8 kHz, +1 dB (brightness)

### Compression
- Ratio: 2.5:1
- Threshold: -16 dB
- Attack: 15 ms
- Release: 150 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Not typically needed. Synths have clean signal paths.

---

## Piano (Live/DI)

### Frequency Ranges
- **Low register**: 27-200 Hz. Bass notes fundamental.
- **Body/warmth**: 200-500 Hz. Fullness of the instrument.
- **Clarity**: 1-3 kHz. Note definition and hammer attack.
- **Brilliance**: 4-8 kHz. Upper harmonics and presence.
- **Sparkle**: 8-16 kHz. String shimmer.

### EQ Moves
- HPF: 40 Hz, 18 dB/oct
- Cut: 250 Hz, -2 dB, Q=1.5 (reduce mud in live room)
- Boost: 1 kHz, +1 dB, Q=1.5 (body)
- Boost: 5 kHz, +2 dB, Q=1.0 (clarity)
- Shelf: 10 kHz, +1.5 dB (sparkle)

### Compression
- Ratio: 2.5:1
- Threshold: -20 dB
- Attack: 20 ms (preserve hammer transient)
- Release: 200 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Not typically needed. Use only if stage bleed is severe.

---

## Accordion

### Frequency Ranges
- **Low body**: 80-250 Hz. Bellows resonance.
- **Mid body**: 250-1 kHz. Core tone of the reeds.
- **Reed clarity**: 2-5 kHz. Individual reed articulation.
- **Reed harshness**: 5-8 kHz. Can be piercing on certain notes.
- **Air**: 8-12 kHz.

### EQ Moves
- HPF: 60 Hz, 18 dB/oct
- Cut: 200 Hz, -2 dB, Q=1.5 (reduce boom)
- Boost: 1 kHz, +1 dB, Q=1.5 (body)
- Boost: 3.5 kHz, +2 dB, Q=1.5 (reed clarity)
- Cut: 8 kHz, -1 dB, Q=1.0 (tame harsh reeds)

### Compression
- Ratio: 3:1
- Threshold: -16 dB
- Attack: 10 ms
- Release: 150 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -42 dB
- Range: 10 dB

---

## Trumpet

### Frequency Ranges
- **Low body**: 150-400 Hz.
- **Warmth/body**: 400-1.5 kHz.
- **Brilliance**: 2-5 kHz. The bright, cutting character.
- **Harshness**: 5-8 kHz. Can be piercing when played loudly.
- **Air**: 8-12 kHz.

### EQ Moves
- HPF: 120 Hz, 18 dB/oct
- Cut: 300 Hz, -1.5 dB, Q=1.5 (boxiness)
- Boost: 1.5 kHz, +1.5 dB, Q=1.5 (body warmth)
- Boost: 5 kHz, +2 dB, Q=1.5 (brilliance)
- Cut: 8 kHz, -1 dB, Q=2.0 (harshness taming)

### Compression
- Ratio: 3:1
- Threshold: -14 dB
- Attack: 5 ms (fast — trumpet dynamics can be extreme)
- Release: 120 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -38 dB
- Attack: 1 ms
- Hold: 60 ms
- Release: 200 ms
- Range: 15 dB

---

## Saxophone

### Frequency Ranges
- **Low body**: 100-300 Hz (varies by type: soprano higher, baritone lower).
- **Honk/nasal**: 250-500 Hz.
- **Body/character**: 500-1.5 kHz. Core tone.
- **Brightness/edge**: 2-5 kHz. Presence and cut.
- **Air/breath**: 6-10 kHz. Breath sound and overtones.

### EQ Moves
- HPF: 80 Hz (alto/tenor) / 40 Hz (baritone) / 150 Hz (soprano)
- Cut: 250 Hz, -2 dB, Q=1.5 (honk reduction)
- Boost: 800 Hz, +1.5 dB, Q=1.5 (body)
- Boost: 3 kHz, +2 dB, Q=1.5 (presence)
- Shelf: 8 kHz, +1 dB (air)

### Compression
- Ratio: 3:1
- Threshold: -16 dB
- Attack: 8 ms
- Release: 140 ms
- Detector: RMS
- Knee: Medium (2)

### Gate
- Threshold: -40 dB
- Attack: 1 ms
- Hold: 70 ms
- Release: 250 ms
- Range: 12 dB

---

## Playback / Backing Track

### Frequency Ranges
Pre-mixed stereo content. Full frequency range 20 Hz - 20 kHz.

### EQ Moves
- HPF: 20 Hz (only remove subsonic content)
- Minimal EQ recommended — the track is already mixed
- Optional: gentle cut at 200 Hz, -1 dB, Q=0.7 (reduce low-mid buildup with live bass)

### Compression
- Not recommended. Playback tracks are already mastered with compression applied.
- Exception: If the track has extreme dynamic range, use ratio 2:1, threshold -10 dB.

### Gate
- Not needed. Signal is clean and continuous.
