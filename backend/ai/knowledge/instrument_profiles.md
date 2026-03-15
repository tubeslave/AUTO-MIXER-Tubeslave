# Instrument Profiles for Live Sound Mixing

Comprehensive profiles for all 24 instrument classes used by AUTO-MIXER.
Each profile contains frequency range, key frequencies, dynamics, and
recommended processing settings.

## Lead Vocal

- **Frequency Range**: 80 Hz - 12 kHz (fundamental), harmonics to 16 kHz
- **Key Frequencies**:
  - 80-100 Hz: Proximity effect rumble (cut)
  - 200-300 Hz: Warmth/body (cut if muddy, +1 dB if thin)
  - 800-1200 Hz: Nasal/honk (cut 2-3 dB if present)
  - 2-4 kHz: Presence and intelligibility (boost 2-3 dB)
  - 5-7 kHz: Sibilance zone (de-ess or notch)
  - 10-14 kHz: Air and breathiness (gentle shelf boost)
- **Dynamic Range**: 20-35 dB (highly variable, depends on singer)
- **Gate**: Not recommended (lose quiet passages)
- **Compressor**: Ratio 3:1, threshold -20 to -16 dB, attack 8-15 ms, release 80-150 ms
- **EQ**: HPF 80 Hz 18 dB/oct, cut 250 Hz -2 dB Q=1.5, boost 3 kHz +2 dB Q=1.0
- **Pan**: Center (0.0) always
- **Common Mics**: Shure SM58, Sennheiser e945, Shure Beta 58A, Neumann KMS 105

## Backing Vocal

- **Frequency Range**: 100 Hz - 12 kHz
- **Key Frequencies**:
  - 200-400 Hz: Proximity/mud (cut more aggressively than lead)
  - 1-3 kHz: Presence (boost slightly less than lead to sit behind)
  - 5-7 kHz: Sibilance (de-ess)
  - 10+ kHz: Air (boost 1 dB for blend)
- **Dynamic Range**: 15-25 dB
- **Gate**: Optional, threshold -50 dB with slow release
- **Compressor**: Ratio 3:1, threshold -18 dB, attack 10 ms, release 100 ms
- **EQ**: HPF 100 Hz 18 dB/oct, cut 250 Hz -3 dB, boost 3.5 kHz +1.5 dB
- **Pan**: Spread L/R for multiple BVs (-0.4 to +0.4)
- **Common Mics**: Shure SM58, Sennheiser e835, Audio-Technica AE4100

## Kick Drum

- **Frequency Range**: 30 Hz - 10 kHz
- **Key Frequencies**:
  - 40-60 Hz: Sub/weight (boost 2-4 dB for thump)
  - 60-100 Hz: Fundamental punch
  - 200-400 Hz: Boxiness/cardboard (cut 3-5 dB)
  - 2-4 kHz: Beater click/attack (boost 2-4 dB)
  - 5-8 kHz: High-end snap
- **Dynamic Range**: 15-25 dB
- **Gate**: Threshold -30 dB, attack 0.5 ms, hold 100 ms, release 50 ms, range -40 dB
- **Compressor**: Ratio 4:1, threshold -16 dB, attack 15-25 ms, release 50-80 ms
- **EQ**: No HPF, boost 60 Hz +3 dB Q=1.0, cut 350 Hz -4 dB Q=2.0, boost 4 kHz +3 dB Q=1.2
- **Pan**: Center (0.0)
- **Common Mics**: Shure Beta 52A, AKG D112, Audix D6, Sennheiser e902

## Snare Drum

- **Frequency Range**: 80 Hz - 12 kHz
- **Key Frequencies**:
  - 100-200 Hz: Body and weight (boost 2 dB for fatness)
  - 400-800 Hz: Ring/boxiness (sweep and cut 2-3 dB)
  - 900-1200 Hz: Ping/ring (narrow cut if ring is excessive)
  - 4-6 kHz: Snap/crack (boost 2-3 dB for cut)
  - 8-10 kHz: Brightness/air
- **Dynamic Range**: 15-25 dB (ghost notes to rimshots)
- **Gate**: Threshold -25 dB, attack 0.5 ms, hold 80 ms, release 40 ms, range -30 dB
- **Compressor**: Ratio 3:1, threshold -14 dB, attack 5-10 ms, release 80-120 ms
- **EQ**: HPF 80 Hz, boost 200 Hz +2 dB, cut 800 Hz -2 dB Q=2.0, boost 5 kHz +3 dB
- **Pan**: Center or slight right (+0.05 to +0.1, audience perspective)
- **Common Mics**: Shure SM57 (top), Shure SM57 or SM81 (bottom, phase inverted)

## Hi-Hat

- **Frequency Range**: 200 Hz - 16 kHz
- **Key Frequencies**:
  - 200-500 Hz: Low-end bleed from kit (cut aggressively)
  - 2-4 kHz: Stick definition
  - 6-10 kHz: Shimmer and presence (boost 1-2 dB)
  - 10-14 kHz: Air and sizzle
- **Dynamic Range**: 10-20 dB (closed to open)
- **Gate**: Not recommended (continuous source)
- **Compressor**: Ratio 2:1, threshold -12 dB, attack 5 ms, release 50 ms
- **EQ**: HPF 200 Hz 24 dB/oct, cut 400 Hz -3 dB, boost 8 kHz +2 dB
- **Pan**: Slight right (+0.25 to +0.35, audience perspective)
- **Common Mics**: AKG C451, Shure SM81, Neumann KM 184

## Overhead (Drum Kit)

- **Frequency Range**: 100 Hz - 20 kHz
- **Key Frequencies**:
  - 100-300 Hz: Low bleed from toms/kick (HPF to taste)
  - 400-600 Hz: Muddy kit sound (cut 2 dB)
  - 3-6 kHz: Cymbal body
  - 8-12 kHz: Cymbal shimmer (boost 1-2 dB)
  - 12-16 kHz: Air/detail
- **Dynamic Range**: 15-25 dB
- **Gate**: Not recommended (ambient mics)
- **Compressor**: Ratio 2:1, threshold -14 dB, attack 20 ms, release 150 ms (gentle)
- **EQ**: HPF 120-200 Hz, cut 400 Hz -2 dB, shelf boost 10 kHz +2 dB
- **Pan**: Hard left and right (L=-0.8, R=+0.8)
- **Common Mics**: AKG C451 (pair), Rode NT5 (pair), Neumann KM 184 (pair)

## Rack Tom

- **Frequency Range**: 60 Hz - 8 kHz
- **Key Frequencies**:
  - 80-150 Hz: Fundamental resonance (boost 2 dB at resonant freq)
  - 300-500 Hz: Boxiness (cut 3 dB)
  - 3-5 kHz: Attack/stick definition (boost 2 dB)
- **Dynamic Range**: 20-30 dB
- **Gate**: Threshold -28 dB, attack 0.5 ms, hold 120 ms, release 60 ms, range -40 dB
- **Compressor**: Ratio 3:1, threshold -14 dB, attack 10-15 ms, release 80 ms
- **EQ**: HPF 80 Hz, boost 120 Hz +2 dB, cut 400 Hz -3 dB, boost 4 kHz +2 dB
- **Pan**: Left-center (-0.2 to -0.4, audience perspective, high to low)
- **Common Mics**: Sennheiser e604, Sennheiser MD 421, Shure SM57

## Floor Tom

- **Frequency Range**: 50 Hz - 6 kHz
- **Key Frequencies**:
  - 60-100 Hz: Fundamental/weight (boost 2-3 dB)
  - 200-400 Hz: Boxiness (cut 3 dB)
  - 3-5 kHz: Attack (boost 2 dB)
- **Dynamic Range**: 20-30 dB
- **Gate**: Threshold -28 dB, attack 0.5 ms, hold 150 ms, release 80 ms, range -40 dB
- **Compressor**: Ratio 3:1, threshold -14 dB, attack 10-15 ms, release 80 ms
- **EQ**: HPF 60 Hz, boost 80 Hz +3 dB, cut 350 Hz -3 dB, boost 3.5 kHz +2 dB
- **Pan**: Right-center (+0.3 to +0.5, audience perspective)
- **Common Mics**: Sennheiser e602, Sennheiser MD 421, AKG D112

## Bass Guitar

- **Frequency Range**: 30 Hz - 5 kHz (fundamental), harmonics to 8 kHz
- **Key Frequencies**:
  - 30-50 Hz: Sub-bass rumble (cut or control)
  - 60-100 Hz: Fundamental weight (boost 2 dB)
  - 150-250 Hz: Low-mid body
  - 250-500 Hz: Mud zone (cut 2-3 dB)
  - 600-1000 Hz: Growl and note definition (boost 1-2 dB)
  - 2-4 kHz: String noise/fret buzz (cut if present)
- **Dynamic Range**: 20-35 dB (very dynamic)
- **Gate**: Not recommended
- **Compressor**: Ratio 3:1 to 4:1, threshold -18 dB, attack 10-20 ms, release 100-200 ms
- **EQ**: HPF 30 Hz (only sub-sonic), boost 80 Hz +2 dB, cut 250 Hz -2 dB, boost 700 Hz +1.5 dB
- **Pan**: Center (0.0) always
- **Common DI/Mics**: Radial J48 DI, Countryman Type 85, Sennheiser e906 (amp)

## Electric Guitar (Clean)

- **Frequency Range**: 80 Hz - 12 kHz
- **Key Frequencies**:
  - 80-150 Hz: Rumble (HPF)
  - 200-400 Hz: Warmth/body
  - 800-1200 Hz: Mid-range character
  - 2-3 kHz: Presence (boost 2 dB for cut-through)
  - 5-8 kHz: Brightness/pick attack
- **Dynamic Range**: 15-25 dB
- **Gate**: Not typically needed
- **Compressor**: Ratio 3:1, threshold -16 dB, attack 15-25 ms, release 100-150 ms
- **EQ**: HPF 80 Hz, cut 300 Hz -2 dB if muddy, boost 2.5 kHz +2 dB
- **Pan**: Off-center (-0.3 or +0.3), hard pan if two guitars
- **Common Mics**: Shure SM57, Sennheiser e906, Royer R-121

## Electric Guitar (Distorted)

- **Frequency Range**: 80 Hz - 10 kHz (amp filtering limits bandwidth)
- **Key Frequencies**:
  - 80-150 Hz: Low-end mud (HPF aggressively)
  - 300-500 Hz: Thickness (genre dependent)
  - 800-1500 Hz: Midrange growl
  - 2-4 kHz: Presence/bite (boost or cut to taste)
  - 5-7 kHz: Fizz (often cut 1-2 dB)
- **Dynamic Range**: 5-15 dB (self-compressing)
- **Gate**: Optional, threshold -45 dB for inter-song noise
- **Compressor**: Often unnecessary. If used, ratio 2:1, gentle
- **EQ**: HPF 80-100 Hz, cut 400 Hz -2 dB, shape 2 kHz to taste
- **Pan**: Off-center or hard pan (-0.6 to +0.6)
- **Common Mics**: Shure SM57, Sennheiser e906, Shure SM7B

## Acoustic Guitar

- **Frequency Range**: 80 Hz - 14 kHz
- **Key Frequencies**:
  - 80-150 Hz: Body resonance/boominess (HPF or cut)
  - 200-300 Hz: Warmth (cut if fighting vocal)
  - 2-3 kHz: Clarity and pick attack (boost 2 dB)
  - 5-7 kHz: String presence
  - 10-14 kHz: Sparkle and air (boost 1.5 dB)
- **Dynamic Range**: 15-30 dB (fingerpicking to strumming)
- **Gate**: Not recommended
- **Compressor**: Ratio 2:1 to 3:1, threshold -18 dB, attack 15-25 ms, release 100-200 ms
- **EQ**: HPF 80 Hz, cut 200 Hz -3 dB, boost 3 kHz +2 dB, shelf 10 kHz +1.5 dB
- **Pan**: Off-center (+0.3 or -0.3)
- **Common Mics**: DPA 4099, AKG C451, Neumann KM 184, DI from undersaddle pickup

## Keys / Piano (Stereo)

- **Frequency Range**: 27 Hz (A0) - 14 kHz
- **Key Frequencies**:
  - 30-60 Hz: Rumble from pedal/stage vibration (HPF if not playing low register)
  - 200-400 Hz: Mud/boominess (cut 2 dB)
  - 1-3 kHz: Definition and attack
  - 5-8 kHz: Brightness
  - 10+ kHz: Air
- **Dynamic Range**: 25-40 dB (very dynamic instrument)
- **Gate**: Not recommended
- **Compressor**: Ratio 2:1 to 3:1, threshold -18 dB, attack 15-25 ms, release 100-200 ms
- **EQ**: HPF 40-60 Hz, cut 250 Hz -2 dB, boost 3 kHz +1.5 dB
- **Pan**: Stereo — low notes left, high notes right (L=-0.3, R=+0.3) or centered mono
- **Common Sources**: DI (stereo pair), Radial ProD2 stereo DI

## Organ

- **Frequency Range**: 16 Hz (pedals) - 10 kHz
- **Key Frequencies**:
  - 16-60 Hz: Pedal notes (powerful sub-bass, manage carefully)
  - 100-300 Hz: Body and warmth
  - 500-1000 Hz: Midrange character
  - 2-4 kHz: Presence
  - 5-8 kHz: Key click (characteristic, preserve)
- **Dynamic Range**: 10-20 dB (volume controlled by player)
- **Gate**: Not recommended
- **Compressor**: Ratio 2:1, threshold -16 dB, attack 20 ms, release 150 ms
- **EQ**: HPF 40 Hz (only if no pedal parts), cut 300 Hz -2 dB, boost 3 kHz +1.5 dB
- **Pan**: Off-center (+0.2) or stereo spread for Leslie effect
- **Common Sources**: DI from keyboard, mic on Leslie cabinet (Shure SM57)

## Synthesizer

- **Frequency Range**: 20 Hz - 20 kHz (full range instrument)
- **Key Frequencies**: Depends entirely on patch — treat each patch differently
  - Bass synth: Manage sub (30-60 Hz), boost note (80-200 Hz)
  - Pad synth: Cut lows, shape mids, control high end
  - Lead synth: Treat like a vocal for frequency space
- **Dynamic Range**: 5-20 dB (often controlled by player/sequencer)
- **Gate**: Not typically needed
- **Compressor**: Depends on patch. Pads: gentle 2:1. Bass: 3:1-4:1. Lead: 3:1
- **EQ**: HPF based on lowest note needed
- **Pan**: Center for bass synth, spread for pads, off-center for leads
- **Common Sources**: DI (stereo or mono)

## Strings (Violin, Viola, Cello)

- **Frequency Range**: Violin 200 Hz-10 kHz, Viola 130 Hz-8 kHz, Cello 65 Hz-6 kHz
- **Key Frequencies**:
  - Fundamental range varies by instrument
  - 200-500 Hz: Body resonance
  - 1-3 kHz: Bow attack and presence
  - 3-5 kHz: Rosin/scratch (cut if harsh)
  - 7-10 kHz: Air and overtones
- **Dynamic Range**: 25-40 dB
- **Gate**: Not recommended
- **Compressor**: Ratio 2:1, threshold -18 dB, attack 15-20 ms, release 150 ms
- **EQ**: HPF 80-150 Hz (depending on instrument), cut harshness at 2-4 kHz, boost 8 kHz +1 dB
- **Pan**: Section spread (violins left, violas center-left, cellos center-right)
- **Common Mics**: DPA 4099, Neumann KM 184, clip-on condenser

## Brass (Trumpet, Trombone, Saxophone)

- **Frequency Range**: Trumpet 180 Hz-10 kHz, Trombone 60 Hz-8 kHz, Sax 100 Hz-10 kHz
- **Key Frequencies**:
  - 100-300 Hz: Fundamental warmth
  - 500-1000 Hz: Body and tone
  - 1-3 kHz: Bite and presence
  - 3-5 kHz: Brightness/harshness (cut 1-2 dB for fatigue)
  - 6-10 kHz: Air
- **Dynamic Range**: 25-40 dB (brass is extremely dynamic)
- **Gate**: Not recommended
- **Compressor**: Ratio 3:1, threshold -16 dB, attack 10-20 ms, release 100 ms
- **EQ**: HPF 80 Hz, cut 3 kHz -2 dB if harsh, boost 1 kHz +1 dB for warmth
- **Pan**: Spread across stereo field matching stage position
- **Common Mics**: Shure SM57, Sennheiser e906, AKG C414

## Percussion (Congas, Bongos, Tambourine, Shaker)

- **Frequency Range**: 60 Hz - 16 kHz (varies by instrument)
- **Key Frequencies**:
  - Congas: 100-200 Hz (slap), 3-5 kHz (attack)
  - Bongos: 200-400 Hz (tone), 4-6 kHz (attack)
  - Tambourine: 2-4 kHz (jingle body), 8-12 kHz (shimmer)
  - Shaker: 4-10 kHz (main energy)
- **Dynamic Range**: 15-25 dB
- **Gate**: Not typically needed; optional on hand drums
- **Compressor**: Ratio 2:1, threshold -14 dB, attack 5-15 ms, release 80 ms
- **EQ**: HPF 80-200 Hz depending on source, boost attack frequencies
- **Pan**: Off-center, matching stage position (-0.3 to +0.5)
- **Common Mics**: Shure SM57, AKG C451, Sennheiser e604

## Harmonica

- **Frequency Range**: 200 Hz - 8 kHz
- **Key Frequencies**:
  - 200-500 Hz: Body
  - 1-2 kHz: Mid-range presence
  - 2-4 kHz: Brightness
  - 5-8 kHz: Air and overtones
- **Dynamic Range**: 15-25 dB
- **Gate**: Not recommended
- **Compressor**: Ratio 3:1, threshold -16 dB, attack 10 ms, release 100 ms
- **EQ**: HPF 200 Hz, cut 400 Hz -2 dB, boost 2 kHz +2 dB
- **Pan**: Off-center (-0.2 to +0.2)
- **Common Mics**: Shure SM57 (on amp), Shure Green Bullet (cupped), vocal mic

## Banjo

- **Frequency Range**: 100 Hz - 12 kHz
- **Key Frequencies**:
  - 100-200 Hz: Resonator body
  - 300-500 Hz: Boxiness (cut)
  - 2-4 kHz: Pick attack and clarity
  - 6-10 kHz: String brightness
- **Dynamic Range**: 15-25 dB
- **Gate**: Not typically needed
- **Compressor**: Ratio 2:1, threshold -16 dB, attack 10-20 ms, release 100 ms
- **EQ**: HPF 100 Hz, cut 400 Hz -2 dB, boost 3 kHz +2 dB
- **Pan**: Off-center (+0.3)
- **Common Mics**: AKG C451, Neumann KM 184, Shure SM81

## Mandolin

- **Frequency Range**: 150 Hz - 12 kHz
- **Key Frequencies**:
  - 150-300 Hz: Body (cut if boomy)
  - 1-2 kHz: Chop/percussive attack
  - 3-5 kHz: Presence and clarity
  - 8-12 kHz: Sparkle
- **Dynamic Range**: 15-25 dB
- **Gate**: Not recommended
- **Compressor**: Ratio 2:1, threshold -16 dB, attack 10-15 ms, release 100 ms
- **EQ**: HPF 120 Hz, cut 250 Hz -2 dB, boost 3.5 kHz +2 dB
- **Pan**: Off-center (-0.2 or +0.2)
- **Common Mics**: DPA 4099, AKG C451, Neumann KM 184

## Upright/Double Bass

- **Frequency Range**: 40 Hz - 8 kHz
- **Key Frequencies**:
  - 40-80 Hz: Fundamental (boost 2 dB for weight)
  - 100-250 Hz: Body
  - 250-500 Hz: Mud (cut 2-3 dB)
  - 600-1000 Hz: Note definition and bow attack
  - 2-4 kHz: String noise/detail
- **Dynamic Range**: 25-35 dB
- **Gate**: Not recommended
- **Compressor**: Ratio 3:1, threshold -18 dB, attack 15-20 ms, release 150 ms
- **EQ**: HPF 35 Hz, boost 70 Hz +2 dB, cut 300 Hz -2 dB, boost 800 Hz +1.5 dB
- **Pan**: Center (0.0)
- **Common Mics/DI**: DPA 4099 clip-on, Realist pickup, Fishman Full Circle

## Click Track / Playback

- **Frequency Range**: Full range (20 Hz - 20 kHz)
- **Key Frequencies**: N/A — line-level stereo source
- **Dynamic Range**: Fixed by source (typically 0-6 dB)
- **Gate**: Not applicable
- **Compressor**: Not typically needed on playback
- **EQ**: Flat — no EQ unless compensating for PA response
- **Pan**: Stereo as supplied, or mono center for click
- **Notes**: Click track must NEVER be routed to FOH. Only to monitor/IEM buses.
  Use a dedicated stereo DI or the Wing's USB playback input.

## Audience / Ambience Mics

- **Frequency Range**: 20 Hz - 20 kHz
- **Key Frequencies**: Full range capture — minimal processing
- **Dynamic Range**: 40+ dB
- **Gate**: Not recommended
- **Compressor**: Gentle 2:1 to control crowd peaks, threshold -12 dB
- **EQ**: HPF 60 Hz (remove HVAC/traffic rumble), gentle rolloff above 14 kHz
- **Pan**: Stereo pair, hard left and right (L=-1.0, R=+1.0)
- **Notes**: Keep faders down during performance, bring up between songs or for
  recording ambiance. Useful for broadcast and recording.
- **Common Mics**: Small-diaphragm condenser pair (Rode NT5, AKG C451)
