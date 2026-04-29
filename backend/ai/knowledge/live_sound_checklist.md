# Live Sound Checklist

Comprehensive checklists for every phase of a live concert, from load-in
to teardown. Designed for FOH engineers using the Behringer Wing Rack
with AUTO-MIXER automation.

## Pre-Show Setup Checklist

### Physical Setup (Before Power On)
- Verify all XLR cables are connected and labeled at both ends (stage and mixer)
- Check that all DI boxes are in place and set to correct ground lift position
- Confirm microphone stands are positioned correctly per the stage plot
- Verify all wireless mic receivers are racked and antenna cables connected
- Check that monitor wedges or IEM transmitters are positioned and cabled
- Confirm main PA is rigged and cabled (L/R/Sub, confirm polarity markings)
- Ensure UPS (uninterruptible power supply) is connected for mixer and critical gear
- Check that all power conditioners are on and reading correct voltage (118-122V for US)
- Verify network cables for Dante/AES50 stage boxes are connected and redundant

### Power On Sequence
1. Power on audio snake / stage box first (AES50 or Dante units)
2. Power on Behringer Wing Rack — allow 30 seconds for full boot
3. Power on wireless receivers and transmitters
4. Power on effects processors and outboard gear
5. Power on amplifiers / powered speakers LAST (PA, monitors)
6. Verify Wing Rack network connectivity (ping mixer IP)
7. Launch AUTO-MIXER backend and confirm OSC handshake
8. Verify WebSocket connection to frontend is established

### Mixer Configuration
- Load the appropriate show file or scene for the band
- Verify input patch: every channel is receiving the correct source
- Confirm all output routing: main L/R, subs, monitors, recording feeds
- Set all channel faders to -infinity before line check
- Verify all main and bus limiters are engaged (safety limiters)
- Confirm talkback microphone is working and routed correctly
- Check that automation scenes are loaded and sequenced if needed
- Verify clock source: internal 48 kHz or external word clock
- Confirm DCA group assignments match the mixing workflow

### Patch Verification (Channel by Channel)
For each input channel on the Wing Rack:
1. Confirm source routing matches stage plot
2. Verify phantom power setting (on for condensers, off for dynamics/ribbons)
3. Check that channel name and color are set correctly
4. Confirm DCA group assignments match the mixing workflow
5. Verify bus sends are configured (monitor sends pre-fader, FX sends post-fader)
6. Ensure HPF is engaged and set to appropriate frequency for the source
7. Confirm that channel EQ, comp, and gate have starting positions from the template

### Gain Setting Procedure (Ring-Out)
For each channel, with musician playing or speaking at performance level:
1. Set preamp gain so peaks read -18 to -12 dBFS on the channel meter
2. Verify there is no clipping indicator on the preamp stage
3. If using a pad, engage it for hot sources (drum mics near cymbals, loud amps)
4. Set channel fader to 0 dB (unity)
5. Check that the signal sounds clean — no distortion, hum, or buzz
6. Document gain settings for recall during the show

### Monitor Ring-Out Procedure
For each monitor send (wedge or IEM):
1. Turn up the monitor send with the vocal mic first
2. Slowly increase level until feedback begins
3. Identify the feedback frequency using RTA or ear training
4. Apply a narrow notch filter (Q=8-12, depth -3 to -6 dB) at that frequency
5. Continue raising level and notching until 3-4 notches are placed
6. Mark the maximum safe level for that monitor mix
7. Repeat for other instruments in that monitor mix
8. Document all monitor EQ settings

## Line Check Procedure

The line check verifies every input is working and properly routed.
Do this systematically from channel 1 through the last input.

### Order of Line Check
1. **Kick drum** — engineer says "kick please," drummer plays kick only
2. **Snare** — top and bottom separately, verify phase between them
3. **Hi-hat**
4. **Toms** — high to low (rack 1, rack 2, floor)
5. **Overheads** — verify stereo image and phase coherence with close mics
6. **Bass guitar** — DI first, then amp mic, verify phase alignment
7. **Electric guitars** — each guitar separately
8. **Acoustic guitars**
9. **Keys / Piano** — verify stereo routing if applicable
10. **Other instruments** (horns, strings, percussion)
11. **Backing tracks / playback** — verify stereo and click track routing
12. **Backing vocals** — each mic individually
13. **Lead vocal** — last (or first if time is limited and vocalist is available)

### During Each Line Check
- Verify signal appears on the correct channel
- Listen for buzz, hum, crackle, or other artifacts
- Check that phantom power is correct (no pops on engagement)
- Verify mute group and DCA assignments
- Briefly check HPF and basic EQ
- Confirm monitor send reaches the correct wedge/IEM

## Sound Check Order

Sound check builds the mix layer by layer. This order ensures each
element is heard in context.

### Phase 1: Drums
1. Kick drum alone — set level, basic EQ, gate, compression
2. Snare alone — set level, basic EQ, gate, compression, check phase with kick
3. Kick + Snare together — verify balance and low-end interaction
4. Add overheads — set level, HPF, blend with close mics. Check phase.
5. Pan Overhead L/R first so kick and snare leakage remains centered.
6. Add toms, hi-hat, ride, and other drum spot mics. Pan each one where it is
   heard in the already-panned overhead image.
7. Full kit — refine balance, verify in main PA

### Phase 2: Bass
1. Bass guitar alone — set level, EQ, compression
2. Bass + Kick drum — verify low-end separation and complement
3. Bass + Full drums — refine bass level in context

### Phase 3: Guitars and Keys
1. Each guitar individually — set level, EQ, pan position
2. Keys/Piano — set level, EQ, pan
3. All rhythm section together — guitars + keys + bass + drums
4. Refine balance, check for frequency masking between instruments

### Phase 4: Vocals
1. Lead vocal alone — set level, EQ chain, compression, de-esser
2. Lead vocal + band — adjust vocal level to sit properly above instruments
3. Each backing vocal individually
4. All backing vocals together — adjust blend and pan positions
5. Full band + all vocals — final balance adjustments

### Phase 5: Effects
1. Add vocal reverb — adjust send level, decay time, pre-delay
2. Add instrument reverbs/delays
3. Verify effects levels in full-band context
4. Check that effects are not muddying the mix

### Phase 6: Full Band Run-Through
1. Ask the band to play a representative song (verse, chorus, bridge)
2. Walk the venue — check sound from different positions
3. Adjust main system EQ if needed (graphic or parametric on output)
4. Verify subwoofer level and crossover point
5. Check monitor levels with the full band playing

## Show Mixing Workflow

### Verse/Chorus Management
- Vocals typically need +1 to +2 dB boost going into choruses
- Guitar levels may increase in choruses — be ready to manage
- Cymbal wash increases in choruses — may need to duck overheads slightly
- Bass level usually remains constant — it anchors the mix
- Use DCA groups for quick section-level adjustments:
  - DCA 1: All drums
  - DCA 2: Bass
  - DCA 3: Guitars
  - DCA 4: Keys
  - DCA 5: Lead vocal
  - DCA 6: Backing vocals
  - DCA 7: Effects returns
  - DCA 8: Playback / click

### Dynamic Song Sections
- **Intro/Outro**: Often quieter — pull instruments back, let featured parts shine
- **Verse**: Vocal is primary — everything else supports
- **Pre-Chorus**: Building energy — gradually raise guitars and drums
- **Chorus**: Full energy — all elements at peak. Vocal must still be above.
- **Bridge**: Often a contrast — may need dramatic level changes
- **Solo**: Feature the solo instrument (+2-3 dB), slightly reduce other mid-range
  instruments to give it space

### Real-Time Adjustments
- Watch fader positions — if most are above +5 dB, the mix is too quiet overall.
  Pull everything down 3 dB and raise the main bus.
- Watch gain reduction meters on compressors — if GR exceeds 10 dB consistently,
  the threshold is too low.
- Keep an eye on main bus meters — target -6 to -3 dBFS peaks
- Listen for feedback constantly — especially during quiet passages and monitor checks

## Between-Song Protocols

### Quick Check (15-30 seconds)
1. Reset any song-specific effects (e.g., special delay, distortion on vocal)
2. Bring vocal effects back to default reverb settings
3. Check main bus meters — are we at a good level?
4. Mute any channels that were only used for the previous song
5. Unmute any channels needed for the next song
6. If the setlist is known, recall the next scene/snapshot

### Extended Break (Between Sets)
1. Play walk-in music at appropriate level
2. Mute all stage channels
3. Check for any issues reported by musicians on monitors
4. Review main bus limiter — has it been hitting too hard?
5. Adjust system EQ if the room has changed (audience adds absorption)
6. Check wireless mic battery levels (replace if below 40%)
7. Reset all effects to default show settings

## Emergency Procedures

### Feedback Emergency
1. **Immediately** reduce the master fader 3-5 dB
2. Identify the offending channel (look for the channel with highest gain reduction
   or use RTA to spot the frequency)
3. Mute the offending channel
4. Apply a narrow notch at the feedback frequency (-6 to -12 dB, Q=10)
5. Slowly unmute and raise the channel
6. If feedback returns, apply deeper cut or reduce the channel's monitor send

### Signal Loss (No Audio from a Channel)
1. Check if the channel is muted or fader is down
2. Check if DCA group or mute group is engaged
3. Verify the preamp is receiving signal (check input meter)
4. If no signal at preamp, check cable connection at stage box
5. Try a known-good cable
6. Try a known-good mic or DI
7. Re-patch to a spare input channel if available
8. If the issue is on the musician's end (dead battery, broken cable), mute the
   channel and wait for them to resolve it

### Power Failure
1. Do NOT panic — UPS should keep mixer alive for 5-15 minutes
2. If PA goes silent, communicate with stage manager
3. If mixer is on UPS, save the current scene immediately
4. When power returns, wait 30 seconds before powering on amplifiers
5. Verify all connections and clock sync before resuming
6. Check that all scenes and automation recalled correctly

### Excessive SPL / Safety
1. If SPL readings exceed 105 dBA sustained, reduce main fader immediately
2. If a single channel is causing excessive SPL (e.g., runaway feedback),
   mute it immediately
3. If limiters are engaging constantly, the overall level is too high
4. Communicate with production manager about SPL requirements
5. Document any SPL limit exceedances for compliance purposes

### Wireless Mic Dropout
1. Check if the transmitter is still powered on (battery may be dead)
2. Check RF signal strength on the receiver
3. If intermittent, suspect antenna placement or interference
4. Switch to backup frequency if available
5. As a last resort, replace with a wired mic

## Post-Show Teardown

### Power Down Sequence (Reverse of Power On)
1. Fade main fader to -infinity
2. Mute all outputs
3. Power off amplifiers and powered speakers FIRST
4. Power off outboard effects and processors
5. Power off wireless receivers
6. Save show file on the Wing Rack
7. Power off the Wing Rack mixer
8. Power off stage boxes
9. Disconnect and coil all cables neatly

### Documentation
- Save show file with date and venue name
- Note any issues encountered (problem channels, feedback frequencies)
- Note wireless frequencies used (for future coordination)
- Document any changes made from the default template
- Note any equipment failures or concerns
- Record AUTO-MIXER session log for analysis

## Monitor Mixing Considerations

### IEM (In-Ear Monitor) Specific
- Always use a limiter on IEM bus outputs — maximum +10 dBu to protect hearing
- Target 85 dBA average in IEM mixes (OSHA hearing safety guidelines)
- Provide ambient mics (crowd/room) in IEM mixes for natural feel
- Click track must be in a separate bus or clearly in one ear only
- IEM mixes are typically mono per musician — check before assuming stereo
- Bass player and drummer should have click and each other in their mix
- Provide a "more me" bus control if possible

### Wedge Monitor Specific
- Maximum 2-3 open mics per wedge mix to prevent feedback
- Wedge EQ should be ringed out before the show (see Ring-Out procedure)
- Vocal monitors: voice + a little guitar/keys for pitch reference
- Drum fill: kick, snare, vocals, and bass
- Side fills: broader mix for keyboard players or horn sections
- Never point a wedge directly at a microphone — angle at 45 degrees minimum
- Keep wedge levels as low as possible while meeting musician needs

### Common Monitor Requests and Solutions
- "More me": Increase their channel in their mix by 2-3 dB
- "Less band": Reduce everything except their channel by 2-3 dB
- "More low end": Boost below 200 Hz in their monitor send, or adjust the wedge EQ
- "It sounds harsh": Roll off high end above 6 kHz in their monitor send
- "I can't hear myself": Often caused by excessive stage volume — address source

## RF Frequency Management

### Wireless Microphone Coordination
- Survey the RF environment before the show using a spectrum analyzer
- Avoid TV broadcast frequencies (check local TV channel assignments)
- Maintain minimum 250 kHz spacing between wireless channels
- Use intermodulation calculation software for 4+ simultaneous systems
- Keep spare frequencies documented for emergency re-tuning
- Check that all transmitters are on the correct frequency group/channel
- Replace batteries before every show — do not rely on remaining charge
- Always carry spare batteries for every wireless system

### Antenna Placement
- Use directional (paddle) antennas when possible, pointed at the stage
- Mount antennas at least 1 meter above the floor
- Keep antennas away from metal surfaces and other RF sources
- Use antenna distribution systems for 4+ receivers
- Maximum cable run from antenna to receiver: 15 meters with low-loss cable
- Place antennas at stage left and stage right for diversity

## Stage Volume Control

### Managing Stage Volume
- The single biggest challenge in live sound. Excessive stage volume
  undermines the FOH mix.
- Guitar amp volume: Ask guitarists to point amps across the stage (not at audience)
  or use amp shields/isolation cabinets.
- Drum volume: Drum screens (Perspex shields) help but do not solve the problem.
  Electronic drum triggers can supplement mic'ed acoustic drums.
- Monitor levels: Start low and build up. Musicians tend to ask for "more"
  continuously — resist creeping levels.
- IEM adoption: In-ear monitors dramatically reduce stage volume. Encourage adoption.
- Communication: Talk to the band about stage volume before the show.
  Explain that lower stage volume = better FOH mix = better audience experience.

## Venue-Specific Adjustments

### Indoor Small Club (Under 300 capacity)
- Reduce sub bass (-3 dB below 60 Hz on mains)
- Room reflections are strong: increase HPF frequencies on vocals
- Lower overall SPL target: 95 dBA
- Reverb may not be needed — room provides natural ambience
- Stage volume is the dominant challenge

### Indoor Theater / Auditorium (300-2000 capacity)
- Standard SPL target: 100 dBA
- More reverb from room; reduce FX send levels
- Watch for slap-back echo from back wall
- Check for standing waves in the bass region
- PA delay towers or fill speakers may be needed for coverage

### Outdoor Festival Stage
- Increase sub bass (+3 dB below 80 Hz to compensate for no room gain)
- Increase high-frequency (+2 dB above 8 kHz for air absorption over distance)
- Higher SPL target: 103-105 dBA (competition from ambient noise)
- Wind protection on all microphones is essential
- Monitor mixes need more level due to open-air dispersion
- Subwoofer cardioid arrays to reduce stage low-end bleed

### House of Worship
- Conservative dynamics (less compression, more natural feel)
- Speech intelligibility is the top priority
- Lower SPL target: 85-90 dBA average
- Wide stereo image for choir and congregation
- Longer reverb tails acceptable for hymns and worship songs
