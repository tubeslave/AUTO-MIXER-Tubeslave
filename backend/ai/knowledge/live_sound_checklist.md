# Live Sound Checklist — Complete Show Procedure

A comprehensive pre-show, soundcheck, performance, and post-show procedure for live concerts using the Behringer Wing Rack digital mixer. Follow this checklist systematically to ensure a reliable show.

## Pre-Show System Checks (Before Musicians Arrive)

### Power and Signal Path
1. Verify all power distribution is on and stable (UPS if available).
2. Confirm Wing Rack is powered on and firmware is current.
3. Verify Dante/AES50/USB connections are active and synced.
4. Check clock source and sample rate (48 kHz standard for live).
5. Verify all stage boxes and splitters are powered and connected.
6. Test headphone/monitor outputs from the console.
7. Confirm AUTO-MIXER Tubeslave software is running and connected to the Wing via OSC.
8. Verify WebSocket connection between frontend and backend is active.

### FOH PA System
1. Power on amplifiers/powered speakers in correct sequence (processors first, then amps).
2. Run pink noise through main L/R at low level to verify all cabinets are working.
3. Check for any rattles, buzzes, or distortion from cabinets.
4. Verify delay towers/fills are timed correctly if applicable.
5. Confirm system EQ/processing is loaded and correct for the venue.
6. Walk the venue to check coverage — listen for dead spots and excessive reflections.

### Monitor System
1. Verify all monitor wedges/IEM transmitters are powered on.
2. Send pink noise to each monitor send individually to confirm routing.
3. Label all monitor mixes clearly on the console.
4. Have spare IEM receivers and earpieces ready.
5. Check IEM RF frequencies for conflicts with other wireless systems.

### Console Preparation
1. Load the show file or start from a clean scene.
2. Verify routing: all inputs assigned to correct channels.
3. Set all faders to -infinity initially.
4. Reset all EQ, dynamics, and effects to default/bypassed state.
5. Label all channels with source names and colors.
6. Set up DCA groups: Drums, Vocals, Instruments, Effects.
7. Assign talkback microphone and verify routing.
8. Set up effects returns (reverbs, delays) on aux channels.
9. Save the initial scene as a backup before making any changes.

### AUTO-MIXER Tubeslave Preparation
1. Verify the YAML configuration file matches the current show layout.
2. Confirm channel assignments in the config match the input list.
3. Check that the AI knowledge base is indexed and responding.
4. Enable or disable feedback detection based on the show type.
5. Verify Dante routing configuration matches the physical patching (64-channel layout).
6. Test OSC communication: send a query to `/ch/1/name` and confirm response.
7. Confirm the `/xremote` keepalive is running (every 8 seconds).

## Line Check Procedure

The line check verifies every signal path from stage to console before the musician soundcheck. Each channel is tested individually.

### Line Check Steps
1. Start at channel 1 and work through all inputs sequentially.
2. For each channel:
   a. Ask the stage tech to scratch/tap the mic or play the instrument.
   b. Verify signal appears on the correct channel meter on the Wing.
   c. Check the signal is clean (no buzz, hum, crackle, or intermittent).
   d. Verify phase is correct (especially for multi-mic sources like drums).
   e. Set initial gain: peaks at -18 to -12 dBFS.
   f. Enable HPF at the default frequency for the instrument type.
3. Check DI boxes: verify pad switches and ground lifts as needed.
4. Test wireless systems: walk the stage to check for dropouts.
5. Verify 48V phantom power is on only for condenser mics (NOT for ribbons or dynamics that don't need it).

### Common Line Check Issues
- No signal: Check cable, stage box channel, Wing input routing (`/ch/{n}/in/conn/grp` and `/ch/{n}/in/conn/in`).
- Hum/buzz: Check ground lift on DI, cable shielding, power isolation.
- Intermittent signal: Bad cable or connector — replace immediately.
- Phase issues: Flip phase on one mic (`/ch/{n}/in/set/inv`); choose the position with more low end.
- Wrong channel: Re-patch at stage box or reassign in Wing routing.

## Channel-by-Channel Soundcheck Order

Soundcheck is done one instrument at a time in a specific order designed to build the mix from the foundation up.

### 1. Kick Drum
- Have the drummer play kick only at performance level.
- Set gain for peaks at -12 to -6 dBFS.
- Enable HPF at 30 Hz.
- Apply kick EQ preset: cut 400 Hz, boost 60 Hz and 2.5 kHz.
- Set gate: threshold -30 dB, attack 0.5 ms, hold 50 ms, release 200 ms.
- Set compressor: ratio 4:1, threshold -12 dB, attack 20 ms, release 80 ms.
- Bring fader to -6 dB as starting point.

### 2. Snare Drum
- Snare only at performance level (regular hits AND rimshots).
- Set gain, HPF at 80 Hz.
- Apply snare EQ preset.
- Set gate: threshold -28 dB, attack 0.5 ms.
- Set compressor: ratio 4.5:1, threshold -10 dB.
- Check phase against kick — both should sound full together.

### 3. Toms
- Each tom individually, then all toms together.
- Set gain, HPF at 60-80 Hz (depending on tom size).
- Apply tom EQ preset.
- Set gates (toms are the most critical for gating due to bleed).
- Set compressor.
- Pan toms across stereo field (audience perspective).

### 4. Hi-Hat
- Hi-hat at performance level (open and closed).
- Set gain, HPF at 200 Hz (aggressive filtering).
- Apply hi-hat EQ preset.
- Optional gate.
- Pan slightly left (audience perspective).

### 5. Overheads
- Full kit playing while adjusting overheads.
- Set gain, HPF at 100-150 Hz.
- Check phase between overheads (should be equidistant from snare).
- Light compression.
- Pan hard left and right for stereo image.

### 6. Room Mic (if used)
- Full kit playing.
- Blend room mic in for depth and ambience.
- Heavy compression optional for "room sound" effect.

### 7. Bass
- Bass player plays typical riff at performance level.
- Set gain, HPF at 30 Hz.
- Apply bass EQ preset.
- Set compressor: ratio 4:1, attack 25 ms.
- Listen with kick drum together — they must not fight.
- If kick and bass are masking each other, EQ one to emphasize low-end and the other mid-bass.

### 8. Guitars
- Each guitar individually at performance level (clean AND distorted tones).
- Set gain, HPF at 100 Hz.
- Apply guitar EQ preset.
- Set compressor (gentle for already-compressed amp signals).
- Pan guitars to opposite sides if two guitars.

### 9. Keys / Synth
- All patches the musician will use during the show.
- Set gain, HPF as appropriate for the patches.
- Check both quiet pads and loud leads for level consistency.
- Apply compression for level control.

### 10. Vocals (Most Important)
- Lead vocal first, then backing vocals.
- Have the vocalist sing at full performance level AND speak between songs.
- Set gain for peaks at -12 to -6 dBFS.
- HPF at 80 Hz (lead) / 100 Hz (backing).
- Apply vocal EQ preset.
- Set compressor: ratio 3:1, threshold -18 dB.
- This is the most critical channel — spend the most time here.
- Check for feedback by slowly increasing the fader with monitors active.

### 11. Full Band
- Have the entire band play a song section together.
- Balance faders for a cohesive mix.
- Adjust EQ to resolve masking issues between instruments.
- Set overall volume to target SPL.
- Check stereo image and panning.
- Verify monitor mixes with musicians.
- Save the soundcheck scene as a snapshot on the Wing.

## Monitor Setup

### In-Ear Monitors (IEM)
1. Start with a safe, low level to protect musicians' hearing.
2. Build the monitor mix one instrument at a time (same order as FOH soundcheck).
3. Add a small amount of ambient/room mic for natural feel.
4. Limit the maximum output level of the IEM transmitter.
5. Always have a limiter on IEM sends to prevent dangerous levels.
6. Verify stereo panning in IEM mixes if using stereo monitors.

### Wedge Monitors
1. Ring out each wedge: slowly raise the vocal mic level until feedback starts, then notch EQ the feedback frequency.
2. Repeat until you achieve maximum gain before feedback.
3. Build the monitor mix starting with the musician's own instrument/voice.
4. Keep monitor levels as low as possible — less stage volume = cleaner FOH mix.
5. Use graphic EQ on bus outputs for monitor ring-out on the Wing.

## System Measurement (Optional)

### RTA / FFT Measurement
1. Place measurement microphone at mix position (FOH).
2. Play pink noise through the PA system.
3. Measure frequency response using RTA (Real-Time Analyzer).
4. Apply system EQ corrections on the system processor (not the console).
5. Target a smooth, gently falling response from low to high frequencies.

### SPL Monitoring
1. Set up SPL meter at FOH position.
2. Target levels:
   - Rock/pop: 95-100 dBA slow
   - Acoustic/jazz: 85-95 dBA slow
   - Corporate/speech: 80-90 dBA slow
   - Festival main stage: 100-105 dBA slow
3. Monitor throughout the show and adjust master fader as needed.
4. Watch for local noise ordinance limits.

## During the Show

### Active Mixing Checklist
1. Keep lead vocal consistently audible above the band.
2. Watch meters — no channel should be clipping.
3. Monitor master bus headroom (maintain at least 6 dB).
4. Listen for feedback and react immediately with notch EQ or fader pullback.
5. Adjust mix for venue fill as audience arrives (bodies absorb sound, especially high frequencies).
6. Ride vocal fader for dynamics (verse quieter than chorus).
7. Mute unused channels between songs to reduce noise floor.
8. Check IEM/monitor levels with musicians between songs if possible.
9. Watch for instrument changes (acoustic to electric guitar, different vocal mic, etc.) and adjust processing.
10. Monitor AUTO-MIXER Tubeslave dashboard for alerts and signal warnings.

### Song Transition Checklist
1. Mute channels not needed for the next song.
2. Unmute channels needed for the next song.
3. Recall scene/snapshot if programmed for song-specific settings.
4. Verify effects sends are appropriate for the next song's tempo and style.
5. Reset any manual fader rides to starting positions.

### Emergency Procedures
- **Feedback**: Immediately pull down the offending channel fader, then apply notch EQ at the feedback frequency. Slowly bring fader back up. If AUTO-MIXER feedback detection is active, it should catch this automatically.
- **Clip/distortion**: Reduce input gain or fader. Check for internal clip points (EQ boost stacking, compressor makeup gain).
- **Signal loss**: Check cables, stage box, routing. Have backup cables ready. Verify Wing input routing via OSC: `/ch/{n}/in/conn/grp`.
- **Power failure**: If UPS is available, gracefully shut down. Never hot-swap audio during a show.
- **OSC connection lost**: AUTO-MIXER will attempt reconnection automatically. If persistent, verify network connectivity and restart the OSC client.
- **Wireless dropout**: Switch to backup wireless unit or wired mic. Always have a wired backup for critical channels (lead vocal).

## Post-Show Procedures

### Immediate Post-Show
1. Mute all channels and pull master fader to -infinity.
2. Thank the band and crew.
3. Save the final show scene as a snapshot on the Wing. Use a descriptive name including the date and venue.
4. Export the show file from the Wing to USB for backup.
5. Power down in reverse order: amps/speakers first, then processors, then console last.

### Console Reset and Archival
1. Save the complete show file to the Wing's internal storage and to an external USB drive.
2. Document any notes about the show: problem channels, feedback frequencies, monitor preferences for each musician.
3. If using AUTO-MIXER Tubeslave, export the session log for review.
4. Record any EQ notch filters that were applied during the show — these indicate persistent feedback frequencies at this venue.
5. Note the final master fader position and target SPL achieved.

### Equipment Teardown
1. Disconnect all stage box cables, coiling them properly (over-under technique).
2. Check for damaged cables and set aside for repair.
3. Pack microphones in protective cases.
4. Coil and pack all audio snakes and multi-pin connectors.
5. Safely store IEM transmitters and receivers.
6. Shut down and pack the Wing Rack and associated networking equipment.
7. Verify all equipment is accounted for against the load-in inventory.

### Data and Settings Backup
1. Copy the Wing show file to a cloud backup or separate drive.
2. Save the AUTO-MIXER Tubeslave configuration YAML for this venue/show.
3. Archive any Dante routing presets used.
4. Back up the AI knowledge base if custom entries were added during the show.
5. Record venue-specific notes: house PA system EQ, monitor positions, power distribution layout, stage plot.

## Venue Size Adjustments

### Small Venues (Under 200 capacity)

#### Characteristics
- Short throw distances (under 10 meters to back wall).
- Room reflections are significant — early reflections arrive within 10-30 ms.
- Bass buildup in corners and along walls.
- Low ceiling height increases low-frequency problems.
- Audience is close to the PA and stage.

#### Adjustments
- **Overall level**: Target 90-95 dBA. Small rooms get loud fast.
- **Bass management**: Reduce low-end on kick and bass by 2-3 dB compared to standard settings. Room modes will reinforce bass naturally.
- **Reverb**: Use very little or no reverb. The room provides its own natural ambience. If reverb is needed, use short room settings (0.3-0.6 seconds).
- **HPF**: Raise HPF frequencies by 10-20 Hz across all channels to reduce low-end buildup.
- **Monitor levels**: Keep extremely low. Stage volume directly competes with FOH in small rooms.
- **Subwoofers**: Often not needed. If present, reduce sub level by 6-10 dB or disable them entirely.
- **Delay**: No delay speakers needed. Throw distance is short enough for a single point source.
- **Panning**: Reduce stereo width. Many audience members will be off-axis. Keep critical elements (vocals, kick, snare, bass) center.

### Medium Venues (200-1000 capacity)

#### Characteristics
- Moderate throw distances (10-30 meters).
- Mix of direct sound and room reflections.
- More even bass distribution than small rooms.
- May have balconies or mezzanines that need coverage.

#### Adjustments
- **Overall level**: Target 95-100 dBA for rock/pop.
- **Bass management**: Standard settings from instrument profiles work well.
- **Reverb**: Moderate use. Hall or plate reverb at 1.0-2.0 seconds. Adjust based on the room's natural reverb.
- **Delay**: May need front fills for close seating areas. Time them to the main PA (distance in meters / 343 = delay in seconds).
- **Subwoofers**: Standard deployment. Cardioid sub arrangement if available to reduce stage low-end.
- **Panning**: Full stereo width is appropriate. The audience spread justifies left-right imaging.
- **Coverage**: Check for dead spots under balconies. Use delay fills if needed.

### Large Venues (1000-5000 capacity)

#### Characteristics
- Long throw distances (30-60+ meters).
- Significant propagation delay between PA and rear seating.
- Large air volume absorbs high frequencies over distance.
- Multiple seating sections may need independent coverage.

#### Adjustments
- **Overall level**: Target 98-105 dBA at FOH position. Account for inverse square law losses at rear of venue.
- **High-frequency compensation**: Boost high shelf by 1-2 dB on the system processor (not per-channel) to compensate for air absorption over distance.
- **Bass management**: May need to reduce sub level if the venue has significant low-frequency resonances. Use measurement mic to identify problem frequencies.
- **Reverb**: Reduce reverb or eliminate it. Large venues have significant natural reverb (RT60 of 1.5-3.0 seconds). Adding more will muddy the mix.
- **Delay towers**: Essential for rear seating areas. Time delays to the main PA with an additional 10-15 ms offset (Haas effect) so the brain localizes sound to the stage.
- **Subwoofers**: Fly subs or use cardioid ground-stacked configurations to direct bass energy toward the audience and away from the stage.
- **Compression**: Tighten compression ratios slightly (add 0.5:1 to standard settings) to maintain consistency at distance. Dynamics get exaggerated over long throws.
- **DCA management**: Essential. Use DCA groups actively to make broad mix moves during the show. Individual channel adjustments are less effective at these distances.

### Outdoor Venues and Festivals

#### Characteristics
- No room reflections (except ground bounce).
- Sound dissipates freely — no bass buildup from room modes.
- Wind and environmental noise compete with the PA.
- Long throw distances with no walls to help contain sound.
- Temperature and humidity affect sound propagation.

#### Adjustments
- **Overall level**: Target 100-105 dBA at FOH. Outdoor shows need more power due to lack of room reinforcement.
- **Bass boost**: Increase sub level by 3-6 dB compared to indoor settings. No room gain to help.
- **High-frequency boost**: Add 2-3 dB shelf above 8 kHz on the system EQ. Air absorption is significant outdoors, especially in dry or hot conditions.
- **Reverb**: Use more reverb than indoor shows. The absence of room reflections makes dry mixes sound thin and lifeless. Plate reverb at 1.5-2.5 seconds on vocals.
- **Wind protection**: Ensure all outdoor mics have windscreens. Low-frequency wind noise can overload preamps.
- **Delay towers**: Essential for audiences beyond 30 meters. Calculate timing based on temperature-adjusted speed of sound (approximately 331 + 0.6 * temperature_celsius m/s).
- **Monitor levels**: Can be higher outdoors since there is no room to reflect monitor sound into the FOH mics. But IEMs are still preferred for consistency.
- **HPF**: Standard settings. No room modes to worry about, so HPFs can be slightly lower than indoor settings.

## Genre-Specific Mixing Adjustments

### Rock / Hard Rock

#### Mix Philosophy
Aggressive, powerful, and loud. The energy comes from the drums and guitars. Vocals must cut through a dense wall of sound.

#### Key Adjustments
- **Drums**: Emphasize kick attack (boost 3-5 kHz) and snare crack (boost 3-5 kHz). Gate tightly. Heavy drum bus compression for glue.
- **Bass**: Aggressive compression (5:1 ratio). Boost growl at 700-900 Hz for midrange presence. The bass should interlock with the kick — share the low-end space.
- **Guitars**: Let them be loud in the mix. Pan hard left and right if two guitars. Reduce low-end aggressively (HPF at 120-150 Hz) to leave room for bass and kick.
- **Vocals**: Heavy compression (4:1) to keep vocals above the wall of guitars. Boost presence at 3-4 kHz for cut-through. Use short plate reverb (1.0-1.5 seconds) to add depth without washing out.
- **Overall mix**: Drums 40% of the mix energy, guitars 30%, vocals 20%, bass 10% (perceived). Target 98-102 dBA.
- **Effects**: Short reverb, minimal delay. Keep things tight and punchy.

### Pop / Dance

#### Mix Philosophy
Clean, polished, and commercially balanced. Vocals are the absolute priority. The low end should be tight and punchy.

#### Key Adjustments
- **Vocals**: The most important element. Use parallel compression for consistent level. Boost air shelf at 10-12 kHz for openness. Medium plate reverb (1.5-2.0 seconds).
- **Kick/bass**: Tight, punchy, and controlled. Side-chain the bass to the kick for clarity. Heavy HPF on everything else to keep the low-end clean.
- **Keys/synth**: Important for pop arrangements. Keep clean and present. Pan synth pads wide for stereo width.
- **Guitars**: Support role only. Keep lower in the mix than rock. HPF at 150 Hz.
- **Backing vocals**: Spread wide in stereo. Compress harder than leads (4:1) for a smooth, blended sound. Use reverb to push them slightly back.
- **Overall mix**: Vocals 35% of mix energy, rhythm section 35%, instruments 20%, effects 10%. Target 95-100 dBA.
- **Effects**: Lush reverbs on vocals. Tempo-synced delays (quarter note or dotted eighth). Clean, polished effects.

### Jazz / Acoustic

#### Mix Philosophy
Natural, dynamic, and transparent. The mixer's job is to faithfully reproduce what the musicians are doing on stage. Minimal processing.

#### Key Adjustments
- **Dynamics**: Use compression very sparingly. Ratios of 2:1 maximum. Let the natural dynamics of the performance come through.
- **EQ**: Subtractive only. Remove problems but do not shape the tone aggressively. The instruments should sound like themselves.
- **Reverb**: Use natural-sounding hall or chamber reverb (1.5-2.5 seconds). The reverb should enhance the sense of space, not be noticeable as an effect.
- **Drums**: Minimal gating. Jazz drumming includes ghost notes and brush work that gates will destroy. Use light compression only. Overheads are the primary drum source — close mics are supplementary.
- **Bass**: Acoustic bass or upright needs careful EQ to control boominess (cut 100-200 Hz) while preserving warmth. Very gentle compression (2:1). No gate.
- **Piano**: Full-range instrument. Minimal EQ. Light compression to control dynamics between pianissimo and fortissimo passages.
- **Vocals**: Natural sound. Light compression (2:1 to 3:1). Minimal EQ. Avoid sibilance control unless truly necessary — jazz vocals are intimate and detail matters.
- **Overall mix**: Balance should reflect the natural acoustic balance on stage. Target 85-95 dBA.
- **Effects**: Subtle. The audience should not be aware that effects are being used.

### Country / Folk

#### Mix Philosophy
Warm, organic, and vocal-forward. Acoustic instruments should sound natural. Storytelling is central, so vocal intelligibility is paramount.

#### Key Adjustments
- **Vocals**: Priority one. Warm but clear. Boost presence at 2-3 kHz for storytelling intelligibility. Use plate reverb (1.0-1.8 seconds) for warmth. Slapback delay (80-120 ms) is a classic country vocal effect.
- **Acoustic guitar**: Major instrument in country/folk. Boost string clarity at 3 kHz. Use gentle compression to even out strumming dynamics. Pan slightly off-center.
- **Fiddle/violin**: Similar to vocal treatment. HPF at 150 Hz. Boost 2-4 kHz for presence. Watch for harsh upper frequencies (5-7 kHz).
- **Pedal steel/dobro**: Mid-heavy instrument. Boost 1-2 kHz for presence. Cut 300-500 Hz to avoid mud. Pan opposite to acoustic guitar.
- **Electric guitar**: Cleaner tones than rock. Less gain, less compression. Boost midrange character at 800 Hz.
- **Drums**: Slightly less aggressive than rock. Snare should crack without dominating. Tighter gating to keep the mix clean.
- **Bass**: Warm and supportive. Less aggressive than rock. Follow kick drum closely.
- **Overall mix**: Vocals 40%, acoustic instruments 30%, rhythm section 20%, effects 10%. Target 92-98 dBA.

### Electronic / DJ

#### Mix Philosophy
The pre-mixed playback is the main source. The mixer's role is system management, level control, and ensuring the PA reproduces the material faithfully.

#### Key Adjustments
- **Playback channels**: Minimal processing. The material is already mixed and mastered. HPF at 20 Hz for subsonic protection only.
- **EQ**: Avoid per-channel EQ on playback. Use system EQ only for room correction.
- **Compression**: Do not compress the main playback. It is already mastered with dynamics processing.
- **Sub management**: Electronic music relies heavily on sub-bass (30-60 Hz). Ensure the sub system is aligned and has sufficient headroom. Sub energy can be 6-10 dB louder than typical live band shows.
- **Vocal (MC/DJ)**: If there is a live vocal mic, compress aggressively (4:1 to 6:1) and HPF at 100 Hz. The vocal must cut through the dense electronic mix.
- **Monitoring**: DJ monitoring is typically via booth monitor or headphones. Keep the booth feed clean and unprocessed.
- **Limiters**: Essential on the master bus. Electronic music has extreme peak-to-average ratios. Set limiters to protect the PA system.
- **Overall level**: Target 100-105 dBA. Electronic music audiences expect high SPL levels.

### Metal / Punk

#### Mix Philosophy
Maximum aggression and impact. Double bass drums, distorted guitars, and screaming vocals define the genre. Clarity in the chaos is the goal.

#### Key Adjustments
- **Kick drum**: Critical for double bass clarity. Boost beater click at 4-6 kHz aggressively (+4-5 dB). Use fast gate attack (0.3 ms) and short hold (30 ms) for maximum separation between rapid hits.
- **Snare**: Heavy compression (6:1) to maintain consistent crack during blast beats. Boost 3-5 kHz for attack.
- **Guitars**: HPF at 150 Hz minimum — bass guitar owns the low end. Scoop mids slightly (cut 400-600 Hz) for the classic metal tone, but not too much or guitars will disappear in the mix.
- **Bass**: Must be audible through the wall of guitars. Boost 700-1000 Hz for midrange definition. Heavy compression (5:1) for consistency.
- **Vocals**: Screaming/growling vocals need different treatment than clean singing. HPF at 120 Hz. Boost 2-4 kHz for intelligibility. Heavy compression (4:1 to 6:1). Less reverb — keep vocals dry and aggressive.
- **Overall mix**: Massive wall of sound. Drums 35%, guitars 35%, vocals 20%, bass 10%. Target 100-105 dBA.
- **Headroom**: Metal bands are loud. Ensure gain staging leaves sufficient headroom at all stages. Master bus limiter is essential.

### Worship / Church

#### Mix Philosophy
Supporting the congregation, not performing a concert. Intelligibility of lyrics and spoken word is the top priority. The mix should never be distracting.

#### Key Adjustments
- **Vocals**: Absolute priority. Intelligibility above all. Boost presence at 2-4 kHz. Use compression (3:1) for consistent level. Reverb should be subtle — plate at 1.0-1.5 seconds.
- **Spoken word**: Separate from singing. Use a different scene/snapshot for spoken word with tighter compression (4:1), no reverb, and HPF at 100 Hz. Reduce music levels during speech.
- **Band levels**: Keep lower than a concert environment. The band supports the congregation singing — it should not overpower them.
- **Dynamics**: Controlled but not squashed. The music should breathe but without extreme peaks that distract.
- **Acoustic instruments**: Emphasize warmth and clarity. Acoustic guitar and piano are typically featured instruments.
- **Electric guitar**: Keep subdued. Use amp modeling or low stage volume. HPF at 120 Hz.
- **Drums**: Use electronic drums or drum shields when possible to control stage volume. If acoustic drums, gate tightly and keep overheads lower in the mix.
- **Subwoofers**: Reduce by 3-6 dB from concert levels. Excessive bass is distracting in worship.
- **Overall level**: Target 80-92 dBA. Many congregations and venues have strict limits.
- **Room acoustics**: Churches often have long reverb times (RT60 of 2-5 seconds). Use minimal or no added reverb. Consider system delays for distributed speaker systems.
