# Troubleshooting Guide

Comprehensive troubleshooting reference for live sound issues encountered
during concerts. Each section covers symptoms, diagnosis steps, and solutions.

## Feedback Diagnosis and Resolution

### Identifying Feedback
Feedback occurs when a microphone picks up its own amplified signal from a
speaker, creating a self-reinforcing loop. It manifests as a sustained tone,
usually at a single frequency.

### Common Feedback Frequencies
- **250-500 Hz**: Low-frequency feedback, common with vocal mics near wedge monitors.
  Sounds like a low hum or howl.
- **800-1200 Hz**: Mid-frequency feedback, often caused by tom or snare mics near fills.
  Sounds like a horn or honk.
- **2-4 kHz**: Presence-range feedback, most common and most noticeable. Human ear is
  most sensitive here. Sounds like a piercing ring.
- **6-10 kHz**: High-frequency feedback from condenser mics. Sounds like a whistle.

### Resolution Steps
1. **Immediate**: Reduce the master fader or mute the offending channel
2. **Identify**: Use RTA (real-time analyzer) to find the exact frequency, or
   use ear training: sweep a parametric EQ boost to find where it rings
3. **Notch**: Apply a narrow cut (Q=8-12, depth -4 to -8 dB) at the feedback frequency
4. **Prevent**: Reduce monitor send level for that channel, reposition the mic,
   or reposition the monitor wedge
5. **Document**: Log the feedback frequency and channel for future reference

### Prevention Strategies
- Always ring out monitors during sound check
- Use cardioid or hypercardioid microphones for better off-axis rejection
- Position monitor wedges at the microphone's null point (typically 180 degrees
  for cardioid, 120 degrees for hypercardioid)
- Keep the microphone as close to the source as possible (inverse square law)
- Never boost 2-4 kHz on monitor sends
- Engage the Wing's built-in feedback suppressor on monitor buses if available

## Ground Loop Detection and Fixing

### Symptoms
- Constant 50/60 Hz hum (depending on local mains frequency)
- Hum that changes when touching equipment or cables
- Buzz with harmonic content at 100/120 Hz, 150/180 Hz

### Diagnosis
1. Unplug all inputs and listen to each output — hum present means it is in the
   output stage or power supply
2. Plug in one input at a time — hum appears when the problem channel is connected
3. Touch the ground pin of the XLR connector — if hum changes, it is a ground issue
4. Check if the hum correlates with dimmer lights or HVAC systems

### Solutions
- **DI Box Ground Lift**: Engage the ground lift switch on the DI box. This breaks
  the ground loop between the instrument and the mixer. Try this first.
- **Isolation Transformer**: Use an audio isolation transformer (1:1) inline on the
  offending cable. This galvanically isolates the two ground systems.
- **Star Grounding**: Ensure all audio equipment shares a single ground point.
  Avoid daisy-chaining power from multiple circuits.
- **Power Conditioner**: Use a power conditioner to clean mains power and provide
  a consistent ground reference.
- **Lift Pin 1 (Shield)**: As a last resort, disconnect pin 1 (ground/shield) at one
  end of the XLR cable. This is not recommended for permanent installations but
  can solve emergencies.
- **Cable Routing**: Keep audio cables away from power cables. Cross them at 90 degrees
  if they must intersect.

## Latency Issues

### Symptoms
- Musicians complain that the sound feels "behind" or "delayed"
- Flamming effect on drums (close mic and overhead not aligned)
- Comb filtering between close and distant mics

### Diagnosis
1. Measure round-trip latency: play a click through the system and record at the mic
   position. Measure the delay between direct and amplified sound.
2. Check processing chain: each digital processing stage adds latency
3. Check sample rate and buffer size: lower sample rates add more latency per sample
4. Check for double conversion: analog to digital to analog chains multiply latency

### Common Latency Sources
| Source                     | Typical Latency    |
|----------------------------|--------------------|
| Wing Rack internal DSP     | 0.8 ms             |
| AES50 stage box            | 0.25 ms per hop    |
| Dante audio network        | 0.15-5.0 ms        |
| AD/DA conversion           | 0.5-1.5 ms each    |
| Plugin/insert processing   | 0.5-5.0 ms         |
| PA speaker distance (10m)  | 29 ms              |

### Solutions
- Use the Wing Rack's built-in delay on each channel to time-align sources.
  Path: `/$ch/{N}/delay/time` (0-500 ms)
- Minimize the number of digital conversion stages
- Use AES50 instead of Dante for lowest latency to stage boxes
- Align close mics and overheads using GCC-PHAT phase measurement
- For PA alignment: measure delay from each speaker cluster to the mix position
  and apply corresponding delays to closer speakers

## Signal Routing Problems

### Symptoms
- Channel produces no sound despite showing input signal
- Sound comes from unexpected output
- Muting a channel does not stop the sound

### Diagnosis
1. **Check Input Patch**: Verify the channel's source assignment matches the
   physical input. On the Wing: `/$ch/{N}/config/src`
2. **Check Bus Routing**: Verify the channel is routed to the correct bus.
   Check main bus assignment and bus send levels.
3. **Check DCA Assignment**: A muted DCA will silence all assigned channels.
   Check DCA mute states.
4. **Check Mute Groups**: Mute groups can silently affect multiple channels.
   Verify mute group membership.
5. **Check Direct Outputs**: The channel may have a direct output enabled that
   bypasses the normal bus routing.

### Common Routing Mistakes
- Channel routed to bus instead of main (or vice versa)
- Monitor send is post-fader when it should be pre-fader
- FX send accidentally assigned to a monitor bus
- Stereo link enabled on a channel that should be mono
- DCA assignment causing unexpected level changes

### Solutions
- Verify routing systematically: input -> channel -> bus -> output
- Use the Wing's signal flow view to trace the signal path
- Reset the channel strip to default and rebuild routing
- Check for accidental stereo links that pair adjacent channels

## Wireless Mic Dropouts

### Symptoms
- Intermittent loss of audio from wireless microphone
- Static or crackling during movement on stage
- Complete signal loss in certain stage positions

### Diagnosis
1. Check battery level — most dropouts are caused by low batteries
2. Check RF signal strength on the receiver display
3. Walk the stage with the transmitter and note dead spots
4. Check for new RF interference sources (other wireless systems, LED walls,
   digital devices, cell phone towers nearby)
5. Check antenna connections on the receiver

### Solutions
- **Replace Batteries**: Always use fresh, high-quality alkaline or lithium batteries.
  Rechargeable NiMH batteries have lower voltage and die suddenly.
- **Frequency Coordination**: Use Shure Wireless Workbench, Sennheiser WSM, or similar
  software to calculate intermodulation-free frequencies.
- **Antenna Positioning**: Move antennas to line-of-sight with the stage. Avoid placing
  antennas behind metal racks or inside equipment cases.
- **Antenna Distribution**: Use an active antenna distribution system to improve
  signal coverage and reduce cable losses.
- **Squelch Adjustment**: Increase squelch on the receiver to reject weak signals
  and interference (at the cost of slightly reduced range).
- **Backup**: Always have a wired microphone and cable ready as a backup.

## Monitor Mixing Problems

### Vocal Clarity in Monitors
- Most common complaint: "I can't hear myself"
- Root cause is usually excessive stage volume from other sources
- Increasing monitor level creates a feedback risk

### Solutions for Monitor Clarity
1. First, try to reduce other sources in the monitor mix (especially instruments)
2. Add a slight presence boost (2-3 kHz) to the vocal channel monitor send
3. Apply a high-pass filter on the monitor bus (100-200 Hz) to reduce mud
4. Consider switching the musician to IEM — this dramatically improves isolation
5. Use a wedge with better pattern control (conical horn, controlled dispersion)

### Musician Complaints and Responses
| Complaint                | Likely Cause                    | Solution                        |
|--------------------------|---------------------------------|---------------------------------|
| "I can't hear myself"   | Too much band in their mix      | Reduce others, not raise vocal  |
| "It sounds muddy"       | Low-end buildup in wedge        | HPF on monitor bus, cut 200 Hz  |
| "My ears hurt"           | IEM level too high              | Reduce level, check limiter     |
| "I hear echo"            | Delay between monitors and PA   | Reduce PA bleed, isolate IEM    |
| "The bass is boomy"      | Low-end coupling with floor     | Decouple wedge from floor       |
| "I need more click"      | Click too quiet in IEM mix      | Boost click bus, hard pan click  |

## Phase Cancellation Diagnosis

### Symptoms
- Thin or hollow sound when combining multiple mics on one source
- Bass disappears when drum overheads are added
- Comb filtering (metallic, phasey sound)

### Diagnosis Steps
1. **Polarity Check**: Flip the polarity (phase invert) on one mic. If the sound
   gets fuller, the mics are out of polarity. On the Wing: `/$ch/{N}/preamp/invert`
2. **Time Alignment**: Measure the distance from each mic to the source. A difference
   of 1 foot = approximately 1 ms delay. Apply delay to the closer mic.
3. **GCC-PHAT**: Use the AUTO-MIXER's GCC-PHAT (Generalized Cross-Correlation with
   Phase Transform) to automatically measure the time delay between two channels.
   This gives sample-accurate delay measurement.
4. **Visual Check**: Use an oscilloscope or waveform display to compare waveforms
   from two mics. Look for inverted peaks.

### Common Phase Problems
- **Snare top + bottom**: Bottom mic is often inverted. Flip polarity on bottom mic.
- **Kick inside + outside**: Apply 1-3 ms delay to the outside mic (it is farther away).
- **Bass DI + amp mic**: The amp mic is typically 0.5-3 ms behind the DI. Apply delay
  to the DI to match the amp mic, or vice versa.
- **Drum overheads + close mics**: Apply delay to close mics to align with the overhead
  pair (overheads are farther from the drums).
- **Two guitar mics**: If using two mics on one cabinet, verify they are at the same
  distance from the cone. Small misalignment causes phase issues.

### Solutions
- Apply channel delay to align arrival times. Use GCC-PHAT for measurement.
- Flip polarity (180 degrees) on one channel if they are out of phase.
- Physically reposition mics to reduce path length differences.
- The 3:1 rule: place mics at least 3x farther apart than their distance to the source.

## Digital Clipping Resolution

### Symptoms
- Harsh, crackling distortion on peaks
- Clip indicators lighting on channel meters
- Flat-topped waveforms visible in metering

### Diagnosis
1. **Check Preamp**: Is the analog preamp clipping before the digital conversion?
   Look for the preamp clip LED on the stage box.
2. **Check Digital Levels**: Are the channel levels exceeding 0 dBFS? Check post-EQ
   and post-compressor levels.
3. **Check Bus Levels**: Sum of many channels can clip the bus even if individual
   channels are fine.
4. **Check Processing**: EQ boosts and compressor makeup gain can push levels into clipping.

### Solutions
- **Reduce Preamp Gain**: Lower the analog gain until peaks are at -12 to -18 dBFS.
  If the source is too loud even at minimum gain, engage the pad switch (-20 dB).
  On the Wing: `/$ch/{N}/preamp/gain` and `/$ch/{N}/preamp/trim`
- **Reduce EQ Boosts**: Convert boosts to equivalent cuts. Instead of boosting 3 kHz
  by +4 dB, cut everything else by -4 dB and raise the fader.
- **Reduce Bus Level**: If the main bus is clipping, pull all channel faders down by
  3-6 dB simultaneously. Do not just reduce the main fader — the bus is already clipping.
- **Engage Limiter**: Always have a safety limiter on the main bus. Set the ceiling
  at -1 dBFS with a fast attack (0.1 ms).

## Network Audio Issues (Dante, AES50)

### AES50 Issues
- **No Sync**: Check AES50 cable connection. LED on stage box should show solid green.
  Blinking = no sync. Try replacing the cable.
- **Crackling/Dropouts**: AES50 cables must be Category 5e or better, shielded.
  Maximum cable length: 100 meters. Check for damaged connectors.
- **Clock Sync**: The Wing Rack should be the clock master when using AES50 stage boxes.
  Verify in Wing settings: Utility -> Sync -> AES50 A/B -> Internal.
- **Redundancy**: Connect both AES50 A and AES50 B for automatic failover.

### Dante Issues
- **No Audio**: Verify routing in Dante Controller. Check that sample rate matches
  across all Dante devices (must all be 48 kHz).
- **Latency**: Dante latency depends on configuration: 0.15 ms (ultra-low) to
  5 ms (recommended for large networks). Use 1 ms for live sound.
- **Clock Master**: One device must be the Dante clock master (preferred master).
  Verify in Dante Controller that the Wing or a dedicated clock is master.
- **Network Switches**: Use managed Gigabit Ethernet switches with QoS enabled
  (DSCP 46 for Dante audio traffic). Disable EEE (Energy Efficient Ethernet).

### General Network Troubleshooting
1. Check all Ethernet cables for damage (continuity test)
2. Verify all devices are on the same subnet
3. Check for IP address conflicts
4. Verify switch port speeds (must be Gigabit for Dante)
5. Check for excessive network traffic from other devices (isolate audio network)
6. Reboot all network devices in sequence: switches first, then audio devices

## Mixer Communication Errors (OSC)

### Symptoms
- AUTO-MIXER cannot connect to the Wing Rack
- Commands sent but no response from mixer
- Intermittent connection drops

### Diagnosis
1. **Network Check**: Ping the mixer IP address. If no response, check cable and IP config.
2. **Port Check**: Verify UDP port 2222 is not blocked by a firewall.
3. **Keepalive Check**: The Wing Rack requires a keepalive packet (WING\0) every 5-10
   seconds. If missed, the mixer drops the connection.
4. **Subscription Check**: OSC subscriptions expire every 10 seconds. Verify the
   AUTO-MIXER is re-subscribing at the correct interval.
5. **Rate Limit Check**: Sending more than 50 OSC messages per second can overwhelm
   the mixer. Check message send rate.

### Solutions
- Verify network configuration: mixer IP, subnet mask, gateway
- Use a direct Ethernet connection (bypass switches) for testing
- Check that no other software is connected to the mixer on the same port
- Restart the OSC connection manager: it should auto-reconnect
- Increase keepalive frequency to every 3 seconds for more reliability
- Reduce OSC message rate by batching parameter changes
- Check for UDP packet loss with a network analyzer (Wireshark)

### Common OSC Error Codes and Meanings
| Error                        | Meaning                              | Fix                           |
|------------------------------|--------------------------------------|-------------------------------|
| Connection timeout           | Mixer not reachable on network       | Check cables, IP, firewall    |
| No keepalive response        | Mixer dropped the connection         | Reconnect, check send rate    |
| Parameter out of range       | OSC value outside 0.0-1.0            | Clamp values before sending   |
| Unknown OSC address          | Path does not exist on this firmware  | Check firmware version docs   |
| Subscription expired         | Re-subscribe timer too slow           | Reduce re-subscribe interval  |

## Mixer DSP Overload

### Symptoms
- Audio glitches and dropouts
- Wing Rack displays DSP overload warning
- Processing bypassed automatically

### Diagnosis
1. Check DSP usage on the Wing Rack's info screen
2. Count the number of active processing blocks (EQ, comp, gate, FX per channel)
3. Check if all 8 FX slots are in use with heavy algorithms (reverb, pitch shift)

### Solutions
- Disable processing on unused channels (muted channels still consume DSP)
- Use simpler FX algorithms (room reverb instead of convolution)
- Consolidate FX: share reverbs across channels using FX buses instead of
  per-channel inserts
- Reduce the number of active EQ bands per channel
- Bypass gates on channels that do not need them
