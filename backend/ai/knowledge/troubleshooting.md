# Troubleshooting Guide

Comprehensive troubleshooting for AUTO-MIXER Tubeslave and the Behringer Wing Rack live mixing system. Organized by symptom category for quick diagnosis. Covers both the software application and the mixer hardware.

---

## OSC Connection Issues

### Cannot Connect to Wing Mixer

#### Symptoms
- AUTO-MIXER dashboard shows "Disconnected" or "Connection Failed."
- No meter activity on any channels.
- OSC commands have no effect on the mixer.

#### Likely Causes
1. Incorrect IP address in configuration.
2. Wing Rack is not powered on or not on the same network.
3. Firewall blocking UDP ports 2222 (discovery) or 2223 (OSC).
4. Another application already bound to the UDP port.
5. Network switch or cable issue.

#### Solutions
1. Verify the Wing's IP address on the mixer's screen (Setup > Network) and confirm it matches the `ip` field in the YAML config or the OSC manager's `ip` parameter.
2. Ping the Wing from the machine running AUTO-MIXER: `ping <wing_ip>`. If no response, check the physical network connection.
3. Check firewall rules. On Linux: `sudo ufw status` or `sudo iptables -L`. Ensure UDP ports 2222 and 2223 are open for both inbound and outbound.
4. Check if another process is using port 2223: `sudo lsof -i :2223` or `sudo ss -ulnp | grep 2223`. Kill the conflicting process or change the port.
5. Test with a simple OSC probe: send `WING?` to the mixer's IP on UDP port 2222. You should receive a response with the mixer's model and firmware version. If no response, the problem is network-level.
6. Restart the OSC connection in AUTO-MIXER. The system will re-send the `WING?` discovery packet and re-establish the `/xremote` keepalive.

### Connection Drops Periodically

#### Symptoms
- AUTO-MIXER works for a while then loses connection.
- Health monitor reports "unhealthy" after a period of operation.
- Log shows "No response received within health_timeout_sec" warnings.

#### Likely Causes
1. The `/xremote` keepalive is not being sent frequently enough. The Wing requires it every 10 seconds; AUTO-MIXER sends it every 8 seconds by default (OSCManager.XREMOTE_INTERVAL_SEC).
2. Network congestion causing dropped UDP packets.
3. The Wing is being restarted or a firmware update is in progress.
4. Wi-Fi connection instability (if not using wired Ethernet).

#### Solutions
1. Verify the keepalive thread is running. Check the logs for periodic `/xremote` send messages. If absent, the keepalive thread may have crashed.
2. Switch to wired Ethernet. Wi-Fi is unreliable for real-time OSC control. Use 5 GHz if wired is impossible.
3. Check the OSCManager `health_timeout_sec` setting. Default is 15 seconds. If the network has high latency, increase to 20-30 seconds.
4. Reduce the outbound message rate if the network is saturated. Lower `rate_limit_hz` from the default 50 Hz to 25 Hz.
5. Use a dedicated network switch for mixer control. Do not share with Dante audio or general internet traffic.

### OSC Messages Not Taking Effect

#### Symptoms
- Commands are sent (visible in logs) but the mixer state does not change.
- Query responses return unexpected values or no values.

#### Likely Causes
1. Incorrect OSC address syntax. Wing addresses are case-sensitive and use specific numbering.
2. Wrong value type. The Wing expects specific types (float, int, string) for each parameter.
3. Rate limiting is dropping messages. The send queue may be full (max 4096 messages by default in OSCManager).
4. The OSC connection was established but the `/xremote` subscription expired.

#### Solutions
1. Verify OSC address format. Channel numbers are 1-40 with no zero-padding: `/ch/1/fdr` not `/ch/01/fdr`. Bus numbers are 1-16. Aux numbers are 1-8.
2. Check value types. Faders expect float (0.0 to 1.0), mutes expect int (0 or 1), names expect string. Sending the wrong type will be silently ignored by the Wing.
3. Query the parameter first to verify connectivity: send the address with no arguments (e.g., `/ch/1/fdr`) and check for a response. If no response, the connection may be dead.
4. Re-send `/xremote` to refresh the subscription. AUTO-MIXER does this automatically every 8 seconds, but if the keepalive thread has stopped, send it manually.
5. Check the OSCManager send queue size in logs. If it reports "queue full," reduce the message rate or increase the queue size.

---

## WebSocket Connection Issues

### Frontend Cannot Connect to Backend

#### Symptoms
- The web UI shows "Connecting..." indefinitely or displays a connection error.
- Browser console shows WebSocket connection refused errors.
- The AUTO-MIXER dashboard is blank or unresponsive.

#### Likely Causes
1. The backend server is not running.
2. Wrong WebSocket host or port. Default is `localhost:8765`.
3. Firewall blocking TCP port 8765.
4. The backend crashed during startup due to missing dependencies.

#### Solutions
1. Verify the backend process is running: `ps aux | grep server.py` or check the terminal where it was launched.
2. Check the WebSocket URL in the frontend configuration. It should match the `ws_host` and `ws_port` in the AutoMixerServer constructor (default `ws://localhost:8765`).
3. Check if the port is in use: `lsof -i :8765`. If another process is using it, change the port in configuration.
4. Check backend startup logs for import errors. Common missing dependencies: `websockets`, `pythonosc`, `numpy`, `pyyaml`.
5. If running on a remote machine, ensure the `ws_host` is set to `0.0.0.0` (not `localhost`) to accept external connections.

### WebSocket Disconnects During Operation

#### Symptoms
- The UI intermittently shows disconnected state.
- Real-time meter updates freeze and then resume.
- Actions from the UI are delayed or lost.

#### Likely Causes
1. High message volume overloading the WebSocket connection. Each meter update, fader change, and status broadcast is a WebSocket message.
2. The backend's asyncio event loop is blocked by a long-running synchronous operation.
3. Network timeout between frontend and backend (especially if running on different machines).
4. Browser tab was put to sleep by the OS to save resources.

#### Solutions
1. Throttle meter updates. Reduce the frequency of metering broadcasts to 10-15 Hz instead of real-time.
2. Check backend logs for "event loop blocked" or slow handler warnings. Offload CPU-intensive operations (FFT analysis, spectral processing) to separate threads.
3. Increase the WebSocket ping/pong timeout if running across a network.
4. If using Chrome, disable tab sleeping for the AUTO-MIXER tab (chrome://flags > Tab Freezing > Disabled).

---

## Audio Signal Issues

### No Signal on Channel

#### Symptoms
- Channel meter shows no activity on the Wing or in AUTO-MIXER.
- The instrument is playing but nothing appears in the analysis.

#### Likely Causes
1. Physical cable disconnected or faulty.
2. Stage box channel not patched.
3. Wing input routing incorrect: wrong input group or input number.
4. Channel is muted.
5. Fader is at -infinity.
6. Preamp gain is at minimum.
7. 48V phantom power not enabled for condenser microphones.

#### Solutions
1. Check the physical connection at the stage box end and the microphone end.
2. Verify Wing input routing via OSC: query `/ch/{n}/in/conn/grp` (input group) and `/ch/{n}/in/conn/in` (input number within the group).
3. Check mute state: query `/ch/{n}/mute`. Value should be 0 (unmuted).
4. Check fader position: query `/ch/{n}/fdr`. Value should be above 0.0 (which is -infinity in dB).
5. Check preamp gain: query `/ch/{n}/in/set/trim`. Ensure it is set appropriately (not at minimum).
6. For condenser mics, verify 48V is enabled on the correct channel at the stage box or Wing preamp.
7. Check the main send assignment: query `/ch/{n}/main/1/on`. Should be 1 (enabled).

### Clipping and Distortion

#### Symptoms
- Red clip indicators on the Wing channel meters.
- Audible distortion or harshness on the output.
- AUTO-MIXER gain staging module reports clipping warnings.

#### Likely Causes
1. Preamp gain too high. The Wing ADC clips at 0 dBFS with no headroom.
2. Large EQ boosts stacking up through the signal chain.
3. Compressor makeup gain pushing the signal above 0 dBFS.
4. Too many channels summing to a single bus without level reduction.
5. Master fader too high.

#### Solutions
1. Reduce preamp gain (trim) and compensate with the channel fader. Target -18 to -12 dBFS peaks at the preamp stage.
2. Review all EQ boosts on the clipping channel. No single boost should exceed +6 dB. Use subtractive EQ (cutting problems) rather than additive EQ (boosting what you want).
3. Check compressor makeup gain. Reduce it so the post-compressor signal peaks at -6 dBFS.
4. Reduce individual channel fader levels going to the bus. Or reduce the bus fader.
5. On the master, maintain at least 6 dB of headroom. Pull the master fader back if needed.

### Audio Latency and Buffer Underruns

#### Symptoms
- Audible clicks, pops, or dropouts in the audio.
- Visible gaps in the AUTO-MIXER waveform display.
- System logs report buffer underrun warnings.
- Dante Controller shows "Late" packets.

#### Likely Causes
1. CPU overload on the machine running AUTO-MIXER. Audio processing (FFT, spectral analysis) is competing with other tasks.
2. Dante buffer size too small for the network conditions.
3. Network switch not handling PTP (Precision Time Protocol) correctly.
4. USB audio interface buffer size too small.
5. Operating system not configured for real-time audio.

#### Solutions
1. Check CPU usage: `top` or `htop`. If AUTO-MIXER processes are using more than 70% CPU, reduce the number of active analysis modules.
2. Increase the Dante latency setting in Dante Controller. Try 2 ms, then 5 ms if problems persist. Lower latency requires better network infrastructure.
3. Use a network switch that supports IEEE 1588 PTP. Consumer-grade switches often do not handle PTP correctly.
4. Increase the USB audio buffer size. For analysis (not monitoring), 256 or 512 samples at 48 kHz is usually sufficient.
5. On Linux, configure the real-time audio settings: add the user to the `audio` group, set `rtprio` in `/etc/security/limits.conf`, and consider using a low-latency kernel.

---

## Feedback Detection Issues

### False Positive Feedback Detection

#### Symptoms
- AUTO-MIXER applies notch EQ filters when there is no audible feedback.
- Sustained musical notes (e.g., organ, synth pads, sustained vocals) are being treated as feedback.
- Notch filters appear on channels that are not experiencing feedback.

#### Likely Causes
1. The feedback detector's confidence threshold is too low. Sustained tonal content can resemble feedback in the FFT analysis.
2. The peak persistence window is too short. Musical notes can persist for several seconds just like feedback.
3. The detector is analyzing the wrong signal (post-EQ instead of pre-fader, or vice versa).

#### Solutions
1. Increase the feedback detector's confidence threshold. The default detects feedback above a certain confidence score (0.0 to 1.0). Raise it to 0.8 or higher to reduce false positives.
2. Increase the minimum peak persistence time. Feedback typically persists longer than musical notes and grows in amplitude. If peaks are being flagged after only a few frames, increase the required frame count.
3. Verify the detector is analyzing the correct Dante channel. Pre-fader signals (Dante channels 1-24) are the intended input for feedback detection.
4. Review the magnitude history of flagged events. True feedback shows a steadily increasing magnitude over time. Musical content is stable or decreasing.
5. Temporarily disable automatic notch application and run in detection-only mode to verify accuracy before re-enabling automatic correction.

### Notch Filter Artifacts

#### Symptoms
- Audible thinning or coloration of the sound after feedback detection runs.
- Multiple narrow notch filters stacking up on a single channel.
- The original tone of the instrument is lost after several notch filters are applied.

#### Likely Causes
1. Too many notch filters on one channel. The system supports up to 8 per channel (bands 1-4 on main EQ, bands 1-3 on pre-EQ, plus the low shelf repurposed as PEQ).
2. Notch filters are too deep. MAX_NOTCH_DEPTH_DB is -12 dB by default, which is aggressive.
3. Notch filters placed too close together in frequency, creating a wide combined cut.
4. Notch filters are not being removed after the feedback condition clears.

#### Solutions
1. Limit the maximum number of active notch filters per channel to 4-5 instead of the maximum 8. Fewer, well-placed notches are better than many overlapping ones.
2. Reduce the maximum notch depth. Try -6 dB instead of -12 dB. This is usually sufficient to suppress feedback without destroying the channel tone.
3. Implement a minimum frequency spacing rule. Notch filters should be at least 1/3 octave apart. If two feedback frequencies are closer than this, use a single wider notch.
4. Set a time-to-live on notch filters. If a notch filter has been active for more than 5 minutes without the feedback recurring, consider releasing it.
5. Review notch filter status via OSC: query `/ch/{n}/eq/{band}g` and `/ch/{n}/eq/{band}f` for each band to see what filters are currently applied.

### Feedback Not Detected

#### Symptoms
- Audible feedback is present but AUTO-MIXER does not respond.
- No feedback events appear in the log.
- The feedback detector appears to be running but not flagging anything.

#### Likely Causes
1. The feedback detector module is not enabled or not receiving audio data.
2. The confidence threshold is set too high, causing real feedback to be ignored.
3. The Dante channel carrying the signal is not routed to the analysis input.
4. The feedback frequency is outside the detector's configured range.

#### Solutions
1. Verify the feedback detector is enabled and the analysis thread is running. Check logs for "FeedbackDetector started" or similar.
2. Lower the confidence threshold temporarily to see if events are being generated at lower confidence levels. If so, tune the threshold to a value between the false positives and the missed detections.
3. Verify the Dante routing: the pre-fader signal for the affected channel must be routed to the correct Dante output and reaching the analysis input.
4. Check the detector's frequency range configuration. Ensure it covers at least 100 Hz to 12 kHz, which encompasses all common feedback frequencies.

---

## Dante Routing Issues

### No Dante Audio Reaching AUTO-MIXER

#### Symptoms
- All analysis modules report no audio data.
- Dante channels show zero signal in the AUTO-MIXER dashboard.
- The Wing is receiving and passing audio to the PA normally.

#### Likely Causes
1. Dante subscriptions are not established. The receiving device (AUTO-MIXER's Dante interface) is not subscribed to the Wing's Dante transmit channels.
2. Sample rate mismatch between the Wing and the Dante interface. Both must be at 48 kHz.
3. Clock master conflict. Only one device on the Dante network should be the PTP clock master.
4. Dante network is on a different VLAN or subnet from the AUTO-MIXER machine.

#### Solutions
1. Open Dante Controller and verify subscriptions. Each AUTO-MIXER input channel should have a green checkmark showing it is subscribed to the corresponding Wing Dante output. Red X means subscription failed.
2. Check sample rates on all Dante devices. The Wing defaults to 48 kHz. Verify the Dante interface on the AUTO-MIXER machine is also at 48 kHz.
3. In Dante Controller, check the clock status. Ideally, the Wing should be the PTP clock master. If another device has taken mastership, either let it be master or configure preferred master priority.
4. Verify the Dante network is reachable: the AUTO-MIXER machine's Dante interface and the Wing must be on the same network segment (VLAN, subnet).

### Wrong Signals on Dante Channels

#### Symptoms
- AUTO-MIXER analysis shows unexpected signals (e.g., a bus mix instead of individual channels).
- The channel recognition module identifies instruments on wrong channels.
- Pre-EQ and pre-fader signals appear swapped.

#### Likely Causes
1. The Wing's Dante direct out routing does not match the expected 64-channel layout defined in `dante_routing_config.py`.
2. Dante subscriptions in Dante Controller are cross-patched.
3. The Wing output tap points are wrong (e.g., Post-Fader instead of Pre-Fader).

#### Solutions
1. Review the expected Dante layout:
   - Channels 1-24: Individual channels, Pre-Fader (post-EQ/Dynamics)
   - Channels 25-48: Individual channels, Pre-EQ (dry, post-preamp)
   - Channels 49-50: Master L/R (post-fader)
   - Channels 51-52: Drum Bus L/R
   - Channels 53-54: Vocal Bus L/R
   - Channels 55-56: Instrument Bus L/R
   - Channel 57: Measurement Mic (pre-EQ)
   - Channel 58: Ambient Mic (pre-fader)
   - Channels 59-60: Matrix 1/2
   - Channels 61-64: Reserve
2. On the Wing, verify each Direct Out assignment. Go to Routing > Direct Outs and confirm the tap point (Pre-Fader for channels 1-24, Pre-EQ for channels 25-48) and the Dante output channel number.
3. In Dante Controller, verify each subscription matches the expected channel map. Fix any crossed subscriptions.

### Dante Clock Sync Issues

#### Symptoms
- Intermittent clicks, pops, or digital noise in the audio.
- Dante Controller shows clock warnings or "Clock Source Lost" errors.
- Sample rate drift causing gradual timing errors.

#### Likely Causes
1. Multiple devices competing for PTP clock master.
2. Network switch does not support IEEE 1588 PTP.
3. A device on the network is sending conflicting clock signals.

#### Solutions
1. In Dante Controller, designate the Wing as the preferred clock master. Set its priority to 1 (highest).
2. Use a managed network switch that explicitly supports PTP/IEEE 1588. Unmanaged consumer switches may interfere with PTP packets.
3. Remove any non-Dante devices that may be generating PTP traffic from the Dante network.
4. If using a redundant Dante network, ensure both primary and secondary networks use the same clock master configuration.

---

## High CPU Usage

### Symptoms
- The AUTO-MIXER backend process uses excessive CPU (above 80%).
- The system becomes sluggish or unresponsive.
- Audio analysis latency increases, causing delayed reactions.
- Fan noise increases on the host machine.

### Likely Causes
1. Too many analysis modules running simultaneously. Each module (FFT analysis, feedback detection, auto-EQ, phase alignment, LUFS metering, spectral recognition) consumes CPU.
2. FFT processing on all 24+ channels at high frequency.
3. The AI knowledge base is being reindexed frequently or using the ChromaDB + sentence-transformers backend with CPU-only inference.
4. The gain staging controller is running continuous LUFS measurement on many channels simultaneously.
5. The spectral analysis buffer is too large, causing expensive FFT computations.

### Solutions
1. Disable analysis modules that are not needed for the current show. For example, if feedback detection is not needed (IEM-only show), disable it.
2. Reduce the FFT analysis rate. Instead of analyzing every audio frame, skip frames to reduce CPU load. Analyzing at 10-20 Hz is sufficient for most mixing decisions.
3. If using ChromaDB with sentence-transformers for the knowledge base, consider switching to the TF-IDF (sklearn) backend which is less CPU-intensive. Set this in the configuration.
4. Limit the number of simultaneously analyzed channels. Analyze only active channels (those with signal above the gate threshold).
5. Reduce the FFT size. A 2048-point FFT at 48 kHz provides approximately 23 Hz resolution, which is sufficient for most mixing analysis. Using 4096 or 8192 points is only needed for very fine frequency resolution (e.g., feedback detection).
6. Monitor per-module CPU usage in the logs. The AutoMixerServer cleanup method (`cleanup_all_controllers()`) can stop individual controllers if needed.

---

## ML Model and Knowledge Base Issues

### Knowledge Base Fails to Load

#### Symptoms
- AI suggestions are unavailable or return generic responses.
- Log shows "chromadb not available" or "sklearn not available" warnings.
- The knowledge base index is empty despite markdown files being present.

#### Likely Causes
1. Required Python packages are not installed. The knowledge base needs at least one of: `chromadb` + `sentence-transformers`, or `sklearn` (scikit-learn) + `numpy`.
2. The knowledge markdown files are missing or in the wrong directory.
3. The ChromaDB persistent storage directory has permission issues.
4. The sentence-transformers model failed to download (no internet access on the production machine).

#### Solutions
1. Check which backend is available. The knowledge base logs its backend choice at startup:
   - Best: `chromadb` + `sentence-transformers` (vector search)
   - Good: `sklearn` TF-IDF (term frequency search)
   - Fallback: keyword matching (always available, no dependencies)
2. Install the recommended backend: `pip install scikit-learn numpy`. This provides good search quality without requiring a GPU or large model downloads.
3. Verify the knowledge files exist in `backend/ai/knowledge/`: `mixing_rules.md`, `instrument_profiles.md`, `wing_osc_reference.md`, `live_sound_checklist.md`, `troubleshooting.md`.
4. If using ChromaDB, ensure the persistent storage directory exists and is writable. Check file permissions.
5. If the sentence-transformers model fails to download, pre-download it on a machine with internet access and copy the model files to the cache directory (`~/.cache/torch/sentence_transformers/`).

### Knowledge Base Returns Irrelevant Results

#### Symptoms
- AI suggestions do not match the question or context.
- The knowledge base returns chunks from unrelated documents.
- Queries about specific instruments return generic mixing advice.

#### Likely Causes
1. The knowledge base index is stale. Files were updated but the index was not rebuilt.
2. The query is too vague or uses different terminology than the knowledge documents.
3. The TF-IDF backend has limited semantic understanding compared to ChromaDB + sentence-transformers.

#### Solutions
1. Rebuild the knowledge base index by calling `index_all()` on the KnowledgeBase instance. This re-reads all markdown files and re-chunks them by heading.
2. Use specific terminology in queries. The knowledge base splits documents by `#`, `##`, and `###` headings, so queries that match heading text will score higher.
3. If using the keyword fallback backend, consider upgrading to the sklearn TF-IDF backend for better relevance. Install `scikit-learn`.
4. Check that the markdown files follow the expected heading structure. The indexer splits on `#{1,3}` patterns — content under deeper headings (####, #####) is included in the parent chunk.

### Channel Recognition Fails

#### Symptoms
- The channel recognition module fails to identify instruments.
- All channels are labeled "Unknown" or with incorrect instrument types.
- Spectral analysis returns no results.

#### Likely Causes
1. No audio signal on the channels being analyzed. Recognition requires active signal.
2. The Dante routing is sending the wrong tap point (e.g., Post-Fader with the fader down instead of Pre-EQ).
3. The recognition module cannot import required dependencies (`numpy`, `scipy`).
4. The spectral analysis window is too short to capture the instrument's characteristic frequencies.

#### Solutions
1. Ensure the instrument is producing sound during recognition. The channel must have active signal above the noise floor.
2. Verify that Dante channels 25-48 (Pre-EQ dry signals) are being used for recognition. These are the unprocessed signals that best reveal instrument characteristics.
3. Check that `numpy` is installed. `scipy` is optional but provides better peak finding for the spectral fallback method.
4. Increase the analysis window to at least 2 seconds of audio. Short windows may not capture enough spectral information for reliable identification.

---

## Configuration Issues

### Configuration File Not Loading

#### Symptoms
- AUTO-MIXER starts with default settings instead of the saved configuration.
- Log shows "Config file not found" warnings.
- Changes made in the UI are not persisted after restart.

#### Likely Causes
1. The YAML configuration file path is incorrect or the file does not exist.
2. The YAML file has syntax errors (indentation, missing colons, invalid characters).
3. The `pyyaml` package is not installed (falls back to JSON).
4. File permissions prevent reading or writing the config file.

#### Solutions
1. Verify the config file path. The ConfigManager logs the path at startup. Check that the file exists at that location.
2. Validate the YAML syntax using an online validator or `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`. Common errors: tabs instead of spaces, missing colons after keys, unquoted strings with special characters.
3. Install PyYAML: `pip install pyyaml`. Without it, only JSON configuration files are supported.
4. Check file permissions: `ls -la config.yaml`. Ensure the user running AUTO-MIXER has read and write access.

### Configuration Hot-Reload Not Working

#### Symptoms
- Changes to the configuration file are not picked up by the running application.
- The watchdog file monitor is not detecting changes.
- Manual edits to the YAML file have no effect until restart.

#### Likely Causes
1. The `watchdog` package is not installed. Hot-reload requires it.
2. The file system does not support inotify events (some network-mounted filesystems, Docker volumes).
3. The configuration change callback is not registered.

#### Solutions
1. Install watchdog: `pip install watchdog`. The ConfigManager checks for it at startup and logs a warning if missing.
2. If using Docker or a network filesystem, the watchdog observer may not receive filesystem events. In this case, implement a polling-based watcher or restart the application after config changes.
3. Verify that change callbacks are registered via `config_manager.on_change(callback)`. Without callbacks, the config reloads internally but modules are not notified.

### Invalid Channel Assignments

#### Symptoms
- AUTO-MIXER tries to control channels that do not exist on the mixer.
- OSC messages are sent to invalid addresses (e.g., `/ch/45/fdr` when the Wing only has 40 channels).
- Unexpected behavior on wrong channels.

#### Likely Causes
1. The configuration file lists channel numbers outside the valid range (1-40 for channels, 1-16 for buses, 1-8 for aux).
2. The channel assignment was carried over from a different show file and does not match the current stage setup.
3. Off-by-one errors: the configuration uses 0-based numbering but the Wing uses 1-based.

#### Solutions
1. Verify channel numbers in the configuration. Wing channels are 1-40, buses are 1-16, aux are 1-8, main is 1-2, matrix is 1-6, DCA is 1-8.
2. Cross-reference the channel list in the config against the Wing's channel labels. Query channel names via OSC: `/ch/{n}/name` for each channel.
3. The `format_channel_number()` function in `wing_addresses.py` does not zero-pad channel numbers. Ensure the configuration uses plain integers (1, 2, 3...) not zero-padded strings ("01", "02", "03").

---

## Feedback Issues (Live Sound)

### Symptoms
- Sustained ringing tone at a specific frequency.
- Tone increases in volume rapidly if not addressed.
- Most common with vocals and acoustic instruments (open microphones).

### Immediate Response
1. Pull down the fader of the offending channel immediately.
2. If unclear which channel, pull down the master fader or mute the main output.
3. Identify the feedback frequency by its pitch:
   - Low rumble (100-300 Hz): Room mode resonance, often stage monitors.
   - Honk/nasal (400-800 Hz): Vocal mic proximity to monitors.
   - Ring (1-3 kHz): Most common vocal feedback range.
   - Squeal (3-6 kHz): High-mid feedback, piercing.
   - Whistle (6-12 kHz): High-frequency feedback, very noticeable.

### Solutions
1. **Notch EQ**: Apply a narrow cut (-6 to -12 dB, Q=6-10) at the exact feedback frequency. Use the Wing's parametric EQ bands. OSC: `/ch/{n}/eq/{band}f` (frequency), `/ch/{n}/eq/{band}g` (gain), `/ch/{n}/eq/{band}q` (Q factor).
2. **Reduce monitor level**: If feedback is from stage monitors, reduce the send level to the offending wedge.
3. **Microphone technique**: Ask the performer to stay closer to the mic (better signal-to-noise) and not point it at the monitors.
4. **Microphone positioning**: Ensure vocal mics are in the null zone (rejection axis) of the monitor wedge.
5. **HPF adjustment**: Raise the HPF to remove low-frequency feedback susceptibility.
6. **Gain reduction**: Lower the preamp gain and compensate with fader position — this changes the feedback threshold.

### Prevention
- Ring out monitors before the show: slowly raise mic level until feedback, then notch EQ each ring.
- Use high-quality hypercardioid microphones for loud stages.
- Keep stage volume as low as possible.
- Position monitors in the mic's rejection zone.
- Use IEM (in-ear monitors) where possible — eliminates monitor wedge feedback entirely.
- Enable AUTO-MIXER feedback detection during the show for automatic notch filter deployment.

---

## Phase Problems

### Symptoms
- Thin, hollow, or comb-filtered sound when two microphones on the same source are combined.
- Loss of low-end when kick drum mics (inside and outside) are summed.
- Snare sounds weak when combined with overhead mics.
- Bass guitar DI and amp mic sound different combined than individually.

### Diagnosis
1. Solo each mic individually — they should each sound full and normal.
2. Combine them — if the sound gets thinner or loses low-end, phase is the issue.
3. Flip the phase (polarity) on one mic using the Wing's phase invert: `/ch/{n}/in/set/inv`.
4. The correct polarity is whichever sounds fuller and louder when combined.

### Solutions
1. **Polarity invert**: Flip phase on one of the two microphones. On the Wing: set `/ch/{n}/in/set/inv` to 1.
2. **Time alignment**: If the mics are at different distances from the source, add delay to the closer mic. Calculate: distance difference in meters / 343 = delay in seconds. Convert to ms. Use the Wing's input delay: `/ch/{n}/in/set/dly`.
3. **Microphone placement**: Ensure multi-mic setups follow the 3:1 rule (distance between mics should be 3x the distance from mic to source).
4. **Common phase issues by instrument**:
   - **Kick drum** (inside + outside): Almost always needs phase flip on one mic.
   - **Snare top + bottom**: Bottom mic typically needs phase invert (they face opposite directions).
   - **Bass DI + amp**: Check phase between DI (direct electrical) and amp mic (acoustic + propagation delay).
   - **Overheads**: Must be equidistant from snare to avoid comb filtering.
5. **AUTO-MIXER auto-phase**: The PhaseAlignmentController can automatically detect and correct phase issues by analyzing the cross-correlation between paired channels.

### Wing Phase Tools
- Phase invert: `/ch/{n}/in/set/inv` (0 or 1)
- Input delay on: `/ch/{n}/in/set/dlyon` (0 or 1)
- Input delay value: `/ch/{n}/in/set/dly` (value depends on mode)
- Input delay mode: `/ch/{n}/in/set/dlymode` (MS for milliseconds)

---

## Noise Floor Issues

### Symptoms
- Audible hiss, hum, or buzz when channels are unmuted.
- Noise increases as more channels are opened.
- Noise is present even when no input signal is being generated.

### Types of Noise

#### Hum (50/60 Hz and harmonics)
- **Cause**: Ground loop between equipment.
- **Solution**:
  1. Use DI boxes with ground lift switches.
  2. Ensure all audio equipment shares the same ground circuit.
  3. Use balanced cables (XLR) for all connections over 3 meters.
  4. On the Wing, check if the hum is on specific channels — isolate the source.
  5. Check Dante network grounding if using AES67/Dante.

#### Hiss (Broadband high-frequency noise)
- **Cause**: Excessive preamp gain, noisy sources, long cable runs.
- **Solution**:
  1. Reduce preamp gain and increase fader level instead (optimize gain staging).
  2. Check for correct impedance matching (mic into mic input, line into line input).
  3. Use gates to automatically reduce noise during silent passages.
  4. On the Wing, set gate thresholds just above the noise floor.

#### Buzz (Harmonics of mains frequency)
- **Cause**: Electromagnetic interference from power lines, dimmers, LED drivers.
- **Solution**:
  1. Route audio cables away from power cables (cross at 90 degrees if unavoidable).
  2. Use shielded cables for all audio connections.
  3. Request that lighting dimmers be set to a stable level or use LED fixtures with clean drivers.
  4. If a specific channel buzzes, try a different cable and input.

### Noise Floor Best Practices
- Keep unused channels muted or gated.
- Set appropriate gate thresholds on all microphone channels.
- Use HPFs on all channels (removes low-frequency rumble that adds up).
- Maintain proper gain staging: -20 LUFS average, peaks at -6 dBFS.
- On the Wing, verify that input connections match the source (mic level vs. line level).

---

## Signal Routing Problems

### No Signal on Channel
1. Check the physical connection at the stage box.
2. Verify the input routing on the Wing: `/ch/{n}/in/conn/grp` and `/ch/{n}/in/conn/in`.
3. Confirm the channel is not muted: `/ch/{n}/mute` should be 0.
4. Check that the fader is up: `/ch/{n}/fdr` should be above 0.0.
5. Verify preamp gain is set: `/ch/{n}/in/set/trim`.
6. Check that 48V phantom power is enabled for condenser mics.
7. Check the main send is enabled: `/ch/{n}/main/1/on` should be 1.

### Signal on Wrong Channel
1. Check the input routing table on the Wing.
2. Verify physical patching at the stage box matches the Wing's input assignment.
3. Common fix: re-assign the input in Wing routing rather than re-patching cables.
4. Wing input routing: `/ch/{n}/in/conn/grp` (input group) and `/ch/{n}/in/conn/in` (input number within group).

### Bus/Send Routing Issues
1. Verify the send is enabled: `/ch/{n}/send/{b}/on`.
2. Check the send level: `/ch/{n}/send/{b}/lvl`.
3. Confirm send mode (pre/post): `/ch/{n}/send/{b}/mode`.
4. Verify the bus output is routed to the correct physical output.
5. Check the bus fader: `/bus/{n}/fdr`.

---

## Level Management Problems

### Clipping / Distortion
1. **At preamp**: Reduce input gain (trim). The ADC clips at 0 dBFS.
2. **At channel EQ**: Large EQ boosts can cause internal clipping even on digital systems. Reduce boost amounts or lower the channel fader.
3. **At compressor**: Check makeup gain — excessive makeup gain after compression can clip.
4. **At bus**: Too many channels summing to one bus. Reduce individual channel levels.
5. **At master**: Master fader too hot. Pull back to maintain headroom.

### Insufficient Level
1. Check preamp gain — may need to increase.
2. Check for signal loss in the chain: gate threshold too high (cutting off signal).
3. Check compressor threshold — too low may be squashing the signal.
4. Check send levels to buses.
5. Check main output assignment: `/ch/{n}/main/1/on` and `/ch/{n}/main/1/lvl`.

### Inconsistent Levels Between Channels
1. Verify gain staging across all channels (all should average -20 LUFS pre-fader).
2. Use compressors to tame dynamics on inconsistent sources.
3. Check for "gain drift" — preamp gain may have been bumped during the show.
4. Use DCA groups to make overall level adjustments without changing individual gain structure.

---

## Dynamics Processing Issues

### Gate Cutting Off Wanted Signal
- **Symptom**: Quiet passages are being gated.
- **Solution**: Lower the gate threshold. The threshold should be set between the noise floor and the quietest wanted signal.
- **Wing OSC**: `/ch/{n}/gate/thr` — decrease the value (more negative = more open).

### Gate Not Closing (Bleed Passing Through)
- **Symptom**: Bleed from adjacent instruments passes through the gate.
- **Solution**: Raise the gate threshold above the bleed level. Use sidechain filtering to focus the gate on the target instrument's frequency range.
- **Wing OSC**: `/ch/{n}/gate/thr` — increase the value (less negative = tighter).
- **Sidechain**: `/ch/{n}/gatesc/type` to HP12, `/ch/{n}/gatesc/f` to the fundamental frequency.

### Compressor Over-Compressing
- **Symptom**: Sound is squashed, lifeless, "pumping."
- **Solution**: Raise the threshold (less compression), reduce the ratio, slow the release.
- **Quick fix**: `/ch/{n}/dyn/thr` — raise by 3-6 dB.

### Compressor Not Engaging
- **Symptom**: No gain reduction visible, dynamics are uncontrolled.
- **Solution**: Lower the threshold or increase gain staging.
- **Check**: Is the compressor enabled? `/ch/{n}/dyn/on` should be 1.

---

## Network and Connection Issues

### Wing Disconnects from Control Software
1. Send `/xremote` keepalive every 10 seconds from your control app. AUTO-MIXER sends it every 8 seconds by default.
2. Check network cable and switch port.
3. Verify IP addresses are on the same subnet.
4. Restart the OSC client connection (send `WING?` to port 2222 again).

### High Latency / Delayed Fader Response
1. Check network load — too many broadcast packets can cause delays.
2. Reduce OSC message rate (throttle to 10 Hz per parameter). The OSCManager rate_limit_hz controls this.
3. Use a dedicated network for mixer control (separate from Dante audio).
4. Verify the switch is not overloaded.

### Wi-Fi Control Issues
1. Use 5 GHz Wi-Fi for lower latency and less interference.
2. Position the access point with direct line of sight to the mixing position.
3. Dedicate the Wi-Fi network to mixer control only.
4. Set a static IP for the Wing and control device.
5. If Wi-Fi is unreliable, fall back to wired Ethernet.

---

## Dependency and Installation Issues

### Missing Python Dependencies

#### Symptoms
- ImportError on startup for various modules.
- Specific features are unavailable (voice control, auto-EQ, etc.).

#### Required Dependencies (Core)
- `websockets` — WebSocket server for frontend communication
- `pythonosc` (python-osc) — OSC message serialization and transport
- `numpy` — Numerical computation, FFT analysis
- `pyyaml` — YAML configuration file support

#### Optional Dependencies (Feature-Specific)
- `scipy` — Enhanced peak finding for feedback detection, spectral analysis
- `scikit-learn` — TF-IDF knowledge base backend
- `chromadb` + `sentence-transformers` — Vector search knowledge base (best quality)
- `watchdog` — Configuration file hot-reload
- `sounddevice` — Local audio input for analysis

#### Solutions
1. Install all core dependencies: `pip install websockets python-osc numpy pyyaml`
2. Install recommended optional dependencies: `pip install scipy scikit-learn watchdog`
3. For voice control (if needed): check that the voice control module's dependencies are installed. The server gracefully handles missing `VoiceControlSherpa` with a try/except import.
4. If a specific feature is not working, check the startup logs for "not available" or "not installed" warnings that indicate which package is missing.

### Version Conflicts

#### Symptoms
- Unexpected behavior after a package update.
- Functions that previously worked now raise TypeError or AttributeError.

#### Common Conflicts
- `websockets` library changed its import paths between major versions. The server handles this with a try/except import for `WebSocketServerProtocol` from both `websockets.server` and `websockets.legacy.server`.
- `numpy` 2.x removed some deprecated type aliases. If using numpy 2.x, ensure the code does not use `np.float` (use `float` or `np.float64` instead).

#### Solutions
1. Pin dependency versions in a `requirements.txt` file.
2. Use a virtual environment to isolate dependencies: `python -m venv venv && source venv/bin/activate`.
3. If upgrading packages, test thoroughly before deploying to a live show environment.
