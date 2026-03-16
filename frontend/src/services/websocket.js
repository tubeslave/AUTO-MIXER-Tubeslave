class WebSocketService {
  constructor() {
    this.ws = null;
    this.url = 'ws://localhost:8765';
    this.listeners = new Map();
    this.reconnectInterval = 3000;
    this.reconnectTimer = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
          }
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
console.log('WebSocket message received:', data.type, data);
this.notifyListeners(data.type, data);
          } catch (error) {
console.error('Error parsing message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.notifyListeners('disconnected', {});
          this.scheduleReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  scheduleReconnect() {
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        console.log('Attempting to reconnect...');
        this.connect().catch(err => console.error('Reconnect failed:', err));
      }, this.reconnectInterval);
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('Sending WebSocket message:', message);
      try {
        const jsonMessage = JSON.stringify(message);
        console.log('Sending JSON string:', jsonMessage);
this.ws.send(jsonMessage);
} catch (error) {
console.error('Error sending WebSocket message:', error);
      }
    } else {
      const state = this.ws ? this.ws.readyState : 'null';
console.error('WebSocket is not connected. ReadyState:', state);
      console.error('WebSocket object:', this.ws);
      console.error('Message that failed to send:', message);
      if (state === 0) {
        console.warn('WebSocket is still connecting...');
      } else if (state === 2 || state === 3) {
        console.warn('WebSocket is closed. Attempting to reconnect...');
        this.connect().catch(err => console.error('Reconnect failed:', err));
      }
    }
  }

  on(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType).push(callback);
  }

  off(eventType, callback) {
    if (this.listeners.has(eventType)) {
      const callbacks = this.listeners.get(eventType);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  notifyListeners(eventType, data) {
if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).forEach(callback => {
        try {
callback(data);
        } catch (error) {
console.error(`Error in listener for ${eventType}:`, error);
        }
      });
    } else {
}
  }

  // Wing direct connection
  connectWing(ip, sendPort = 2223, receivePort = 2223) {
    this.send({
      type: 'connect_wing',
      ip,
      send_port: sendPort,
      receive_port: receivePort
    });
  }

  // dLive MIDI/TCP connection
  connectDLive(ip, port = 51328, tls = false, midiBaseChannel = 0) {
    this.send({
      type: 'connect_dlive',
      ip,
      port,
      tls,
      midi_base_channel: midiBaseChannel
    });
  }

  // Mixing Station connection
  connectMixingStation(host = '127.0.0.1', oscPort = 8000, restPort = 8080) {
    this.send({
      type: 'connect_mixing_station',
      host,
      osc_port: oscPort,
      rest_port: restPort
    });
  }

  // Discover Mixing Station on common ports
  discoverMixingStation() {
    this.send({
      type: 'discover_mixing_station'
    });
  }

  // Disconnect from mixer (works for both Wing and Mixing Station)
  disconnectMixer() {
    this.send({
      type: 'disconnect_mixer'
    });
  }

  // Routing commands
  routeOutput(outputGroup, outputNumber, sourceGroup, sourceChannel) {
    this.send({
      type: 'route_output',
      output_group: outputGroup,
      output_number: outputNumber,
      source_group: sourceGroup,
      source_channel: sourceChannel
    });
  }

  routeMultipleOutputs(outputGroup, startOutput, numOutputs, sourceGroup, startSourceChannel) {
    this.send({
      type: 'route_multiple_outputs',
      output_group: outputGroup,
      start_output: startOutput,
      num_outputs: numOutputs,
      source_group: sourceGroup,
      start_source_channel: startSourceChannel
    });
  }

  getOutputRouting(outputGroup, outputNumber) {
    this.send({
      type: 'get_output_routing',
      output_group: outputGroup,
      output_number: outputNumber
    });
  }

  // Channel input routing commands
  setChannelInput(channel, sourceGroup, sourceChannel) {
    this.send({
      type: 'set_channel_input',
      channel: channel,
      source_group: sourceGroup,
      source_channel: sourceChannel
    });
  }

  setChannelAltInput(channel, sourceGroup, sourceChannel) {
    this.send({
      type: 'set_channel_alt_input',
      channel: channel,
      source_group: sourceGroup,
      source_channel: sourceChannel
    });
  }

  getChannelInputRouting(channel) {
    this.send({
      type: 'get_channel_input_routing',
      channel: channel
    });
  }

  // Snapshot/Scene commands
  loadSnap(snapName) {
    this.send({
      type: 'load_snap',
      snap_name: snapName
    });
  }

  saveSnap(snapName) {
    this.send({
      type: 'save_snap',
      snap_name: snapName
    });
  }

  // Legacy method for backward compatibility
  disconnectWing() {
    this.disconnectMixer();
  }

  setFader(channel, value) {
    this.send({
      type: 'set_fader',
      channel,
      value
    });
  }

  setGain(channel, value) {
    this.send({
      type: 'set_gain',
      channel,
      value
    });
  }

  setEQ(channel, band, freq, gain, q) {
    this.send({
      type: 'set_eq',
      channel,
      band,
      freq,
      gain,
      q
    });
  }

  setCompressor(channel, params) {
    this.send({
      type: 'set_compressor',
      channel,
      params
    });
  }

  getState() {
    this.send({
      type: 'get_state'
    });
  }

  // Voice control commands
  startVoiceControl(modelSize = 'small', language = 'ru', deviceId = null, channel = 0) {
    const message = {
      type: 'start_voice_control',
      model_size: modelSize,
      language: language === null ? '' : language, // Convert null to empty string for auto-detect
      device_id: deviceId,
      channel: channel
    };
    console.log('startVoiceControl sending:', message);
    this.send(message);
  }

  stopVoiceControl() {
    this.send({
      type: 'stop_voice_control'
    });
  }

  getVoiceControlStatus() {
    this.send({
      type: 'get_voice_control_status'
    });
  }

  // Real-time Peak Correction commands (only remaining gain staging functionality)

  startRealtimeCorrection(deviceId = null, channels = [], channelSettings = {}, channelMapping = {}, options = {}) {
    console.log('websocketService.startRealtimeCorrection() called', {
      deviceId,
      channels,
      channelSettings,
      channelMapping,
      options
    });
    this.send({
      type: 'start_realtime_correction',
      device_id: deviceId,
      channels: channels,
      channel_settings: channelSettings,
      channel_mapping: channelMapping,
      learning_duration_sec: options.learning_duration_sec
    });
  }

  stopRealtimeCorrection() {
    console.log('websocketService.stopRealtimeCorrection() called');
    this.send({
      type: 'stop_realtime_correction'
    });
  }

  getGainStagingStatus() {
    this.send({
      type: 'get_gain_staging_status'
    });
  }

  // Update Safe Gain Settings (learning duration, etc.)
  updateSafeGainSettings(settings) {
    this.send({
      type: 'update_safe_gain_settings',
      settings: settings
    });
  }

  // Channel name scanning
  scanChannelNames(channels = []) {
    this.send({
      type: 'scan_channel_names',
      channels: channels
    });
  }

  // Scan mixer channel names for Audio Device tab
  scanMixerChannelNames() {
    this.send({
      type: 'scan_mixer_channel_names'
    });
  }

  // Reset TRIM to 0dB for selected channels
  resetTrim(channels = []) {
    this.send({
      type: 'reset_trim',
      channels: channels
    });
  }

  // Bypass: Disable all modules and set faders to 0dB
  bypassMixer() {
    this.send({
      type: 'bypass_mixer'
    });
  }

  // ========== Auto-EQ Commands ==========

  // Start Auto-EQ analysis
  startAutoEQ(deviceId, channel, profile = 'custom', autoApply = false) {
    this.send({
      type: 'start_auto_eq',
      device_id: deviceId,
      channel: channel,
      profile: profile,
      auto_apply: autoApply
    });
  }

  // Stop Auto-EQ analysis
  stopAutoEQ() {
    this.send({
      type: 'stop_auto_eq'
    });
  }

  // Set EQ profile
  setEQProfile(profile) {
    this.send({
      type: 'set_eq_profile',
      profile: profile
    });
  }

  // Apply EQ corrections to mixer
  applyEQCorrection() {
    this.send({
      type: 'apply_eq_correction'
    });
  }

  // Reset EQ to flat
  resetEQ() {
    this.send({
      type: 'reset_eq'
    });
  }

  // Get available EQ profiles
  getEQProfiles() {
    this.send({
      type: 'get_eq_profiles'
    });
  }

  // Get Auto-EQ status
  getAutoEQStatus() {
    this.send({
      type: 'get_auto_eq_status'
    });
  }

  // ========== Multi-Channel Auto-EQ Commands ==========

  // Start multi-channel Auto-EQ analysis
  startMultiChannelAutoEQ(deviceId, channelsConfig, mode = 'soundcheck') {
    console.log('startMultiChannelAutoEQ called with:', { deviceId, channelsConfig, mode });
    this.send({
      type: 'start_multi_channel_auto_eq',
      device_id: deviceId,
      channels_config: channelsConfig,
      mode: mode  // 'soundcheck' or 'live'
    });
  }

  // Stop multi-channel Auto-EQ analysis
  stopMultiChannelAutoEQ() {
    this.send({
      type: 'stop_multi_channel_auto_eq'
    });
  }

  // Set profile for a specific channel
  setChannelProfile(channel, profile) {
    this.send({
      type: 'set_channel_profile',
      channel: channel,
      profile: profile
    });
  }

  // Apply corrections for a specific channel
  applyChannelCorrection(channel) {
    this.send({
      type: 'apply_channel_correction',
      channel: channel
    });
  }

  // Apply corrections for all channels
  applyAllCorrections() {
    this.send({
      type: 'apply_all_corrections'
    });
  }

  // ========== Phase Alignment Commands ==========

  // Start phase alignment analysis
  startPhaseAlignment(deviceId, referenceChannel, channels) {
    this.send({
      type: 'start_phase_alignment',
      device_id: deviceId,
      reference_channel: referenceChannel,
      channels: channels
    });
  }

  // Stop phase alignment analysis
  stopPhaseAlignment() {
    this.send({
      type: 'stop_phase_alignment'
    });
  }

  // Apply phase/delay corrections to mixer
  applyPhaseCorrections(measurements = null) {
    this.send({
      type: 'apply_phase_corrections',
      measurements: measurements
    });
  }

  // Get phase alignment status
  getPhaseAlignmentStatus() {
    this.send({
      type: 'get_phase_alignment_status'
    });
  }

  // Reset all phase and delay corrections
  resetPhaseDelay(channels = null) {
    this.send({
      type: 'reset_phase_delay',
      channels: channels
    });
  }

  // ========== Auto Fader Commands ==========

  // Start Real-Time Fader mode
  startRealtimeFader(deviceId, channels, channelSettings, channelMapping, settings = {}) {
    console.log('startRealtimeFader called with:', { deviceId, channels, channelSettings, settings });
    this.send({
      type: 'start_realtime_fader',
      device_id: deviceId,
      channels: channels,
      channel_settings: channelSettings,
      channel_mapping: channelMapping,
      settings: settings
    });
  }

  // Stop Real-Time Fader mode
  stopRealtimeFader() {
    console.log('stopRealtimeFader called');
    this.send({
      type: 'stop_realtime_fader'
    });
  }

  // Start Auto Balance LEARN phase
  startAutoBalance(deviceId, channels, channelSettings, channelMapping, duration = 15, bleedThreshold = -50) {
    console.log('startAutoBalance (LEARN) called with:', { deviceId, channels, duration, bleedThreshold });
    this.send({
      type: 'start_auto_balance',
      device_id: deviceId,
      channels: channels,
      channel_settings: channelSettings,
      channel_mapping: channelMapping,
      duration: duration,
      bleed_threshold: bleedThreshold
    });
  }

  // Apply Auto Balance to mixer
  applyAutoBalance() {
    console.log('applyAutoBalance called');
    this.send({ type: 'apply_auto_balance' });
  }

  // Cancel Auto Balance collection
  cancelAutoBalance() {
    console.log('cancelAutoBalance called');
    this.send({
      type: 'cancel_auto_balance'
    });
  }

  // Set Auto Fader genre profile
  setAutoFaderProfile(profile) {
    this.send({
      type: 'set_auto_fader_profile',
      profile: profile
    });
  }

  // Update Auto Fader settings
  updateAutoFaderSettings(settings) {
    this.send({
      type: 'update_auto_fader_settings',
      settings: settings
    });
  }

  // Get Auto Fader status
  getAutoFaderStatus() {
    this.send({
      type: 'get_auto_fader_status'
    });
  }

  // Save Auto Fader settings as defaults
  saveAutoFaderDefaults(settings) {
    console.log('saveAutoFaderDefaults called with:', settings);
    this.send({
      type: 'save_auto_fader_defaults',
      settings: settings
    });
  }

  // Load saved Auto Fader defaults
  loadAutoFaderDefaults() {
    console.log('loadAutoFaderDefaults called');
    this.send({
      type: 'load_auto_fader_defaults'
    });
  }

  // Save all settings (all sections)
  saveAllSettings(settings) {
    console.log('saveAllSettings called with:', settings);
    this.send({
      type: 'save_all_settings',
      settings: settings
    });
  }

  // Get Dante routing scheme
  getDanteRouting(totalChannels = 64) {
    this.send({
      type: 'get_dante_routing',
      total_channels: totalChannels
    });
  }

  // Load all saved settings
  loadAllSettings() {
    console.log('loadAllSettings called');
    this.send({
      type: 'load_all_settings'
    });
  }

  // ========== Auto Soundcheck Commands ==========

  startAutoSoundcheck(deviceId, channels, channelSettings, channelMapping, timings) {
    console.log('startAutoSoundcheck called with:', { deviceId, channels, timings });
    this.send({
      type: 'start_auto_soundcheck',
      device_id: deviceId,
      channels: channels,
      channel_settings: channelSettings,
      channel_mapping: channelMapping,
      timings: timings
    });
  }

  stopAutoSoundcheck() {
    console.log('stopAutoSoundcheck called');
    this.send({
      type: 'stop_auto_soundcheck'
    });
  }

  getAutoSoundcheckStatus() {
    this.send({
      type: 'get_auto_soundcheck_status'
    });
  }

  // ========== Auto Compressor ==========
  startAutoCompressor(deviceId, channels, channelMapping, channelNames = {}) {
    this.send({
      type: 'start_auto_compressor',
      device_id: deviceId,
      channels: channels,
      channel_mapping: channelMapping,
      channel_names: channelNames
    });
  }
  stopAutoCompressor() {
    this.send({ type: 'stop_auto_compressor' });
  }
  getAutoCompressorStatus() {
    this.send({ type: 'get_auto_compressor_status' });
  }
  startAutoCompressorSoundcheck(genreFactor = 1, mixDensityFactor = 1, bpm = null) {
    this.send({
      type: 'start_auto_compressor_soundcheck',
      genre_factor: genreFactor,
      mix_density_factor: mixDensityFactor,
      bpm: bpm
    });
  }
  stopAutoCompressorSoundcheck() {
    this.send({ type: 'stop_auto_compressor_soundcheck' });
  }
  startAutoCompressorLive(autoCorrect = true) {
    this.send({
      type: 'start_auto_compressor_live',
      auto_correct: autoCorrect
    });
  }
  stopAutoCompressorLive() {
    this.send({ type: 'stop_auto_compressor_live' });
  }
  setAutoCompressorProfile(channel, profile) {
    this.send({
      type: 'set_auto_compressor_profile',
      channel: channel,
      profile: profile
    });
  }
  setAutoCompressorManual(channel, params) {
    this.send({
      type: 'set_auto_compressor_manual',
      channel: channel,
      params: params
    });
  }

  // ========== Auto Panner ==========
  startAutoPanner(deviceId, channels, instrumentTypes, spectralCentroids, genre = 'rock') {
    this.send({
      type: 'start_auto_panner',
      device_id: deviceId,
      channels: channels,
      instrument_types: instrumentTypes,
      spectral_centroids: spectralCentroids,
      genre: genre
    });
  }
  stopAutoPanner() {
    this.send({ type: 'stop_auto_panner' });
  }
  getAutoPannerStatus() {
    this.send({ type: 'get_auto_panner_status' });
  }
  calculateAutoPanning(channels, instrumentTypes, spectralCentroids) {
    this.send({
      type: 'calculate_auto_panning',
      channels: channels,
      instrument_types: instrumentTypes,
      spectral_centroids: spectralCentroids
    });
  }
  applyAutoPanning() {
    this.send({ type: 'apply_auto_panning' });
  }

  // ========== Auto Reverb ==========
  startAutoReverb(deviceId, channels, instrumentTypes, spectralCentroids, spectralFluxes) {
    this.send({
      type: 'start_auto_reverb',
      device_id: deviceId,
      channels: channels,
      instrument_types: instrumentTypes,
      spectral_centroids: spectralCentroids,
      spectral_fluxes: spectralFluxes
    });
  }
  stopAutoReverb() {
    this.send({ type: 'stop_auto_reverb' });
  }
  getAutoReverbStatus() {
    this.send({ type: 'get_auto_reverb_status' });
  }
  calculateAutoReverb(channels, instrumentTypes, spectralCentroids, spectralFluxes) {
    this.send({
      type: 'calculate_auto_reverb',
      channels: channels,
      instrument_types: instrumentTypes,
      spectral_centroids: spectralCentroids,
      spectral_fluxes: spectralFluxes
    });
  }
  applyAutoReverb() {
    this.send({ type: 'apply_auto_reverb' });
  }

  // ========== Auto Gate ==========
  startAutoGate(deviceId, channels, channelConfigs, settings) {
    this.send({
      type: 'start_auto_gate',
      device_id: deviceId,
      channels: channels,
      channel_configs: channelConfigs,
      settings: settings
    });
  }
  stopAutoGate() {
    this.send({ type: 'stop_auto_gate' });
  }
  getAutoGateStatus() {
    this.send({ type: 'get_auto_gate_status' });
  }
  configureGateChannel(channelId, config) {
    this.send({
      type: 'configure_gate_channel',
      channel_id: channelId,
      config: config
    });
  }

  // ========== Auto Effects ==========
  startAutoEffects(deviceId, channels, settings) {
    this.send({
      type: 'start_auto_effects',
      device_id: deviceId,
      channels: channels,
      settings: settings
    });
  }
  stopAutoEffects() {
    this.send({ type: 'stop_auto_effects' });
  }
  getAutoEffectsStatus() {
    this.send({ type: 'get_auto_effects_status' });
  }

  // ========== Cross-Adaptive EQ ==========
  startCrossAdaptiveEQ(deviceId, channels, settings) {
    this.send({
      type: 'start_cross_adaptive_eq',
      device_id: deviceId,
      channels: channels,
      settings: settings
    });
  }
  stopCrossAdaptiveEQ() {
    this.send({ type: 'stop_cross_adaptive_eq' });
  }
  getCrossAdaptiveEQStatus() {
    this.send({ type: 'get_cross_adaptive_eq_status' });
  }
}

export default new WebSocketService();
