const BASE_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 15000;
const RECONNECT_FACTOR = 1.7;
const HEARTBEAT_INTERVAL_MS = 5000;
const HEARTBEAT_MISS_LIMIT = 2;
const OFFLINE_QUEUE_LIMIT = 100;

function resolveWsUrl() {
  // Приоритет: env сборки → localStorage (правится без пересборки) →
  // хост страницы (продакшен за одним хостом) → localhost (dev)
  if (process.env.REACT_APP_WS_URL) return process.env.REACT_APP_WS_URL;
  try {
    const stored = window.localStorage.getItem('automixer_ws_url');
    if (stored) return stored;
  } catch (e) { /* приватный режим */ }
  if (typeof window !== 'undefined' && window.location && window.location.hostname
      && window.location.protocol.startsWith('http')) {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.hostname}:8765`;
  }
  return 'ws://localhost:8765';
}

function withAuthToken(url) {
  try {
    const token = window.localStorage.getItem('automixer_token');
    if (token) {
      return url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
    }
  } catch (e) { /* приватный режим */ }
  return url;
}

class WebSocketService {
  constructor() {
    this.ws = null;
    this.url = resolveWsUrl();
    this.listeners = new Map();
    this.reconnectTimer = null;
    this.reconnectDelay = BASE_RECONNECT_MS;
    this.heartbeatTimer = null;
    this.missedPongs = 0;
    this.mixerConnected = false;
    this.offlineQueue = [];
    this.intentionalClose = false;
  }

  emitConnectionState(state) {
    this.notifyListeners('connection_state', {
      type: 'connection_state',
      state,
      mixerConnected: this.mixerConnected,
      url: this.url,
    });
  }

  startHeartbeat() {
    this.stopHeartbeat();
    this.missedPongs = 0;
    this.heartbeatTimer = setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      this.missedPongs += 1;
      if (this.missedPongs > HEARTBEAT_MISS_LIMIT) {
        // Бэкенд молчит: TCP может «висеть» десятки секунд, не сообщая
        // об обрыве — принудительно рвём соединение, onclose запустит
        // реконнект с backoff
        console.warn('Heartbeat lost, forcing reconnect');
        try { this.ws.close(); } catch (e) { /* уже закрыт */ }
        return;
      }
      try {
        this.ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
      } catch (e) { /* onclose разберётся */ }
    }, HEARTBEAT_INTERVAL_MS);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  flushOfflineQueue() {
    const queued = this.offlineQueue.splice(0, this.offlineQueue.length);
    queued.forEach((message) => this.send(message));
  }

  notifyUnsupportedCommand(commandType, eventType, message) {
    console.warn(`WebSocket command disabled: ${commandType}. ${message}`);
    this.notifyListeners(eventType, {
      type: eventType,
      active: false,
      implemented: false,
      error: message,
      unsupported_command: commandType
    });
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        this.intentionalClose = false;
        this.ws = new WebSocket(withAuthToken(this.url));

        this.ws.onopen = () => {
          console.log('WebSocket connected:', this.url);
          if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
          }
          this.reconnectDelay = BASE_RECONNECT_MS;
          this.startHeartbeat();
          this.emitConnectionState('connected');
          this.notifyListeners('connected', { type: 'connected' });
          this.flushOfflineQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'pong') {
              this.missedPongs = 0;
              if (this.mixerConnected !== Boolean(data.mixer_connected)) {
                this.mixerConnected = Boolean(data.mixer_connected);
                this.emitConnectionState('connected');
              }
              return;
            }
            this.notifyListeners(data.type, data);
          } catch (error) {
            console.error('Error parsing message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = (event) => {
          this.stopHeartbeat();
          this.notifyListeners('disconnected', {});
          if (event && event.code === 4401) {
            // Auth отклонён — реконнект с тем же токеном бессмыслен
            console.error('WebSocket auth rejected (4401), not reconnecting');
            this.emitConnectionState('unauthorized');
            return;
          }
          this.emitConnectionState('reconnecting');
          if (!this.intentionalClose) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    // Экспоненциальный backoff с джиттером: без него упавший бэкенд
    // получает синхронный флуд реконнектов от всех клиентов
    const jitter = 0.8 + Math.random() * 0.4;
    const delay = Math.round(this.reconnectDelay * jitter);
    this.reconnectDelay = Math.min(this.reconnectDelay * RECONNECT_FACTOR, MAX_RECONNECT_MS);
    console.log(`Reconnecting in ${delay} ms...`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect().catch(() => this.scheduleReconnect());
    }, delay);
  }

  disconnect() {
    this.intentionalClose = true;
    this.stopHeartbeat();
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
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
      }
      return;
    }
    // Оффлайн: кладём в очередь (с лимитом) — уйдёт после реконнекта.
    // Терять команды оператора молча нельзя
    if (this.offlineQueue.length >= OFFLINE_QUEUE_LIMIT) {
      this.offlineQueue.shift();
    }
    this.offlineQueue.push(message);
    console.warn(`WebSocket offline, message queued (${this.offlineQueue.length}):`, message.type);
    if (!this.reconnectTimer && (!this.ws || this.ws.readyState === WebSocket.CLOSED)) {
      this.scheduleReconnect();
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
    if (!this.listeners.has(eventType)) return;
    this.listeners.get(eventType).forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in listener for ${eventType}:`, error);
      }
    });
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

  getSupervisedPilotStatus() {
    this.send({
      type: 'get_supervised_pilot_status'
    });
  }

  getOperatorMode() {
    this.send({
      type: 'get_operator_mode'
    });
  }

  setOperatorMode(mode, reason = 'frontend_mode_control') {
    this.send({
      type: 'set_operator_mode',
      mode,
      reason
    });
  }

  getDashboardSnapshot() {
    this.send({
      type: 'get_dashboard_snapshot'
    });
  }

  getChannelInventory(totalChannels = 24) {
    this.send({
      type: 'get_channel_inventory',
      total_channels: totalChannels
    });
  }

  getDecisionQueue(limit = 50) {
    this.send({
      type: 'get_decision_queue',
      limit
    });
  }

  getConnectionTopology() {
    this.send({
      type: 'get_connection_topology'
    });
  }

  runOperatorAnalysis(totalChannels = 24, createProposals = true) {
    this.send({
      type: 'run_operator_analysis',
      total_channels: totalChannels,
      create_proposals: createProposals
    });
  }

  importSafeGainSuggestions() {
    this.send({
      type: 'import_safe_gain_suggestions'
    });
  }

  importSoundcheckRecommendations(bundle = null) {
    const message = {
      type: 'import_soundcheck_recommendations'
    };
    if (bundle) message.bundle = bundle;
    this.send(message);
  }

  createOperatorProposal(proposal) {
    this.send({
      type: 'create_operator_proposal',
      proposal
    });
  }

  acceptOperatorProposal(proposalId) {
    this.send({
      type: 'accept_operator_proposal',
      proposal_id: proposalId
    });
  }

  dismissOperatorProposal(proposalId, reason = 'frontend_dismiss') {
    this.send({
      type: 'dismiss_operator_proposal',
      proposal_id: proposalId,
      reason
    });
  }

  applyOperatorProposal(proposalId, approved = false, reason = 'frontend_supervised_apply') {
    this.send({
      type: 'apply_operator_proposal',
      proposal_id: proposalId,
      approved,
      reason,
      approval_id: `frontend-${proposalId}`
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
    this.notifyUnsupportedCommand(
      'start_auto_panner',
      'auto_panner_status',
      'Auto Panner backend control is not implemented yet.'
    );
  }
  stopAutoPanner() {
    this.notifyUnsupportedCommand(
      'stop_auto_panner',
      'auto_panner_status',
      'Auto Panner backend control is not implemented yet.'
    );
  }
  getAutoPannerStatus() {
    this.send({ type: 'get_auto_panner_status' });
  }
  calculateAutoPanning(channels, instrumentTypes, spectralCentroids) {
    this.notifyUnsupportedCommand(
      'calculate_auto_panning',
      'auto_panner_status',
      'Auto Panner calculation is not implemented yet.'
    );
  }
  applyAutoPanning() {
    this.notifyUnsupportedCommand(
      'apply_auto_panning',
      'auto_panner_status',
      'Auto Panner apply is not implemented yet.'
    );
  }

  // ========== Auto Reverb ==========
  startAutoReverb(deviceId, channels, instrumentTypes, spectralCentroids, spectralFluxes) {
    this.notifyUnsupportedCommand(
      'start_auto_reverb',
      'auto_reverb_status',
      'Auto Reverb backend control is not implemented yet.'
    );
  }
  stopAutoReverb() {
    this.notifyUnsupportedCommand(
      'stop_auto_reverb',
      'auto_reverb_status',
      'Auto Reverb backend control is not implemented yet.'
    );
  }
  getAutoReverbStatus() {
    this.send({ type: 'get_auto_reverb_status' });
  }
  calculateAutoReverb(channels, instrumentTypes, spectralCentroids, spectralFluxes) {
    this.notifyUnsupportedCommand(
      'calculate_auto_reverb',
      'auto_reverb_status',
      'Auto Reverb calculation is not implemented yet.'
    );
  }
  applyAutoReverb() {
    this.notifyUnsupportedCommand(
      'apply_auto_reverb',
      'auto_reverb_status',
      'Auto Reverb apply is not implemented yet.'
    );
  }

  // ========== Auto Gate ==========
  startAutoGate(deviceId, channels, channelConfigs, settings) {
    this.notifyUnsupportedCommand(
      'start_auto_gate',
      'auto_gate_status',
      'Auto Gate backend control is not implemented yet.'
    );
  }
  stopAutoGate() {
    this.notifyUnsupportedCommand(
      'stop_auto_gate',
      'auto_gate_status',
      'Auto Gate backend control is not implemented yet.'
    );
  }
  getAutoGateStatus() {
    this.send({ type: 'get_auto_gate_status' });
  }
  configureGateChannel(channelId, config) {
    this.notifyUnsupportedCommand(
      'configure_gate_channel',
      'auto_gate_status',
      'Auto Gate channel configuration is not implemented yet.'
    );
  }

  // ========== Auto Effects ==========
  startAutoEffects(deviceId, channels, settings) {
    this.notifyUnsupportedCommand(
      'start_auto_effects',
      'auto_effects_status',
      'Auto Effects backend control is not implemented yet.'
    );
  }
  stopAutoEffects() {
    this.notifyUnsupportedCommand(
      'stop_auto_effects',
      'auto_effects_status',
      'Auto Effects backend control is not implemented yet.'
    );
  }
  getAutoEffectsStatus() {
    this.send({ type: 'get_auto_effects_status' });
  }

  // ========== Cross-Adaptive EQ ==========
  startCrossAdaptiveEQ(deviceId, channels, settings) {
    this.notifyUnsupportedCommand(
      'start_cross_adaptive_eq',
      'cross_adaptive_eq_status',
      'Cross-Adaptive EQ backend control is not implemented yet.'
    );
  }
  stopCrossAdaptiveEQ() {
    this.notifyUnsupportedCommand(
      'stop_cross_adaptive_eq',
      'cross_adaptive_eq_status',
      'Cross-Adaptive EQ backend control is not implemented yet.'
    );
  }
  getCrossAdaptiveEQStatus() {
    this.send({ type: 'get_cross_adaptive_eq_status' });
  }
}

const websocketService = new WebSocketService();

export default websocketService;
export { WebSocketService, resolveWsUrl, BASE_RECONNECT_MS, MAX_RECONNECT_MS, OFFLINE_QUEUE_LIMIT };
