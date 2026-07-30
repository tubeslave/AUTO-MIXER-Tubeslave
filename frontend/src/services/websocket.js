import { BaseWebSocketTransport } from './websocketCore.mjs';
import {
  buildDecisionQueueFirstPageRequest,
  buildDecisionQueueNextPageRequest,
  buildProposalApplyRequest,
  buildProposalDetailRequest
} from './decisionQueueV2.mjs';

class WebSocketService extends BaseWebSocketTransport {
  setServerUrl(url) {
    this.setUrl(url);
  }

  getServerUrl() {
    return this.getUrl();
  }

  // Wing direct connection
  connectWing(ip, sendPort = 2223, receivePort = 2223, observationOnly = true) {
    this.send({
      type: 'connect_wing',
      ip,
      send_port: sendPort,
      receive_port: receivePort,
      observation_only: observationOnly
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

  // Scan network for supported mixers (WING + dLive)
  scanMixers(fullScan = false, subnet = null) {
    this.send({
      type: 'scan_mixers',
      full_scan: fullScan,
      subnet
    });
  }

  autoConnectMixer(preferredType = null, preferredIp = null, fullScan = false) {
    this.send({
      type: 'auto_connect',
      preferred_type: preferredType,
      preferred_ip: preferredIp,
      full_scan: fullScan
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

  getManualWriteSupervision() {
    this.send({
      type: 'get_manual_write_supervision'
    });
  }

  resetManualWriteEmergencyStop(approved = false, approvalId = '', reason = 'frontend_operator_reset') {
    this.send({
      type: 'reset_manual_write_emergency_stop',
      approved,
      approval_id: approvalId,
      reason
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

  subscribeMeters() {
    this.send({
      type: 'subscribe_meters'
    });
  }

  unsubscribeMeters() {
    this.send({
      type: 'unsubscribe_meters'
    });
  }

  getChannelInventory(totalChannels = 24) {
    this.send({
      type: 'get_channel_inventory',
      total_channels: totalChannels
    });
  }

  getDecisionQueue() {
    return this.send(buildDecisionQueueFirstPageRequest());
  }

  getDecisionQueueNextPage(cursor, expectedSnapshotRevision) {
    try {
      return this.send(buildDecisionQueueNextPageRequest(cursor, expectedSnapshotRevision));
    } catch (error) {
      console.error('Decision Queue pagination request was rejected:', error);
      return false;
    }
  }

  getOperatorProposalDetail(identity, chunkIndex = 0) {
    try {
      return this.send(buildProposalDetailRequest(identity, chunkIndex));
    } catch (error) {
      console.error('Proposal detail request was rejected:', error);
      return false;
    }
  }

  getDecisionTraces(limit = 50) {
    this.send({
      type: 'get_decision_traces',
      limit
    });
  }

  getSafetyMonitor() {
    this.send({
      type: 'get_safety_monitor'
    });
  }

  getReplayRuns(limit = 20) {
    this.send({
      type: 'get_replay_runs',
      limit
    });
  }

  startReplayRun(runId = '') {
    this.send({
      type: 'start_replay_run',
      run_id: runId
    });
  }

  getChannelIntelligence(totalChannels = 48) {
    this.send({
      type: 'get_channel_intelligence',
      total_channels: totalChannels
    });
  }

  getOperatorAgentRoster() {
    this.send({
      type: 'get_operator_agent_roster'
    });
  }

  getConnectionTopology() {
    this.send({
      type: 'get_connection_topology'
    });
  }

  getGroupsBusesSnapshot() {
    this.send({
      type: 'get_groups_buses_snapshot'
    });
  }

  getMemoryKnowledgeSnapshot() {
    this.send({
      type: 'get_memory_knowledge_snapshot'
    });
  }

  getTrainingSessionsSnapshot() {
    this.send({
      type: 'get_training_sessions_snapshot'
    });
  }

  getRulesLibrarySnapshot() {
    this.send({
      type: 'get_rules_library_snapshot'
    });
  }

  getAgentCommunicationsSnapshot() {
    this.send({
      type: 'get_agent_communications_snapshot'
    });
  }

  getProcessingPipelineSnapshot(includeMethods = false) {
    this.send({
      type: 'get_processing_pipeline_snapshot',
      include_methods: includeMethods
    });
  }

  getMainPipelinePreview(totalChannels = 48, includeMethods = false) {
    this.send({
      type: 'get_main_pipeline_preview',
      total_channels: totalChannels,
      include_methods: includeMethods
    });
  }

  getTelemetryAlertsSnapshot(limit = 50, sessionLimit = 8) {
    this.send({
      type: 'get_telemetry_alerts_snapshot',
      limit,
      session_limit: sessionLimit
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

  applyOperatorProposal(identity, approved = false, reason = 'frontend_supervised_apply') {
    try {
      const proposalId = identity?.proposalId;
      return this.send(buildProposalApplyRequest({
        identity,
        approved,
        reason,
        approvalId: `frontend-${proposalId}`,
        idempotencyKey: `frontend-${proposalId}-${Date.now()}`
      }));
    } catch (error) {
      console.error('Proposal apply was rejected before transport:', error);
      return false;
    }
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

  // AI Mixing Agent commands
  startAgent(mode = 'suggest', channels = [], useLlm = true, allowAutoApply = false) {
    this.send({
      type: 'start_agent',
      mode,
      channels,
      use_llm: useLlm,
      allow_auto_apply: allowAutoApply
    });
  }

  stopAgent() {
    this.send({ type: 'stop_agent' });
  }

  emergencyStop() {
    return this.send({
      type: 'emergency_stop',
      reason: 'frontend_operator_emergency_stop'
    });
  }

  emergencyStopAgent() {
    this.send({ type: 'emergency_stop_agent' });
  }

  getAgentStatus() {
    this.send({ type: 'get_agent_status' });
  }

  setAgentMode(mode = 'suggest', channels = [], useLlm = true, allowAutoApply = false, start = false) {
    this.send({
      type: 'set_agent_mode',
      mode,
      channels,
      use_llm: useLlm,
      allow_auto_apply: allowAutoApply,
      start
    });
  }

  updateAgentState(channels = []) {
    this.send({
      type: 'update_agent_state',
      channels
    });
  }

  getPendingActions() {
    this.send({ type: 'get_pending_actions' });
  }

  approveAction(index) {
    this.send({ type: 'approve_action', index });
  }

  approveAllActions() {
    this.send({ type: 'approve_all_actions' });
  }

  dismissAction(index) {
    this.send({ type: 'dismiss_action', index });
  }

  dismissAllActions() {
    this.send({ type: 'dismiss_all_actions' });
  }

  getActionHistory(limit = 50) {
    this.send({ type: 'get_action_history', limit });
  }

  getAudioCaptureStatus() {
    this.send({ type: 'get_audio_capture_status' });
  }

  // Ф10 (DEC-0010, контракт MSG-CLD-0031): голосовой помощник саундчека v1.
  // Невалидные lang/enabled бэкенд игнорирует молча, статус без изменений.
  setVoiceLanguage(lang) {
    this.send({ type: 'set_voice_language', lang });
  }

  setVoiceEnabled(enabled) {
    this.send({ type: 'set_voice_enabled', enabled });
  }

  getVoiceAssistantStatus() {
    this.send({ type: 'get_voice_assistant_status' });
  }

  startAudioCapturePreview(deviceId = null, numChannels = 48, sampleRate = 48000) {
    this.send({
      type: 'start_audio_capture_preview',
      device_id: deviceId,
      num_channels: numChannels,
      sample_rate: sampleRate
    });
  }

  stopAudioCapturePreview() {
    this.send({ type: 'stop_audio_capture_preview' });
  }

  getChannelMeters() {
    this.send({ type: 'get_channel_meters' });
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
      mode: options.mode || 'soundcheck',
      learning_duration_sec: options.learning_duration_sec,
      safe_gain_settings: options.safe_gain_settings || options.processing || {}
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

  getLiveInputTrimStatus() {
    this.send({
      type: 'get_live_input_trim_status'
    });
  }

  getFreezeStatus() {
    this.send({
      type: 'get_freeze_status'
    });
  }

  getReadyForLive() {
    this.send({
      type: 'get_ready_for_live'
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

  scanAudioDevices() {
    this.send({
      type: 'scan_audio_devices'
    });
  }

  selectAudioDevice(preferredName = null, preferredProtocol = null, minChannels = 2) {
    this.send({
      type: 'select_audio_device',
      preferred_name: preferredName,
      preferred_protocol: preferredProtocol,
      min_channels: minChannels
    });
  }

  createSnapshot(channels = []) {
    this.send({
      type: 'create_snapshot',
      channels
    });
  }

  restoreSnapshot(snapshotPath = null) {
    this.send({
      type: 'undo_restore_snapshot',
      snapshot_path: snapshotPath
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

  // ========== System Measurement / Master EQ ==========

  startSystemMeasurement(
    deviceId,
    referenceChannel,
    measurementChannel,
    durationSec = 6,
    targetBus = 'master',
    targetId = 1,
    correctionMode = 'flat',
    referenceCurve = 'pink_noise_live_pa'
  ) {
    this.send({
      type: 'start_system_measurement',
      device_id: deviceId,
      reference_channel: referenceChannel,
      measurement_channel: measurementChannel,
      duration_sec: durationSec,
      target_bus: targetBus,
      target_id: targetId,
      correction_mode: correctionMode,
      reference_curve: referenceCurve,
    });
  }

  applySystemMeasurement(targetBus = 'master', targetId = 1) {
    this.send({
      type: 'apply_system_measurement',
      target_bus: targetBus,
      target_id: targetId,
    });
  }

  resetSystemMeasurement(targetBus = 'master', targetId = 1) {
    this.send({
      type: 'reset_system_measurement',
      target_bus: targetBus,
      target_id: targetId,
    });
  }

  getSystemMeasurementStatus() {
    this.send({
      type: 'get_system_measurement_status',
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

  startAutoSoundcheck(deviceId, channels, channelSettings, channelMapping, timings, observeOnly = true, mode = 'full_band') {
    console.log('startAutoSoundcheck called with:', { deviceId, channels, timings, observeOnly, mode });
    this.send({
      type: 'start_auto_soundcheck',
      device_id: deviceId,
      channels: channels,
      channel_settings: channelSettings,
      channel_mapping: channelMapping,
      timings: timings,
      observe_only: observeOnly,
      mode: mode
    });
  }

  nextSoundcheckChannel() {
    this.send({ type: 'soundcheck_next_channel' });
  }

  skipSoundcheckChannel() {
    this.send({ type: 'soundcheck_skip_channel' });
  }

  repeatSoundcheckChannel() {
    this.send({ type: 'soundcheck_repeat_channel' });
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

  acceptSoundcheckReplan(runId, draftId, approvalId) {
    return this.send({
      type: 'accept_soundcheck_replan',
      run_id: runId,
      draft_id: draftId,
      approved: true,
      approval_id: approvalId
    });
  }

  reanalyzeSoundcheckStage(runId, approvalId) {
    return this.send({
      type: 'reanalyze_soundcheck_stage',
      run_id: runId,
      approved: true,
      approval_id: approvalId
    });
  }

  finalizeSoundcheckReadiness(runId) {
    return this.send({
      type: 'finalize_soundcheck_readiness',
      run_id: runId
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

}

const websocketService = new WebSocketService();

export default websocketService;
