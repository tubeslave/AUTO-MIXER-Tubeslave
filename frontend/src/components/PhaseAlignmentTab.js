import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './PhaseAlignmentTab.css';

function PhaseAlignmentTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [isActive, setIsActive] = useState(false);
  const [referenceChannel, setReferenceChannel] = useState(null);
  const [channelsToAlign, setChannelsToAlign] = useState([]);
  const [measurements, setMeasurements] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  
  // NEW: GCC-PHAT settings (from Kimi method)
  const [gccPhatSettings, setGccPhatSettings] = useState({
    fftSize: 4096,
    hopSize: 2048,
    maxDelayMs: 50,
    correlationThreshold: 0.5,
    psrThreshold: 5.0,
    updateIntervalMs: 100,
    useTemporalSmoothing: true,
    smoothingWindow: 10
  });
  
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  
  // Initialize reference channel and channels to align
  useEffect(() => {
    if (selectedChannels && selectedChannels.length > 0) {
      if (!referenceChannel) {
        setReferenceChannel(selectedChannels[0]);
      }
      // Channels to align are all selected channels except reference
      const alignChannels = selectedChannels.filter(ch => ch !== referenceChannel);
      setChannelsToAlign(alignChannels);
    }
  }, [selectedChannels, referenceChannel]);
  
  // WebSocket event handlers
  useEffect(() => {
    const handlePhaseAlignmentStatus = (data) => {
      console.log('Phase alignment status update:', data);
      
      if (data.active !== undefined) {
        setIsActive(data.active);
      }
      
      if (data.message) {
        setStatusMessage(data.message);
      }
      
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setIsActive(false);
      }
      
      if (data.reference_channel !== undefined) {
        setReferenceChannel(data.reference_channel);
      }
      
      if (data.channels) {
        setChannelsToAlign(data.channels);
      }
    };
    
    const handlePhaseAlignmentMeasurements = (data) => {
      if (data.measurements) {
        setMeasurements(data.measurements);
      }
    };
    
    const handlePhaseAlignmentApplyResult = (data) => {
      if (data.success) {
        setStatusMessage('Phase/delay corrections applied to mixer');
        if (data.corrections) {
          console.log('Applied corrections:', data.corrections);
        }
      } else {
        setStatusMessage(`Failed to apply corrections: ${data.error || 'Unknown error'}`);
      }
    };
    
    const handlePhaseAlignmentResetResult = (data) => {
      if (data.success) {
        setStatusMessage(data.message || 'Phase and delay reset for all channels');
        setMeasurements({});
      } else {
        setStatusMessage(`Failed to reset: ${data.error || 'Unknown error'}`);
      }
    };
    
    websocketService.on('phase_alignment_status', handlePhaseAlignmentStatus);
    websocketService.on('phase_alignment_measurements', handlePhaseAlignmentMeasurements);
    websocketService.on('phase_alignment_apply_result', handlePhaseAlignmentApplyResult);
    websocketService.on('phase_alignment_reset_result', handlePhaseAlignmentResetResult);
    
    websocketService.getPhaseAlignmentStatus();
    
    return () => {
      websocketService.off('phase_alignment_status', handlePhaseAlignmentStatus);
      websocketService.off('phase_alignment_measurements', handlePhaseAlignmentMeasurements);
      websocketService.off('phase_alignment_apply_result', handlePhaseAlignmentApplyResult);
      websocketService.off('phase_alignment_reset_result', handlePhaseAlignmentResetResult);
    };
  }, []);
  
  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };
  
  const handleReferenceChannelChange = (e) => {
    const newRef = parseInt(e.target.value);
    setReferenceChannel(newRef);
    const alignChannels = selectedChannels.filter(ch => ch !== newRef);
    setChannelsToAlign(alignChannels);
  };
  
  const handleGccPhatSettingChange = (setting, value) => {
    setGccPhatSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };
  
  const handleStartAnalysis = () => {
    if (!selectedDevice) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    if (!referenceChannel) {
      setStatusMessage('Please select a reference channel');
      return;
    }
    
    if (channelsToAlign.length === 0) {
      setStatusMessage('Please select channels to align');
      return;
    }
    
    setStatusMessage('Starting GCC-PHAT phase alignment analysis...');
    
    // NEW: Send GCC-PHAT settings along with command
    websocketService.send({
      type: 'start_phase_alignment',
      device_id: selectedDevice,
      reference_channel: referenceChannel,
      channels: channelsToAlign,
      settings: gccPhatSettings
    });
  };
  
  const handleStopAnalysis = () => {
    setStatusMessage('Stopping phase alignment analysis...');
    websocketService.send({ type: 'stop_phase_alignment' });
  };
  
  const handleApplyCorrections = () => {
    websocketService.send({ type: 'apply_phase_corrections' });
    setStatusMessage('Applying phase/delay corrections...');
  };
  
  const handleResetPhaseDelay = () => {
    if (!window.confirm('This will reset and disable all phase inversions and delays on selected channels. Continue?')) {
      return;
    }
    
    setStatusMessage('Resetting phase and delay for all channels...');
    const channelsToReset = selectedChannels && selectedChannels.length > 0 
      ? selectedChannels 
      : channelsToAlign;
    websocketService.send({
      type: 'reset_phase_delay',
      channels: channelsToReset
    });
  };
  
  const formatDelay = (ms) => {
    if (ms === undefined || ms === null) return '--';
    const sign = ms >= 0 ? '+' : '';
    return `${sign}${ms.toFixed(2)} ms`;
  };
  
  const getCorrelationColor = (correlation) => {
    if (correlation >= 0.8) return 'correlation-excellent';
    if (correlation >= 0.5) return 'correlation-good';
    if (correlation >= 0.3) return 'correlation-poor';
    return 'correlation-bad';
  };
  
  const getConfidenceIndicator = (confidence) => {
    if (confidence >= 0.8) return '✓✓✓';
    if (confidence >= 0.6) return '✓✓';
    if (confidence >= 0.4) return '✓';
    return '?';
  };
  
  if (!selectedChannels || selectedChannels.length === 0) {
    return (
      <div className="phase-alignment-tab">
        <div className="phase-alignment-section">
          <h2>PHASE ALIGNMENT</h2>
          <div className="no-channels-message">
            <p>No channels selected for processing.</p>
            <p>Please go to the "Mixer Connection" tab and select channels to process.</p>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="phase-alignment-tab">
      <div className="phase-alignment-section">
        <h2>GCC-PHAT PHASE ALIGNMENT</h2>
        
        <div className="phase-alignment-header">
          <p className="section-description">
            Automatic time alignment using GCC-PHAT (Generalized Cross-Correlation Phase Transform).
            Based on Intelligent Music Production method (Knapp & Carter, 1976).
          </p>
        </div>
        
        {/* NEW: GCC-PHAT Settings Panel */}
        <div className="gcc-phat-settings-panel">
          <div className="settings-header" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
            <h3>GCC-PHAT Settings</h3>
            <span className="toggle-icon">{showAdvancedSettings ? '▼' : '▶'}</span>
          </div>
          
          <div className="settings-main">
            <div className="setting-group">
              <label>FFT Size</label>
              <div className="setting-control">
                <select
                  value={gccPhatSettings.fftSize}
                  onChange={(e) => handleGccPhatSettingChange('fftSize', parseInt(e.target.value))}
                  disabled={isActive}
                >
                  <option value={2048}>2048 (Lower latency)</option>
                  <option value={4096}>4096 (Balanced)</option>
                  <option value={8192}>8192 (Higher precision)</option>
                </select>
                <small>Larger = better precision, more latency</small>
              </div>
            </div>
            
            <div className="setting-group">
              <label>Max Delay</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={gccPhatSettings.maxDelayMs}
                  onChange={(e) => handleGccPhatSettingChange('maxDelayMs', parseInt(e.target.value))}
                  disabled={isActive}
                />
                <span className="setting-value">{gccPhatSettings.maxDelayMs} ms</span>
              </div>
            </div>
            
            <div className="setting-group">
              <label>Correlation Threshold</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={gccPhatSettings.correlationThreshold}
                  onChange={(e) => handleGccPhatSettingChange('correlationThreshold', parseFloat(e.target.value))}
                  disabled={isActive}
                />
                <span className="setting-value">{gccPhatSettings.correlationThreshold}</span>
              </div>
              <small>Minimum correlation to accept measurement</small>
            </div>
          </div>
          
          {showAdvancedSettings && (
            <div className="settings-advanced">
              <div className="settings-section">
                <h4>Quality Metrics</h4>
                
                <div className="setting-group">
                  <label>PSR Threshold</label>
                  <div className="setting-control">
                    <input
                      type="range"
                      min="3"
                      max="15"
                      step="0.5"
                      value={gccPhatSettings.psrThreshold}
                      onChange={(e) => handleGccPhatSettingChange('psrThreshold', parseFloat(e.target.value))}
                      disabled={isActive}
                    />
                    <span className="setting-value">{gccPhatSettings.psrThreshold} dB</span>
                  </div>
                  <small>Peak-to-Sidelobe Ratio threshold</small>
                </div>
                
                <div className="setting-group">
                  <label>Update Interval</label>
                  <div className="setting-control">
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="10"
                      value={gccPhatSettings.updateIntervalMs}
                      onChange={(e) => handleGccPhatSettingChange('updateIntervalMs', parseInt(e.target.value))}
                      disabled={isActive}
                    />
                    <span className="setting-value">{gccPhatSettings.updateIntervalMs} ms</span>
                  </div>
                  <small>Rate limiting for OSC commands</small>
                </div>
              </div>
              
              <div className="settings-section">
                <h4>Temporal Smoothing</h4>
                
                <div className="setting-group">
                  <label>Enable Smoothing</label>
                  <div className="setting-control">
                    <input
                      type="checkbox"
                      checked={gccPhatSettings.useTemporalSmoothing}
                      onChange={(e) => handleGccPhatSettingChange('useTemporalSmoothing', e.target.checked)}
                      disabled={isActive}
                    />
                    <span>Use median filtering</span>
                  </div>
                </div>
                
                {gccPhatSettings.useTemporalSmoothing && (
                  <div className="setting-group">
                    <label>Smoothing Window</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="3"
                        max="20"
                        value={gccPhatSettings.smoothingWindow}
                        onChange={(e) => handleGccPhatSettingChange('smoothingWindow', parseInt(e.target.value))}
                        disabled={isActive}
                      />
                      <span className="setting-value">{gccPhatSettings.smoothingWindow} frames</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Channel Selection */}
        <div className="channel-selection-section">
          <div className="control-group">
            <label>Reference Channel:</label>
            <select
              value={referenceChannel || ''}
              onChange={handleReferenceChannelChange}
              disabled={isActive}
              className="channel-select"
            >
              {selectedChannels.map(ch => (
                <option key={ch} value={ch}>
                  {getChannelName(ch)}
                </option>
              ))}
            </select>
            <small>The reference channel is the timing baseline</small>
          </div>
          
          <div className="control-group">
            <label>Channels to Align:</label>
            <div className="channels-to-align-list">
              {channelsToAlign.length > 0 ? (
                channelsToAlign.map(ch => (
                  <span key={ch} className="channel-tag">
                    {getChannelName(ch)}
                  </span>
                ))
              ) : (
                <span className="no-channels">No channels selected (reference excluded)</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Control Buttons */}
        <div className="control-buttons">
          {!isActive ? (
            <button
              className="btn-phase start"
              onClick={handleStartAnalysis}
              disabled={!selectedDevice || channelsToAlign.length === 0}
            >
              Start GCC-PHAT Analysis
            </button>
          ) : (
            <button
              className="btn-phase stop"
              onClick={handleStopAnalysis}
            >
              Stop Analysis
            </button>
          )}
          
          <button
            className="btn-phase apply"
            onClick={handleApplyCorrections}
            disabled={!isActive && Object.keys(measurements).length === 0}
          >
            Apply Corrections
          </button>
          
          <button
            className="btn-phase reset"
            onClick={handleResetPhaseDelay}
            disabled={selectedChannels.length === 0}
          >
            Reset All
          </button>
        </div>
        
        {statusMessage && (
          <div className="status-message">
            {statusMessage}
          </div>
        )}
        
        {/* Measurements Table */}
        {Object.keys(measurements).length > 0 && (
          <div className="measurements-table">
            <h3>Measurements</h3>
            <div className="table-header">
              <div className="col-channel">Channel</div>
              <div className="col-delay">Delay</div>
              <div className="col-correlation">Correlation</div>
              <div className="col-psr">PSR</div>
              <div className="col-confidence">Conf</div>
              <div className="col-status">Status</div>
            </div>
            
            <div className="table-body">
              {Object.entries(measurements).map(([channelId, data]) => (
                <div key={channelId} className="table-row">
                  <div className="col-channel">
                    {getChannelName(parseInt(channelId))}
                  </div>
                  <div className="col-delay">
                    {formatDelay(data.delay_ms)}
                  </div>
                  <div className={`col-correlation ${getCorrelationColor(data.correlation)}`}>
                    {(data.correlation * 100).toFixed(1)}%
                  </div>
                  <div className="col-psr">
                    {data.psr?.toFixed(1) || '--'} dB
                  </div>
                  <div className="col-confidence">
                    {getConfidenceIndicator(data.confidence)}
                  </div>
                  <div className="col-status">
                    {data.valid ? (
                      <span className="status-valid">✓ Valid</span>
                    ) : (
                      <span className="status-invalid">✗ Invalid</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Help Text */}
        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li><strong>Select reference channel</strong> — typically the closest mic (e.g., Kick In)</li>
            <li><strong>Add channels to align</strong> — select other mics of the same source (e.g., Kick Out, OH)</li>
            <li><strong>Start analysis</strong> — GCC-PHAT computes delays between reference and each channel</li>
            <li><strong>Apply corrections</strong> — delays are sent to mixer via OSC</li>
          </ol>
          
          <h4>Quality Indicators:</h4>
          <ul>
            <li><strong>Correlation</strong> — how similar are the signals (higher = better)</li>
            <li><strong>PSR</strong> — Peak-to-Sidelobe Ratio (higher = more confident)</li>
            <li><strong>Confidence</strong> — overall measurement quality (✓✓✓ = excellent)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default PhaseAlignmentTab;
