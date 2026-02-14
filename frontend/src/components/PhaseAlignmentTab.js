import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './PhaseAlignmentTab.css';

function PhaseAlignmentTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [isActive, setIsActive] = useState(false);
  const [referenceChannel, setReferenceChannel] = useState(null);
  const [channelsToAlign, setChannelsToAlign] = useState([]);
  const [measurements, setMeasurements] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  
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
          // Можно показать примененные коррекции
          console.log('Applied corrections:', data.corrections);
        }
      } else {
        setStatusMessage(`Failed to apply corrections: ${data.error || 'Unknown error'}`);
      }
    };
    
    const handlePhaseAlignmentResetResult = (data) => {
      if (data.success) {
        setStatusMessage(data.message || 'Phase and delay reset for all channels');
        // Очистить измерения после сброса
        setMeasurements({});
      } else {
        setStatusMessage(`Failed to reset: ${data.error || 'Unknown error'}`);
      }
    };
    
    websocketService.on('phase_alignment_status', handlePhaseAlignmentStatus);
    websocketService.on('phase_alignment_measurements', handlePhaseAlignmentMeasurements);
    websocketService.on('phase_alignment_apply_result', handlePhaseAlignmentApplyResult);
    websocketService.on('phase_alignment_reset_result', handlePhaseAlignmentResetResult);
    
    // Get initial status
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
    // Update channels to align
    const alignChannels = selectedChannels.filter(ch => ch !== newRef);
    setChannelsToAlign(alignChannels);
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
    
    setStatusMessage('Starting phase alignment analysis...');
    websocketService.startPhaseAlignment(selectedDevice, referenceChannel, channelsToAlign);
  };
  
  const handleStopAndApply = () => {
    setStatusMessage('Stopping analysis and applying corrections...');
    // Остановка анализа автоматически применит коррекции на бэкенде
    websocketService.stopPhaseAlignment();
  };
  
  const handleResetPhaseDelay = () => {
    if (!window.confirm('This will reset and disable all phase inversions and delays on all selected channels. Continue?')) {
      return;
    }
    
    setStatusMessage('Resetting phase and delay for all channels...');
    // Сбрасываем для всех выбранных каналов (включая reference channel)
    const channelsToReset = selectedChannels && selectedChannels.length > 0 ? selectedChannels : channelsToAlign;
    websocketService.resetPhaseDelay(channelsToReset);
  };
  
  const formatDelay = (ms) => {
    if (ms === undefined || ms === null) return '--';
    const sign = ms >= 0 ? '+' : '';
    return `${sign}${ms.toFixed(2)} ms`;
  };
  
  const formatPhase = (deg) => {
    if (deg === undefined || deg === null) return '--';
    const sign = deg >= 0 ? '+' : '';
    return `${sign}${deg.toFixed(1)}°`;
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
        <h2>PHASE ALIGNMENT</h2>
        
        <div className="phase-alignment-header">
          <p className="section-description">
            Analyze and correct phase and delay relationships between channels.
            Select a reference channel and channels to align relative to it.
          </p>
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
          </div>
          
          <div className="channels-to-align">
            <label>Channels to Align:</label>
            <div className="channels-list">
              {channelsToAlign.length > 0 ? (
                channelsToAlign.map(ch => (
                  <div key={ch} className="channel-tag">
                    {getChannelName(ch)}
                  </div>
                ))
              ) : (
                <span className="no-channels">No channels selected (select channels in Mixer Connection tab)</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Control Buttons */}
        <div className="control-buttons">
          {!isActive ? (
            <button
              className="btn-phase-align start"
              onClick={handleStartAnalysis}
              disabled={!selectedDevice || !referenceChannel || channelsToAlign.length === 0}
              title={!selectedDevice ? 'Select audio device first' : 
                     !referenceChannel ? 'Select reference channel' :
                     channelsToAlign.length === 0 ? 'Select channels to align' : 
                     'Start phase alignment analysis'}
            >
              Auto-Align
            </button>
          ) : (
            <button
              className="btn-phase-align stop"
              onClick={handleStopAndApply}
            >
              Stop and Add
            </button>
          )}
          
          <button
            className="btn-phase-align reset"
            onClick={handleResetPhaseDelay}
            disabled={isActive || !selectedChannels || selectedChannels.length === 0}
            title={isActive ? 'Stop analysis first' : 
                   (!selectedChannels || selectedChannels.length === 0) ? 'Select channels first' :
                   'Reset and disable all phase inversions and delays'}
          >
            Reset Phase/Delay
          </button>
        </div>
        
        {/* Status Indicator */}
        <div className="status-section">
          <div className={`status-indicator ${isActive ? 'active' : 'inactive'}`}>
            <span className="status-dot"></span>
            <span>{isActive ? 'Analyzing' : 'Stopped'}</span>
          </div>
          
          {statusMessage && (
            <div className="status-message">
              {statusMessage}
            </div>
          )}
        </div>
        
        {/* Measurements Table */}
        {Object.keys(measurements).length > 0 && (
          <div className="measurements-section">
            <h3>Measurements</h3>
            <div className="measurements-table">
              <div className="table-header">
                <div className="col-channel">Channel</div>
                <div className="col-delay">Delay</div>
                <div className="col-phase">Phase</div>
                <div className="col-coherence">Coherence</div>
                <div className="col-phase-invert">Phase Invert</div>
              </div>
              
              <div className="table-body">
                {Object.entries(measurements).map(([pairKey, meas]) => {
                  // Parse pair key (format: "(ref_ch, ch)")
                  const match = pairKey.match(/\((\d+),\s*(\d+)\)/);
                  if (!match) return null;
                  
                  const refCh = parseInt(match[1]);
                  const ch = parseInt(match[2]);
                  
                  return (
                    <div key={pairKey} className="table-row">
                      <div className="col-channel">
                        {getChannelName(ch)}
                        <span className="ref-label">(ref: {getChannelName(refCh)})</span>
                      </div>
                      <div className="col-delay">
                        {formatDelay(meas.optimal_delay_ms)}
                      </div>
                      <div className="col-phase">
                        {formatPhase(meas.phase_diff_deg)}
                      </div>
                      <div className="col-coherence">
                        {meas.coherence ? meas.coherence.toFixed(3) : '--'}
                      </div>
                      <div className="col-phase-invert">
                        {meas.phase_invert ? 'Yes' : 'No'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PhaseAlignmentTab;
