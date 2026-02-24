import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoEffectsTab.css';

// Feature definitions for display
const FEATURES = [
  { id: 'rms', name: 'RMS', tier: 1, freq: '100Hz', unit: 'dB', color: '#00d4ff' },
  { id: 'peak', name: 'Peak', tier: 1, freq: '100Hz', unit: 'dB', color: '#00c851' },
  { id: 'crest_factor', name: 'Crest', tier: 1, freq: '100Hz', unit: 'dB', color: '#ff9800' },
  { id: 'os_peak', name: 'OS Peak', tier: 1, freq: '100Hz', unit: 'dB', color: '#ff4444' },
  { id: 'attack', name: 'Attack', tier: 1, freq: '100Hz', unit: 'ms', color: '#9c27b0' },
  { id: 'loudness', name: 'Loudness', tier: 2, freq: '50Hz', unit: 'LUFS', color: '#00d4ff' },
  { id: 'centroid', name: 'Centroid', tier: 2, freq: '50Hz', unit: 'Hz', color: '#00c851' },
  { id: 'flux', name: 'Flux', tier: 3, freq: '30Hz', unit: '', color: '#ff9800' },
  { id: 'spread', name: 'Spread', tier: 3, freq: '30Hz', unit: 'Hz', color: '#ff4444' },
  { id: 'correlation', name: 'Corr', tier: 4, freq: '10Hz', unit: '', color: '#00d4ff' },
  { id: 'flatness', name: 'Flatness', tier: 4, freq: '10Hz', unit: '', color: '#00c851' },
  { id: 'energy_low', name: 'Low', tier: 4, freq: '10Hz', unit: '', color: '#ff9800' },
  { id: 'energy_mid', name: 'Mid', tier: 4, freq: '10Hz', unit: '', color: '#ff4444' },
  { id: 'energy_high', name: 'High', tier: 4, freq: '10Hz', unit: '', color: '#9c27b0' },
  { id: 'lra', name: 'LRA', tier: 5, freq: '5Hz', unit: 'dB', color: '#00d4ff' }
];

const STATES = [
  { id: 'silent', name: 'Silent', color: '#666', icon: '🔇' },
  { id: 'quiet', name: 'Quiet', color: '#00d4ff', icon: '🔉' },
  { id: 'active', name: 'Active', color: '#00c851', icon: '🔊' },
  { id: 'loud', name: 'Loud', color: '#ff9800', icon: '📢' },
  { id: 'peak', name: 'Peak', color: '#ff4444', icon: '⚠️' }
];

function AutoEffectsTab({ selectedChannels, availableChannels, selectedDevice, audioClients }) {
  const [active, setActive] = useState(false);
  const [mode, setMode] = useState('off');
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({});
  const [matrixData, setMatrixData] = useState([]);
  const [selectedChannel, setSelectedChannel] = useState(null);
  const [showMatrix, setShowMatrix] = useState(false);
  
  // Settings
  const [settings, setSettings] = useState({
    numChannels: 16,
    sampleRate: 48000,
    deadband: 0.02,
    updateFreq: 60
  });

  useEffect(() => {
    const handleAutoEffectsStatus = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.mode !== undefined) setMode(data.mode);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      if (data.channel_data) {
        setChannelData(data.channel_data);
      }
      if (data.matrix) {
        setMatrixData(data.matrix);
      }
    };

    websocketService.on('auto_effects_status', handleAutoEffectsStatus);
    websocketService.getAutoEffectsStatus();
    
    return () => {
      websocketService.off('auto_effects_status', handleAutoEffectsStatus);
    };
  }, []);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    websocketService.startAutoEffects(
      selectedDevice,
      selectedChannels,
      settings
    );
    
    setStatusMessage('Starting Auto Effects Automation...');
  };

  const handleStop = () => {
    websocketService.stopAutoEffects();
    setStatusMessage('Stopping Auto Effects...');
  };

  const formatValue = (value, unit) => {
    if (value === undefined || value === null) return '--';
    if (unit === 'dB' || unit === 'LUFS') {
      return `${value.toFixed(1)} ${unit}`;
    }
    if (unit === 'Hz') {
      return `${Math.round(value)} ${unit}`;
    }
    return `${value.toFixed(2)}`;
  };

  const getStateInfo = (stateId) => {
    return STATES.find(s => s.id === stateId) || STATES[0];
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-effects-tab">
      <section className="auto-effects-section">
        <h2>🎛️ Auto Effects Automation</h2>
        
        <p className="section-description">
          Cross-adaptive effects automation with 13 audio features and coherence matrix.
        </p>

        {/* Architecture Info */}
        <div className="architecture-info">
          <h4>4-Module Architecture</h4>
          <div className="modules-grid">
            <div className="module-card">
              <span className="module-num">1</span>
              <span className="module-name">AudioCore</span>
              <small>13 features, tiered updates</small>
            </div>
            <div className="module-card">
              <span className="module-num">2</span>
              <span className="module-name">AnalysisEngine</span>
              <small>Cross-adaptive matrix</small>
            </div>
            <div className="module-card">
              <span className="module-num">3</span>
              <span className="module-name">StateManager</span>
              <small>State machine + mapping</small>
            </div>
            <div className="module-card">
              <span className="module-num">4</span>
              <span className="module-name">OSCInterface</span>
              <small>Rate limiting + deadband</small>
            </div>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="control-buttons">
          {!active ? (
            <button
              className="btn-primary"
              onClick={handleStart}
              disabled={!canStart}
            >
              Start Auto Effects
            </button>
          ) : (
            <button
              className="btn-stop"
              onClick={handleStop}
            >
              Stop Auto Effects
            </button>
          )}
          
          <button
            className="btn-secondary"
            onClick={() => setShowMatrix(!showMatrix)}
            disabled={!active}
          >
            {showMatrix ? 'Hide Matrix' : 'Show Matrix'}
          </button>
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {/* Channel Selector */}
        {active && channels.length > 0 && (
          <div className="channel-selector">
            <label>Select Channel:</label>
            <select
              value={selectedChannel || ''}
              onChange={(e) => setSelectedChannel(parseInt(e.target.value))}
            >
              <option value="">All Channels</option>
              {channels.map(ch => (
                <option key={ch} value={ch}>
                  {getChannelName(ch)}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Features Display */}
        {active && Object.keys(channelData).length > 0 && (
          <div className="features-display">
            <h4>Audio Features (13)</h4>
            
            {selectedChannel !== null && channelData[selectedChannel] ? (
              // Single channel detailed view
              <div className="single-channel-features">
                <div className="channel-header">
                  <h5>{getChannelName(selectedChannel)}</h5>
                  <span className="state-badge" style={{ 
                    color: getStateInfo(channelData[selectedChannel].state).color 
                  }}>
                    {getStateInfo(channelData[selectedChannel].state).icon} 
                    {getStateInfo(channelData[selectedChannel].state).name}
                  </span>
                </div>
                
                <div className="features-grid">
                  {FEATURES.map(feature => {
                    const value = channelData[selectedChannel]?.features?.[feature.id];
                    return (
                      <div key={feature.id} className="feature-item">
                        <div className="feature-header">
                          <span className="feature-name">{feature.name}</span>
                          <span className="feature-tier">T{feature.tier}</span>
                        </div>
                        <div className="feature-value" style={{ color: feature.color }}>
                          {formatValue(value, feature.unit)}
                        </div>
                        <small>{feature.freq}</small>
                      </div>
                    );
                  })}
                </div>
                
                {channelData[selectedChannel].gain !== undefined && (
                  <div className="gain-display">
                    <label>Cross-Adaptive Gain:</label>
                    <span className="gain-value">
                      {(20 * Math.log10(channelData[selectedChannel].gain + 1e-10)).toFixed(1)} dB
                    </span>
                  </div>
                )}
              </div>
            ) : (
              // All channels summary
              <div className="channels-summary">
                <div className="summary-header">
                  <div>Channel</div>
                  <div>State</div>
                  <div>RMS</div>
                  <div>Peak</div>
                  <div>Loudness</div>
                  <div>Gain</div>
                </div>
                {channels.map(ch => {
                  const data = channelData[ch];
                  if (!data) return null;
                  const state = getStateInfo(data.state);
                  return (
                    <div key={ch} className="summary-row" onClick={() => setSelectedChannel(ch)}>
                      <div>{getChannelName(ch)}</div>
                      <div style={{ color: state.color }}>
                        {state.icon} {state.name}
                      </div>
                      <div>{data.features?.rms?.toFixed(1) || '--'} dB</div>
                      <div>{data.features?.peak?.toFixed(1) || '--'} dB</div>
                      <div>{data.features?.loudness?.toFixed(1) || '--'} LUFS</div>
                      <div>{(20 * Math.log10(data.gain + 1e-10)).toFixed(1)} dB</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Coherence Matrix */}
        {showMatrix && matrixData.length > 0 && (
          <div className="matrix-display">
            <h4>Coherence Matrix (Cross-Adaptive)</h4>
            <div className="matrix-grid">
              {matrixData.map((row, i) => (
                <div key={i} className="matrix-row">
                  {row.map((value, j) => (
                    <div 
                      key={j} 
                      className="matrix-cell"
                      style={{
                        backgroundColor: `rgba(0, 212, 255, ${value})`,
                        opacity: i === j ? 0.3 : 1  // Diagonal is less prominent
                      }}
                      title={`Channel ${i} ↔ ${j}: ${value.toFixed(2)}`}
                    >
                      {value > 0.5 ? '●' : value > 0.2 ? '○' : ''}
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <small>Blue intensity = correlation strength</small>
          </div>
        )}

        {/* Feature Tiers Legend */}
        <div className="tiers-legend">
          <h4>Feature Tiers</h4>
          <div className="tiers-list">
            <div className="tier-item">
              <span className="tier-badge t1">T1</span>
              <span>100Hz: RMS, Peak, Crest, OS_Peak, Attack</span>
            </div>
            <div className="tier-item">
              <span className="tier-badge t2">T2</span>
              <span>50Hz: Loudness, Spectral Centroid</span>
            </div>
            <div className="tier-item">
              <span className="tier-badge t3">T3</span>
              <span>30Hz: Spectral Flux, Spread</span>
            </div>
            <div className="tier-item">
              <span className="tier-badge t4">T4</span>
              <span>10Hz: Correlation, Flatness, 3-Band Energy</span>
            </div>
            <div className="tier-item">
              <span className="tier-badge t5">T5</span>
              <span>5Hz: LRA (Loudness Range)</span>
            </div>
          </div>
        </div>

        {/* OSC Addresses */}
        <div className="osc-info">
          <h4>OSC Addresses</h4>
          <div className="osc-addresses">
            <code>/track/{'{n}'}/rms f [0-1] 100Hz</code>
            <code>/track/{'{n}'}/peak f [0-1] 100Hz</code>
            <code>/track/{'{n}'}/loudness f [0-1] 50Hz</code>
            <code>/track/{'{n}'}/centroid f [0-1] 50Hz</code>
            <code>/cross/matrix ffff... [136] 20Hz</code>
            <code>/cross/gain/{'{n}'} f [0-2] 20Hz</code>
          </div>
        </div>

        {/* Help Text */}
        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li><strong>AudioCore</strong> — Extract 13 features with tiered frequencies (100Hz→5Hz)</li>
            <li><strong>AnalysisEngine</strong> — Compute coherence matrix and apply Gain Sharing</li>
            <li><strong>StateManager</strong> — Track states (silent→quiet→active→loud→peak)</li>
            <li><strong>OSCInterface</strong> — Send with 2% deadband and transient bypass</li>
          </ol>
          
          <h4>Key Features:</h4>
          <ul>
            <li><strong>Cross-Adaptive:</strong> Each channel's gain depends on all others</li>
            <li><strong>Coherence Matrix:</strong> 136 elements track inter-channel relationships</li>
            <li><strong>Tiered Updates:</strong> Critical features at 100Hz, slow at 5Hz</li>
            <li><strong>Low Latency:</strong> 2.67ms audio frame, {'<5ms'} end-to-end</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default AutoEffectsTab;
