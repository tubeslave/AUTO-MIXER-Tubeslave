import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoFaderTab.css';

// Scenario definitions for display
const SCENARIOS = [
  { id: 'silence', name: 'Silence', color: '#666', icon: '🔇', range: '< -60 dB' },
  { id: 'quiet', name: 'Quiet', color: '#00d4ff', icon: '🔉', range: '-60 to -40 dB' },
  { id: 'normal', name: 'Normal', color: '#00c851', icon: '🔊', range: '-40 to -20 dB' },
  { id: 'loud', name: 'Loud', color: '#ff9800', icon: '📢', range: '-20 to -10 dB' },
  { id: 'peak', name: 'Peak', color: '#ff4444', icon: '⚠️', range: '-10 to -6 dB' },
  { id: 'emergency', name: 'Emergency', color: '#ff0000', icon: '🚨', range: '> -3 dB' }
];

// Legacy profiles for compatibility
const LEGACY_PROFILES = [
  { id: 'base', name: 'Base' },
  { id: 'punch', name: 'Punch' },
  { id: 'control', name: 'Control' },
  { id: 'gentle', name: 'Gentle' },
  { id: 'aggressive', name: 'Aggressive' },
  { id: 'broadcast', name: 'Broadcast' }
];

function AutoFaderTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [mode, setMode] = useState('off'); // off, manual, auto_assist, full_auto
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({}); // Real-time channel data
  const [masterLevel, setMasterLevel] = useState({ hybrid: -100, peak: -100 });
  const [emergencyActive, setEmergencyActive] = useState(false);
  
  // NEW: 6-Level Hybrid Method settings
  const [useHybridMethod, setUseHybridMethod] = useState(true);
  const [hybridSettings, setHybridSettings] = useState({
    targetLufs: -18.0,
    loopFrequencyHz: 100,
    kp: 0.5,           // Proportional gain
    ki: 0.05,          // Integral gain
    maxGainDb: 10.0,
    minGainDb: -60.0,
    safetyHeadroomDb: 10.0,
    maxTruePeakDb: -1.0
  });
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  // WebSocket event handlers
  useEffect(() => {
    const handleAutoFaderStatus = (data) => {
      console.log('Auto Fader status:', data);
      
      if (data.active !== undefined) setActive(data.active);
      if (data.mode !== undefined) setMode(data.mode);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      
      // NEW: Handle hybrid method data
      if (data.channel_data) {
        setChannelData(data.channel_data);
      }
      
      if (data.master_level) {
        setMasterLevel(data.master_level);
      }
      
      if (data.emergency_active !== undefined) {
        setEmergencyActive(data.emergency_active);
      }
    };

    websocketService.on('auto_fader_status', handleAutoFaderStatus);
    websocketService.getAutoFaderStatus();
    
    return () => {
      websocketService.off('auto_fader_status', handleAutoFaderStatus);
    };
  }, []);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleHybridSettingChange = (setting, value) => {
    setHybridSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    const settings = useHybridMethod ? {
      method: 'hybrid_6level',
      hybrid_settings: hybridSettings
    } : {
      method: 'legacy'
    };

    websocketService.send({
      type: 'start_auto_fader',
      device_id: selectedDevice,
      channels: selectedChannels,
      settings: settings
    });
    
    setStatusMessage('Starting 6-Level Hybrid Auto Fader...');
  };

  const handleStop = () => {
    websocketService.send({ type: 'stop_auto_fader' });
    setStatusMessage('Stopping Auto Fader...');
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    websocketService.send({
      type: 'set_auto_fader_mode',
      mode: newMode
    });
  };

  const getScenarioInfo = (scenarioId) => {
    return SCENARIOS.find(s => s.id === scenarioId) || SCENARIOS[0];
  };

  const getScenarioColor = (scenarioId) => {
    return getScenarioInfo(scenarioId).color;
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-fader-tab">
      <section className="auto-fader-section">
        <h2>🎚️ Auto Fader (6-Level Hybrid)</h2>
        
        {/* Emergency Alert */}
        {emergencyActive && (
          <div className="emergency-alert">
            <span className="emergency-icon">🚨</span>
            <span className="emergency-text">EMERGENCY: Master level approaching limit!</span>
          </div>
        )}
        
        <p className="section-description">
          NEW: 6-Level Hybrid Architecture with adaptive scenario detection.
          Automatically balances faders based on LUFS + RMS + Peak metrics.
        </p>

        {/* Method Selection */}
        <div className="method-selection">
          <label className="method-toggle">
            <input
              type="checkbox"
              checked={useHybridMethod}
              onChange={(e) => setUseHybridMethod(e.target.checked)}
              disabled={active}
            />
            <span>Use 6-Level Hybrid Method</span>
          </label>
          <small>
            {useHybridMethod 
              ? "Hybrid: 0.45·LUFS + 0.35·RMS + 0.20·Peak with scenario detection"
              : "Use legacy fuzzy logic method"
            }
          </small>
        </div>

        {/* Hybrid Settings Panel */}
        {useHybridMethod && (
          <div className="hybrid-settings-panel">
            <div className="settings-header" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
              <h3>6-Level Hybrid Settings</h3>
              <span className="toggle-icon">{showAdvancedSettings ? '▼' : '▶'}</span>
            </div>
            
            <div className="settings-main">
              <div className="setting-group">
                <label>Target LUFS</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="-30"
                    max="-12"
                    step="1"
                    value={hybridSettings.targetLufs}
                    onChange={(e) => handleHybridSettingChange('targetLufs', parseFloat(e.target.value))}
                    disabled={active}
                  />
                  <span className="setting-value">{hybridSettings.targetLufs} LUFS</span>
                </div>
                <small>Streaming: -14 LUFS, Broadcast: -23 LUFS</small>
              </div>

              <div className="setting-group">
                <label>Loop Frequency</label>
                <div className="setting-control">
                  <select
                    value={hybridSettings.loopFrequencyHz}
                    onChange={(e) => handleHybridSettingChange('loopFrequencyHz', parseInt(e.target.value))}
                    disabled={active}
                  >
                    <option value={50}>50 Hz (20ms)</option>
                    <option value={100}>100 Hz (10ms)</option>
                    <option value={200}>200 Hz (5ms)</option>
                  </select>
                </div>
                <small>Control loop update rate</small>
              </div>

              <div className="setting-group">
                <label>Safety Headroom</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="6"
                    max="15"
                    step="1"
                    value={hybridSettings.safetyHeadroomDb}
                    onChange={(e) => handleHybridSettingChange('safetyHeadroomDb', parseFloat(e.target.value))}
                    disabled={active}
                  />
                  <span className="setting-value">{hybridSettings.safetyHeadroomDb} dB</span>
                </div>
              </div>
            </div>

            {showAdvancedSettings && (
              <div className="settings-advanced">
                <div className="settings-section">
                  <h4>PI Controller</h4>
                  
                  <div className="setting-group">
                    <label>Proportional Gain (Kp)</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="0.1"
                        max="2.0"
                        step="0.1"
                        value={hybridSettings.kp}
                        onChange={(e) => handleHybridSettingChange('kp', parseFloat(e.target.value))}
                        disabled={active}
                      />
                      <span className="setting-value">{hybridSettings.kp}</span>
                    </div>
                    <small>Higher = faster response</small>
                  </div>

                  <div className="setting-group">
                    <label>Integral Gain (Ki)</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="0.01"
                        max="0.2"
                        step="0.01"
                        value={hybridSettings.ki}
                        onChange={(e) => handleHybridSettingChange('ki', parseFloat(e.target.value))}
                        disabled={active}
                      />
                      <span className="setting-value">{hybridSettings.ki}</span>
                    </div>
                    <small>Higher = eliminates steady-state error faster</small>
                  </div>
                </div>

                <div className="settings-section">
                  <h4>Hard Limits</h4>
                  
                  <div className="setting-group">
                    <label>Max Gain</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="0"
                        max="20"
                        step="1"
                        value={hybridSettings.maxGainDb}
                        onChange={(e) => handleHybridSettingChange('maxGainDb', parseFloat(e.target.value))}
                        disabled={active}
                      />
                      <span className="setting-value">{hybridSettings.maxGainDb} dB</span>
                    </div>
                  </div>

                  <div className="setting-group">
                    <label>Max True Peak</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="-6"
                        max="0"
                        step="0.5"
                        value={hybridSettings.maxTruePeakDb}
                        onChange={(e) => handleHybridSettingChange('maxTruePeakDb', parseFloat(e.target.value))}
                        disabled={active}
                      />
                      <span className="setting-value">{hybridSettings.maxTruePeakDb} dBTP</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Mode Selection */}
        <div className="mode-selection">
          <label>Operating Mode:</label>
          <div className="mode-buttons">
            <button
              className={`mode-btn ${mode === 'off' ? 'active' : ''}`}
              onClick={() => handleModeChange('off')}
              disabled={!active}
            >
              Off
            </button>
            <button
              className={`mode-btn ${mode === 'manual' ? 'active' : ''}`}
              onClick={() => handleModeChange('manual')}
              disabled={!active}
            >
              Manual
            </button>
            <button
              className={`mode-btn ${mode === 'auto_assist' ? 'active' : ''}`}
              onClick={() => handleModeChange('auto_assist')}
              disabled={!active}
            >
              Auto Assist
            </button>
            <button
              className={`mode-btn ${mode === 'full_auto' ? 'active' : ''}`}
              onClick={() => handleModeChange('full_auto')}
              disabled={!active}
            >
              Full Auto
            </button>
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
              Start Auto Fader
            </button>
          ) : (
            <button
              className="btn-stop"
              onClick={handleStop}
            >
              Stop Auto Fader
            </button>
          )}
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {/* Master Level Display */}
        {active && (
          <div className="master-level-display">
            <h4>Master Level</h4>
            <div className="level-bars">
              <div className="level-bar">
                <label>Hybrid</label>
                <div className="bar-container">
                  <div 
                    className="bar-fill hybrid"
                    style={{ 
                      width: `${Math.max(0, Math.min(100, (masterLevel.hybrid + 60) / 60 * 100))}%`,
                      backgroundColor: getScenarioColor(
                        masterLevel.hybrid > -6 ? 'peak' : 
                        masterLevel.hybrid > -10 ? 'loud' : 
                        masterLevel.hybrid > -20 ? 'normal' : 'quiet'
                      )
                    }}
                  />
                </div>
                <span className="level-value">{masterLevel.hybrid.toFixed(1)} dB</span>
              </div>
              <div className="level-bar">
                <label>Peak</label>
                <div className="bar-container">
                  <div 
                    className="bar-fill peak"
                    style={{ 
                      width: `${Math.max(0, Math.min(100, (masterLevel.peak + 60) / 60 * 100))}%`,
                      backgroundColor: masterLevel.peak > -3 ? '#ff0000' : 
                                      masterLevel.peak > -10 ? '#ff9800' : '#00c851'
                    }}
                  />
                </div>
                <span className="level-value">{masterLevel.peak.toFixed(1)} dBTP</span>
              </div>
            </div>
          </div>
        )}

        {/* Scenario Legend */}
        {useHybridMethod && (
          <div className="scenario-legend">
            <h4>Scenarios & Rate Limits</h4>
            <div className="scenarios-grid">
              {SCENARIOS.map(scenario => (
                <div key={scenario.id} className="scenario-item">
                  <span className="scenario-icon" style={{ color: scenario.color }}>
                    {scenario.icon}
                  </span>
                  <span className="scenario-name">{scenario.name}</span>
                  <small>{scenario.range}</small>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Channels Table */}
        {Object.keys(channelData).length > 0 && (
          <div className="channels-table">
            <h4>Channel Status</h4>
            <div className="table-header">
              <div>Channel</div>
              <div>Hybrid</div>
              <div>Scenario</div>
              <div>Gain</div>
              <div>Correction</div>
            </div>
            <div className="table-body">
              {Object.entries(channelData).map(([channelId, data]) => {
                const scenario = getScenarioInfo(data.scenario);
                return (
                  <div key={channelId} className="table-row">
                    <div className="col-channel">
                      {getChannelName(parseInt(channelId))}
                    </div>
                    <div className="col-hybrid">
                      {data.hybrid_db?.toFixed(1) || '--'} dB
                    </div>
                    <div className="col-scenario" style={{ color: scenario.color }}>
                      {scenario.icon} {scenario.name}
                    </div>
                    <div className="col-gain">
                      {data.gain_db?.toFixed(1) || '--'} dB
                    </div>
                    <div className="col-correction">
                      <span className={data.correction_db > 0 ? 'positive' : data.correction_db < 0 ? 'negative' : ''}>
                        {data.correction_db > 0 ? '+' : ''}{data.correction_db?.toFixed(1) || '0.0'} dB
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Architecture Info */}
        <div className="architecture-info">
          <h4>6-Level Hybrid Architecture</h4>
          <div className="levels-list">
            <div className="level-item">
              <span className="level-num">0</span>
              <span className="level-name">Hardware Safety</span>
              <small>Hard limiter -1 dBFS, emergency mute</small>
            </div>
            <div className="level-item">
              <span className="level-num">1</span>
              <span className="level-name">Signal Analyzer</span>
              <small>Peak 5ms + RMS 50ms + LUFS 400ms</small>
            </div>
            <div className="level-item">
              <span className="level-num">2</span>
              <span className="level-name">Hybrid Fusion</span>
              <small>0.45·LUFS + 0.35·RMS + 0.20·Peak</small>
            </div>
            <div className="level-item">
              <span className="level-num">3</span>
              <span className="level-name">Scenario Detector</span>
              <small>6 scenarios with adaptive rate limits</small>
            </div>
            <div className="level-item">
              <span className="level-num">4</span>
              <span className="level-name">Decision Engine</span>
              <small>PI-controller with anti-windup</small>
            </div>
            <div className="level-item">
              <span className="level-num">5</span>
              <span className="level-name">Safety Validator</span>
              <small>10dB headroom, emergency override</small>
            </div>
          </div>
        </div>

        {/* Help Text */}
        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li><strong>Signal Analysis</strong> — Extract Peak (5ms), RMS (50ms), LUFS (400ms)</li>
            <li><strong>Hybrid Metric</strong> — Combine: 45% LUFS + 35% RMS + 20% Peak</li>
            <li><strong>Scenario Detection</strong> — Classify: Silence → Normal → Loud → Emergency</li>
            <li><strong>PI Control</strong> — Calculate correction with adaptive rate limiting</li>
            <li><strong>Safety</strong> — Validate hard limits and emergency conditions</li>
            <li><strong>Output</strong> — Send OSC commands to mixer</li>
          </ol>
          
          <h4>Modes:</h4>
          <ul>
            <li><strong>Off</strong> — No automatic control</li>
            <li><strong>Manual</strong> — Display recommendations only</li>
            <li><strong>Auto Assist</strong> — Suggest corrections, manual apply</li>
            <li><strong>Full Auto</strong> — Automatic control with safety limits</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default AutoFaderTab;
