import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoGateTab.css';

// Instrument groups
const GROUPS = [
  { id: 'drums', name: 'Drums', icon: '🥁', color: '#ff6b6b' },
  { id: 'bass', name: 'Bass', icon: '🎸', color: '#4ecdc4' },
  { id: 'vocals', name: 'Vocals', icon: '🎤', color: '#45b7d1' },
  { id: 'guitars', name: 'Guitars', icon: '🎸', color: '#96ceb4' },
  { id: 'keys', name: 'Keys', icon: '🎹', color: '#dda0dd' },
  { id: 'other', name: 'Other', icon: '🎵', color: '#f0e68c' }
];

// Drum types
const DRUM_TYPES = [
  { id: 'kick', name: 'Kick', priority: 4, attack: 0.5, release: 80 },
  { id: 'snare', name: 'Snare', priority: 3, attack: 0.2, release: 50 },
  { id: 'tom', name: 'Tom', priority: 2, attack: 0.3, release: 60 },
  { id: 'overheads', name: 'Overheads', priority: 1, attack: 0.5, release: 100 }
];

// Gate states
const GATE_STATES = [
  { id: 'closed', name: 'Closed', color: '#666' },
  { id: 'opening', name: 'Opening', color: '#ff9800' },
  { id: 'open', name: 'Open', color: '#00c851' },
  { id: 'hold', name: 'Hold', color: '#00d4ff' },
  { id: 'releasing', name: 'Releasing', color: '#ff6b6b' }
];

function AutoGateTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({});
  const [channelConfigs, setChannelConfigs] = useState({});
  const [noiseFloor, setNoiseFloor] = useState(-70);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Settings
  const [settings, setSettings] = useState({
    frameSize: 64,
    baseMarginDb: 6.0,
    maxGroupInfluence: 9.0,
    enableCrossAdaptive: true,
    enableDrumRules: true
  });

  useEffect(() => {
    const handleAutoGateStatus = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      if (data.channel_data) {
        setChannelData(data.channel_data);
      }
      if (data.noise_floor !== undefined) {
        setNoiseFloor(data.noise_floor);
      }
    };

    websocketService.on('auto_gate_status', handleAutoGateStatus);
    websocketService.getAutoGateStatus();
    
    return () => {
      websocketService.off('auto_gate_status', handleAutoGateStatus);
    };
  }, []);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleConfigureChannel = (channelId, config) => {
    setChannelConfigs(prev => ({
      ...prev,
      [channelId]: { ...prev[channelId], ...config }
    }));
    
    // Send configuration to backend
    websocketService.configureGateChannel(channelId, config);
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    websocketService.startAutoGate(
      selectedDevice,
      selectedChannels,
      channelConfigs,
      settings
    );
    
    setStatusMessage('Starting Cross-Adaptive Intelligent Gate...');
  };

  const handleStop = () => {
    websocketService.stopAutoGate();
    setStatusMessage('Stopping Auto Gate...');
  };

  const getStateInfo = (stateId) => {
    return GATE_STATES.find(s => s.id === stateId) || GATE_STATES[0];
  };

  const getGroupInfo = (groupId) => {
    return GROUPS.find(g => g.id === groupId) || GROUPS[5];
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-gate-tab">
      <section className="auto-gate-section">
        <h2>🚪 Auto Gate (Cross-Adaptive Intelligent Gate)</h2>
        
        <p className="section-description">
          Cross-Adaptive Intelligent Gate (CAIG) with adaptive threshold and group-based processing.
          Automatically reduces bleed and noise while preserving transients.
        </p>

        {/* Architecture Info */}
        <div className="architecture-info">
          <h4>CAIG Architecture</h4>
          <div className="architecture-components">
            <div className="comp">
              <span className="comp-name">Feature Extraction</span>
              <small>64 samples (1.33ms)</small>
              <small>RMS, Peak, Crest, LF Energy</small>
            </div>
            <div className="comp-arrow">→</div>
            <div className="comp">
              <span className="comp-name">Group Analysis</span>
              <small>Cross-adaptive influence</small>
              <small>0-9dB threshold offset</small>
            </div>
            <div className="comp-arrow">→</div>
            <div className="comp">
              <span className="comp-name">Adaptive Threshold</span>
              <small>Noise floor + 6dB</small>
              <small>Crest factor offset</small>
            </div>
            <div className="comp-arrow">→</div>
            <div className="comp">
              <span className="comp-name">Drum Kit Rules</span>
              <small>Priority processing</small>
              <small>Kick → Snare suppression</small>
            </div>
            <div className="comp-arrow">→</div>
            <div className="comp">
              <span className="comp-name">Gate Processor</span>
              <small>State machine</small>
              <small>Attack/Hold/Release</small>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <div className="gate-settings-panel">
          <div className="settings-header" onClick={() => setShowAdvanced(!showAdvanced)}>
            <h3>Gate Settings</h3>
            <span className="toggle-icon">{showAdvanced ? '▼' : '▶'}</span>
          </div>
          
          <div className="settings-main">
            <div className="setting-row">
              <label>Noise Floor:</label>
              <span className="value">{noiseFloor.toFixed(1)} dB</span>
              <small>(auto-tracking)</small>
            </div>
            
            <div className="setting-row">
              <label>Base Margin:</label>
              <span className="value">{settings.baseMarginDb} dB</span>
              <small>above noise floor</small>
            </div>
            
            <div className="setting-row">
              <label>Max Group Influence:</label>
              <span className="value">{settings.maxGroupInfluence} dB</span>
              <small>cross-adaptive offset</small>
            </div>
          </div>
          
          {showAdvanced && (
            <div className="settings-advanced">
              <div className="setting-group">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.enableCrossAdaptive}
                    onChange={(e) => setSettings(s => ({ ...s, enableCrossAdaptive: e.target.checked }))}
                    disabled={active}
                  />
                  Enable Cross-Adaptive Processing
                </label>
              </div>
              
              <div className="setting-group">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.enableDrumRules}
                    onChange={(e) => setSettings(s => ({ ...s, enableDrumRules: e.target.checked }))}
                    disabled={active}
                  />
                  Enable Drum Kit Priority Rules
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Group Legend */}
        <div className="groups-legend">
          <h4>Instrument Groups</h4>
          <div className="groups-grid">
            {GROUPS.map(group => (
              <div key={group.id} className="group-item" style={{ borderColor: group.color }}>
                <span className="group-icon">{group.icon}</span>
                <span className="group-name">{group.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Drum Kit Rules */}
        <div className="drum-rules">
          <h4>Drum Kit Priority Rules</h4>
          <div className="drum-priorities">
            {DRUM_TYPES.map(drum => (
              <div key={drum.id} className="drum-item">
                <span className="drum-name">{drum.name}</span>
                <div className="priority-bar">
                  <div className="priority-fill" style={{ width: `${drum.priority * 25}%` }} />
                </div>
                <span className="priority-value">P{drum.priority}</span>
                <small>{drum.attack}ms / {drum.release}ms</small>
              </div>
            ))}
          </div>
          <div className="rules-list">
            <div className="rule">✓ Kick suppresses Snare (+6dB threshold during Kick)</div>
            <div className="rule">✓ Overheads never fully closed (min -10dB)</div>
            <div className="rule">✓ Transients get +6dB crest offset</div>
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
              Start Auto Gate
            </button>
          ) : (
            <button
              className="btn-stop"
              onClick={handleStop}
            >
              Stop Auto Gate
            </button>
          )}
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {/* Channel Configuration */}
        {channels.length > 0 && (
          <div className="channel-config">
            <h4>Channel Configuration</h4>
            {channels.map(ch => {
              const config = channelConfigs[ch] || {};
              const data = channelData[ch];
              
              return (
                <div key={ch} className="channel-config-row">
                  <div className="channel-name">{getChannelName(ch)}</div>
                  
                  <select
                    value={config.group || 'other'}
                    onChange={(e) => handleConfigureChannel(ch, { group: e.target.value })}
                    disabled={active}
                  >
                    {GROUPS.map(g => (
                      <option key={g.id} value={g.id}>{g.name}</option>
                    ))}
                  </select>
                  
                  {config.group === 'drums' && (
                    <select
                      value={config.drum_type || 'kick'}
                      onChange={(e) => {
                        const drum = DRUM_TYPES.find(d => d.id === e.target.value);
                        handleConfigureChannel(ch, {
                          drum_type: e.target.value,
                          is_drum: true,
                          attack_ms: drum.attack,
                          release_ms: drum.release
                        });
                      }}
                      disabled={active}
                    >
                      {DRUM_TYPES.map(d => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                      ))}
                    </select>
                  )}
                  
                  {data && (
                    <div className="gate-status">
                      <span 
                        className="state-badge"
                        style={{ color: getStateInfo(data.state).color }}
                      >
                        {getStateInfo(data.state).name}
                      </span>
                      <span className="gain-value">{data.gain_db?.toFixed(1)}dB</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Real-time Display */}
        {active && Object.keys(channelData).length > 0 && (
          <div className="realtime-display">
            <h4>Gate Status</h4>
            <div className="channels-status">
              {channels.map(ch => {
                const data = channelData[ch];
                if (!data) return null;
                
                const stateInfo = getStateInfo(data.state);
                const config = channelConfigs[ch] || {};
                const groupInfo = getGroupInfo(config.group || 'other');
                
                return (
                  <div key={ch} className="channel-status-row">
                    <div className="channel-info">
                      <span className="channel-name">{getChannelName(ch)}</span>
                      <span className="group-icon" style={{ color: groupInfo.color }}>
                        {groupInfo.icon}
                      </span>
                    </div>
                    
                    <div className="gate-visual">
                      <div className="gate-bar">
                        <div 
                          className="gate-fill"
                          style={{ 
                            width: `${Math.max(0, Math.min(100, (data.gain_db + 80) / 80 * 100))}%`,
                            backgroundColor: stateInfo.color
                          }}
                        />
                      </div>
                    </div>
                    
                    <div className="gate-details">
                      <span className="state" style={{ color: stateInfo.color }}>
                        {stateInfo.name}
                      </span>
                      <span className="gain">{data.gain_db?.toFixed(1)}dB</span>
                      <span className="threshold">thr: {data.threshold_db?.toFixed(1)}dB</span>
                      {data.group_influence > 0 && (
                        <span className="influence">+{data.group_influence?.toFixed(1)}dB</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Formula Display */}
        <div className="formula-display">
          <h4>Adaptive Threshold Formula</h4>
          <div className="formula">
            <code>threshold = noise_floor + 6dB + crest_offset + group_influence</code>
          </div>
          <div className="formula-breakdown">
            <div>• noise_floor: auto-tracked minimum RMS</div>
            <div>• crest_offset: +6dB (transient) / -3dB (sustain)</div>
            <div>• group_influence: 0-9dB based on group activity</div>
          </div>
        </div>

        {/* OSC Commands */}
        <div className="osc-commands">
          <h4>OSC Commands</h4>
          <div className="osc-list">
            <code>/ch/{'{n}'}/gate/on i [0/1]</code>
            <code>/ch/{'{n}'}/gate/thr f [-80 to 0] dB</code>
            <code>/ch/{'{n}'}/gate/atk f [0.1 to 10] ms</code>
            <code>/ch/{'{n}'}/gate/hold f [1 to 50] ms</code>
            <code>/ch/{'{n}'}/gate/rel f [10 to 500] ms</code>
            <code>/ch/{'{n}'}/gate/rng f [-80 to 0] dB</code>
          </div>
        </div>

        {/* Help Text */}
        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li><strong>Feature Extraction</strong> — RMS, Peak, Crest Factor, LF Energy every 64 samples</li>
            <li><strong>Group Analysis</strong> — Calculate group activity and dominant channel</li>
            <li><strong>Adaptive Threshold</strong> — noise_floor + 6dB + crest_offset + group_influence</li>
            <li><strong>Drum Kit Rules</strong> — Apply priority rules (Kick → Snare suppression)</li>
            <li><strong>Gate Processing</strong> — State machine with Attack/Hold/Release phases</li>
          </ol>
          
          <h4>Key Features:</h4>
          <ul>
            <li><strong>Cross-Adaptive:</strong> Channel gates interact based on group activity</li>
            <li><strong>Adaptive Threshold:</strong> Automatically tracks noise floor and signal dynamics</li>
            <li><strong>Drum Kit Rules:</strong> Priority-based processing for drum kits</li>
            <li><strong>Low Latency:</strong> 1.33ms frame size for real-time performance</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default AutoGateTab;
