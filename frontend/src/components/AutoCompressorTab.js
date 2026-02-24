import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoCompressorTab.css';

// NEW: CF Classification profiles
const CF_PROFILES = [
  { id: 'auto', name: 'Auto (CF-based)', description: 'Automatic based on Crest Factor' },
  { id: 'percussion', name: 'Percussion', description: 'CF > 18 dB: Attack 5ms, Release 80ms, Ratio 6:1' },
  { id: 'drums', name: 'Drums', description: 'CF 12-18 dB: Attack 10ms, Release 150ms, Ratio 4:1' },
  { id: 'vocal', name: 'Vocal', description: 'CF 8-12 dB: Attack 15ms, Release 200ms, Ratio 3:1' },
  { id: 'bass', name: 'Bass', description: 'CF 5-8 dB: Attack 40ms, Release 250ms, Ratio 5:1' },
  { id: 'pad', name: 'Pad/Synth', description: 'CF 3-5 dB: Attack 60ms, Release 400ms, Ratio 2:1' },
  { id: 'flat', name: 'Flat/Sine', description: 'CF < 3 dB: Attack 100ms, Release 1000ms, Ratio 1.2:1' }
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

function AutoCompressorTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [soundcheckRunning, setSoundcheckRunning] = useState(false);
  const [liveRunning, setLiveRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [currentParams, setCurrentParams] = useState({});
  const [liveStatus, setLiveStatus] = useState({});
  const [soundcheckProgress, setSoundcheckProgress] = useState({ current: 0, total: 0, channelName: '' });
  const [selectedChannel, setSelectedChannel] = useState(null);
  const [profile, setProfile] = useState('auto');
  const [autoCorrect, setAutoCorrect] = useState(true);
  const [channelNames, setChannelNames] = useState({});
  
  // NEW: CF-based settings
  const [useCFMethod, setUseCFMethod] = useState(true);
  const [cfSettings, setCfSettings] = useState({
    targetLufs: -18.0,
    headroomDb: 3.0,
    maxGrDb: 12.0,
    updateIntervalMs: 100,
    useAdaptive: true,
    rmsWindowMs: 10,
    cfSmoothingMs: 100
  });
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  
  // NEW: Real-time CF display
  const [cfMeasurements, setCfMeasurements] = useState({}); // { channel: { cf_db, cf_class, params } }

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.soundcheck_running !== undefined) setSoundcheckRunning(data.soundcheck_running);
      if (data.live_running !== undefined) setLiveRunning(data.live_running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Error: ${data.error}`);
      if (data.current_params) setCurrentParams(data.current_params);
      
      // NEW: Handle CF-based measurements
      if (data.cf_measurements) {
        setCfMeasurements(data.cf_measurements);
      }
      
      if (data.soundcheck) {
        if (data.progress !== undefined && data.total_channels !== undefined) {
          setSoundcheckProgress({
            current: data.progress,
            total: data.total_channels,
            channelName: data.channel_name || ''
          });
        }
        if (data.complete) setSoundcheckRunning(false);
      }
      if (data.live && data.channel !== undefined) {
        setLiveStatus(prev => ({
          ...prev,
          [data.channel]: {
            gr_estimate: data.gr_estimate,
            lufs: data.lufs,
            status: data.status,
            cf_db: data.cf_db,
            cf_class: data.cf_class
          }
        }));
      }
      if (data.notification === 'operator_attention') {
        setStatusMessage('Attention: multiple auto-corrections on a channel. Check mixer.');
      }
    };
    websocketService.on('auto_compressor_status', handle);
    websocketService.getAutoCompressorStatus();
    return () => websocketService.off('auto_compressor_status', handle);
  }, []);

  const buildChannelMapping = () => {
    const mapping = {};
    (selectedChannels || []).forEach((chId, idx) => {
      const ch = typeof chId === 'number' ? chId : parseInt(chId, 10);
      mapping[ch] = ch;
    });
    return mapping;
  };

  const buildChannelNames = () => {
    const names = {};
    (availableChannels || []).forEach(c => {
      const id = typeof c.id === 'number' ? c.id : parseInt(c.id, 10);
      if (selectedChannels && selectedChannels.includes(id)) {
        names[id] = c.name || `Ch ${id}`;
      }
    });
    return names;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }
    const mapping = buildChannelMapping();
    const names = buildChannelNames();
    setChannelNames(names);
    
    // NEW: Send CF settings if using CF method
    const settings = useCFMethod ? {
      method: 'cf_lufs',
      cf_settings: cfSettings
    } : {
      method: 'legacy'
    };
    
    websocketService.startAutoCompressor(
      selectedDevice,
      selectedChannels,
      mapping,
      names,
      settings
    );
  };

  const handleStop = () => {
    websocketService.stopAutoCompressor();
  };

  const handleSoundcheckStart = () => {
    const settings = useCFMethod ? { method: 'cf_lufs', cf_settings: cfSettings } : { method: 'legacy' };
    websocketService.startAutoCompressorSoundcheck(1, 1, null, settings);
  };

  const handleSoundcheckStop = () => {
    websocketService.stopAutoCompressorSoundcheck();
  };

  const handleLiveStart = () => {
    const settings = useCFMethod ? { method: 'cf_lufs', cf_settings: cfSettings } : { method: 'legacy' };
    websocketService.startAutoCompressorLive(autoCorrect, settings);
  };

  const handleLiveStop = () => {
    websocketService.stopAutoCompressorLive();
  };

  const handleSetProfile = () => {
    if (selectedChannel == null) return;
    websocketService.setAutoCompressorProfile(selectedChannel, profile);
  };

  const handleCFSettingChange = (setting, value) => {
    setCfSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const getCFClassColor = (cfClass) => {
    switch (cfClass) {
      case 'percussion': return '#ff4444'; // Red
      case 'drums': return '#ff9800'; // Orange
      case 'vocal': return '#00d4ff'; // Cyan
      case 'bass': return '#00c851'; // Green
      case 'pad': return '#9c27b0'; // Purple
      case 'flat': return '#888888'; // Gray
      default: return '#ffffff';
    }
  };

  const getCFClassIcon = (cfClass) => {
    switch (cfClass) {
      case 'percussion': return '⚡';
      case 'drums': return '🥁';
      case 'vocal': return '🎤';
      case 'bass': return '🎸';
      case 'pad': return '🎹';
      case 'flat': return '〰️';
      default: return '?';
    }
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-compressor-tab">
      <section className="auto-compressor-section">
        <h2>Auto Compressor (CF-LUFS Method)</h2>
        <p className="section-description">
          NEW: Crest Factor (CF) + LUFS adaptive compression. 
          Automatically adjusts attack, release, and ratio based on signal dynamics.
          Signal is analyzed <strong>post-fader</strong>.
        </p>

        {/* NEW: Method Selection */}
        <div className="method-selection">
          <label className="method-toggle">
            <input
              type="checkbox"
              checked={useCFMethod}
              onChange={(e) => setUseCFMethod(e.target.checked)}
              disabled={active}
            />
            <span>Use CF-LUFS Adaptive Method</span>
          </label>
          <small>
            {useCFMethod 
              ? "Automatically adapts parameters based on Crest Factor and LUFS"
              : "Use legacy preset-based method"
            }
          </small>
        </div>

        {/* NEW: CF Settings Panel */}
        {useCFMethod && (
          <div className="cf-settings-panel">
            <div className="settings-header" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
              <h3>CF-LUFS Settings</h3>
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
                    value={cfSettings.targetLufs}
                    onChange={(e) => handleCFSettingChange('targetLufs', parseFloat(e.target.value))}
                    disabled={active}
                  />
                  <span className="setting-value">{cfSettings.targetLufs} LUFS</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Headroom</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="1"
                    max="6"
                    step="0.5"
                    value={cfSettings.headroomDb}
                    onChange={(e) => handleCFSettingChange('headroomDb', parseFloat(e.target.value))}
                    disabled={active}
                  />
                  <span className="setting-value">{cfSettings.headroomDb} dB</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Max Gain Reduction</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="6"
                    max="20"
                    step="1"
                    value={cfSettings.maxGrDb}
                    onChange={(e) => handleCFSettingChange('maxGrDb', parseFloat(e.target.value))}
                    disabled={active}
                  />
                  <span className="setting-value">{cfSettings.maxGrDb} dB</span>
                </div>
                <small>"Less is more" - warning if GR exceeds this</small>
              </div>
            </div>

            {showAdvancedSettings && (
              <div className="settings-advanced">
                <div className="setting-group">
                  <label>RMS Window</label>
                  <div className="setting-control">
                    <input
                      type="range"
                      min="5"
                      max="20"
                      step="1"
                      value={cfSettings.rmsWindowMs}
                      onChange={(e) => handleCFSettingChange('rmsWindowMs', parseFloat(e.target.value))}
                      disabled={active}
                    />
                    <span className="setting-value">{cfSettings.rmsWindowMs} ms</span>
                  </div>
                  <small>Window for RMS calculation</small>
                </div>

                <div className="setting-group">
                  <label>CF Smoothing</label>
                  <div className="setting-control">
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="10"
                      value={cfSettings.cfSmoothingMs}
                      onChange={(e) => handleCFSettingChange('cfSmoothingMs', parseFloat(e.target.value))}
                      disabled={active}
                    />
                    <span className="setting-value">{cfSettings.cfSmoothingMs} ms</span>
                  </div>
                  <small>Temporal smoothing for Crest Factor</small>
                </div>

                <div className="setting-group">
                  <label>Update Interval</label>
                  <div className="setting-control">
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="10"
                      value={cfSettings.updateIntervalMs}
                      onChange={(e) => handleCFSettingChange('updateIntervalMs', parseFloat(e.target.value))}
                      disabled={active}
                    />
                    <span className="setting-value">{cfSettings.updateIntervalMs} ms</span>
                  </div>
                </div>

                <div className="setting-group">
                  <label>Adaptive Formulas</label>
                  <div className="setting-control">
                    <input
                      type="checkbox"
                      checked={cfSettings.useAdaptive}
                      onChange={(e) => handleCFSettingChange('useAdaptive', e.target.checked)}
                      disabled={active}
                    />
                    <span>Blend base params with adaptive formulas</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* CF Classification Legend */}
        {useCFMethod && (
          <div className="cf-legend">
            <h4>CF Classification</h4>
            <div className="cf-classes">
              {CF_PROFILES.filter(p => p.id !== 'auto').map(profile => (
                <div key={profile.id} className="cf-class-item">
                  <span className="cf-icon">{getCFClassIcon(profile.id)}</span>
                  <span className="cf-name">{profile.name}</span>
                  <small>{profile.description}</small>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="auto-compressor-actions">
          <button
            className="btn-primary"
            onClick={handleStart}
            disabled={!canStart || active}
          >
            Start Auto Compressor
          </button>
          <button
            className="btn-secondary"
            onClick={handleStop}
            disabled={!active}
          >
            Stop
          </button>

          <div className="mode-buttons">
            <button
              className="btn-soundcheck"
              onClick={soundcheckRunning ? handleSoundcheckStop : handleSoundcheckStart}
              disabled={!active}
            >
              {soundcheckRunning ? 'Stop Soundcheck' : 'Soundcheck'}
            </button>
            <button
              className="btn-live"
              onClick={liveRunning ? handleLiveStop : handleLiveStart}
              disabled={!active}
            >
              {liveRunning ? 'Stop Live' : 'Live'}
            </button>
          </div>
        </div>

        {soundcheckRunning && (
          <div className="soundcheck-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${(soundcheckProgress.current / soundcheckProgress.total) * 100}%` }}
              />
            </div>
            <div className="progress-text">
              Analyzing {soundcheckProgress.channelName} ({soundcheckProgress.current + 1} / {soundcheckProgress.total})
            </div>
          </div>
        )}

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {/* NEW: CF Measurements Display */}
        {useCFMethod && Object.keys(cfMeasurements).length > 0 && (
          <div className="cf-measurements">
            <h4>Real-time CF Analysis</h4>
            <div className="cf-table">
              <div className="cf-table-header">
                <div>Channel</div>
                <div>CF</div>
                <div>Class</div>
                <div>Attack</div>
                <div>Release</div>
                <div>Ratio</div>
              </div>
              {Object.entries(cfMeasurements).map(([channelId, data]) => (
                <div key={channelId} className="cf-table-row">
                  <div>{channelNames[channelId] || `Ch ${channelId}`}</div>
                  <div>{data.cf_db?.toFixed(1) || '--'} dB</div>
                  <div style={{ color: getCFClassColor(data.cf_class) }}>
                    {getCFClassIcon(data.cf_class)} {data.cf_class}
                  </div>
                  <div>{data.params?.attack_ms?.toFixed(0) || '--'} ms</div>
                  <div>{data.params?.release_ms?.toFixed(0) || '--'} ms</div>
                  <div>{data.params?.ratio?.toFixed(1) || '--'}:1</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Legacy Profile Selection (when not using CF method) */}
        {!useCFMethod && (
          <div className="profile-section">
            <h4>Legacy Profile</h4>
            <div className="profile-row">
              <select
                value={selectedChannel || ''}
                onChange={(e) => setSelectedChannel(parseInt(e.target.value))}
                disabled={channels.length === 0}
              >
                <option value="">Select channel</option>
                {channels.map(ch => (
                  <option key={ch} value={ch}>
                    {channelNames[ch] || `Channel ${ch}`}
                  </option>
                ))}
              </select>
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                disabled={selectedChannel == null}
              >
                {LEGACY_PROFILES.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <button
                onClick={handleSetProfile}
                disabled={selectedChannel == null}
              >
                Apply
              </button>
            </div>
          </div>
        )}

        <div className="live-controls">
          <label className="auto-correct-toggle">
            <input
              type="checkbox"
              checked={autoCorrect}
              onChange={(e) => setAutoCorrect(e.target.checked)}
              disabled={liveRunning}
            />
            <span>Auto-correct (live)</span>
          </label>
        </div>

        {channels.length > 0 && (
          <div className="channels-list">
            <h4>Selected Channels</h4>
            <ul>
              {channels.map(ch => {
                const status = liveStatus[ch];
                return (
                  <li key={ch}>
                    <span className="channel-name">{channelNames[ch] || `Channel ${ch}`}</span>
                    {status && (
                      <span className="channel-status">
                        GR: {status.gr_estimate?.toFixed(1) || '--'} dB
                        {status.cf_db !== undefined && (
                          <span style={{ marginLeft: '10px', color: getCFClassColor(status.cf_class) }}>
                            {getCFClassIcon(status.cf_class)} CF: {status.cf_db.toFixed(1)} dB
                          </span>
                        )}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Help Text */}
        <div className="help-text">
          <h4>How CF-LUFS Method Works:</h4>
          <ol>
            <li><strong>Analyze Crest Factor (CF)</strong> — ratio of peak to RMS (transient vs sustained)</li>
            <li><strong>Classify signal type</strong> — percussion, drums, vocal, bass, pad, or flat</li>
            <li><strong>Calculate parameters</strong> — attack, release, ratio based on CF class</li>
            <li><strong>Adapt to LUFS</strong> — threshold and makeup gain based on loudness</li>
            <li><strong>Apply smoothly</strong> — gradual parameter transitions to avoid artifacts</li>
          </ol>
          
          <h4>CF Classes:</h4>
          <ul>
            <li><strong>Percussion (CF &gt; 18 dB)</strong> — Very dynamic, fast attack needed</li>
            <li><strong>Drums (CF 12-18 dB)</strong> — Dynamic, preserve transients</li>
            <li><strong>Vocal (CF 8-12 dB)</strong> — Moderate dynamics, smooth compression</li>
            <li><strong>Bass (CF 5-8 dB)</strong> — Sustained, slower attack</li>
            <li><strong>Pad (CF 3-5 dB)</strong> — Very sustained, gentle compression</li>
            <li><strong>Flat (CF &lt; 3 dB)</strong> — Minimal dynamics, very gentle</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default AutoCompressorTab;
