import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './CrossAdaptiveEQTab.css';

// Frequency bands for display
const FREQUENCY_BANDS = [
  { id: 'sub', name: 'Sub', freq: '40Hz', color: '#8B4513' },
  { id: 'bass', name: 'Bass', freq: '120Hz', color: '#D2691E' },
  { id: 'low_mid', name: 'Low Mid', freq: '375Hz', color: '#FF8C00' },
  { id: 'mid', name: 'Mid', freq: '1.25kHz', color: '#FFD700' },
  { id: 'high_mid', name: 'High Mid', freq: '3kHz', color: '#9ACD32' },
  { id: 'high', name: 'High', freq: '6kHz', color: '#00CED1' },
  { id: 'air', name: 'Air', freq: '12kHz', color: '#87CEEB' }
];

// Priority levels
const PRIORITIES = [
  { value: 1, label: 'High (Lead)', color: '#00c851' },
  { value: 2, label: 'Medium', color: '#ff9800' },
  { value: 3, label: 'Low (Support)', color: '#666' }
];

function CrossAdaptiveEQTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [adjustments, setAdjustments] = useState([]);
  const [channelPriorities, setChannelPriorities] = useState({});
  const [channelBandEnergy, setChannelBandEnergy] = useState({});
  const [showAdjustments, setShowAdjustments] = useState(false);
  
  // Settings
  const [settings, setSettings] = useState({
    overlapToleranceDb: 6.0,
    maxCutDb: -6.0,
    maxBoostDb: 3.0,
    minBandLevelDb: -80.0
  });

  useEffect(() => {
    const handleCrossAdaptiveEQStatus = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      if (data.adjustments) {
        setAdjustments(data.adjustments);
      }
      if (data.channel_band_energy) {
        setChannelBandEnergy(data.channel_band_energy);
      }
    };

    websocketService.on('cross_adaptive_eq_status', handleCrossAdaptiveEQStatus);
    websocketService.getCrossAdaptiveEQStatus();
    
    return () => {
      websocketService.off('cross_adaptive_eq_status', handleCrossAdaptiveEQStatus);
    };
  }, []);

  // Initialize channel priorities when selected channels change
  useEffect(() => {
    const newPriorities = {};
    selectedChannels.forEach((ch, index) => {
      // Default: first channels get higher priority (lead instruments)
      newPriorities[ch] = channelPriorities[ch] || Math.min(3, 1 + Math.floor(index / 3));
    });
    setChannelPriorities(newPriorities);
  }, [selectedChannels]);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    if (selectedChannels.length < 2) {
      setStatusMessage('Select at least 2 channels for cross-adaptive processing.');
      return;
    }

    websocketService.startCrossAdaptiveEQ(
      selectedDevice,
      selectedChannels,
      {
        ...settings,
        channel_priorities: channelPriorities
      }
    );
    
    setStatusMessage('Starting Cross-Adaptive EQ analysis...');
  };

  const handleStop = () => {
    websocketService.stopCrossAdaptiveEQ();
    setStatusMessage('Stopping Cross-Adaptive EQ...');
  };

  const handlePriorityChange = (channel, priority) => {
    setChannelPriorities(prev => ({
      ...prev,
      [channel]: parseInt(priority)
    }));
  };

  const getPriorityLabel = (priority) => {
    const p = PRIORITIES.find(p => p.value === priority);
    return p ? p.label : 'Medium';
  };

  const getPriorityColor = (priority) => {
    const p = PRIORITIES.find(p => p.value === priority);
    return p ? p.color : '#ff9800';
  };

  const formatGain = (gain) => {
    if (gain === undefined || gain === null) return '--';
    const sign = gain > 0 ? '+' : '';
    return `${sign}${gain.toFixed(1)} dB`;
  };

  return (
    <div className="cross-adaptive-eq-tab">
      <section className="cae-section">
        <div className="cae-header">
          <h2>Cross-Adaptive EQ</h2>
          <div className={`cae-status-indicator ${active ? 'active' : 'inactive'}`}>
            {active ? '● Active' : '○ Inactive'}
          </div>
        </div>

        <div className="cae-description">
          <p>
            Intelligent mirror EQ based on IMP 7.3 research. Automatically resolves frequency 
            conflicts between channels by cutting overlapping frequencies in lower-priority 
            channels and boosting them in higher-priority channels.
          </p>
        </div>

        {statusMessage && (
          <div className={`cae-status-message ${statusMessage.includes('Error') ? 'error' : ''}`}>
            {statusMessage}
          </div>
        )}
      </section>

      <section className="cae-section">
        <h3>Channel Priorities</h3>
        <p className="cae-help-text">
          Assign priority levels to channels. Higher priority (Lead) channels will be boosted 
          while lower priority (Support) channels will be cut in overlapping frequency bands.
        </p>
        
        {selectedChannels.length === 0 ? (
          <div className="cae-no-channels">
            Select channels in the Mixer Connection tab to configure priorities.
          </div>
        ) : (
          <div className="cae-priorities-grid">
            {selectedChannels.map(channel => (
              <div key={channel} className="cae-priority-item">
                <span className="cae-channel-name">{getChannelName(channel)}</span>
                <select
                  value={channelPriorities[channel] || 2}
                  onChange={(e) => handlePriorityChange(channel, e.target.value)}
                  disabled={active}
                  className="cae-priority-select"
                  style={{ borderColor: getPriorityColor(channelPriorities[channel] || 2) }}
                >
                  {PRIORITIES.map(p => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="cae-section">
        <h3>Settings</h3>
        <div className="cae-settings-grid">
          <div className="cae-setting-item">
            <label>Overlap Tolerance (dB)</label>
            <input
              type="number"
              value={settings.overlapToleranceDb}
              onChange={(e) => setSettings(prev => ({ ...prev, overlapToleranceDb: parseFloat(e.target.value) }))}
              disabled={active}
              step={0.5}
              min={1}
              max={12}
            />
            <small>Max level difference to consider as overlap</small>
          </div>
          
          <div className="cae-setting-item">
            <label>Max Cut (dB)</label>
            <input
              type="number"
              value={settings.maxCutDb}
              onChange={(e) => setSettings(prev => ({ ...prev, maxCutDb: parseFloat(e.target.value) }))}
              disabled={active}
              step={0.5}
              min={-12}
              max={-1}
            />
            <small>Maximum reduction in masker channels</small>
          </div>
          
          <div className="cae-setting-item">
            <label>Max Boost (dB)</label>
            <input
              type="number"
              value={settings.maxBoostDb}
              onChange={(e) => setSettings(prev => ({ ...prev, maxBoostDb: parseFloat(e.target.value) }))}
              disabled={active}
              step={0.5}
              min={0.5}
              max={6}
            />
            <small>Maximum boost in masked channels</small>
          </div>
          
          <div className="cae-setting-item">
            <label>Min Band Level (dB)</label>
            <input
              type="number"
              value={settings.minBandLevelDb}
              onChange={(e) => setSettings(prev => ({ ...prev, minBandLevelDb: parseFloat(e.target.value) }))}
              disabled={active}
              step={5}
              min={-100}
              max={-40}
            />
            <small>Minimum level for band processing</small>
          </div>
        </div>
      </section>

      <section className="cae-section">
        <div className="cae-actions">
          <button
            className={`cae-btn ${active ? 'stop' : 'start'}`}
            onClick={active ? handleStop : handleStart}
            disabled={!selectedDevice || selectedChannels.length === 0}
          >
            {active ? 'Stop Analysis' : 'Start Analysis'}
          </button>
          
          {adjustments.length > 0 && (
            <button
              className="cae-btn toggle"
              onClick={() => setShowAdjustments(!showAdjustments)}
            >
              {showAdjustments ? 'Hide Adjustments' : `Show Adjustments (${adjustments.length})`}
            </button>
          )}
        </div>
      </section>

      {showAdjustments && adjustments.length > 0 && (
        <section className="cae-section">
          <h3>EQ Adjustments</h3>
          <div className="cae-adjustments-table">
            <table>
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Frequency</th>
                  <th>Gain</th>
                  <th>Q Factor</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {adjustments.map((adj, index) => (
                  <tr key={index}>
                    <td>{getChannelName(adj.channel_id)}</td>
                    <td>{Math.round(adj.frequency_hz)} Hz</td>
                    <td className={adj.gain_db > 0 ? 'positive' : 'negative'}>
                      {formatGain(adj.gain_db)}
                    </td>
                    <td>{adj.q_factor.toFixed(1)}</td>
                    <td>{adj.gain_db > 0 ? 'Boost' : 'Cut'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {Object.keys(channelBandEnergy).length > 0 && (
        <section className="cae-section">
          <h3>Frequency Band Analysis</h3>
          <div className="cae-band-energy">
            {selectedChannels.map(channel => {
              const bands = channelBandEnergy[channel];
              if (!bands) return null;
              
              return (
                <div key={channel} className="cae-channel-bands">
                  <div className="cae-channel-label">{getChannelName(channel)}</div>
                  <div className="cae-bands-visual">
                    {FREQUENCY_BANDS.map(band => {
                      const level = bands[band.id];
                      const normalizedLevel = level ? Math.max(0, (level + 80) / 80) : 0;
                      
                      return (
                        <div
                          key={band.id}
                          className="cae-band-bar"
                          style={{
                            height: `${normalizedLevel * 100}%`,
                            backgroundColor: band.color
                          }}
                          title={`${band.name} (${band.freq}): ${level ? level.toFixed(1) : '--'} dB`}
                        />
                      );
                    })}
                  </div>
                  <div className="cae-bands-labels">
                    {FREQUENCY_BANDS.map(band => (
                      <span key={band.id} className="cae-band-label">{band.name}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="cae-section cae-info">
        <h3>About Cross-Adaptive EQ</h3>
        <ul>
          <li><strong>Mirror EQ Strategy:</strong> Boosts one track and cuts another at the same frequency to resolve conflicts</li>
          <li><strong>Subtractive EQ (Cuts):</strong> Uses Q=4.0 for surgical precision in masker channels</li>
          <li><strong>Additive EQ (Boosts):</strong> Uses Q=2.0 for musicality in masked channels</li>
          <li><strong>Boost Amount:</strong> Half of the cut amount for conservative processing</li>
          <li><strong>Reference:</strong> Based on IMP 7.3 research (De Man, Reiss & Stables)</li>
        </ul>
      </section>
    </div>
  );
}

export default CrossAdaptiveEQTab;
