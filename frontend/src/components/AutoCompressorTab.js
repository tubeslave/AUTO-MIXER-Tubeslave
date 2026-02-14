import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoCompressorTab.css';

const PROFILES = [
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
  const [profile, setProfile] = useState('base');
  const [autoCorrect, setAutoCorrect] = useState(true);
  const [channelNames, setChannelNames] = useState({});

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.soundcheck_running !== undefined) setSoundcheckRunning(data.soundcheck_running);
      if (data.live_running !== undefined) setLiveRunning(data.live_running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Error: ${data.error}`);
      if (data.current_params) setCurrentParams(data.current_params);
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
            status: data.status
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
    websocketService.startAutoCompressor(
      selectedDevice,
      selectedChannels,
      mapping,
      names
    );
  };

  const handleStop = () => {
    websocketService.stopAutoCompressor();
  };

  const handleSoundcheckStart = () => {
    websocketService.startAutoCompressorSoundcheck(1, 1, null);
  };

  const handleSoundcheckStop = () => {
    websocketService.stopAutoCompressorSoundcheck();
  };

  const handleLiveStart = () => {
    websocketService.startAutoCompressorLive(autoCorrect);
  };

  const handleLiveStop = () => {
    websocketService.stopAutoCompressorLive();
  };

  const handleSetProfile = () => {
    if (selectedChannel == null) return;
    websocketService.setAutoCompressorProfile(selectedChannel, profile);
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-compressor-tab">
      <section className="auto-compressor-section">
        <h2>Auto Compressor</h2>
        <p className="section-description">
          Signal is analyzed <strong>post-fader</strong>. Configure the mixer so that post-fader output of each channel is routed to the selected audio device (e.g. Dante).
        </p>

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
        </div>
        {active && (
          <div className="indicator active">Active (post-fader capture)</div>
        )}
      </section>

      <section className="auto-compressor-section">
        <h2>Soundcheck</h2>
        <p className="section-description">Record 5–10 s per channel, analyze, then apply compressor settings.</p>
        <div className="auto-compressor-actions">
          <button
            className="btn-primary"
            onClick={handleSoundcheckStart}
            disabled={!active || soundcheckRunning}
          >
            Start Soundcheck
          </button>
          <button
            className="btn-secondary"
            onClick={handleSoundcheckStop}
            disabled={!soundcheckRunning}
          >
            Stop
          </button>
        </div>
        {soundcheckRunning && soundcheckProgress.total > 0 && (
          <div className="soundcheck-progress">
            Channel {soundcheckProgress.current + 1}/{soundcheckProgress.total} {soundcheckProgress.channelName && `— ${soundcheckProgress.channelName}`}
          </div>
        )}
      </section>

      <section className="auto-compressor-section">
        <h2>Live</h2>
        <p className="section-description">Monitor levels and optionally auto-correct over/under compression.</p>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={autoCorrect}
            onChange={(e) => setAutoCorrect(e.target.checked)}
            disabled={liveRunning}
          />
          Auto-correct
        </label>
        <div className="auto-compressor-actions">
          <button
            className="btn-primary"
            onClick={handleLiveStart}
            disabled={!active || liveRunning}
          >
            Start Live
          </button>
          <button
            className="btn-secondary"
            onClick={handleLiveStop}
            disabled={!liveRunning}
          >
            Stop Live
          </button>
        </div>
      </section>

      <section className="auto-compressor-section">
        <h2>Channels</h2>
        <div className="channel-list">
          {channels.map(chId => {
            const ch = typeof chId === 'number' ? chId : parseInt(chId, 10);
            const name = channelNames[ch] || availableChannels?.find(c => (c.id === ch || c.id === String(ch)))?.name || `Ch ${ch}`;
            const params = currentParams[ch] || {};
            const live = liveStatus[ch];
            return (
              <div
                key={ch}
                className={`channel-row ${selectedChannel === ch ? 'selected' : ''}`}
                onClick={() => setSelectedChannel(ch)}
              >
                <span className="channel-name">{name}</span>
                <span className="channel-params">
                  Thr: {params.threshold != null ? params.threshold.toFixed(0) : '—'} dB
                  Ratio: {params.ratio_wing || '—'}
                  Att: {params.attack_ms != null ? params.attack_ms : '—'} ms
                  Rel: {params.release_ms != null ? params.release_ms : '—'} ms
                </span>
                {live && (
                  <span className={`channel-live status-${live.status || 'normal'}`}>
                    GR: {live.gr_estimate} dB · {live.status || 'normal'}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        {selectedChannel != null && (
          <div className="profile-row">
            <select value={profile} onChange={(e) => setProfile(e.target.value)}>
              {PROFILES.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <button className="btn-small" onClick={handleSetProfile}>Apply profile</button>
          </div>
        )}
      </section>

      <section className="auto-compressor-section status-section">
        <p className="status-message">{statusMessage || '—'}</p>
      </section>
    </div>
  );
}

export default AutoCompressorTab;
