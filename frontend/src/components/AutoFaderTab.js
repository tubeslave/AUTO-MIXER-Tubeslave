import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoFaderTab.css';
import SignalHint from './SignalHint';

function AutoFaderTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelLevels, setChannelLevels] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    bleedThreshold: -50, targetLoudness: -18, vocalistBoost: 3,
    smoothingAlpha: 0.3, updateInterval: 0.1, faderRangeMin: -20, faderRangeMax: 10,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined) setRunning(data.running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.levels) setChannelLevels(data.levels);
    };
    const handleLoaded = (data) => {
      if (data.settings?.autoFader) setSettings(s => ({ ...s, ...data.settings.autoFader }));
    };
    websocketService.on('auto_fader_status', handle);
    websocketService.on('all_settings_loaded', handleLoaded);
    websocketService.getAutoFaderStatus();
    websocketService.loadAllSettings();
    return () => {
      websocketService.off('auto_fader_status', handle);
      websocketService.off('all_settings_loaded', handleLoaded);
    };
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    const mapping = {}; selectedChannels.forEach(ch => { mapping[ch] = ch; });
    const chSettings = {}; selectedChannels.forEach(ch => { chSettings[ch] = { preset: 'custom' }; });
    websocketService.startRealtimeFader(selectedDevice, selectedChannels, chSettings, mapping, settings);
  };
  const handleStop = () => websocketService.stopRealtimeFader();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_fader" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-fader-tab">
      <SignalHint moduleKey="auto_fader" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={running ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {running ? 'Стоп' : 'Старт'}
          </button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <div className="settings-panel">
          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            <span>Настройки</span><span>{showSettings ? '▼' : '▶'}</span>
          </div>
          {showSettings && (
            <div className="settings-body">
              <div className="setting-row">
                <label>Bleed Threshold</label>
                <input type="range" min="-70" max="-30" value={settings.bleedThreshold}
                  onChange={e => setSettings(s => ({...s, bleedThreshold: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.bleedThreshold} dB</span>
              </div>
              <div className="setting-row">
                <label>Target Loudness</label>
                <input type="range" min="-30" max="-12" value={settings.targetLoudness}
                  onChange={e => setSettings(s => ({...s, targetLoudness: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.targetLoudness} LUFS</span>
              </div>
              <div className="setting-row">
                <label>Smoothing</label>
                <input type="range" min="0.05" max="0.9" step="0.05" value={settings.smoothingAlpha}
                  onChange={e => setSettings(s => ({...s, smoothingAlpha: parseFloat(e.target.value)}))} disabled={running} />
                <span className="val">{settings.smoothingAlpha}</span>
              </div>
            </div>
          )}
        </div>

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Level</th><th>Fader</th></tr></thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const level = channelLevels[ch];
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td>{level?.lufs?.toFixed(1) || '--'} LUFS</td>
                  <td>{level?.fader?.toFixed(1) || '--'} dB</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoFaderTab;
