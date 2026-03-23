import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoFaderTab.css';
import SignalHint from './SignalHint';
import ChannelPresetSelect from './ChannelPresetSelect';
import { mapPresetForMethod } from '../constants/instrumentPresets';

const FADER_MODES = [
  { id: 'off', label: 'Off' },
  { id: 'manual', label: 'Manual' },
  { id: 'auto_assist', label: 'Assist' },
  { id: 'full_auto', label: 'Auto' },
];

function AutoFaderTab({
  selectedChannels,
  availableChannels,
  selectedDevice,
  audioDevices,
  globalMode,
  channelPresets = {},
  setChannelPreset = () => {},
  detectInstrumentPreset = () => 'custom',
}) {
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState('off');
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
      if (data.mode !== undefined) setMode(data.mode);
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
    const chSettings = {};
    selectedChannels.forEach(ch => {
      const inferred = detectInstrumentPreset(availableChannels?.find(c => c.id === ch)?.name);
      const preset = channelPresets[String(ch)] || inferred;
      chSettings[ch] = { preset: mapPresetForMethod(preset, 'gain') };
    });
    websocketService.startRealtimeFader(selectedDevice, selectedChannels, chSettings, mapping, settings);
  };
  const handleStop = () => websocketService.stopRealtimeFader();

  const handleModeChange = (m) => {
    setMode(m);
    websocketService.send({ type: 'set_auto_fader_mode', mode: m });
  };

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
          <div className="mode-pills">
            {FADER_MODES.map(m => (
              <button key={m.id} className={`pill ${mode === m.id ? 'active' : ''}`}
                onClick={() => handleModeChange(m.id)}>
                {m.label}
              </button>
            ))}
          </div>
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
          <thead>
            <tr>
              <th>Канал</th>
              <th>Пресет</th>
              <th>Level</th>
              <th>Fader</th>
            </tr>
          </thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const inferred = detectInstrumentPreset(name);
              const preset = channelPresets[String(ch)] || inferred;
              const level = channelLevels[ch];
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td>
                    <ChannelPresetSelect
                      value={preset}
                      onChange={(presetId) => setChannelPreset(ch, presetId)}
                      disabled={running}
                    />
                  </td>
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
