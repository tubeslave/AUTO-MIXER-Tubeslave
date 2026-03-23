import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoCompressorTab.css';
import SignalHint from './SignalHint';
import { INSTRUMENT_PRESETS, mapPresetForMethod } from '../constants/instrumentPresets';

const CF_ICONS = { percussion: '⚡', drums: '🥁', vocal: '🎤', bass: '🎸', pad: '🎹', flat: '〰️' };
const CF_COLORS = { percussion: '#f85149', drums: '#f0883e', vocal: '#00d4ff', bass: '#3fb950', pad: '#a371f7', flat: '#484f58' };

function AutoCompressorTab({
  selectedChannels,
  availableChannels,
  selectedDevice,
  audioDevices,
  globalMode,
  channelPresets = {},
  setChannelPreset = () => {},
  detectInstrumentPreset = () => 'custom',
}) {
  const [active, setActive] = useState(false);
  const [soundcheckRunning, setSoundcheckRunning] = useState(false);
  const [liveRunning, setLiveRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [liveStatus, setLiveStatus] = useState({});
  const [cfMeasurements, setCfMeasurements] = useState({});
  const [channelNames, setChannelNames] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [cfSettings, setCfSettings] = useState({
    targetLufs: -18.0, headroomDb: 3.0, maxGrDb: 12.0,
    updateIntervalMs: 100, useAdaptive: true, rmsWindowMs: 10, cfSmoothingMs: 100
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.soundcheck_running !== undefined) setSoundcheckRunning(data.soundcheck_running);
      if (data.live_running !== undefined) setLiveRunning(data.live_running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.cf_measurements) setCfMeasurements(data.cf_measurements);
      if (data.soundcheck?.complete) setSoundcheckRunning(false);
      if (data.live && data.channel !== undefined) {
        setLiveStatus(p => ({ ...p, [data.channel]: {
          gr_estimate: data.gr_estimate, lufs: data.lufs,
          cf_db: data.cf_db, cf_class: data.cf_class
        }}));
      }
    };
    websocketService.on('auto_compressor_status', handle);
    websocketService.getAutoCompressorStatus();
    return () => websocketService.off('auto_compressor_status', handle);
  }, []);

  const buildMapping = () => {
    const m = {};
    (selectedChannels || []).forEach(id => { m[typeof id === 'number' ? id : parseInt(id)] = id; });
    return m;
  };
  const buildNames = () => {
    const n = {};
    (availableChannels || []).forEach(c => {
      if (selectedChannels?.includes(c.id)) n[c.id] = c.name || `Ch ${c.id}`;
    });
    return n;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    setChannelNames(buildNames());
    const channelSettings = {};
    selectedChannels.forEach(ch => {
      const inferred = detectInstrumentPreset(availableChannels?.find(c => c.id === ch)?.name);
      const preset = channelPresets[String(ch)] || inferred;
      channelSettings[ch] = { preset: mapPresetForMethod(preset, 'compressor') };
    });
    websocketService.startAutoCompressor(
      selectedDevice,
      selectedChannels,
      buildMapping(),
      buildNames(),
      channelSettings,
      { method: 'cf_lufs', cf_settings: cfSettings },
    );
  };
  const handleStop = () => websocketService.stopAutoCompressor();

  const handleModeAction = () => {
    if (globalMode === 'soundcheck') {
      if (soundcheckRunning) websocketService.stopAutoCompressorSoundcheck();
      else websocketService.startAutoCompressorSoundcheck(1, 1, null, { method: 'cf_lufs', cf_settings: cfSettings });
    } else {
      if (liveRunning) websocketService.stopAutoCompressorLive();
      else websocketService.startAutoCompressorLive(true, { method: 'cf_lufs', cf_settings: cfSettings });
    }
  };

  const isModeRunning = globalMode === 'soundcheck' ? soundcheckRunning : liveRunning;
  const channels = selectedChannels || [];

  if (!channels.length) return (<div><SignalHint moduleKey="auto_compressor" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-compressor-tab">
      <SignalHint moduleKey="auto_compressor" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${active ? 'stop' : 'go'}`} onClick={active ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {active ? 'Стоп' : 'Старт'}
          </button>
          <button className={`btn-sm ${isModeRunning ? 'running' : ''}`} onClick={handleModeAction} disabled={!active}>
            {isModeRunning ? `Стоп ${globalMode}` : globalMode === 'soundcheck' ? 'Soundcheck' : 'Live'}
          </button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <div className="settings-panel">
          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            <span>Настройки CF-LUFS</span><span>{showSettings ? '▼' : '▶'}</span>
          </div>
          {showSettings && (
            <div className="settings-body">
              <div className="setting-row">
                <label>Target LUFS</label>
                <input type="range" min="-30" max="-12" step="1" value={cfSettings.targetLufs}
                  onChange={e => setCfSettings(s => ({...s, targetLufs: parseFloat(e.target.value)}))} disabled={active} />
                <span className="val">{cfSettings.targetLufs}</span>
              </div>
              <div className="setting-row">
                <label>Headroom</label>
                <input type="range" min="1" max="6" step="0.5" value={cfSettings.headroomDb}
                  onChange={e => setCfSettings(s => ({...s, headroomDb: parseFloat(e.target.value)}))} disabled={active} />
                <span className="val">{cfSettings.headroomDb} dB</span>
              </div>
              <div className="setting-row">
                <label>Max GR</label>
                <input type="range" min="6" max="20" step="1" value={cfSettings.maxGrDb}
                  onChange={e => setCfSettings(s => ({...s, maxGrDb: parseFloat(e.target.value)}))} disabled={active} />
                <span className="val">{cfSettings.maxGrDb} dB</span>
              </div>
            </div>
          )}
        </div>

        {/* CF Measurements */}
        {Object.keys(cfMeasurements).length > 0 && (
          <table className="data-table">
            <thead><tr><th>Канал</th><th>CF</th><th>Класс</th><th>Attack</th><th>Release</th><th>Ratio</th></tr></thead>
            <tbody>
              {Object.entries(cfMeasurements).map(([id, d]) => (
                <tr key={id}>
                  <td>{channelNames[id] || `Ch ${id}`}</td>
                  <td>{d.cf_db?.toFixed(1) || '--'} dB</td>
                  <td style={{color: CF_COLORS[d.cf_class]}}>{CF_ICONS[d.cf_class]} {d.cf_class}</td>
                  <td>{d.params?.attack_ms?.toFixed(0) || '--'} ms</td>
                  <td>{d.params?.release_ms?.toFixed(0) || '--'} ms</td>
                  <td>{d.params?.ratio?.toFixed(1) || '--'}:1</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Channel list with live status */}
        {Object.keys(cfMeasurements).length === 0 && (
          <table className="data-table">
            <thead><tr><th>Канал</th><th>Инструмент</th><th>GR</th><th>CF</th></tr></thead>
            <tbody>
              {channels.map(ch => {
                const s = liveStatus[ch];
                const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
                const inferred = detectInstrumentPreset(name);
                const preset = channelPresets[String(ch)] || inferred;
                return (
                  <tr key={ch}>
                    <td>{name}</td>
                    <td>
                      <select value={preset} onChange={e => setChannelPreset(ch, e.target.value)} disabled={active}>
                        {INSTRUMENT_PRESETS.map(p => (
                          <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                      </select>
                    </td>
                    <td>{s?.gr_estimate?.toFixed(1) || '--'} dB</td>
                    <td style={{color: CF_COLORS[s?.cf_class]}}>
                      {s?.cf_class ? `${CF_ICONS[s.cf_class]} ${s.cf_db?.toFixed(1)} dB` : '--'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default AutoCompressorTab;
