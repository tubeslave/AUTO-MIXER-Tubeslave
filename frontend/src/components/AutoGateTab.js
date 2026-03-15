import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoGateTab.css';
import SignalHint from './SignalHint';

function AutoGateTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelGateStatus, setChannelGateStatus] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    threshold: -40, holdMs: 100, releaseMs: 200, rangeDb: -80,
    lookaheadMs: 5, hysteresisDb: 6, useCrossAdaptive: true,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined) setRunning(data.running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.channel_status) setChannelGateStatus(data.channel_status);
    };
    websocketService.on('auto_gate_status', handle);
    websocketService.getAutoGateStatus();
    return () => websocketService.off('auto_gate_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    const configs = {};
    selectedChannels.forEach(ch => { configs[ch] = { threshold: settings.threshold }; });
    websocketService.startAutoGate(selectedDevice, selectedChannels, configs, settings);
  };
  const handleStop = () => websocketService.stopAutoGate();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_gate" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-gate-tab">
      <SignalHint moduleKey="auto_gate" />
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
                <label>Threshold</label>
                <input type="range" min="-60" max="-20" value={settings.threshold}
                  onChange={e => setSettings(s => ({...s, threshold: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.threshold} dB</span>
              </div>
              <div className="setting-row">
                <label>Hold</label>
                <input type="range" min="10" max="500" step="10" value={settings.holdMs}
                  onChange={e => setSettings(s => ({...s, holdMs: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.holdMs} ms</span>
              </div>
              <div className="setting-row">
                <label>Release</label>
                <input type="range" min="50" max="500" step="10" value={settings.releaseMs}
                  onChange={e => setSettings(s => ({...s, releaseMs: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.releaseMs} ms</span>
              </div>
              <div className="setting-row">
                <label>Range</label>
                <input type="range" min="-80" max="-20" value={settings.rangeDb}
                  onChange={e => setSettings(s => ({...s, rangeDb: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.rangeDb} dB</span>
              </div>
            </div>
          )}
        </div>

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Gate</th></tr></thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const gs = channelGateStatus[ch];
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td className={gs?.open ? 'gate-open' : 'gate-closed'}>
                    {gs ? (gs.open ? '● Open' : '○ Closed') : '--'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoGateTab;
