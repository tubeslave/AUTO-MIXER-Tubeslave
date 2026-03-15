import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './CrossAdaptiveEQTab.css';
import SignalHint from './SignalHint';

function CrossAdaptiveEQTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [eqAdjustments, setEqAdjustments] = useState({});
  const [bandAnalysis, setBandAnalysis] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    reductionStrength: 0.5, frequencyBands: 6, updateInterval: 500,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.eq_adjustments) setEqAdjustments(data.eq_adjustments);
      if (data.band_analysis) setBandAnalysis(data.band_analysis);
    };
    websocketService.on('cross_adaptive_eq_status', handle);
    websocketService.getCrossAdaptiveEQStatus();
    return () => websocketService.off('cross_adaptive_eq_status', handle);
  }, []);

  const handleToggle = () => {
    if (active) {
      websocketService.stopCrossAdaptiveEQ();
    } else {
      if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
      websocketService.startCrossAdaptiveEQ(selectedDevice, selectedChannels, settings);
    }
  };

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="cross_adaptive_eq" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="cross-adaptive-eq-tab">
      <SignalHint moduleKey="cross_adaptive_eq" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${active ? 'stop' : 'go'}`}
            onClick={handleToggle}
            disabled={!selectedDevice || !channels.length}>
            {active ? 'Стоп' : 'Старт'}
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
                <label>Сила коррекции</label>
                <input type="range" min="0.1" max="1.0" step="0.1" value={settings.reductionStrength}
                  onChange={e => setSettings(s => ({...s, reductionStrength: parseFloat(e.target.value)}))} disabled={active} />
                <span className="val">{(settings.reductionStrength * 100).toFixed(0)}%</span>
              </div>
              <div className="setting-row">
                <label>Полос EQ</label>
                <select value={settings.frequencyBands} onChange={e => setSettings(s => ({...s, frequencyBands: parseInt(e.target.value)}))} disabled={active}>
                  <option value={4}>4</option><option value={6}>6</option><option value={8}>8</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {Object.keys(eqAdjustments).length > 0 && (
          <table className="data-table">
            <thead><tr><th>Канал</th><th>EQ коррекция</th></tr></thead>
            <tbody>
              {Object.entries(eqAdjustments).map(([chId, adj]) => {
                const name = availableChannels?.find(c => c.id === parseInt(chId))?.name || `Ch ${chId}`;
                return (
                  <tr key={chId}>
                    <td>{name}</td>
                    <td>{Array.isArray(adj) ? adj.map(a => `${a.freq}Hz: ${a.gain?.toFixed(1)}dB`).join(', ') : JSON.stringify(adj)}</td>
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

export default CrossAdaptiveEQTab;
