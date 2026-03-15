import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './PhaseAlignmentTab.css';
import SignalHint from './SignalHint';

function PhaseAlignmentTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [measurements, setMeasurements] = useState({});
  const [referenceChannel, setReferenceChannel] = useState(selectedChannels?.[0] || 1);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    fftSize: 4096, analysisTime: 5, coherenceThreshold: 0.5,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined || data.active !== undefined) setRunning(data.running ?? data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.measurements) setMeasurements(data.measurements);
    };
    websocketService.on('phase_alignment_status', handle);
    websocketService.getPhaseAlignmentStatus();
    return () => websocketService.off('phase_alignment_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    websocketService.startPhaseAlignment(selectedDevice, referenceChannel, selectedChannels);
  };
  const handleStop = () => websocketService.stopPhaseAlignment();
  const handleApply = () => websocketService.applyPhaseCorrections(measurements);
  const handleReset = () => {
    websocketService.resetPhaseDelay(selectedChannels);
    setStatusMessage('Phase/Delay сброшен');
  };

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_phase" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="phase-alignment-tab">
      <SignalHint moduleKey="auto_phase" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={running ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {running ? 'Стоп' : 'Анализ'}
          </button>
          <button className="btn-sm" onClick={handleApply}
            disabled={Object.keys(measurements).length === 0}>
            Применить
          </button>
          <button className="btn-sm" onClick={handleReset}
            disabled={!channels.length}>
            Сброс
          </button>
          <div className="ref-select">
            <label>Ref:</label>
            <select value={referenceChannel} onChange={e => setReferenceChannel(parseInt(e.target.value))} disabled={running}>
              {channels.map(ch => {
                const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
                return <option key={ch} value={ch}>{name}</option>;
              })}
            </select>
          </div>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <div className="settings-panel">
          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            <span>Настройки GCC-PHAT</span><span>{showSettings ? '▼' : '▶'}</span>
          </div>
          {showSettings && (
            <div className="settings-body">
              <div className="setting-row">
                <label>FFT Size</label>
                <select value={settings.fftSize} onChange={e => setSettings(s => ({...s, fftSize: parseInt(e.target.value)}))} disabled={running}>
                  <option value={2048}>2048</option><option value={4096}>4096</option><option value={8192}>8192</option>
                </select>
              </div>
              <div className="setting-row">
                <label>Время анализа</label>
                <input type="range" min="2" max="15" value={settings.analysisTime}
                  onChange={e => setSettings(s => ({...s, analysisTime: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.analysisTime}s</span>
              </div>
              <div className="setting-row">
                <label>Coherence min</label>
                <input type="range" min="0.1" max="0.9" step="0.1" value={settings.coherenceThreshold}
                  onChange={e => setSettings(s => ({...s, coherenceThreshold: parseFloat(e.target.value)}))} disabled={running} />
                <span className="val">{settings.coherenceThreshold}</span>
              </div>
            </div>
          )}
        </div>

        {Object.keys(measurements).length > 0 && (
          <table className="data-table">
            <thead><tr><th>Канал</th><th>Delay</th><th>Phase</th><th>Coherence</th></tr></thead>
            <tbody>
              {Object.entries(measurements).map(([chId, m]) => {
                const name = availableChannels?.find(c => c.id === parseInt(chId))?.name || `Ch ${chId}`;
                return (
                  <tr key={chId}>
                    <td>{name}</td>
                    <td>{m.delay_ms?.toFixed(2) || '--'} ms</td>
                    <td className={m.invert_phase ? 'phase-inv' : ''}>{m.invert_phase ? '⟳ Invert' : '✓ OK'}</td>
                    <td>{m.coherence?.toFixed(2) || '--'}</td>
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

export default PhaseAlignmentTab;
