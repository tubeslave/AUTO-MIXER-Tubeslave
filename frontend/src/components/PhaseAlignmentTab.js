import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './PhaseAlignmentTab.css';
import SignalHint from './SignalHint';

function PhaseAlignmentTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [measurements, setMeasurements] = useState({});
  const [applyDetail, setApplyDetail] = useState({});
  const [referenceChannel, setReferenceChannel] = useState(selectedChannels?.[0] || 1);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    fftSize: 4096,
    analysisWindowSec: 10,
    referenceCoherenceMin: 0.1,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined || data.active !== undefined) setRunning(data.running ?? data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.measurements) setMeasurements(data.measurements);
      if (data.detail) setApplyDetail(data.detail);
    };
    const handleMeasurements = (data) => {
      if (data.measurements) setMeasurements(data.measurements);
    };
    const handleApply = (data) => {
      if (data.detail) setApplyDetail(data.detail);
      if (data.message) setStatusMessage(data.message);
    };
    websocketService.on('phase_alignment_status', handle);
    websocketService.on('phase_alignment_measurements', handleMeasurements);
    websocketService.on('phase_alignment_apply_result', handleApply);
    websocketService.getPhaseAlignmentStatus();
    return () => {
      websocketService.off('phase_alignment_status', handle);
      websocketService.off('phase_alignment_measurements', handleMeasurements);
      websocketService.off('phase_alignment_apply_result', handleApply);
    };
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    setMeasurements({});
    // Исключаем reference из списка — backend ожидает каналы для выравнивания (без ref)
    const channelsToAlign = selectedChannels.filter(ch => ch !== referenceChannel);
    websocketService.startPhaseAlignment(selectedDevice, referenceChannel, channelsToAlign, settings);
  };
  const handleStop = () => {
    setRunning(false);  // Оптимистичное обновление — UI сразу показывает «Анализ»
    websocketService.stopPhaseAlignment();
  };
  const handleApply = () => websocketService.applyPhaseCorrections();
  const handleReset = () => {
    websocketService.resetPhaseDelay(selectedChannels);
    setStatusMessage('Phase/Delay сброшен');
  };

  const channels = selectedChannels || [];
  const measurementByChannel = {};
  Object.entries(measurements || {}).forEach(([pairKey, m]) => {
    const match = pairKey.match(/\((\d+),\s*(\d+)\)/);
    const ch = match ? parseInt(match[2], 10) : parseInt(pairKey, 10);
    if (!Number.isNaN(ch)) measurementByChannel[ch] = m;
  });
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
            <span>Настройки поиска референса</span><span>{showSettings ? '▼' : '▶'}</span>
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
                <label>Окно анализа, сек</label>
                <input type="range" min="5" max="20" value={settings.analysisWindowSec}
                  onChange={e => setSettings(s => ({...s, analysisWindowSec: parseInt(e.target.value, 10)}))} disabled={running} />
                <span className="val">{settings.analysisWindowSec}s</span>
              </div>
              <div className="setting-row">
                <label>Coherence min</label>
                <input type="range" min="0.1" max="0.9" step="0.05" value={settings.referenceCoherenceMin}
                  onChange={e => setSettings(s => ({...s, referenceCoherenceMin: parseFloat(e.target.value)}))} disabled={running} />
                <span className="val">{settings.referenceCoherenceMin.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>

        {Object.keys(measurements).length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Канал</th><th>Delay GCC</th><th>Phase</th><th>Coherence</th>
                <th>Статус</th><th>Применено delay</th><th>Примечание</th>
              </tr>
            </thead>
            <tbody>
              {/* Reference channel row */}
              <tr key={`ref-${referenceChannel}`} className="ref-row">
                <td>
                  {availableChannels?.find(c => c.id === referenceChannel)?.name || `Ch ${referenceChannel}`}
                  <span className="ref-badge">Ref</span>
                </td>
                <td>0 ms</td>
                <td>—</td>
                <td>—</td>
                <td>{applyDetail[referenceChannel]?.eligible_for_alignment ? 'Участвует' : (Object.keys(applyDetail).length ? '—' : '—')}</td>
                <td>{applyDetail[referenceChannel]?.applied_delay_ms != null ? `${applyDetail[referenceChannel].applied_delay_ms.toFixed(2)} ms` : '—'}</td>
                <td>{applyDetail[referenceChannel]?.ignored_reason || '—'}</td>
              </tr>
              {/* Show every selected channel (except reference) even without fresh measurements */}
              {channels.filter(ch => ch !== referenceChannel).map((ch) => {
                const m = measurementByChannel[ch];
                const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
                const d = applyDetail[ch];
                const status = d?.eligible_for_alignment ? 'Участвует' : (d ? 'Игнор' : '—');
                const note = d?.ignored_reason === 'not_detected'
                  ? 'Сигнал референса не найден'
                  : (d?.ignored_reason === 'excluded_preset'
                    ? 'Исключен по пресету'
                    : (d?.ignored_reason === 'insufficient_hits'
                      ? 'Недостаточно подтверждений'
                      : (d?.ignored_reason === 'delay_above_10ms' ? 'Delay > 10 ms' : '—')));
                return (
                  <tr key={`phase-row-${ch}`} className={d?.eligible_for_alignment ? 'eligible' : (d ? 'ignored' : '')}>
                    <td>{name}</td>
                    <td>{m?.delay_ms != null ? `${m.delay_ms.toFixed(2)} ms` : '--'}</td>
                    <td className={m?.invert_phase ? 'phase-inv' : ''}>
                      {m ? (m.invert_phase ? '⟳ Invert' : '✓ OK') : '--'}
                    </td>
                    <td>{m?.coherence != null ? m.coherence.toFixed(2) : '--'}</td>
                    <td>{status}</td>
                    <td>{d?.applied_delay_ms != null ? `${d.applied_delay_ms.toFixed(2)} ms` : '—'}</td>
                    <td>{note}</td>
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
