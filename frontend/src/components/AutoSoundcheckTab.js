import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoSoundcheckTab.css';
import SignalHint from './SignalHint';

const CYCLE_STEPS = [
  { id: 'gain', label: 'Gain Staging' },
  { id: 'phase', label: 'Phase' },
  { id: 'eq', label: 'EQ' },
  { id: 'comp', label: 'Compressor' },
  { id: 'gate', label: 'Gate' },
  { id: 'fader', label: 'Fader' },
];

function AutoSoundcheckTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [observeOnly, setObserveOnly] = useState(true);
  const [statusMessage, setStatusMessage] = useState('');
  const [currentStep, setCurrentStep] = useState('');
  const [progress, setProgress] = useState(0);
  const [log, setLog] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [timings, setTimings] = useState({
    gainDuration: 30, phaseDuration: 15, eqDuration: 20,
    compDuration: 15, gateDuration: 10, faderDuration: 15,
  });

  useEffect(() => {
    const handle = (data) => {
      if (data.is_running !== undefined) {
        setRunning(data.is_running);
      } else if (data.running !== undefined) {
        setRunning(data.running);
      }
      if (data.message) {
        setStatusMessage(data.message);
        setLog(prev => [...prev.slice(-20), data.message]);
      }
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.current_step) setCurrentStep(data.current_step);
      if (data.step_progress !== undefined) {
        setProgress(data.step_progress);
      } else if (data.progress !== undefined) {
        setProgress(data.progress);
      }
    };
    const handleObservation = (data) => {
      if (data.message) {
        setLog(prev => [...prev.slice(-80), data.message]);
      }
    };
    websocketService.on('auto_soundcheck_status', handle);
    websocketService.on('auto_soundcheck_observation', handleObservation);
    websocketService.getAutoSoundcheckStatus();
    return () => {
      websocketService.off('auto_soundcheck_status', handle);
      websocketService.off('auto_soundcheck_observation', handleObservation);
    };
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    const mapping = {}; selectedChannels.forEach(ch => { mapping[ch] = ch; });
    const chSettings = {}; selectedChannels.forEach(ch => { chSettings[ch] = { preset: 'custom' }; });
    const backendTimings = {
      gain_staging: timings.gainDuration,
      phase_alignment: timings.phaseDuration,
      auto_eq: timings.eqDuration,
      auto_fader: timings.faderDuration,
    };
    setLog([]);
    websocketService.startAutoSoundcheck(
      selectedDevice,
      selectedChannels,
      chSettings,
      mapping,
      backendTimings,
      observeOnly
    );
  };
  const handleStop = () => websocketService.stopAutoSoundcheck();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_soundcheck" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-soundcheck-tab">
      <SignalHint moduleKey="auto_soundcheck" />
      <div className="module-card">
        <div className="module-actions">
          <label className="settings-toggle" style={{ cursor: running ? 'default' : 'pointer' }}>
            <span>Observation Mode</span>
            <input
              type="checkbox"
              checked={observeOnly}
              disabled={running}
              onChange={e => setObserveOnly(e.target.checked)}
            />
          </label>
          <button className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={running ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {running ? 'Стоп' : 'Запустить Soundcheck'}
          </button>
        </div>
        {observeOnly && !running && (
          <div className="module-status">Режим наблюдения: пульт не изменяется, все команды только логируются.</div>
        )}
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        {running && (
          <div className="progress-section">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="step-indicators">
              {CYCLE_STEPS.map(s => (
                <span key={s.id} className={`step-chip ${currentStep === s.id ? 'active' : ''}`}>
                  {s.label}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="settings-panel">
          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            <span>Тайминги</span><span>{showSettings ? '▼' : '▶'}</span>
          </div>
          {showSettings && (
            <div className="settings-body">
              {Object.entries(timings).map(([key, val]) => (
                <div className="setting-row" key={key}>
                  <label>{key.replace('Duration', '')}</label>
                  <input type="range" min="5" max="60" step="5" value={val}
                    onChange={e => setTimings(t => ({...t, [key]: parseInt(e.target.value)}))} disabled={running} />
                  <span className="val">{val}s</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {log.length > 0 && (
          <div className="soundcheck-log">
            {log.map((msg, i) => <div key={i} className="log-line">{msg}</div>)}
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoSoundcheckTab;
