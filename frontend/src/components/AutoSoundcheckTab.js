import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoSoundcheckTab.css';
import SignalHint from './SignalHint';
import { buildAutofohSessionSummary } from './autoSoundcheckSummary.mjs';

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
  const [showSessionDetails, setShowSessionDetails] = useState(false);
  const [sessionStatus, setSessionStatus] = useState({});
  const [timings, setTimings] = useState({
    gainDuration: 30, phaseDuration: 15, eqDuration: 20,
    compDuration: 15, gateDuration: 10, faderDuration: 15,
  });

  useEffect(() => {
    const handle = (data) => {
      setSessionStatus(prev => ({ ...prev, ...data }));
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
    setShowSessionDetails(false);
    setSessionStatus({});
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
  const sessionSummary = buildAutofohSessionSummary(sessionStatus);
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
        {!running && sessionSummary && (
          <div className="autofoh-session-card">
            <div className="autofoh-session-head">
              <div>
                <div className="autofoh-session-kicker">AutoFOH</div>
                <div className="autofoh-session-title">{sessionSummary.title}</div>
              </div>
              {sessionSummary.hasExpandableDetails && (
                <button
                  type="button"
                  className="autofoh-session-toggle"
                  onClick={() => setShowSessionDetails(open => !open)}
                >
                  {showSessionDetails ? 'Скрыть детали' : 'Показать детали'}
                </button>
              )}
            </div>
            <div className="autofoh-session-summary">{sessionSummary.summaryText}</div>
            {sessionSummary.chips.length > 0 && (
              <div className="autofoh-session-chips">
                {sessionSummary.chips.map(chip => (
                  <div key={chip.label} className="autofoh-session-chip">
                    <span className="autofoh-session-chip-label">{chip.label}</span>
                    <span className="autofoh-session-chip-value">{chip.value}</span>
                  </div>
                ))}
              </div>
            )}
            {!showSessionDetails && sessionSummary.detailLines.length > 0 && (
              <div className="autofoh-session-details">
                {sessionSummary.detailLines.slice(0, 2).map((line, index) => (
                  <div key={`${line}-${index}`} className="autofoh-session-detail">{line}</div>
                ))}
              </div>
            )}
            {showSessionDetails && sessionSummary.sections.length > 0 && (
              <div className="autofoh-session-report">
                {sessionSummary.sections.map(section => (
                  <div key={section.title} className="autofoh-session-report-section">
                    <div className="autofoh-session-report-title">{section.title}</div>
                    <div className="autofoh-session-report-items">
                      {section.items.map((item, index) => (
                        <div key={`${section.title}-${index}`} className="autofoh-session-report-item">{item}</div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

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
