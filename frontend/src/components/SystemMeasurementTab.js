import React, { useEffect, useMemo, useState } from 'react';
import websocketService from '../services/websocket';
import SignalHint from './SignalHint';
import './SystemMeasurementTab.css';

const TARGET_OPTIONS = [
  { value: 'master', label: 'Main L/R' },
  { value: 'group', label: 'Bus / Group' },
  { value: 'matrix', label: 'Matrix' },
];

const CORRECTION_MODES = [
  { value: 'flat', label: 'Flat TF' },
  { value: 'pink_noise_reference', label: 'Pink ref' },
];

const REFERENCE_CURVES = [
  { value: 'pink_noise_live_pa', label: 'Live PA' },
  { value: 'pink_noise_flat', label: 'Flat pink' },
];

function SystemMeasurementTab({ selectedDevice, availableChannels }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [result, setResult] = useState(null);
  const [settings, setSettings] = useState({
    referenceChannel: 59,
    measurementChannel: 57,
    durationSec: 6,
    targetBus: 'master',
    targetId: 1,
    correctionMode: 'flat',
    referenceCurve: 'pink_noise_live_pa',
  });

  useEffect(() => {
    const onStatus = (data) => {
      if (data.active !== undefined) setRunning(Boolean(data.active));
      if (data.message) setStatusMessage(data.message);
    };
    const onResult = (data) => {
      setRunning(false);
      if (data.success === false) {
        setStatusMessage(`Ошибка: ${data.error}`);
        return;
      }
      setResult(data);
      if (data.quality !== undefined) {
        setStatusMessage(`Измерение готово, quality ${(data.quality * 100).toFixed(0)}%`);
      }
    };
    const onApply = (data) => {
      setStatusMessage(data.success ? (data.message || 'Коррекция применена') : `Ошибка: ${data.error}`);
    };
    const onReset = (data) => {
      setStatusMessage(data.success ? (data.message || 'EQ сброшен') : `Ошибка: ${data.error}`);
    };

    websocketService.on('system_measurement_status', onStatus);
    websocketService.on('system_measurement_result', onResult);
    websocketService.on('system_measurement_apply_result', onApply);
    websocketService.on('system_measurement_reset_result', onReset);
    websocketService.getSystemMeasurementStatus();

    return () => {
      websocketService.off('system_measurement_status', onStatus);
      websocketService.off('system_measurement_result', onResult);
      websocketService.off('system_measurement_apply_result', onApply);
      websocketService.off('system_measurement_reset_result', onReset);
    };
  }, []);

  const topCorrections = useMemo(() => {
    const corrections = result?.corrections || [];
    return corrections.slice(0, 8);
  }, [result]);

  const handleAnalyze = () => {
    if (!selectedDevice) {
      setStatusMessage('Выберите аудио-устройство');
      return;
    }
    setRunning(true);
    setStatusMessage('Идёт захват...');
    websocketService.startSystemMeasurement(
      selectedDevice,
      settings.referenceChannel,
      settings.measurementChannel,
      settings.durationSec,
      settings.targetBus,
      settings.targetId,
      settings.correctionMode,
      settings.referenceCurve,
    );
  };

  const handleApply = () => {
    websocketService.applySystemMeasurement(settings.targetBus, settings.targetId);
  };

  const handleReset = () => {
    websocketService.resetSystemMeasurement(settings.targetBus, settings.targetId);
  };

  return (
    <div className="system-measurement-tab">
      <SignalHint moduleKey="system_measurement" />
      <div className="module-card">
        <div className="module-actions">
          <button
            className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={handleAnalyze}
            disabled={running || !selectedDevice}
          >
            {running ? 'Измерение...' : 'Измерить'}
          </button>
          <button className="btn-sm" onClick={handleApply} disabled={!result?.corrections?.length}>
            Применить
          </button>
          <button className="btn-sm" onClick={handleReset}>
            Сброс EQ
          </button>
        </div>

        <div className="module-status">
          Route `reference` на Dante/вход, `measurement mic` на отдельный Dante-канал и подайте в PA тестовый материал или pink noise.
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <div className="settings-grid">
          <label>
            Ref ch
            <input
              type="number"
              value={settings.referenceChannel}
              onChange={e => setSettings(s => ({ ...s, referenceChannel: parseInt(e.target.value || '1', 10) }))}
              disabled={running}
            />
          </label>
          <label>
            Meas mic
            <input
              type="number"
              value={settings.measurementChannel}
              onChange={e => setSettings(s => ({ ...s, measurementChannel: parseInt(e.target.value || '1', 10) }))}
              disabled={running}
            />
          </label>
          <label>
            Duration
            <input
              type="number"
              min="3"
              max="20"
              value={settings.durationSec}
              onChange={e => setSettings(s => ({ ...s, durationSec: parseInt(e.target.value || '6', 10) }))}
              disabled={running}
            />
          </label>
          <label>
            Target
            <select
              value={settings.targetBus}
              onChange={e => setSettings(s => ({ ...s, targetBus: e.target.value }))}
              disabled={running}
            >
              {TARGET_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            Target ID
            <input
              type="number"
              min="1"
              max="16"
              value={settings.targetId}
              onChange={e => setSettings(s => ({ ...s, targetId: parseInt(e.target.value || '1', 10) }))}
              disabled={running}
            />
          </label>
          <label>
            Mode
            <select
              value={settings.correctionMode}
              onChange={e => setSettings(s => ({ ...s, correctionMode: e.target.value }))}
              disabled={running}
            >
              {CORRECTION_MODES.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            Curve
            <select
              value={settings.referenceCurve}
              onChange={e => setSettings(s => ({ ...s, referenceCurve: e.target.value }))}
              disabled={running || settings.correctionMode !== 'pink_noise_reference'}
            >
              {REFERENCE_CURVES.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            Device channels
            <input type="text" value={availableChannels?.length || 0} disabled />
          </label>
        </div>

        {result && (
          <>
            <div className="system-measurement-summary">
              <div className="summary-chip">Quality: {(Number(result.quality || 0) * 100).toFixed(0)}%</div>
              <div className="summary-chip">Corrections: {result.num_corrections || 0}</div>
              <div className="summary-chip">Target: {result.target_bus || settings.targetBus} {result.target_id || settings.targetId}</div>
              <div className="summary-chip">Mode: {result.correction_mode || settings.correctionMode}</div>
            </div>

            <table className="data-table">
              <thead>
                <tr>
                  <th>Freq</th>
                  <th>Gain</th>
                  <th>Q</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {topCorrections.map((corr, idx) => (
                  <tr key={`${corr.frequency}-${idx}`}>
                    <td>{Math.round(corr.frequency)} Hz</td>
                    <td>{Number(corr.gain_db).toFixed(1)} dB</td>
                    <td>{Number(corr.q).toFixed(2)}</td>
                    <td>{corr.type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}

export default SystemMeasurementTab;
