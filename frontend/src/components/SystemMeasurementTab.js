import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './SystemMeasurementTab.css';
import SignalHint from './SignalHint';

function SystemMeasurementTab({ selectedDevice, mixerClient, globalMode }) {
  const [statusMessage, setStatusMessage] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [numPositions, setNumPositions] = useState(3);
  const [currentPosition, setCurrentPosition] = useState(0);
  const [results, setResults] = useState(null);

  useEffect(() => {
    const handle = (data) => {
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.playing !== undefined) setIsPlaying(data.playing);
      if (data.position !== undefined) setCurrentPosition(data.position);
      if (data.results) setResults(data.results);
    };
    websocketService.on('system_measurement_status', handle);
    return () => websocketService.off('system_measurement_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice) { setStatusMessage('Выберите аудио устройство'); return; }
    websocketService.send({
      type: 'start_system_measurement',
      device_id: selectedDevice,
      num_positions: numPositions,
    });
  };

  const handleRecord = () => {
    websocketService.send({ type: 'record_measurement_position' });
  };

  const handleAnalyze = () => {
    websocketService.send({ type: 'analyze_measurements' });
  };

  return (
    <div className="system-measurement-tab">
      <SignalHint moduleKey="system_measurement" />
      <div className="module-card">
        <div className="module-actions">
          <button className="btn-start go" onClick={handleStart} disabled={!selectedDevice || isPlaying}>
            Запустить измерение
          </button>
          <div className="setting-row">
            <label>Точек</label>
            <input type="number" min="1" max="10" value={numPositions}
              onChange={e => setNumPositions(parseInt(e.target.value))} disabled={isPlaying}
              style={{width: '50px', padding: '4px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '4px', color: '#e6e6e6'}} />
          </div>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        {isPlaying && (
          <div className="measurement-controls">
            <button className="btn-sm" onClick={handleRecord} disabled={currentPosition >= numPositions}>
              Записать точку {currentPosition + 1}/{numPositions}
            </button>
            <button className="btn-sm" onClick={handleAnalyze}>
              Анализ
            </button>
          </div>
        )}

        {results && (
          <div className="measurement-results">
            <table className="data-table">
              <thead><tr><th>Параметр</th><th>Значение</th></tr></thead>
              <tbody>
                {Object.entries(results).map(([key, val]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{typeof val === 'number' ? val.toFixed(2) : String(val)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default SystemMeasurementTab;
