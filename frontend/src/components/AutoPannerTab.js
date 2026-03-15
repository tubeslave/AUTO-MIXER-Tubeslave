import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoPannerTab.css';
import SignalHint from './SignalHint';

function AutoPannerTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [panPositions, setPanPositions] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [genre, setGenre] = useState('rock');

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined) setRunning(data.running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.pan_positions) setPanPositions(data.pan_positions);
    };
    websocketService.on('auto_panner_status', handle);
    websocketService.getAutoPannerStatus();
    return () => websocketService.off('auto_panner_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    const types = {}; const centroids = {};
    selectedChannels.forEach(ch => {
      types[ch] = 'custom'; centroids[ch] = 1000;
    });
    websocketService.startAutoPanner(selectedDevice, selectedChannels, types, centroids, genre);
  };
  const handleStop = () => websocketService.stopAutoPanner();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_panner" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-panner-tab">
      <SignalHint moduleKey="auto_panner" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={running ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {running ? 'Стоп' : 'Старт'}
          </button>
          <select value={genre} onChange={e => setGenre(e.target.value)} disabled={running}
            className="genre-select">
            <option value="rock">Rock</option>
            <option value="jazz">Jazz</option>
            <option value="pop">Pop</option>
            <option value="classical">Classical</option>
            <option value="electronic">Electronic</option>
          </select>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Pan</th></tr></thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const pan = panPositions[ch];
              const panStr = pan != null ? (pan < 0 ? `L${Math.abs(pan)}` : pan > 0 ? `R${pan}` : 'C') : '--';
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td>{panStr}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoPannerTab;
