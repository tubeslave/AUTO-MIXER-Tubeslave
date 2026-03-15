import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoReverbTab.css';
import SignalHint from './SignalHint';

function AutoReverbTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({});

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.channel_data) setChannelData(data.channel_data);
    };
    websocketService.on('auto_reverb_status', handle);
    websocketService.getAutoReverbStatus();
    return () => websocketService.off('auto_reverb_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    const types = {}; const centroids = {}; const fluxes = {};
    selectedChannels.forEach(ch => { types[ch] = 'custom'; centroids[ch] = 1000; fluxes[ch] = 0.5; });
    websocketService.startAutoReverb(selectedDevice, selectedChannels, types, centroids, fluxes);
  };
  const handleStop = () => websocketService.stopAutoReverb();
  const handleApply = () => websocketService.applyAutoReverb();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_reverb" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-reverb-tab">
      <SignalHint moduleKey="auto_reverb" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${active ? 'stop' : 'go'}`}
            onClick={active ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {active ? 'Стоп' : 'Старт'}
          </button>
          <button className="btn-sm" onClick={handleApply}
            disabled={!active || Object.keys(channelData).length === 0}>
            Применить
          </button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Reverb</th></tr></thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const d = channelData[ch];
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td>{d ? `${d.reverb_type || 'calculated'}` : '--'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoReverbTab;
