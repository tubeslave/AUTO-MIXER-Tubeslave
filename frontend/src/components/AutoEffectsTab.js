import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoEffectsTab.css';
import SignalHint from './SignalHint';

function AutoEffectsTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [running, setRunning] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelEffects, setChannelEffects] = useState({});

  useEffect(() => {
    const handle = (data) => {
      if (data.running !== undefined) setRunning(data.running);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.channel_effects) setChannelEffects(data.channel_effects);
    };
    websocketService.on('auto_effects_status', handle);
    websocketService.getAutoEffectsStatus();
    return () => websocketService.off('auto_effects_status', handle);
  }, []);

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels?.length) { setStatusMessage('Выберите устройство и каналы'); return; }
    websocketService.startAutoEffects(selectedDevice, selectedChannels, {});
  };
  const handleStop = () => websocketService.stopAutoEffects();

  const channels = selectedChannels || [];
  if (!channels.length) return (<div><SignalHint moduleKey="auto_effects" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-effects-tab">
      <SignalHint moduleKey="auto_effects" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${running ? 'stop' : 'go'}`}
            onClick={running ? handleStop : handleStart}
            disabled={!selectedDevice || !channels.length}>
            {running ? 'Стоп' : 'Старт'}
          </button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Эффекты</th></tr></thead>
          <tbody>
            {channels.map(ch => {
              const name = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const fx = channelEffects[ch];
              return (
                <tr key={ch}>
                  <td>{name}</td>
                  <td>{fx ? (fx.active_effects || []).join(', ') || 'none' : '--'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoEffectsTab;
