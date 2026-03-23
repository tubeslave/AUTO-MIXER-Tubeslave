import React, { useState, useEffect, useRef, useCallback } from 'react';
import websocketService from '../services/websocket';
import './AutoEQTab.css';
import SignalHint from './SignalHint';
import { INSTRUMENT_PRESETS, mapPresetForMethod } from '../constants/instrumentPresets';

function AutoEQTab({
  selectedChannels,
  availableChannels,
  selectedDevice,
  audioDevices,
  globalMode,
  channelPresets = {},
  setChannelPreset = () => {},
  detectInstrumentPreset = () => 'custom',
}) {
  const [isActive, setIsActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelStatus, setChannelStatus] = useState({});
  const [channelCorrections, setChannelCorrections] = useState({});
  const [channelCorrectionsReady, setChannelCorrectionsReady] = useState({});
  const eqMode = globalMode || 'soundcheck';
  const effectiveDevice = selectedDevice || (audioDevices?.length > 0 ? audioDevices[0].id : null);

  useEffect(() => {
    if (selectedChannels && selectedChannels.length > 0 && availableChannels) {
      selectedChannels.forEach(id => {
        const key = String(id);
        if (!channelPresets[key]) {
          const ch = availableChannels.find(c => c.id === id);
          setChannelPreset(id, detectInstrumentPreset(ch?.name));
        }
      });
    }
  }, [selectedChannels, availableChannels, channelPresets, setChannelPreset, detectInstrumentPreset]);

  useEffect(() => {
    const onStatus = (data) => {
      if (data.active !== undefined) {
        setIsActive(data.active);
        if (!data.active && data.message?.includes('Soundcheck complete')) {
          setChannelStatus(prev => {
            const s = { ...prev };
            Object.keys(s).forEach(ch => { if (s[ch] !== 'Applied') s[ch] = 'Applied'; });
            return s;
          });
        }
      }
      if (data.message) setStatusMessage(data.message);
    };
    const onSpectrum = (data) => {
      const ch = data.channel;
      setChannelStatus(prev => {
        if (channelCorrectionsReady[ch] || prev[ch] === 'Ready' || prev[ch] === 'Applied') return prev;
        return { ...prev, [ch]: 'Analyzing' };
      });
    };
    const onCorrections = (data) => {
      const ch = data.channel;
      setChannelCorrections(p => ({ ...p, [ch]: data.corrections || [] }));
      setChannelStatus(p => ({ ...p, [ch]: eqMode === 'soundcheck' ? 'Ready' : 'Ready' }));
      if (data.corrections_ready) setChannelCorrectionsReady(p => ({ ...p, [ch]: true }));
    };
    const onApply = (data) => {
      if (data.success && data.channel) setChannelStatus(p => ({ ...p, [data.channel]: 'Applied' }));
      else if (data.success) setStatusMessage(data.message || 'Applied');
      else setStatusMessage(`Ошибка: ${data.error}`);
    };
    const onResetAll = (data) => {
      setStatusMessage(data.success ? (data.message || 'EQ сброшен') : `Ошибка: ${data.error}`);
    };

    websocketService.on('multi_channel_status', onStatus);
    websocketService.on('multi_channel_auto_eq_status', onStatus);
    websocketService.on('multi_channel_spectrum', onSpectrum);
    websocketService.on('multi_channel_corrections', onCorrections);
    websocketService.on('multi_channel_apply_result', onApply);
    websocketService.on('reset_all_eq_result', onResetAll);
    websocketService.on('auto_eq_status', (d) => { if (d.active !== undefined) setIsActive(d.active); if (d.message) setStatusMessage(d.message); });

    return () => {
      websocketService.off('multi_channel_status', onStatus);
      websocketService.off('multi_channel_auto_eq_status', onStatus);
      websocketService.off('multi_channel_spectrum', onSpectrum);
      websocketService.off('multi_channel_corrections', onCorrections);
      websocketService.off('multi_channel_apply_result', onApply);
      websocketService.off('reset_all_eq_result', onResetAll);
    };
  }, [eqMode, channelCorrectionsReady]);

  const handleToggle = () => {
    if (isActive) {
      websocketService.stopMultiChannelAutoEQ();
      setIsActive(false);
      setChannelStatus({});
      setChannelCorrectionsReady({});
    } else {
      if (!selectedChannels?.length) {
        setStatusMessage('Выберите каналы');
        return;
      }
      const cfg = selectedChannels.map(id => {
        const ch = availableChannels?.find(c => c.id === id);
        const instrumentPreset = channelPresets[String(id)] || detectInstrumentPreset(ch?.name);
        return {
          channel: id,
          profile: mapPresetForMethod(instrumentPreset, 'eq'),
          auto_apply: eqMode === 'live',
        };
      });
      let deviceId = effectiveDevice;
      if (effectiveDevice != null && effectiveDevice !== '') {
        if (typeof effectiveDevice === 'string') {
          const dev = audioDevices?.find(d => d.id === effectiveDevice || String(d.index) === String(effectiveDevice));
          if (dev?.index !== undefined) deviceId = dev.index;
          else { const p = parseInt(effectiveDevice); if (!isNaN(p)) deviceId = p; }
        }
      } else {
        deviceId = null;
      }
      websocketService.startMultiChannelAutoEQ(deviceId, cfg, eqMode);
      setIsActive(true);
      setChannelCorrectionsReady({});
      setChannelStatus({});
      setStatusMessage('Анализ...');
    }
  };

  const handleResetAll = () => {
    if (!selectedChannels?.length) return;
    websocketService.send({ type: 'reset_all_eq', channels: selectedChannels });
    setStatusMessage('Сброс EQ...');
  };

  const handleApplyAll = () => {
    websocketService.applyAllCorrections();
    setStatusMessage('Применяю...');
  };

  if (!selectedChannels?.length) return (<div><SignalHint moduleKey="auto_eq" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);

  return (
    <div className="auto-eq-tab">
      <SignalHint moduleKey="auto_eq" />
      <div className="module-card">
        <div className="module-actions">
          <button type="button" className={`btn-start ${isActive ? 'stop' : 'go'}`} onClick={handleToggle}
            disabled={!selectedChannels?.length}
            title={isActive ? 'Измерение — нажмите для остановки' : 'Готов к анализу'}>
            {isActive ? 'Стоп' : (eqMode === 'soundcheck' ? 'EQ' : 'Live')}
          </button>
          {eqMode === 'soundcheck' && (
            <button className="btn-sm" onClick={handleApplyAll}
              disabled={Object.keys(channelCorrections).length === 0}>
              Применить все
            </button>
          )}
          <button className="btn-sm" onClick={handleResetAll}
            disabled={!selectedChannels?.length}>
            Сброс EQ
          </button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <table className="data-table">
          <thead><tr><th>Канал</th><th>Профиль</th><th>Статус</th></tr></thead>
          <tbody>
            {selectedChannels.map(id => {
              const ch = availableChannels?.find(c => c.id === id);
              const name = ch?.name || `Ch ${id}`;
              const instrumentPreset = channelPresets[String(id)] || detectInstrumentPreset(ch?.name);
              return (
                <tr key={id}>
                  <td>{name}</td>
                  <td>
                    <select value={instrumentPreset}
                      onChange={e => setChannelPreset(id, e.target.value)}
                      disabled={isActive}>
                      {INSTRUMENT_PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                    </select>
                  </td>
                  <td className={`eq-status ${(channelStatus[id] || 'ready').toLowerCase()}`}>
                    {channelStatus[id] || 'Ready'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AutoEQTab;
