import React, { useState, useEffect, useRef, useCallback } from 'react';
import websocketService from '../services/websocket';
import './AutoEQTab.css';
import SignalHint from './SignalHint';

const EQ_PROFILES = [
  { id: 'kick', name: 'Kick', cat: 'Drums' }, { id: 'kick_rock', name: 'Kick (Rock)', cat: 'Drums' },
  { id: 'kick_metal', name: 'Kick (Metal)', cat: 'Drums' }, { id: 'kick_jazz', name: 'Kick (Jazz)', cat: 'Drums' },
  { id: 'kick_pop', name: 'Kick (Pop)', cat: 'Drums' },
  { id: 'snare', name: 'Snare', cat: 'Drums' }, { id: 'snare_rock', name: 'Snare (Rock)', cat: 'Drums' },
  { id: 'snare_metal', name: 'Snare (Metal)', cat: 'Drums' }, { id: 'snare_jazz', name: 'Snare (Jazz)', cat: 'Drums' },
  { id: 'tom', name: 'Tom', cat: 'Drums' }, { id: 'ftom', name: 'Floor Tom', cat: 'Drums' },
  { id: 'hihat', name: 'Hi-Hat', cat: 'Drums' }, { id: 'ride', name: 'Ride', cat: 'Drums' },
  { id: 'cymbals', name: 'Cymbals', cat: 'Drums' }, { id: 'overheads', name: 'OH', cat: 'Drums' },
  { id: 'leadvocal', name: 'Lead Vocal', cat: 'Vocals' },
  { id: 'vocal_rock', name: 'Vocal Rock', cat: 'Vocals' }, { id: 'vocal_pop', name: 'Vocal Pop', cat: 'Vocals' },
  { id: 'vocal_jazz', name: 'Vocal Jazz', cat: 'Vocals' }, { id: 'vocal_metal', name: 'Vocal Metal', cat: 'Vocals' },
  { id: 'vocal_rnb', name: 'Vocal R&B', cat: 'Vocals' }, { id: 'vocal_classical', name: 'Vocal Classical', cat: 'Vocals' },
  { id: 'backvocal', name: 'Back Vocal', cat: 'Vocals' },
  { id: 'guitar', name: 'Electric Guitar', cat: 'Guitar' }, { id: 'guitar_rock', name: 'Guitar Rock', cat: 'Guitar' },
  { id: 'guitar_metal', name: 'Guitar Metal', cat: 'Guitar' }, { id: 'guitar_jazz', name: 'Guitar Jazz', cat: 'Guitar' },
  { id: 'guitar_clean', name: 'Guitar Clean', cat: 'Guitar' }, { id: 'guitar_lead', name: 'Guitar Lead', cat: 'Guitar' },
  { id: 'acousticguitar', name: 'Acoustic Guitar', cat: 'Guitar' },
  { id: 'bass', name: 'Bass', cat: 'Bass' }, { id: 'bass_rock', name: 'Bass Rock', cat: 'Bass' },
  { id: 'bass_metal', name: 'Bass Metal', cat: 'Bass' }, { id: 'bass_jazz', name: 'Bass Jazz', cat: 'Bass' },
  { id: 'bass_funk', name: 'Bass Funk', cat: 'Bass' },
  { id: 'keys', name: 'Keys', cat: 'Keys' }, { id: 'piano', name: 'Piano', cat: 'Keys' },
  { id: 'synth', name: 'Synth', cat: 'Keys' },
  { id: 'violin', name: 'Violin', cat: 'Orch' }, { id: 'cello', name: 'Cello', cat: 'Orch' },
  { id: 'trumpet', name: 'Trumpet', cat: 'Orch' }, { id: 'saxophone', name: 'Sax', cat: 'Orch' },
  { id: 'flute', name: 'Flute', cat: 'Orch' },
  { id: 'accordion', name: 'Accordion', cat: 'Other' }, { id: 'playback', name: 'Playback', cat: 'Other' },
  { id: 'custom', name: 'Custom', cat: 'Other' }
];

function detectProfile(name) {
  if (!name) return 'custom';
  const n = name.toUpperCase();
  if (n.includes('HI HAT') || n.includes('HI-HAT') || n.includes('HIHAT')) return 'hihat';
  if (n === 'PB' || n.includes('PLAYBACK')) return 'playback';
  if (n.includes('ACCORD')) return 'accordion';
  if (n.includes('KICK') || n.includes('BD') || n.includes('BASS DRUM')) return 'kick';
  if (n.includes('SNARE') || n.includes('SD')) return 'snare';
  if (n.includes('FLOOR') || n.includes('FTOM')) return 'ftom';
  if (n.includes('TOM')) return 'tom';
  if (n.includes('RIDE')) return 'ride';
  if (n.includes('OVERHEAD') || n.includes('OH')) return 'overheads';
  if (n.includes('CYMBAL') || n.includes('CRASH')) return 'cymbals';
  if (n.includes('VOCAL') || n.includes('VOC')) {
    return n.includes('BACK') || n.includes('BG') ? 'backvocal' : 'leadvocal';
  }
  if (n.includes('GUITAR') || n.includes('GTR')) {
    return n.includes('ACOUSTIC') || n.includes('AC') ? 'acousticguitar' : 'guitar';
  }
  if (n.includes('BASS') || n.includes('BS')) return 'bass';
  if (n.includes('PIANO')) return 'piano';
  if (n.includes('KEY')) return 'keys';
  if (n.includes('SYNTH')) return 'synth';
  return 'custom';
}

function AutoEQTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [isActive, setIsActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelProfiles, setChannelProfiles] = useState({});
  const [channelStatus, setChannelStatus] = useState({});
  const [channelCorrections, setChannelCorrections] = useState({});
  const [channelCorrectionsReady, setChannelCorrectionsReady] = useState({});
  const eqMode = globalMode || 'soundcheck';

  useEffect(() => {
    if (selectedChannels && selectedChannels.length > 0 && availableChannels) {
      const newP = {};
      selectedChannels.forEach(id => {
        if (!channelProfiles[id]) {
          const ch = availableChannels.find(c => c.id === id);
          newP[id] = detectProfile(ch?.name);
        }
      });
      if (Object.keys(newP).length > 0) setChannelProfiles(prev => ({ ...prev, ...newP }));
    }
  }, [selectedChannels, availableChannels, channelProfiles]);

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
      if (data.channel && data.profile) setChannelProfiles(p => ({ ...p, [data.channel]: data.profile }));
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
      setChannelStatus(p => ({ ...p, [ch]: eqMode === 'soundcheck' ? 'Applied' : 'Ready' }));
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
      if (!selectedDevice || !selectedChannels?.length) {
        setStatusMessage('Выберите устройство и каналы');
        return;
      }
      const cfg = selectedChannels.map(id => {
        const ch = availableChannels?.find(c => c.id === id);
        const profile = channelProfiles[id] || detectProfile(ch?.name);
        return { channel: id, profile, auto_apply: eqMode === 'live' };
      });
      let deviceId = selectedDevice;
      if (typeof selectedDevice === 'string') {
        const dev = audioDevices?.find(d => d.id === selectedDevice || String(d.index) === String(selectedDevice));
        if (dev?.index !== undefined) deviceId = dev.index;
        else { const p = parseInt(selectedDevice); if (!isNaN(p)) deviceId = p; }
      }
      websocketService.startMultiChannelAutoEQ(deviceId, cfg, eqMode);
      setIsActive(true);
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
          <button className={`btn-start ${isActive ? 'stop' : 'go'}`} onClick={handleToggle}
            disabled={!selectedDevice || !selectedChannels?.length}>
            {isActive ? 'Стоп' : (eqMode === 'soundcheck' ? 'Soundcheck' : 'Live')}
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
              const profile = channelProfiles[id] || detectProfile(ch?.name);
              return (
                <tr key={id}>
                  <td>{name}</td>
                  <td>
                    <select value={profile}
                      onChange={e => setChannelProfiles(p => ({...p, [id]: e.target.value}))}
                      disabled={isActive}>
                      {EQ_PROFILES.map(p => <option key={p.id} value={p.id}>{p.cat}: {p.name}</option>)}
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
