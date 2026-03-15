import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './GainStagingTab.css';
import SignalHint from './SignalHint';

const SIGNAL_PRESETS = {
  kick: { name: 'Kick' }, snare: { name: 'Snare' }, tom: { name: 'Tom' },
  hihat: { name: 'Hi-Hat' }, ride: { name: 'Ride' }, cymbals: { name: 'Cymbals' },
  overheads: { name: 'Overheads' }, room: { name: 'Room' }, bass: { name: 'Bass' },
  electricGuitar: { name: 'Electric Guitar' }, acousticGuitar: { name: 'Acoustic Guitar' },
  accordion: { name: 'Accordion' }, synth: { name: 'Synth / Keys' },
  playback: { name: 'Playback' }, leadVocal: { name: 'Lead Vocal' },
  backVocal: { name: 'Back Vocal' }, custom: { name: 'Custom' }
};

const DEFAULT_SETTINGS = {
  targetLufs: -23, truePeakLimit: -1, ratio: 4, learningDurationSec: 30,
  emaFastAlpha: 0.4, emaSlowAlpha: 0.08, switchThresholdDb: 3.0,
  maxRateDbPerFrame: 2.0, hysteresisDb: 0.5,
};

const recognizeSignalType = (channelName) => {
  if (!channelName || !channelName.trim()) return 'custom';
  const n = channelName.toLowerCase().trim();
  if (/\b(room|рум)\b/i.test(n)) return 'room';
  if (/\b(ohl|ohr|over[\s-]?head|overhead)\b/i.test(n)) return 'overheads';
  if (/\b(snare|sd|sn|малый|снэйр)\b/i.test(n)) return 'snare';
  if (/\b(kick|bd|bass\s*drum|бочка|кик)\b/i.test(n)) return 'kick';
  if (/\b(tom|том|floor\s*tom|флор)\b/i.test(n)) return 'tom';
  if (/\b(hi[\s-]?hat|hh|хай[\s-]?хэт)\b/i.test(n)) return 'hihat';
  if (/\b(ride|райд)\b/i.test(n)) return 'ride';
  if (/\b(crash|splash|china|cymbal)\b/i.test(n)) return 'cymbals';
  if (/\b(bass|бас|sub)(?![\s-]?(drum|бочка))/i.test(n)) return 'bass';
  if (/\b(acoustic|акустик|agtr)\b/i.test(n)) return 'acousticGuitar';
  if (/\b(electric|электро|egtr|gtr|гитар|guitar)\b/i.test(n)) return 'electricGuitar';
  if (/\b(accordion|accord|bayan|баян|аккордеон)\b/i.test(n)) return 'accordion';
  if (/\b(synth|keys|keyboard|piano|клавиш|синт)\b/i.test(n)) return 'synth';
  if (/\b(playback|pb|track|backing|минус)\b/i.test(n)) return 'playback';
  if (/\b(lead\s*vox|lead\s*vocal|лид[\s-]?вок|main\s*vox|vox\s*1)\b/i.test(n)) return 'leadVocal';
  if (/\b(back[\s-]?vox|bvox|бэк[\s-]?вок|choir|хор|bgv)\b/i.test(n)) return 'backVocal';
  if (/\b(vox|vocal|вокал|голос|voice|mic)\b/i.test(n)) return 'leadVocal';
  return 'custom';
};

function GainStagingTab({ selectedChannels, availableChannels, selectedDevice, audioDevices, globalMode }) {
  const [channelSettings, setChannelSettings] = useState({});
  const [running, setRunning] = useState(false);
  const [measuredLevels, setMeasuredLevels] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const handleStatus = (data) => {
      if (data.realtime_enabled !== undefined) setRunning(data.realtime_enabled);
      if (data.message) setStatusMessage(data.message);
      if (data.error) { setStatusMessage(`Ошибка: ${data.error}`); setRunning(false); }
      if (data.status_type === 'realtime_correction_started') { setRunning(true); setStatusMessage('AGC запущен'); }
      if (data.status_type === 'realtime_correction_stopped') { setRunning(false); setStatusMessage('AGC остановлен'); }
      if (data.status_type === 'levels_update' && data.channels) {
        const newLevels = {};
        Object.entries(data.channels).forEach(([ch, d]) => {
          newLevels[parseInt(ch)] = {
            peak: d.measured_peak ?? d.true_peak ?? -60,
            lufs: d.lufs ?? -60,
            truePeak: d.true_peak ?? d.measured_peak ?? -60,
            gain: d.gain ?? 0,
            status: d.status ?? 'idle',
            signalPresent: d.signal_present || false,
          };
        });
        setMeasuredLevels(newLevels);
      }
    };
    const handleNames = (data) => {
      if (!data.channel_names) return;
      setChannelSettings(prev => {
        const updated = { ...prev };
        selectedChannels.forEach(id => {
          const num = typeof id === 'number' ? id : parseInt(id);
          const name = data.channel_names[num] || data.channel_names[String(num)];
          if (name && name.trim()) {
            updated[id] = { ...updated[id], preset: recognizeSignalType(name), scannedName: name.trim() };
          }
        });
        return updated;
      });
    };
    const handleSettingsLoaded = (data) => {
      if (data.settings?.gainStaging) setSettings(s => ({ ...s, ...data.settings.gainStaging }));
    };

    websocketService.on('gain_staging_status', handleStatus);
    websocketService.on('mixer_channel_names', handleNames);
    websocketService.on('all_settings_loaded', handleSettingsLoaded);
    websocketService.getGainStagingStatus();
    websocketService.loadAllSettings();

    return () => {
      websocketService.off('gain_staging_status', handleStatus);
      websocketService.off('mixer_channel_names', handleNames);
      websocketService.off('all_settings_loaded', handleSettingsLoaded);
    };
  }, [selectedChannels]);

  useEffect(() => {
    const s = {};
    selectedChannels.forEach(id => {
      if (channelSettings[id]) { s[id] = channelSettings[id]; }
      else {
        const ch = availableChannels.find(c => c.id === id);
        s[id] = { preset: ch?.name ? recognizeSignalType(ch.name) : 'custom' };
      }
    });
    setChannelSettings(s);
  }, [selectedChannels, availableChannels]);

  const handleToggle = () => {
    if (running) {
      websocketService.stopRealtimeCorrection();
    } else {
      if (!selectedDevice || selectedChannels.length === 0) {
        setStatusMessage('Выберите устройство и каналы');
        return;
      }
      const mapping = {};
      selectedChannels.forEach(ch => { mapping[ch] = ch; });
      websocketService.startRealtimeCorrection(selectedDevice, selectedChannels, channelSettings, mapping, {
        learning_duration_sec: settings.learningDurationSec,
        processing: {
          target_lufs: settings.targetLufs, true_peak_limit: settings.truePeakLimit,
          ratio: settings.ratio, ema_fast_alpha: settings.emaFastAlpha,
          ema_slow_alpha: settings.emaSlowAlpha, switch_threshold_db: settings.switchThresholdDb,
          max_rate_db_per_frame: settings.maxRateDbPerFrame, hysteresis_db: settings.hysteresisDb,
        }
      });
    }
  };

  const handleResetTrim = () => {
    if (selectedChannels.length === 0) return;
    websocketService.resetTrim(selectedChannels);
    setStatusMessage('TRIM → 0dB');
  };

  const getChannelName = (id) => {
    const ch = availableChannels.find(c => c.id === id);
    return ch?.name || `Ch ${id}`;
  };
  const fmtDb = (v) => { if (v == null || v === -60) return '--'; return `${v >= 0 ? '+' : ''}${v.toFixed(1)}`; };

  if (selectedChannels.length === 0) {
    return (<div><SignalHint moduleKey="gain_staging" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);
  }

  return (
    <div className="gain-staging-tab">
      <SignalHint moduleKey="gain_staging" />
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${running ? 'stop' : 'go'}`} onClick={handleToggle}
            disabled={!selectedDevice || selectedChannels.length === 0}>
            {running ? 'Стоп' : 'Старт'}
          </button>
          <button className="btn-sm" onClick={handleResetTrim} disabled={running || selectedChannels.length === 0}>
            Reset TRIM
          </button>
          <div className={`run-indicator ${running ? 'on' : ''}`}>{running ? 'AGC ●' : ''}</div>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        {/* Compact Settings */}
        <div className="settings-panel">
          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            <span>Настройки</span><span>{showSettings ? '▼' : '▶'}</span>
          </div>
          {showSettings && (
            <div className="settings-body">
              <div className="setting-row">
                <label>Target LUFS</label>
                <input type="range" min="-30" max="-14" step="1" value={settings.targetLufs}
                  onChange={e => setSettings(s => ({...s, targetLufs: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.targetLufs}</span>
              </div>
              <div className="setting-row">
                <label>Ratio</label>
                <select value={settings.ratio} onChange={e => setSettings(s => ({...s, ratio: parseInt(e.target.value)}))} disabled={running}>
                  <option value={2}>2:1</option><option value={4}>4:1</option><option value={8}>8:1</option>
                </select>
              </div>
              <div className="setting-row">
                <label>True Peak</label>
                <input type="range" min="-6" max="0" step="0.5" value={settings.truePeakLimit}
                  onChange={e => setSettings(s => ({...s, truePeakLimit: parseFloat(e.target.value)}))} disabled={running} />
                <span className="val">{settings.truePeakLimit} dBTP</span>
              </div>
              <div className="setting-row">
                <label>Анализ</label>
                <input type="range" min="10" max="60" step="5" value={settings.learningDurationSec}
                  onChange={e => setSettings(s => ({...s, learningDurationSec: parseInt(e.target.value)}))} disabled={running} />
                <span className="val">{settings.learningDurationSec}s</span>
              </div>
            </div>
          )}
        </div>

        {/* Channels table */}
        <table className="data-table">
          <thead>
            <tr>
              <th>Канал</th>
              <th>Тип</th>
              <th>Peak</th>
              {running && <><th>LUFS</th><th>Gain</th><th>Статус</th></>}
            </tr>
          </thead>
          <tbody>
            {selectedChannels.map(id => {
              const s = channelSettings[id] || {};
              const l = measuredLevels[id] || {};
              return (
                <tr key={id} className={l.signalPresent ? 'has-signal' : ''}>
                  <td>{getChannelName(id)}</td>
                  <td>
                    <select value={s.preset || 'custom'}
                      onChange={e => setChannelSettings(p => ({...p, [id]: {...p[id], preset: e.target.value}}))}
                      disabled={running}>
                      {Object.entries(SIGNAL_PRESETS).map(([k,v]) => (
                        <option key={k} value={k}>{v.name}</option>
                      ))}
                    </select>
                  </td>
                  <td className={l.peak > -6 ? 'hot' : ''}>{fmtDb(l.peak)}</td>
                  {running && (
                    <>
                      <td>{l.lufs > -60 ? `${l.lufs?.toFixed(1)}` : '--'}</td>
                      <td className={l.gain > 0 ? 'positive' : l.gain < 0 ? 'negative' : ''}>{fmtDb(l.gain)}</td>
                      <td>{l.status || 'idle'}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default GainStagingTab;
