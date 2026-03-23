import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './GainStagingTab.css';
import SignalHint from './SignalHint';
import ChannelPresetSelect from './ChannelPresetSelect';
import { mapPresetForMethod } from '../constants/instrumentPresets';

const DEFAULT_SETTINGS = {
  targetLufs: -23, truePeakLimit: -1, ratio: 4,
  emaFastAlpha: 0.4, emaSlowAlpha: 0.08, switchThresholdDb: 3.0,
  maxRateDbPerFrame: 2.0, hysteresisDb: 0.5,
};

function GainStagingTab({
  selectedChannels,
  availableChannels,
  selectedDevice,
  audioDevices,
  globalMode,
  channelPresets = {},
  setChannelPreset = () => {},
  detectInstrumentPreset = () => 'custom',
}) {
  const [running, setRunning] = useState(false);
  const [measuredLevels, setMeasuredLevels] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisCompleted, setAnalysisCompleted] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const handleStatus = (data) => {
      if (data.realtime_enabled !== undefined) setRunning(data.realtime_enabled);
      if (data.message) setStatusMessage(data.message);
      if (data.error) { setStatusMessage(`Ошибка: ${data.error}`); setRunning(false); }
      if (data.status_type === 'realtime_correction_started') { setRunning(true); setStatusMessage('AGC запущен'); }
      if (data.status_type === 'realtime_correction_stopped') {
        setRunning(false);
        setAnalysisCompleted(false);
        setStatusMessage('AGC остановлен');
        setAnalysisProgress(0);
      }
      if (data.status_type === 'safe_gain_started') {
        setRunning(true);
        setAnalysisCompleted(false);
        setAnalysisProgress(0);
      }
      if (data.status_type === 'safe_gain_progress' && data.progress != null) {
        setAnalysisProgress(Math.round(data.progress * 100));
      }
      if (data.status_type === 'safe_gain_ready' || data.status_type === 'realtime_correction_stopped') {
        setAnalysisProgress(100);
      }
      if (data.status_type === 'safe_gain_completed') {
        setRunning(false);
        setAnalysisCompleted(true);
        setAnalysisProgress(100);
      }
      const buildLevels = (channels, defaults = {}) => {
        const newLevels = {};
        Object.entries(channels).forEach(([ch, d]) => {
          newLevels[parseInt(ch)] = {
            peak: d.measured_peak ?? d.peak ?? d.peak_db ?? d.true_peak ?? -60,
            lufs: d.lufs ?? -60,
            truePeak: d.true_peak ?? d.peak ?? d.peak_db ?? d.measured_peak ?? -60,
            gain: d.gain ?? d.suggested_gain_db ?? 0,
            status: d.status_message ?? d.status ?? defaults.status ?? 'idle',
            signalPresent: d.signal_present ?? (d.signal_presence > 0) ?? false,
            readyForFinalize: d.ready_for_finalize ?? false,
            limitedBy: d.limited_by ?? defaults.limitedBy ?? 'none',
          };
        });
        return newLevels;
      };
      if (data.status_type === 'levels_update' && data.channels) {
        setMeasuredLevels(buildLevels(data.channels));
      }
      if (data.status_type === 'safe_gain_progress' && data.channels) {
        setMeasuredLevels(buildLevels(data.channels, { status: 'learning' }));
      }
      if (data.status_type === 'safe_gain_ready' && data.suggestions) {
        const fromSuggestions = {};
        Object.entries(data.suggestions).forEach(([ch, d]) => {
          fromSuggestions[ch] = {
            peak_db: d.peak_db,
            lufs: d.lufs,
            suggested_gain_db: d.suggested_gain_db,
            status: 'ready',
            signal_presence: d.signal_presence ?? 0,
            limited_by: d.limited_by,
          };
        });
        setMeasuredLevels(buildLevels(fromSuggestions, { status: 'ready' }));
      }
      if (data.status_type === 'safe_gain_applied' && data.message) {
        setStatusMessage(data.message);
      }
    };
    const handleNames = (data) => {
      if (!data.channel_names) return;
      selectedChannels.forEach((id) => {
        const num = typeof id === 'number' ? id : parseInt(id);
        const name = data.channel_names[num] || data.channel_names[String(num)];
        if (!name || !String(name).trim()) return;
        const key = String(id);
        const current = channelPresets[key];
        if (current && current !== 'custom') return;
        setChannelPreset(id, detectInstrumentPreset(name));
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
  }, [selectedChannels, setChannelPreset, detectInstrumentPreset, channelPresets]);

  const handleToggle = () => {
    if (running) {
      websocketService.stopRealtimeCorrection();
    } else {
      if (!selectedDevice || selectedChannels.length === 0) {
        setStatusMessage('Выберите устройство и каналы');
        return;
      }
      const mapping = {};
      const channelSettings = {};
      selectedChannels.forEach(ch => { mapping[ch] = ch; });
      selectedChannels.forEach(ch => {
        const preset = channelPresets[String(ch)]
          || detectInstrumentPreset(availableChannels.find(c => c.id === ch)?.name);
        channelSettings[ch] = { preset: mapPresetForMethod(preset, 'gain') };
      });
      websocketService.startRealtimeCorrection(selectedDevice, selectedChannels, channelSettings, mapping, {
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
  const getStatusBadge = (statusText, ready) => {
    const text = statusText || 'idle';
    const baseBadge = (() => {
      if (ready || text === 'Готов к применению') {
        return <span className="status-badge ready">✅ Готов к применению</span>;
      }
      if (text === 'Подайте основной сигнал') {
        return <span className="status-badge waiting">🟡 Подайте основной сигнал</span>;
      }
      return <span className="status-badge neutral">{text}</span>;
    })();
    return baseBadge;
  };

  const getStereoBadge = (limitedBy) => {
    if (limitedBy === 'stereo_pair_balance') {
      return <span className="status-badge stereo">🎧 Stereo pair (-6 dB)</span>;
    }
    return null;
  };

  const showResultColumns = running || analysisCompleted;

  const getStatusCell = (statusText, ready, limitedBy) => {
    const stereoBadge = getStereoBadge(limitedBy);
    if (!stereoBadge) {
      return getStatusBadge(statusText, ready);
    }
    return (
      <div className="status-stack">
        {getStatusBadge(statusText, ready)}
        {stereoBadge}
      </div>
    );
  };

  if (selectedChannels.length === 0) {
    return (<div><SignalHint moduleKey="gain_staging" /><div className="no-channels">Выберите каналы на вкладке Connect</div></div>);
  }

  return (
    <div className="gain-staging-tab">
      <SignalHint moduleKey="gain_staging" />
      <div className="module-card">
        <div className="module-actions">
          <button
            className={`btn-start ${running ? 'stop' : 'go'} ${analysisCompleted ? 'completed' : ''}`}
            onClick={handleToggle}
            disabled={!selectedDevice || selectedChannels.length === 0}>
            {running ? 'Stop' : (analysisCompleted ? 'Gain ✓' : 'Gain')}
          </button>
          <button className="btn-sm" onClick={handleResetTrim} disabled={running || selectedChannels.length === 0}>
            Reset TRIM
          </button>
          <div className={`run-indicator ${running ? 'on' : ''}`}>{running ? 'AGC ●' : ''}</div>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}
        {running && analysisProgress > 0 && analysisProgress < 100 && (
          <div className="analysis-progress">
            <div className="progress-bar" style={{ width: `${analysisProgress}%` }} />
          </div>
        )}

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
                <span className="val">Авто-стоп по готовности каналов</span>
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
              {showResultColumns && <><th>Gain</th><th>Статус</th></>}
            </tr>
          </thead>
          <tbody>
            {selectedChannels.map(id => {
              const preset = channelPresets[String(id)]
                || detectInstrumentPreset(getChannelName(id));
              const l = measuredLevels[id] || {};
              return (
                <tr key={id} className={`${l.signalPresent ? 'has-signal' : ''} ${l.readyForFinalize ? 'is-ready' : ''}`}>
                  <td>{getChannelName(id)}</td>
                  <td>
                    <ChannelPresetSelect
                      value={preset || 'custom'}
                      onChange={(presetId) => setChannelPreset(id, presetId)}
                    />
                  </td>
                  <td className={l.peak > -6 ? 'hot' : ''}>{fmtDb(l.peak)}</td>
                  {showResultColumns && (
                    <>
                      <td className={l.gain > 0 ? 'positive' : l.gain < 0 ? 'negative' : ''}>{fmtDb(l.gain)}</td>
                      <td>{getStatusCell(l.status, l.readyForFinalize, l.limitedBy)}</td>
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
