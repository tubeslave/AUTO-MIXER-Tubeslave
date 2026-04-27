import React, { useEffect, useMemo, useState } from 'react';
import websocketService from '../services/websocket';
import './IPhoneControlSurface.css';

const MIXER_DEFAULTS = {
  wing: { label: 'WING', port: 2223 },
  dlive: { label: 'dLive', port: 51328 },
  mixing_station: { label: 'Mixing Station', port: 8000 },
};

function meterPercent(value) {
  if (value === undefined || value === null) return 0;
  return Math.max(0, Math.min(100, ((Number(value) + 90) / 72) * 100));
}

function formatAction(action) {
  const type = action?.type || action?.action || 'correction';
  const channel = action?.channel ? `Ch ${action.channel}` : 'Mix';
  const value = action?.parameters?.gain_db ?? action?.parameters?.freq ?? action?.value ?? '';
  return { title: `${channel}: ${type}`, value };
}

function IPhoneControlSurface({
  serverConnected,
  mixerConnected,
  statusMessage,
  mixerIp,
  mixerPort,
  onMixerIpChange,
  onMixerPortChange,
  connecting,
  onConnect,
  audioDevices,
  selectedDevice,
  onDeviceChange,
  availableChannels,
  selectedChannels,
  onChannelToggle,
  onSelectAllChannels,
  onScanMixer,
  onBypass,
  globalMode,
  onGlobalModeChange,
}) {
  const [view, setView] = useState('live');
  const [serverUrl, setServerUrl] = useState(websocketService.getServerUrl());
  const [mixerType, setMixerType] = useState('wing');
  const [mixingStationRestPort, setMixingStationRestPort] = useState(8080);
  const [dliveTls, setDliveTls] = useState(false);
  const [agentStatus, setAgentStatus] = useState({ is_running: false });
  const [pendingActions, setPendingActions] = useState([]);
  const [history, setHistory] = useState([]);
  const [meters, setMeters] = useState({});
  const [audioStatus, setAudioStatus] = useState({ running: false });
  const [lastBackup, setLastBackup] = useState({ path: null, label: 'нет backup' });
  const [message, setMessage] = useState('');

  useEffect(() => {
    const onAgentStatus = (data) => setAgentStatus(data);
    const onAgentStarted = (data) => {
      setAgentStatus(data);
      setMessage('AI Agent запущен в режиме подтверждения');
      websocketService.getPendingActions();
    };
    const onAgentStopped = () => {
      setAgentStatus(prev => ({ ...prev, is_running: false }));
      setMessage('AI Agent остановлен');
    };
    const onEmergencyStopped = () => {
      setAgentStatus(prev => ({ ...prev, is_running: false, emergency_stop: true }));
      setMessage('Аварийный стоп выполнен');
    };
    const onPending = (data) => setPendingActions(data.actions || []);
    const onHistory = (data) => setHistory(data.history || []);
    const onMeters = (data) => setMeters(data.channels || {});
    const onAudioStatus = (data) => setAudioStatus(data);
    const onSnapshot = (data) => {
      if (data.success) {
        setLastBackup({
          path: data.path,
          label: `${data.success_count || 0} ch · только что`,
        });
        setMessage('Backup выбранных каналов создан');
      } else {
        setMessage(`Backup: ${data.error || 'ошибка'}`);
      }
    };
    const onRestore = (data) => {
      setMessage(data.success ? 'Откат применен' : `Откат: ${data.error || 'ошибка'}`);
    };
    const onBypassResult = (data) => {
      setMessage(data.success ? `Bypass: ${data.success_count}/${data.total_count || 40}` : `Bypass: ${data.error}`);
    };
    const onAutoConnect = (data) => {
      if (data.message) setMessage(data.message);
    };

    websocketService.on('agent_status', onAgentStatus);
    websocketService.on('agent_started', onAgentStarted);
    websocketService.on('agent_stopped', onAgentStopped);
    websocketService.on('agent_emergency_stopped', onEmergencyStopped);
    websocketService.on('pending_actions', onPending);
    websocketService.on('action_history', onHistory);
    websocketService.on('channel_meters', onMeters);
    websocketService.on('audio_capture_status', onAudioStatus);
    websocketService.on('snapshot_result', onSnapshot);
    websocketService.on('restore_result', onRestore);
    websocketService.on('bypass_result', onBypassResult);
    websocketService.on('auto_connect_status', onAutoConnect);

    websocketService.getAgentStatus();
    websocketService.getPendingActions();
    websocketService.getActionHistory(12);
    websocketService.getAudioCaptureStatus();
    websocketService.getChannelMeters();

    const timer = setInterval(() => {
      websocketService.getAgentStatus();
      websocketService.getPendingActions();
      websocketService.getAudioCaptureStatus();
      websocketService.getChannelMeters();
    }, 2500);

    return () => {
      clearInterval(timer);
      websocketService.off('agent_status', onAgentStatus);
      websocketService.off('agent_started', onAgentStarted);
      websocketService.off('agent_stopped', onAgentStopped);
      websocketService.off('agent_emergency_stopped', onEmergencyStopped);
      websocketService.off('pending_actions', onPending);
      websocketService.off('action_history', onHistory);
      websocketService.off('channel_meters', onMeters);
      websocketService.off('audio_capture_status', onAudioStatus);
      websocketService.off('snapshot_result', onSnapshot);
      websocketService.off('restore_result', onRestore);
      websocketService.off('bypass_result', onBypassResult);
      websocketService.off('auto_connect_status', onAutoConnect);
    };
  }, []);

  const selectedChannelSet = useMemo(
    () => new Set((selectedChannels || []).map(id => String(id))),
    [selectedChannels]
  );

  const channelRows = useMemo(() => {
    const source = selectedChannels?.length
      ? selectedChannels.map(id => availableChannels?.find(ch => String(ch.id) === String(id)) || { id, name: `Ch ${id}` })
      : (availableChannels || []).slice(0, 8);
    return source.filter(Boolean);
  }, [availableChannels, selectedChannels]);

  const primaryRows = channelRows.slice(0, 6);

  const mixerPortValue = Number(mixerPort) || MIXER_DEFAULTS[mixerType].port;
  const selectedDeviceName = audioDevices?.find(device => String(device.id) === String(selectedDevice))?.name
    || audioDevices?.find(device => String(device.index) === String(selectedDevice))?.name
    || 'не выбран';
  const pendingCount = pendingActions.length;
  const hasChannels = Boolean(selectedChannels?.length);

  const applyServerUrl = () => {
    websocketService.setServerUrl(serverUrl);
    setMessage(`Backend URL: ${serverUrl}`);
  };

  const connectMixer = () => {
    if (mixerConnected) {
      websocketService.disconnectMixer();
      return;
    }

    if (mixerType === 'wing') {
      onConnect();
      return;
    }

    if (mixerType === 'dlive') {
      websocketService.connectDLive(mixerIp, mixerPortValue, dliveTls);
      setMessage('Подключаю dLive...');
      return;
    }

    websocketService.connectMixingStation(mixerIp || '127.0.0.1', mixerPortValue, Number(mixingStationRestPort) || 8080);
    setMessage('Подключаю Mixing Station...');
  };

  const createBackup = () => {
    if (!hasChannels) {
      setMessage('Выберите каналы для backup');
      return;
    }
    websocketService.createSnapshot(selectedChannels);
    setMessage('Создаю backup выбранных каналов...');
  };

  const restoreBackup = () => {
    if (!window.confirm('Откатить выбранные каналы к последнему backup?')) return;
    websocketService.restoreSnapshot(lastBackup.path);
    setMessage('Выполняю откат...');
  };

  const startAgent = () => {
    if (!hasChannels) {
      setMessage('Выберите каналы для AI Agent');
      return;
    }
    websocketService.startAgent(globalMode === 'live' ? 'auto' : 'suggest', selectedChannels, true, false, false);
    setMessage('Запускаю AI Agent...');
  };

  const stopAgent = () => {
    websocketService.stopAgent();
    setMessage('Останавливаю AI Agent...');
  };

  const emergencyStop = () => {
    websocketService.emergencyStopAgent();
    setMessage('Аварийный стоп...');
  };

  const applyPending = () => {
    if (!pendingCount) {
      setMessage('Нет коррекций для применения');
      return;
    }
    websocketService.approveAllActions();
    setMessage('Применяю подтвержденные коррекции...');
  };

  const dismissPending = () => {
    websocketService.dismissAllActions();
    setMessage('Очередь коррекций очищена');
  };

  const confirmBypass = () => {
    if (!window.confirm('Bypass отключит обработку и сбросит фейдеры WING на 0 dB. Продолжить?')) return;
    onBypass();
  };

  const renderChannelCard = (channel) => {
    const meter = meters[channel.id] || meters[String(channel.id)] || {};
    const active = selectedChannelSet.has(String(channel.id));
    const rms = meter.rms_db;
    return (
      <button
        key={channel.id}
        className={`iphone-channel ${active ? 'selected' : ''}`}
        type="button"
        onClick={() => onChannelToggle(channel.id)}
      >
        <span>
          <strong>{String(channel.name || `Ch ${channel.id}`).slice(0, 22)}</strong>
          <small>{active ? 'обработка включена' : 'monitor only'}</small>
        </span>
        <span className="iphone-meter-stack">
          <i className="iphone-meter"><b style={{ width: `${meterPercent(rms)}%` }} /></i>
          <em>{rms !== undefined ? `${rms} dB` : '--'}</em>
        </span>
      </button>
    );
  };

  return (
    <div className="iphone-remote-shell">
      <header className="iphone-remote-top">
        <div className="iphone-title-row">
          <div>
            <h2>Live Remote</h2>
            <p>{MIXER_DEFAULTS[mixerType].label} · {selectedDeviceName}</p>
          </div>
          <div className={`iphone-status ${serverConnected ? 'ok' : 'bad'}`}>
            <span />
            {serverConnected ? 'Backend' : 'Offline'}
          </div>
        </div>

        <div className="iphone-mode-switch" role="group" aria-label="Режим">
          <button
            type="button"
            className={globalMode === 'soundcheck' ? 'active' : ''}
            onClick={() => onGlobalModeChange('soundcheck')}
          >
            Soundcheck
          </button>
          <button
            type="button"
            className={globalMode === 'live' ? 'active' : ''}
            onClick={() => onGlobalModeChange('live')}
          >
            Live
          </button>
        </div>
      </header>

      {message && <div className="iphone-banner">{message}</div>}
      {!message && pendingCount > 0 && (
        <div className="iphone-banner">{pendingCount} коррекции ждут подтверждения · backup: {lastBackup.label}</div>
      )}

      <main className="iphone-remote-body">
        {view === 'live' && (
          <>
            <section className="iphone-card compact">
              <div className="iphone-card-head">
                <span>Состояние</span>
                <strong>{mixerConnected ? 'Mixer OK' : 'Mixer off'}</strong>
              </div>
              <div className="iphone-health-grid">
                <div><b>{agentStatus.is_running ? 'Run' : 'Idle'}</b><small>AI Agent</small></div>
                <div><b>{pendingCount}</b><small>Queue</small></div>
                <div><b>{audioStatus.running ? 'On' : 'Off'}</b><small>Audio</small></div>
              </div>
            </section>

            <section className="iphone-section">
              <div className="iphone-section-title">
                <span>Каналы</span>
                <strong>{selectedChannels?.length || 0} on</strong>
              </div>
              <div className="iphone-channel-list">
                {primaryRows.length ? primaryRows.map(renderChannelCard) : (
                  <div className="iphone-empty">Выберите аудиоинтерфейс и каналы в Setup</div>
                )}
              </div>
            </section>

            <section className="iphone-section">
              <div className="iphone-section-title">
                <span>Коррекции</span>
                <strong>{pendingCount ? 'review' : 'clear'}</strong>
              </div>
              <div className="iphone-correction-list">
                {pendingActions.slice(0, 3).map((action, index) => {
                  const rendered = formatAction(action);
                  return (
                    <div className="iphone-correction" key={`${action.index ?? index}-${action.timestamp ?? index}`}>
                      <div>
                        <b>{rendered.title}</b>
                        <small>{action.reason || 'ожидает подтверждения'}</small>
                      </div>
                      <strong>{rendered.value || 'safe'}</strong>
                    </div>
                  );
                })}
                {!pendingCount && (
                  <div className="iphone-correction muted">
                    <div>
                      <b>Очередь пуста</b>
                      <small>Запустите AI Agent или обновите состояние</small>
                    </div>
                    <strong>OK</strong>
                  </div>
                )}
              </div>
            </section>

            <section className="iphone-card">
              <div className="iphone-card-head">
                <span>Управление</span>
                <strong>{agentStatus.auto_apply_enabled ? 'auto apply' : 'confirm'}</strong>
              </div>
              <div className="iphone-inline-actions">
                <button type="button" onClick={agentStatus.is_running ? stopAgent : startAgent}>
                  {agentStatus.is_running ? 'Пауза AI' : 'Старт AI'}
                </button>
                <button type="button" onClick={createBackup}>Backup</button>
              </div>
            </section>
          </>
        )}

        {view === 'channels' && (
          <section className="iphone-section">
            <div className="iphone-section-title">
              <span>Выбор каналов</span>
              <button type="button" onClick={onSelectAllChannels}>
                {selectedChannels?.length === availableChannels?.length ? 'Снять' : 'Все'}
              </button>
            </div>
            <div className="iphone-channel-list">
              {(availableChannels || []).map(renderChannelCard)}
              {!availableChannels?.length && <div className="iphone-empty">Нет каналов от выбранного аудиоинтерфейса</div>}
            </div>
          </section>
        )}

        {view === 'queue' && (
          <section className="iphone-section">
            <div className="iphone-section-title">
              <span>Очередь коррекций</span>
              <button type="button" onClick={() => websocketService.getPendingActions()}>Обновить</button>
            </div>
            <div className="iphone-correction-list">
              {pendingActions.map((action, index) => {
                const rendered = formatAction(action);
                return (
                  <div className="iphone-correction" key={`${action.index ?? index}-${action.timestamp ?? index}`}>
                    <div>
                      <b>{rendered.title}</b>
                      <small>{action.reason || action.source || 'pending'}</small>
                    </div>
                    <strong>{rendered.value || 'review'}</strong>
                  </div>
                );
              })}
              {!pendingActions.length && <div className="iphone-empty">Коррекций на подтверждение нет</div>}
            </div>
            <div className="iphone-inline-actions">
              <button type="button" onClick={applyPending}>Apply all</button>
              <button type="button" className="warn" onClick={dismissPending}>Clear</button>
            </div>
            <div className="iphone-history">
              {history.slice(0, 5).map((action, index) => (
                <div key={`${index}-${action.timestamp || ''}`}>
                  Ch {action.channel || '-'} · {action.type || action.action || 'action'}
                </div>
              ))}
            </div>
          </section>
        )}

        {view === 'setup' && (
          <section className="iphone-setup">
            <div className="iphone-field">
              <label>Backend на MacBook</label>
              <div className="iphone-field-row">
                <input value={serverUrl} onChange={event => setServerUrl(event.target.value)} />
                <button type="button" onClick={applyServerUrl}>OK</button>
              </div>
            </div>

            <div className="iphone-field">
              <label>Микшерный пульт</label>
              <select
                value={mixerType}
                onChange={event => {
                  const next = event.target.value;
                  setMixerType(next);
                  onMixerPortChange(MIXER_DEFAULTS[next].port);
                }}
              >
                <option value="wing">Behringer WING</option>
                <option value="dlive">Allen & Heath dLive</option>
                <option value="mixing_station">Mixing Station</option>
              </select>
            </div>

            <div className="iphone-field two">
              <div>
                <label>{mixerType === 'mixing_station' ? 'Host' : 'IP'}</label>
                <input value={mixerIp} onChange={event => onMixerIpChange(event.target.value)} />
              </div>
              <div>
                <label>{mixerType === 'mixing_station' ? 'OSC' : 'Port'}</label>
                <input type="number" value={mixerPortValue} onChange={event => onMixerPortChange(Number(event.target.value))} />
              </div>
            </div>

            {mixerType === 'mixing_station' && (
              <div className="iphone-field">
                <label>REST port</label>
                <input type="number" value={mixingStationRestPort} onChange={event => setMixingStationRestPort(event.target.value)} />
              </div>
            )}

            {mixerType === 'dlive' && (
              <label className="iphone-checkbox">
                <input type="checkbox" checked={dliveTls} onChange={event => setDliveTls(event.target.checked)} />
                <span>TLS для dLive</span>
              </label>
            )}

            <div className="iphone-inline-actions">
              <button type="button" onClick={connectMixer} disabled={!serverConnected || connecting}>
                {mixerConnected ? 'Отключить' : 'Подключить'}
              </button>
              <button type="button" onClick={() => websocketService.autoConnectMixer(mixerType, mixerIp, false)}>
                Auto
              </button>
            </div>

            <div className="iphone-field">
              <label>Аудиоинтерфейс</label>
              <select value={selectedDevice} onChange={event => onDeviceChange(event.target.value)}>
                <option value="">Не выбран</option>
                {(audioDevices || []).map(device => (
                  <option key={device.id ?? device.index ?? device.name} value={device.id ?? device.index}>
                    {device.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="iphone-inline-actions">
              <button type="button" onClick={() => websocketService.scanAudioDevices()}>Scan audio</button>
              <button type="button" onClick={onScanMixer} disabled={!mixerConnected}>Имена каналов</button>
            </div>

            <div className="iphone-note">
              {statusMessage || 'iPhone должен быть в той же Wi-Fi сети, что и MacBook.'}
            </div>
          </section>
        )}
      </main>

      <footer className="iphone-command-pad">
        <button type="button" className="apply" onClick={applyPending}>Apply</button>
        <button type="button" className="rollback" onClick={restoreBackup}>Откат</button>
        <button type="button" className="bypass" onClick={confirmBypass}>Bypass</button>
        <button type="button" className="stop" onClick={emergencyStop}>СТОП</button>
      </footer>

      <nav className="iphone-bottom-nav" aria-label="Remote navigation">
        <button type="button" className={view === 'live' ? 'active' : ''} onClick={() => setView('live')}>Live</button>
        <button type="button" className={view === 'channels' ? 'active' : ''} onClick={() => setView('channels')}>Channels</button>
        <button type="button" className={view === 'queue' ? 'active' : ''} onClick={() => setView('queue')}>Queue</button>
        <button type="button" className={view === 'setup' ? 'active' : ''} onClick={() => setView('setup')}>Setup</button>
      </nav>
    </div>
  );
}

export default IPhoneControlSurface;
