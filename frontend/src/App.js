import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import websocketService from './services/websocket';
import VoiceControlTab from './components/VoiceControlTab';
import GainStagingTab from './components/GainStagingTab';
import AutoEQTab from './components/AutoEQTab';
import PhaseAlignmentTab from './components/PhaseAlignmentTab';
import AutoFaderTab from './components/AutoFaderTab';
import AutoSoundcheckTab from './components/AutoSoundcheckTab';
import AutoCompressorTab from './components/AutoCompressorTab';
import AutoEffectsTab from './components/AutoEffectsTab';
import AutoPannerTab from './components/AutoPannerTab';
import AutoGateTab from './components/AutoGateTab';
import AutoReverbTab from './components/AutoReverbTab';
import CrossAdaptiveEQTab from './components/CrossAdaptiveEQTab';
import SystemMeasurementTab from './components/SystemMeasurementTab';
import SettingsTab from './components/SettingsTab';
import ChannelPresetSelect from './components/ChannelPresetSelect';
import { detectInstrumentPreset } from './constants/instrumentPresets';

// Color map for routing roles
const ROLE_COLORS = {
  channel_analysis: '#00d4ff',
  channel_dry: '#f0883e',
  master: '#f85149',
  drum_bus: '#3fb950',
  vocal_bus: '#a371f7',
  instrument_bus: '#d29922',
  measurement_mic: '#39d353',
  ambient_mic: '#f778ba',
  matrix: '#8b949e',
  reserve: '#484f58',
};

const NAV_ITEMS = [
  { id: 'mixer', icon: '🔌', label: 'Connect' },
  { id: 'gainStaging', icon: '📊', label: 'Gain' },
  { id: 'phaseAlignment', icon: '⟳', label: 'Phase' },
  { id: 'autoEQ', icon: '〰', label: 'EQ' },
  { id: 'autoCompressor', icon: '⬇', label: 'Comp' },
  { id: 'autoGate', icon: '🚪', label: 'Gate' },
  { id: 'autoFader', icon: '🎚', label: 'Fader' },
  { id: 'autoPanner', icon: '🎧', label: 'Pan' },
  { id: 'autoReverb', icon: '🌊', label: 'Reverb' },
  { id: 'autoEffects', icon: '✨', label: 'FX' },
  { id: 'crossAdaptiveEQ', icon: '🔀', label: 'X-EQ' },
  { id: 'autoSoundcheck', icon: '🎯', label: 'Soundcheck' },
  { id: 'systemMeasurement', icon: '📐', label: 'Measure' },
  { id: 'voice', icon: '🎙', label: 'Voice' },
  { id: 'settings', icon: '⚙', label: 'Settings' },
];

function App() {
  const [serverConnected, setServerConnected] = useState(false);
  const [mixerConnected, setMixerConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [mixerIp, setMixerIp] = useState('127.0.0.1');
  const [mixerPort, setMixerPort] = useState(2222);
  const [mixerType, setMixerType] = useState('wing');
  const [dliveIp, setDliveIp] = useState('192.168.1.70');
  const [dlivePort, setDlivePort] = useState(51328);
  const [abletonSendPort, setAbletonSendPort] = useState(11000);
  const [abletonRecvPort, setAbletonRecvPort] = useState(11001);
  const [channelOffset, setChannelOffset] = useState(0);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [availableChannels, setAvailableChannels] = useState([]);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [activeTab, setActiveTab] = useState('mixer');
  const [globalMode, setGlobalMode] = useState('soundcheck');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [routingScheme, setRoutingScheme] = useState([]);
  const [routingChannelCount, setRoutingChannelCount] = useState(64);
  const [channelPresets, setChannelPresets] = useState({});
  const setChannelPreset = useCallback((channelId, presetId) => {
    const key = String(channelId);
    setChannelPresets(prev => ({ ...prev, [key]: presetId }));
  }, []);

  const presetsLoadedRef = useRef(false);
  const presetsSaveTimerRef = useRef(null);
  const connectSelectionLoadedRef = useRef(false);
  const connectSelectionRef = useRef({ selectedDevice: '', selectedChannels: [] });
  const connectSelectionSaveTimerRef = useRef(null);
  const audioDevicesRef = useRef([]);

  const applySavedConnectSelection = useCallback((devices) => {
    const deviceList = Array.isArray(devices) ? devices : [];
    audioDevicesRef.current = deviceList;

    if (deviceList.length === 0) {
      setSelectedDevice('');
      setAvailableChannels([]);
      setSelectedChannels([]);
      return;
    }

    if (connectSelectionLoadedRef.current) {
      const savedDeviceId = connectSelectionRef.current.selectedDevice;
      if (savedDeviceId === '' || savedDeviceId == null) {
        setSelectedDevice('');
        setAvailableChannels([]);
        setSelectedChannels([]);
        return;
      }

      const matchedDevice = deviceList.find(d => String(d.id) === String(savedDeviceId));
      if (matchedDevice) {
        const channels = matchedDevice.channels || [];
        const validChannelIds = new Set(channels.map(ch => String(ch.id)));
        const validSelectedChannels = (connectSelectionRef.current.selectedChannels || [])
          .filter(chId => validChannelIds.has(String(chId)));

        setSelectedDevice(matchedDevice.id);
        setAvailableChannels(channels);
        setSelectedChannels(validSelectedChannels);
        return;
      }
    }

    const danteDevice = deviceList.find(device =>
      device.name && device.name.toLowerCase().includes('dante')
    );
    const fallbackDevice = danteDevice || deviceList[0];
    setSelectedDevice(fallbackDevice.id);
    setAvailableChannels(fallbackDevice.channels || []);
  }, []);

  useEffect(() => {
    websocketService.on('connection_status', (data) => {
      setConnecting(false);
      setMixerConnected(data.connected);
      if (data.connected) {
        const mode = data.mode || 'wing';
        const label = mode === 'ableton' ? 'Ableton' : mode === 'dlive' ? 'dLive' : 'Wing';
        setStatusMessage(`${label}: ${data.ip || mixerIp}`);
      } else {
        if (data.error) {
          setStatusMessage(`Ошибка: ${data.error}`);
        } else {
          setStatusMessage('Отключен');
        }
      }
    });

    websocketService.on('audio_devices', (data) => {
      const devices = data.devices || [];
      setAudioDevices(devices);
      applySavedConnectSelection(devices);
    });

    websocketService.on('bypass_result', (data) => {
      if (data.error) {
        setStatusMessage(`Bypass: ${data.error}`);
      } else if (data.success) {
        setStatusMessage(`Bypass OK: ${data.success_count}/40`);
      }
    });

    const isDefaultChannelName = (name) => {
      if (!name || !name.trim()) return true;
      const trimmedName = name.trim();
      const defaultPatterns = [
        /^ch\s*\d+$/i,
        /^channel\s*\d+$/i,
        /^\d+$/,
        /^input\s*\d+$/i,
        /^in\s*\d+$/i,
      ];
      return defaultPatterns.some(pattern => pattern.test(trimmedName));
    };

    websocketService.on('mixer_channel_names', (data) => {
      if (data.error) {
        setStatusMessage(`Scan: ${data.error}`);
        return;
      }
      if (data.channel_names) {
        setChannelPresets(prev => {
          const next = { ...prev };
          Object.entries(data.channel_names).forEach(([ch, name]) => {
            if (!name || !String(name).trim()) return;
            const key = String(ch);
            if (!next[key] || next[key] === 'custom') {
              next[key] = detectInstrumentPreset(name);
            }
          });
          return next;
        });
      }
      if (data.channel_names) {
        setAvailableChannels(prevChannels => {
          const updatedChannels = prevChannels.map(channel => {
            const channelNum = typeof channel.id === 'number' ? channel.id : parseInt(channel.id);
            let newName = null;
            if (!isNaN(channelNum)) {
              newName = data.channel_names[channelNum];
              if (!newName) newName = data.channel_names[String(channelNum)];
            }
            if (newName && newName.trim()) {
              return { ...channel, name: newName.trim() };
            }
            const nameMatch = channel.name?.match(/(\d+)/);
            if (nameMatch) {
              const numFromName = parseInt(nameMatch[1]);
              if (!isNaN(numFromName)) {
                let nameFromMatch = data.channel_names[numFromName] || data.channel_names[String(numFromName)];
                if (nameFromMatch && nameFromMatch.trim()) {
                  return { ...channel, name: nameFromMatch.trim() };
                }
              }
            }
            return channel;
          });

          const channelsWithNames = updatedChannels.filter(channel => {
            return !isDefaultChannelName(channel.name || '');
          });
          if (channelsWithNames.length > 0) {
            setSelectedChannels(channelsWithNames.map(ch => ch.id));
            setStatusMessage(`Найдено ${channelsWithNames.length} каналов`);
          } else {
            setStatusMessage('Каналы просканированы');
          }
          return updatedChannels;
        });
      }
    });

    websocketService.on('disconnected', () => {
      setServerConnected(false);
      setMixerConnected(false);
      setStatusMessage('Переподключение...');
    });

    const handleAllSettingsLoaded = (data) => {
      if (data.settings && data.settings.mixer) {
        const m = data.settings.mixer;
        if (m.mixerIp) setMixerIp(m.mixerIp);
        if (m.mixerPort) setMixerPort(m.mixerPort);
        if (m.mixerType) setMixerType(m.mixerType);
        if (m.dliveIp) setDliveIp(m.dliveIp);
        if (m.dlivePort != null) setDlivePort(m.dlivePort);
        if (m.abletonSendPort != null) setAbletonSendPort(m.abletonSendPort);
        if (m.abletonRecvPort != null) setAbletonRecvPort(m.abletonRecvPort);
        if (m.channelOffset != null) setChannelOffset(m.channelOffset);
      }

      const hasConnectSection = Boolean(
        data.settings && Object.prototype.hasOwnProperty.call(data.settings, 'connect')
      );
      connectSelectionLoadedRef.current = hasConnectSection;
      if (hasConnectSection) {
        const savedConnect = data.settings.connect || {};
        connectSelectionRef.current = {
          selectedDevice: savedConnect.selectedDevice ?? '',
          selectedChannels: Array.isArray(savedConnect.selectedChannels)
            ? savedConnect.selectedChannels
            : [],
        };
      } else {
        connectSelectionRef.current = { selectedDevice: '', selectedChannels: [] };
      }
      applySavedConnectSelection(audioDevicesRef.current);

      const savedPresets = data.settings?.channelPresets?.channels;
      if (savedPresets && typeof savedPresets === 'object') {
        setChannelPresets(savedPresets);
      }
      presetsLoadedRef.current = true;
      setSettingsLoaded(true);
    };
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);

    websocketService.on('dante_routing', (data) => {
      if (data.routing_scheme) setRoutingScheme(data.routing_scheme);
    });

    websocketService.connect()
      .then(() => {
        setServerConnected(true);
        setStatusMessage('Backend OK');
        websocketService.send({ type: 'get_audio_devices' });
        websocketService.loadAllSettings();
        websocketService.getDanteRouting(64);
      })
      .catch(err => {
        setStatusMessage('Backend недоступен');
        console.error(err);
      });

    return () => {
      websocketService.off('connection_status', () => {});
      websocketService.off('audio_devices', () => {});
      websocketService.off('bypass_result', () => {});
      websocketService.off('mixer_channel_names', () => {});
      websocketService.off('disconnected', () => {});
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
      websocketService.off('dante_routing', () => {});
      websocketService.disconnect();
    };
  }, [applySavedConnectSelection]);

  useEffect(() => {
    if (!serverConnected || !presetsLoadedRef.current) return;
    if (presetsSaveTimerRef.current) clearTimeout(presetsSaveTimerRef.current);
    presetsSaveTimerRef.current = setTimeout(() => {
      websocketService.saveAllSettings({
        channelPresets: { channels: channelPresets },
      });
    }, 350);
    return () => {
      if (presetsSaveTimerRef.current) clearTimeout(presetsSaveTimerRef.current);
    };
  }, [channelPresets, serverConnected]);

  useEffect(() => {
    if (!serverConnected || !settingsLoaded) return;
    if (connectSelectionSaveTimerRef.current) clearTimeout(connectSelectionSaveTimerRef.current);

    connectSelectionLoadedRef.current = true;
    connectSelectionRef.current = {
      selectedDevice,
      selectedChannels,
    };

    connectSelectionSaveTimerRef.current = setTimeout(() => {
      websocketService.saveAllSettings({
        connect: {
          selectedDevice,
          selectedChannels,
        },
      });
    }, 350);

    return () => {
      if (connectSelectionSaveTimerRef.current) clearTimeout(connectSelectionSaveTimerRef.current);
    };
  }, [selectedDevice, selectedChannels, serverConnected, settingsLoaded]);

  const handleDeviceChange = (deviceId) => {
    setSelectedDevice(deviceId);
    const device = audioDevices.find(d => d.id === deviceId);
    if (device) {
      setAvailableChannels(device.channels || []);
      setSelectedChannels([]);
    }
  };

  const handleChannelToggle = (channelId) => {
    setSelectedChannels(prev =>
      prev.includes(channelId) ? prev.filter(id => id !== channelId) : [...prev, channelId]
    );
  };

  const handleSelectAllChannels = () => {
    if (selectedChannels.length === availableChannels.length) {
      setSelectedChannels([]);
    } else {
      setSelectedChannels(availableChannels.map(ch => ch.id));
    }
  };

  const handleScanMixer = () => {
    if (!mixerConnected) {
      setStatusMessage('Сначала подключитесь');
      return;
    }
    setStatusMessage('Сканирую...');
    websocketService.scanMixerChannelNames();
  };

  const handleBypass = () => {
    if (!mixerConnected) return;
    if (!window.confirm('Сбросить все модули и фейдеры на 0dB?')) return;
    setStatusMessage('Bypass...');
    websocketService.bypassMixer();
  };

  const handleConnect = () => {
    if (mixerConnected) {
      websocketService.disconnectMixer();
    } else {
      setConnecting(true);
      setStatusMessage('Подключаю...');
      if (mixerType === 'ableton') {
        websocketService.connectAbleton(mixerIp, abletonSendPort, abletonRecvPort, channelOffset);
      } else if (mixerType === 'dlive') {
        websocketService.connectDLive(dliveIp, dlivePort);
      } else {
        websocketService.connectWing(mixerIp, mixerPort, mixerPort);
      }
    }
  };

  const handleResetConnectSelection = () => {
    connectSelectionLoadedRef.current = true;
    connectSelectionRef.current = { selectedDevice: '', selectedChannels: [] };
    setSelectedDevice('');
    setAvailableChannels([]);
    setSelectedChannels([]);
    if (serverConnected) {
      websocketService.saveAllSettings({
        connect: {
          selectedDevice: '',
          selectedChannels: [],
        },
      });
    }
  };

  const sharedProps = {
    selectedChannels,
    availableChannels,
    selectedDevice,
    audioDevices,
    globalMode,
    channelPresets,
    setChannelPreset,
    detectInstrumentPreset,
  };

  return (
    <div className="App">
      {/* Sidebar */}
      <nav className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <h1 className="logo">{sidebarCollapsed ? 'AM' : 'AutoMixer'}</h1>
          <button className="btn-collapse" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
            {sidebarCollapsed ? '▶' : '◀'}
          </button>
        </div>

        <div className="sidebar-status">
          <div className={`status-dot ${serverConnected ? 'on' : 'off'}`} title={serverConnected ? 'Backend Online' : 'Backend Offline'} />
          <div className={`status-dot ${mixerConnected ? 'on' : 'off'}`} title={mixerConnected ? 'Mixer OK' : 'Mixer Off'} />
          {!sidebarCollapsed && (
            <span className="status-text-small">
              {serverConnected ? (mixerConnected ? 'Всё ОК' : 'Mixer ✕') : 'Offline'}
            </span>
          )}
        </div>

        {/* Global Mode Toggle */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${globalMode === 'soundcheck' ? 'active' : ''}`}
            onClick={() => setGlobalMode('soundcheck')}
            title="Soundcheck: анализ → применить"
          >
            {sidebarCollapsed ? 'SC' : 'Soundcheck'}
          </button>
          <button
            className={`mode-btn ${globalMode === 'live' ? 'active' : ''}`}
            onClick={() => setGlobalMode('live')}
            title="Live: непрерывная коррекция"
          >
            {sidebarCollapsed ? 'LV' : 'Live'}
          </button>
        </div>

        <div className="nav-items">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
              title={item.label}
            >
              <span className="nav-icon">{item.icon}</span>
              {!sidebarCollapsed && <span className="nav-label">{item.label}</span>}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* Top bar with mode indicator */}
        <div className="topbar">
          <span className="topbar-title">
            {NAV_ITEMS.find(n => n.id === activeTab)?.icon}{' '}
            {NAV_ITEMS.find(n => n.id === activeTab)?.label}
          </span>
          <div className={`mode-badge ${globalMode}`}>
            {globalMode === 'soundcheck' ? '🎯 Soundcheck' : '🔴 Live'}
          </div>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'mixer' && (
            <div className="connect-page">
              <div className="connect-card">
                <div className="connect-row">
                  {mixerType === 'ableton' && (
                    <>
                      <div className="field">
                        <label>Ableton IP</label>
                        <input
                          type="text"
                          value={mixerIp}
                          onChange={(e) => setMixerIp(e.target.value)}
                          disabled={mixerConnected}
                          placeholder="127.0.0.1"
                        />
                      </div>
                      <div className="field">
                        <label>Send</label>
                        <input
                          type="number"
                          value={abletonSendPort}
                          onChange={(e) => setAbletonSendPort(parseInt(e.target.value))}
                          disabled={mixerConnected}
                        />
                      </div>
                      <div className="field">
                        <label>Recv</label>
                        <input
                          type="number"
                          value={abletonRecvPort}
                          onChange={(e) => setAbletonRecvPort(parseInt(e.target.value))}
                          disabled={mixerConnected}
                        />
                      </div>
                      <div className="field">
                        <label>Ch Offset</label>
                        <input
                          type="number"
                          min="0"
                          value={channelOffset}
                          onChange={(e) => setChannelOffset(parseInt(e.target.value) || 0)}
                          disabled={mixerConnected}
                        />
                      </div>
                    </>
                  )}
                  {mixerType === 'dlive' && (
                    <>
                      <div className="field">
                        <label>dLive IP</label>
                        <input
                          type="text"
                          value={dliveIp}
                          onChange={(e) => setDliveIp(e.target.value)}
                          disabled={mixerConnected}
                        />
                      </div>
                      <div className="field">
                        <label>TCP Port</label>
                        <input
                          type="number"
                          value={dlivePort}
                          onChange={(e) => setDlivePort(parseInt(e.target.value))}
                          disabled={mixerConnected}
                        />
                      </div>
                    </>
                  )}
                  {mixerType === 'wing' && (
                    <>
                      <div className="field">
                        <label>Wing IP</label>
                        <input
                          type="text"
                          value={mixerIp}
                          onChange={(e) => setMixerIp(e.target.value)}
                          disabled={mixerConnected}
                          placeholder="127.0.0.1 (virtual) или 192.168.x.x (Wing)"
                        />
                      </div>
                      <div className="field">
                        <label>OSC Port</label>
                        <input
                          type="number"
                          value={mixerPort}
                          onChange={(e) => setMixerPort(parseInt(e.target.value))}
                          disabled={mixerConnected}
                        />
                      </div>
                    </>
                  )}
                  <button
                    className={`btn-connect ${mixerConnected ? 'connected' : ''}`}
                    onClick={handleConnect}
                    disabled={!serverConnected || connecting}
                  >
                    {connecting ? '...' : mixerConnected ? 'Отключить' : 'Подключить'}
                  </button>
                </div>
                {statusMessage && <p className="status-msg">{statusMessage}</p>}
              </div>

              <div className="connect-card">
                <div className="field full">
                  <label>Аудио устройство</label>
                  <select
                    value={selectedDevice}
                    onChange={(e) => handleDeviceChange(e.target.value)}
                    disabled={audioDevices.length === 0}
                  >
                    {audioDevices.length === 0 ? (
                      <option value="">Нет устройств</option>
                    ) : (
                      <>
                        <option value="">Выберите устройство</option>
                        {audioDevices.map(device => (
                          <option key={device.id} value={device.id}>
                            {device.name}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                </div>
              </div>

              {/* Routing Scheme Card */}
              <div className="connect-card">
                <div className="routing-scheme">
                  <div className="routing-scheme-header">
                    <h3>Схема маршрутизации Dante</h3>
                    <div className="scheme-selector">
                      <button
                        className={`scheme-btn ${routingChannelCount === 64 ? 'active' : ''}`}
                        onClick={() => { setRoutingChannelCount(64); websocketService.getDanteRouting(64); }}
                      >64 ch</button>
                      <button
                        className={`scheme-btn ${routingChannelCount === 32 ? 'active' : ''}`}
                        onClick={() => { setRoutingChannelCount(32); websocketService.getDanteRouting(32); }}
                      >32 ch</button>
                    </div>
                  </div>
                  {routingScheme.map((row, idx) => (
                    <div className="routing-row" key={idx}>
                      <div className="routing-role-bar" style={{ background: ROLE_COLORS[row.role] || '#484f58' }} />
                      <div className="routing-body">
                        <div className="routing-top">
                          <span className="routing-channels" style={{ color: ROLE_COLORS[row.role] || '#e6e6e6' }}>
                            {row.label_short}
                          </span>
                          <span className="routing-tap">{row.tap_point}</span>
                          <span className={`routing-badge ${row.required ? 'required' : 'optional'}`}>
                            {row.required ? 'Обязательно' : 'Опционально'}
                          </span>
                        </div>
                        <div className="routing-label">{row.label_full}</div>
                        {row.wing_routing_hint && (
                          <div className="routing-wing-hint">{row.wing_routing_hint}</div>
                        )}
                        {row.used_by && row.used_by.length > 0 && (
                          <div className="routing-modules">
                            {row.used_by.map((mod, i) => (
                              <span className="routing-module-tag" key={i}>{mod}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {availableChannels.length > 0 && (
                <div className="connect-card">
                  <div className="channels-toolbar">
                    <span className="ch-count">{selectedChannels.length}/{availableChannels.length}</span>
                    <button className="btn-sm" onClick={handleSelectAllChannels}>
                      {selectedChannels.length === availableChannels.length ? 'Снять все' : 'Выбрать все'}
                    </button>
                    <button className="btn-sm" onClick={handleScanMixer} disabled={!mixerConnected}>
                      Скан
                    </button>
                    <button className="btn-sm danger" onClick={handleBypass} disabled={!mixerConnected}>
                      Bypass
                    </button>
                  </div>

                  <div className="channels-table-wrap">
                    <table className="channels-table">
                      <thead>
                        <tr>
                          <th style={{ width: 36 }} />
                          <th>Канал</th>
                          <th style={{ minWidth: 140 }}>Пресет</th>
                        </tr>
                      </thead>
                      <tbody>
                        {availableChannels.map((channel) => {
                          const id = channel.id;
                          const name = channel.name || `Ch ${id}`;
                          const preset =
                            channelPresets[String(id)] || detectInstrumentPreset(name);
                          const selected = selectedChannels.includes(id);
                          return (
                            <tr
                              key={id}
                              className={selected ? 'channel-row-selected' : ''}
                            >
                              <td>
                                <input
                                  id={`ch-toggle-${id}`}
                                  type="checkbox"
                                  checked={selected}
                                  onChange={() => handleChannelToggle(id)}
                                  aria-label={`Выбрать ${name}`}
                                />
                              </td>
                              <td>
                                <label
                                  style={{ cursor: 'pointer' }}
                                  htmlFor={`ch-toggle-${id}`}
                                >
                                  {name}
                                </label>
                              </td>
                              <td>
                                <ChannelPresetSelect
                                  value={preset}
                                  onChange={(presetId) => setChannelPreset(id, presetId)}
                                />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'gainStaging' && <GainStagingTab {...sharedProps} />}
          {activeTab === 'phaseAlignment' && <PhaseAlignmentTab {...sharedProps} />}
          {activeTab === 'autoEQ' && <AutoEQTab {...sharedProps} />}
          {activeTab === 'autoFader' && <AutoFaderTab {...sharedProps} />}
          {activeTab === 'autoSoundcheck' && <AutoSoundcheckTab {...sharedProps} />}
          {activeTab === 'autoCompressor' && <AutoCompressorTab {...sharedProps} />}
          {activeTab === 'autoEffects' && <AutoEffectsTab {...sharedProps} />}
          {activeTab === 'autoPanner' && <AutoPannerTab {...sharedProps} />}
          {activeTab === 'autoGate' && <AutoGateTab {...sharedProps} />}
          {activeTab === 'autoReverb' && <AutoReverbTab {...sharedProps} />}
          {activeTab === 'crossAdaptiveEQ' && <CrossAdaptiveEQTab {...sharedProps} />}
          {activeTab === 'systemMeasurement' && (
            <SystemMeasurementTab selectedDevice={selectedDevice} mixerClient={null} globalMode={globalMode} />
          )}
          {activeTab === 'voice' && <VoiceControlTab globalMode={globalMode} />}
          {activeTab === 'settings' && (
            <SettingsTab
              mixerIp={mixerIp}
              mixerPort={mixerPort}
              onResetConnectSelection={handleResetConnectSelection}
              onResetChannelPresets={() => setChannelPresets({})}
              onMixerSettingsChange={(key, value) => {
                if (key === 'mixerIp') setMixerIp(value);
                if (key === 'mixerPort') setMixerPort(value);
              }}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
