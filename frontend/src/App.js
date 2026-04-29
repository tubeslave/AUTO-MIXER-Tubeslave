import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import websocketService from './services/websocket';
import VoiceControlTab from './components/VoiceControlTab';
import GainStagingTab from './components/GainStagingTab';
import AutoEQTab from './components/AutoEQTab';
import PhaseAlignmentTab from './components/PhaseAlignmentTab';
import SystemMeasurementTab from './components/SystemMeasurementTab';
import AutoFaderTab from './components/AutoFaderTab';
import AutoSoundcheckTab from './components/AutoSoundcheckTab';
import AutoCompressorTab from './components/AutoCompressorTab';
import SettingsTab from './components/SettingsTab';
import AIAgentTab from './components/AIAgentTab';
import IPhoneControlSurface from './components/IPhoneControlSurface';

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
  { id: 'remote', icon: '▣', label: 'Remote' },
  { id: 'mixer', icon: '🔌', label: 'Connect' },
  { id: 'gainStaging', icon: '📊', label: 'Gain' },
  { id: 'phaseAlignment', icon: '⟳', label: 'Phase' },
  { id: 'systemMeasurement', icon: '📐', label: 'Master' },
  { id: 'autoEQ', icon: '〰', label: 'EQ' },
  { id: 'autoCompressor', icon: '⬇', label: 'Comp' },
  { id: 'autoFader', icon: '🎚', label: 'Fader' },
  { id: 'autoSoundcheck', icon: '🎯', label: 'Soundcheck' },
  { id: 'aiAgent', icon: 'AI', label: 'Agent' },
  { id: 'voice', icon: '🎙', label: 'Voice' },
  { id: 'settings', icon: '⚙', label: 'Settings' },
];

function App() {
  const [serverConnected, setServerConnected] = useState(false);
  const [mixerConnected, setMixerConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [mixerIp, setMixerIp] = useState('192.168.1.102');
  const [mixerPort, setMixerPort] = useState(2223);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [availableChannels, setAvailableChannels] = useState([]);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [connecting, setConnecting] = useState(false);
  const [activeTab, setActiveTab] = useState(() => {
    if (typeof window !== 'undefined' && window.innerWidth <= 700) return 'remote';
    return 'mixer';
  });
  const [globalMode, setGlobalMode] = useState('soundcheck');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [routingScheme, setRoutingScheme] = useState([]);
  const [routingChannelCount, setRoutingChannelCount] = useState(64);
  const mixerIpRef = useRef(mixerIp);

  useEffect(() => {
    mixerIpRef.current = mixerIp;
  }, [mixerIp]);

  useEffect(() => {
    const handleConnectionStatus = (data) => {
      setConnecting(false);
      setMixerConnected(data.connected);
      if (data.connected) {
        setStatusMessage(`${data.mode || 'wing'}: ${data.ip || data.host || mixerIpRef.current}`);
      } else {
        if (data.error) {
          setStatusMessage(`Ошибка: ${data.error}`);
        } else {
          setStatusMessage('Отключен');
        }
      }
    };

    const handleAudioDevices = (data) => {
      setAudioDevices(data.devices || []);
      if (data.devices && data.devices.length > 0) {
        const danteDevice = data.devices.find(device =>
          device.name && device.name.toLowerCase().includes('dante')
        );
        const deviceToSelect = danteDevice || data.devices[0];
        setSelectedDevice(deviceToSelect.id);
        setAvailableChannels(deviceToSelect.channels || []);
      }
    };

    const handleBypassResult = (data) => {
      if (data.error) {
        setStatusMessage(`Bypass: ${data.error}`);
      } else if (data.success) {
        setStatusMessage(`Bypass OK: ${data.success_count}/40`);
      }
    };

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

    const handleMixerChannelNames = (data) => {
      if (data.error) {
        setStatusMessage(`Scan: ${data.error}`);
        return;
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
    };

    const handleDisconnected = () => {
      setServerConnected(false);
      setMixerConnected(false);
      setStatusMessage('Соединение потеряно, переподключение...');
    };

    const handleAllSettingsLoaded = (data) => {
      if (data.settings && data.settings.mixer) {
        const m = data.settings.mixer;
        if (m.mixerIp || m.ip) setMixerIp(m.mixerIp || m.ip);
        if (m.mixerPort || m.port) setMixerPort(m.mixerPort || m.port);
      }
    };

    const handleDanteRouting = (data) => {
      if (data.routing_scheme) setRoutingScheme(data.routing_scheme);
    };

    const handleAudioDeviceSelected = (data) => {
      if (data.device) {
        const deviceId = data.device.id ?? data.device.index;
        if (deviceId !== undefined) {
          setSelectedDevice(deviceId);
        }
        setStatusMessage(`Audio: ${data.device.name}`);
      } else if (data.error) {
        setStatusMessage(`Audio: ${data.error}`);
      }
    };

    websocketService.on('connection_status', handleConnectionStatus);
    websocketService.on('audio_devices', handleAudioDevices);
    websocketService.on('bypass_result', handleBypassResult);
    websocketService.on('mixer_channel_names', handleMixerChannelNames);
    websocketService.on('disconnected', handleDisconnected);
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);
    websocketService.on('dante_routing', handleDanteRouting);
    websocketService.on('audio_device_selected', handleAudioDeviceSelected);

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
      websocketService.off('connection_status', handleConnectionStatus);
      websocketService.off('audio_devices', handleAudioDevices);
      websocketService.off('bypass_result', handleBypassResult);
      websocketService.off('mixer_channel_names', handleMixerChannelNames);
      websocketService.off('disconnected', handleDisconnected);
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
      websocketService.off('dante_routing', handleDanteRouting);
      websocketService.off('audio_device_selected', handleAudioDeviceSelected);
      websocketService.disconnect();
    };
  }, []);

  const handleDeviceChange = (deviceId) => {
    setSelectedDevice(deviceId);
    const device = audioDevices.find(d =>
      String(d.id) === String(deviceId) || String(d.index) === String(deviceId)
    );
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
      websocketService.connectWing(mixerIp, mixerPort, mixerPort);
    }
  };

  const sharedProps = {
    selectedChannels,
    availableChannels,
    selectedDevice,
    audioDevices,
    globalMode,
  };

  return (
    <div className={`App ${activeTab === 'remote' ? 'remote-mode' : ''}`}>
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
          {activeTab === 'remote' && (
            <IPhoneControlSurface
              serverConnected={serverConnected}
              mixerConnected={mixerConnected}
              statusMessage={statusMessage}
              mixerIp={mixerIp}
              mixerPort={mixerPort}
              onMixerIpChange={setMixerIp}
              onMixerPortChange={setMixerPort}
              connecting={connecting}
              onConnect={handleConnect}
              audioDevices={audioDevices}
              selectedDevice={selectedDevice}
              onDeviceChange={handleDeviceChange}
              availableChannels={availableChannels}
              selectedChannels={selectedChannels}
              onChannelToggle={handleChannelToggle}
              onSelectAllChannels={handleSelectAllChannels}
              onScanMixer={handleScanMixer}
              onBypass={handleBypass}
              globalMode={globalMode}
              onGlobalModeChange={setGlobalMode}
            />
          )}

          {activeTab === 'mixer' && (
            <div className="connect-page">
              <div className="connect-card">
                <div className="connect-row">
                  <div className="field">
                    <label>Wing Rack IP</label>
                    <input
                      type="text"
                      value={mixerIp}
                      onChange={(e) => setMixerIp(e.target.value)}
                      disabled={mixerConnected}
                      placeholder="192.168.1.102"
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
                      audioDevices.map(device => (
                        <option key={device.id} value={device.id}>
                          {device.name}
                        </option>
                      ))
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

                  <div className="channels-grid">
                    {availableChannels.map(channel => (
                      <label key={channel.id} className={`ch-chip ${selectedChannels.includes(channel.id) ? 'selected' : ''}`}>
                        <input
                          type="checkbox"
                          checked={selectedChannels.includes(channel.id)}
                          onChange={() => handleChannelToggle(channel.id)}
                        />
                        <span>{channel.name || `Ch ${channel.id}`}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'gainStaging' && <GainStagingTab {...sharedProps} />}
          {activeTab === 'phaseAlignment' && <PhaseAlignmentTab {...sharedProps} />}
          {activeTab === 'systemMeasurement' && <SystemMeasurementTab {...sharedProps} />}
          {activeTab === 'autoEQ' && <AutoEQTab {...sharedProps} />}
          {activeTab === 'autoFader' && <AutoFaderTab {...sharedProps} />}
          {activeTab === 'autoSoundcheck' && <AutoSoundcheckTab {...sharedProps} />}
          {activeTab === 'aiAgent' && <AIAgentTab {...sharedProps} />}
          {activeTab === 'autoCompressor' && <AutoCompressorTab {...sharedProps} />}
          {activeTab === 'voice' && <VoiceControlTab globalMode={globalMode} />}
          {activeTab === 'settings' && (
            <SettingsTab
              mixerIp={mixerIp}
              mixerPort={mixerPort}
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
