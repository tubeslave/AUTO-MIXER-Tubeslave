import React, { useState, useEffect, useRef } from 'react';
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
import PilotDeploymentCockpit from './components/PilotDeploymentCockpit';
import {
  AgentsConsoleView,
  ChannelListView,
  ConnectionsView,
  DashboardView,
  TransportBar,
} from './components/ControlCenterViews';

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
  { section: 'Overview', id: 'dashboard', icon: '▣', label: 'Dashboard' },
  { section: 'Channels', id: 'channelList', icon: '☰', label: 'Channel List' },
  { section: 'Channels', id: 'gainStaging', icon: '⌘', label: 'Groups / Buses' },
  { section: 'AI Agents', id: 'agentsConsole', icon: '△', label: 'Agents Console' },
  { section: 'AI Agents', id: 'autoEQ', icon: '▤', label: 'Decision Log' },
  { section: 'Learning', id: 'autoSoundcheck', icon: '◌', label: 'Training Sessions' },
  { section: 'Learning', id: 'systemMeasurement', icon: '□', label: 'Rules Library' },
  { section: 'System', id: 'connections', icon: '▣', label: 'Connections' },
  { section: 'System', id: 'settings', icon: '⚙', label: 'Settings' },
  { section: 'System', id: 'voice', icon: '▤', label: 'Logs' },
];

const WING_QUARANTINED_TABS = new Set();
const OPERATOR_MODES = [
  { key: 'analyze', label: 'Analyze', short: 'ANL' },
  { key: 'assist', label: 'Assist', short: 'AST' },
  { key: 'supervised', label: 'Supervised', short: 'SUP' },
  { key: 'auto', label: 'Auto', short: 'AUT' },
];

function getInitialControlCenterTab() {
  const hashTab = window.location.hash.replace(/^#\/?/, '');
  return NAV_ITEMS.some(item => item.id === hashTab) ? hashTab : 'dashboard';
}

function App() {
  const [serverConnected, setServerConnected] = useState(false);
  const [mixerConnected, setMixerConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [mixerIp, setMixerIp] = useState('192.168.1.102');
  const [mixerPort, setMixerPort] = useState(2223);
  const [mixerType, setMixerType] = useState('wing');
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [availableChannels, setAvailableChannels] = useState([]);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [connecting, setConnecting] = useState(false);
  const [activeTab, setActiveTab] = useState(getInitialControlCenterTab);
  const [globalMode] = useState('soundcheck');
  const [operatorModeStatus, setOperatorModeStatus] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [routingScheme, setRoutingScheme] = useState([]);
  const [routingChannelCount, setRoutingChannelCount] = useState(64);
  const [pilotCockpit, setPilotCockpit] = useState(null);
  const [dashboardSnapshot, setDashboardSnapshot] = useState(null);
  const [channelInventory, setChannelInventory] = useState(null);
  const [decisionQueue, setDecisionQueue] = useState(null);
  const [connectionTopology, setConnectionTopology] = useState(null);
  const [operatorAnalysisReport, setOperatorAnalysisReport] = useState(null);
  const mixerIpRef = useRef(mixerIp);

  useEffect(() => {
    mixerIpRef.current = mixerIp;
  }, [mixerIp]);

  useEffect(() => {
    const refreshPilotCockpit = () => {
      websocketService.getSupervisedPilotStatus();
    };

    const refreshProductState = () => {
      websocketService.getDashboardSnapshot();
      websocketService.getChannelInventory(24);
      websocketService.getDecisionQueue(50);
      websocketService.getConnectionTopology();
    };

    const handleConnectionStatus = (data) => {
      setConnecting(false);
      setMixerConnected(data.connected);
      if (data.connected) {
        setStatusMessage(`Wing: ${data.ip || mixerIpRef.current}`);
        refreshPilotCockpit();
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
      setPilotCockpit(null);
      setStatusMessage('Переподключение...');
    };

    const handleAllSettingsLoaded = (data) => {
      if (data.settings && data.settings.mixer) {
        const m = data.settings.mixer;
        if (m.mixerType) setMixerType(m.mixerType);
        if (m.mixerIp) setMixerIp(m.mixerIp);
        if (m.mixerPort) setMixerPort(m.mixerPort);
      }
    };

    const handleDanteRouting = (data) => {
      if (data.routing_scheme) setRoutingScheme(data.routing_scheme);
    };

    const handleSupervisedPilotStatus = (data) => {
      setPilotCockpit(data);
      if (data.operator_mode) setOperatorModeStatus(data.operator_mode);
    };

    const handleOperatorModeStatus = (data) => {
      setOperatorModeStatus(data);
      if (data.label) setStatusMessage(`Mode: ${data.label}`);
    };

    const handleDashboardSnapshot = (data) => {
      setDashboardSnapshot(data);
      if (data.operator_mode) setOperatorModeStatus(data.operator_mode);
    };

    const handleChannelInventory = (data) => {
      setChannelInventory(data);
    };

    const handleDecisionQueue = (data) => {
      setDecisionQueue(data);
      if (data.operator_mode) setOperatorModeStatus(data.operator_mode);
    };

    const handleConnectionTopology = (data) => {
      setConnectionTopology(data);
      if (data.connection?.connected !== undefined) {
        setMixerConnected(Boolean(data.connection.connected));
      }
    };

    const handleOperatorAnalysisReport = (data) => {
      setOperatorAnalysisReport(data);
      setStatusMessage(`Analysis: ${data.status || 'ok'} · ${data.created_count || 0} proposals`);
      refreshProductState();
    };

    const handleOperatorProposalsImported = (data) => {
      setStatusMessage(`Import: ${data.created_count || 0} new · ${data.existing_count || 0} existing`);
      refreshProductState();
    };

    const handlePilotRefreshTrigger = () => {
      refreshPilotCockpit();
      refreshProductState();
    };

    websocketService.on('connection_status', handleConnectionStatus);
    websocketService.on('audio_devices', handleAudioDevices);
    websocketService.on('bypass_result', handleBypassResult);
    websocketService.on('mixer_channel_names', handleMixerChannelNames);
    websocketService.on('disconnected', handleDisconnected);
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);
    websocketService.on('dante_routing', handleDanteRouting);
    websocketService.on('supervised_pilot_status', handleSupervisedPilotStatus);
    websocketService.on('operator_mode_status', handleOperatorModeStatus);
    websocketService.on('dashboard_snapshot', handleDashboardSnapshot);
    websocketService.on('channel_inventory', handleChannelInventory);
    websocketService.on('decision_queue', handleDecisionQueue);
    websocketService.on('connection_topology', handleConnectionTopology);
    websocketService.on('operator_analysis_report', handleOperatorAnalysisReport);
    websocketService.on('operator_proposals_imported', handleOperatorProposalsImported);
    websocketService.on('operator_proposal_created', handlePilotRefreshTrigger);
    websocketService.on('operator_proposal_accepted', handlePilotRefreshTrigger);
    websocketService.on('operator_proposal_dismissed', handlePilotRefreshTrigger);
    websocketService.on('operator_proposal_apply_result', handlePilotRefreshTrigger);
    websocketService.on('operator_proposal_apply_blocked', handlePilotRefreshTrigger);
    websocketService.on('manual_write_supervision', handlePilotRefreshTrigger);
    websocketService.on('manual_write_session', handlePilotRefreshTrigger);
    websocketService.on('manual_write_applied', handlePilotRefreshTrigger);
    websocketService.on('manual_write_blocked', handlePilotRefreshTrigger);
    websocketService.on('manual_write_error', handlePilotRefreshTrigger);
    websocketService.on('manual_write_rollback_applied', handlePilotRefreshTrigger);
    websocketService.on('manual_write_rollback_blocked', handlePilotRefreshTrigger);
    websocketService.on('manual_write_rollback_error', handlePilotRefreshTrigger);
    websocketService.on('emergency_stop_applied', handlePilotRefreshTrigger);

    websocketService.connect()
      .then(() => {
        setServerConnected(true);
        setStatusMessage('Backend OK');
        websocketService.send({ type: 'get_audio_devices' });
        websocketService.loadAllSettings();
        websocketService.getDanteRouting(64);
        websocketService.getOperatorMode();
        refreshPilotCockpit();
        refreshProductState();
      })
      .catch(err => {
        setStatusMessage('Backend недоступен');
        console.error(err);
      });

    const cockpitPollTimer = setInterval(() => {
      if (websocketService.ws && websocketService.ws.readyState === WebSocket.OPEN) {
        refreshPilotCockpit();
        refreshProductState();
      }
    }, 5000);

    return () => {
      websocketService.off('connection_status', handleConnectionStatus);
      websocketService.off('audio_devices', handleAudioDevices);
      websocketService.off('bypass_result', handleBypassResult);
      websocketService.off('mixer_channel_names', handleMixerChannelNames);
      websocketService.off('disconnected', handleDisconnected);
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
      websocketService.off('dante_routing', handleDanteRouting);
      websocketService.off('supervised_pilot_status', handleSupervisedPilotStatus);
      websocketService.off('operator_mode_status', handleOperatorModeStatus);
      websocketService.off('dashboard_snapshot', handleDashboardSnapshot);
      websocketService.off('channel_inventory', handleChannelInventory);
      websocketService.off('decision_queue', handleDecisionQueue);
      websocketService.off('connection_topology', handleConnectionTopology);
      websocketService.off('operator_analysis_report', handleOperatorAnalysisReport);
      websocketService.off('operator_proposals_imported', handleOperatorProposalsImported);
      websocketService.off('operator_proposal_created', handlePilotRefreshTrigger);
      websocketService.off('operator_proposal_accepted', handlePilotRefreshTrigger);
      websocketService.off('operator_proposal_dismissed', handlePilotRefreshTrigger);
      websocketService.off('operator_proposal_apply_result', handlePilotRefreshTrigger);
      websocketService.off('operator_proposal_apply_blocked', handlePilotRefreshTrigger);
      websocketService.off('manual_write_supervision', handlePilotRefreshTrigger);
      websocketService.off('manual_write_session', handlePilotRefreshTrigger);
      websocketService.off('manual_write_applied', handlePilotRefreshTrigger);
      websocketService.off('manual_write_blocked', handlePilotRefreshTrigger);
      websocketService.off('manual_write_error', handlePilotRefreshTrigger);
      websocketService.off('manual_write_rollback_applied', handlePilotRefreshTrigger);
      websocketService.off('manual_write_rollback_blocked', handlePilotRefreshTrigger);
      websocketService.off('manual_write_rollback_error', handlePilotRefreshTrigger);
      websocketService.off('emergency_stop_applied', handlePilotRefreshTrigger);
      clearInterval(cockpitPollTimer);
      websocketService.disconnect();
    };
  }, []);

  useEffect(() => {
    const handleHashChange = () => {
      setActiveTab(getInitialControlCenterTab());
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

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
      websocketService.connectWing(mixerIp, mixerPort, mixerPort);
    }
  };

  const sharedProps = {
    selectedChannels,
    availableChannels,
    selectedDevice,
    audioDevices,
    globalMode,
    operatorMode: operatorModeStatus?.mode || 'assist',
    operatorModeStatus,
    mixerType,
    wingSafetyQuarantine: mixerType === 'wing',
  };

  const navItems = NAV_ITEMS.map(item => (
    mixerType === 'wing' && WING_QUARANTINED_TABS.has(item.id)
      ? { ...item, quarantined: true }
      : item
  ));

  const operatorMode = operatorModeStatus?.mode || 'assist';
  const operatorModeLabel = operatorModeStatus?.label || OPERATOR_MODES.find(item => item.key === operatorMode)?.label || 'Assist';
  const operatorModeCapabilities = operatorModeStatus?.capabilities || {};
  const operatorModeCanApply = operatorModeCapabilities.can_apply_to_console === true;
  const proposalActions = {
    onAccept: (proposalId) => websocketService.acceptOperatorProposal(proposalId),
    onDismiss: (proposalId) => websocketService.dismissOperatorProposal(proposalId),
    onAnalyze: () => websocketService.runOperatorAnalysis(24, true),
    onImportSafeGain: () => websocketService.importSafeGainSuggestions(),
    onImportSoundcheck: () => websocketService.importSoundcheckRecommendations(),
    onApply: (proposalId) => {
      if (!window.confirm('Применить это предложение через supervised gate?')) return;
      websocketService.applyOperatorProposal(proposalId, true);
    },
  };

  const selectTab = (tabId) => {
    setActiveTab(tabId);
    window.history.replaceState(null, '', `#/${tabId}`);
  };

  return (
    <div className="App">
      {/* Sidebar */}
      <nav className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <h1 className="logo">{sidebarCollapsed ? 'AM' : <><span>AutoMixer</span><small>Tubeslave</small></>}</h1>
          <button className="btn-collapse" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
            {sidebarCollapsed ? '▶' : '◀'}
          </button>
        </div>

        <div className="nav-items">
          {navItems.map((item, index) => {
            const previous = navItems[index - 1];
            const showSection = !sidebarCollapsed && item.section !== previous?.section;
            return (
              <React.Fragment key={item.id}>
                {showSection && <div className="nav-section-label">{item.section}</div>}
                <button
                  className={`nav-item ${activeTab === item.id ? 'active' : ''} ${item.quarantined ? 'quarantined' : ''}`}
                  onClick={() => selectTab(item.id)}
                  title={item.quarantined ? `${item.label} • WING на паузе` : item.label}
                >
                  <span className="nav-icon">{item.icon}</span>
                  {!sidebarCollapsed && (
                    <>
                      <span className="nav-label">{item.label}</span>
                      {item.quarantined && <span className="nav-pill">WING ПАУЗА</span>}
                    </>
                  )}
                </button>
              </React.Fragment>
            );
          })}
        </div>
        {!sidebarCollapsed && (
          <div className="sidebar-health">
            <strong>System Health</strong>
            <span className="state-green">✓ All systems normal</span>
            <p>CPU: 18% · RAM: 32%</p>
            <p>DISK: 512 GB free</p>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* Top bar with session and safety indicators */}
        <div className="topbar">
          <div className="topbar-session">
            <span>Session</span>
            <strong>2025-05-25 · Live in Club · Soundcheck</strong>
            <small>Live in Club · Soundcheck</small>
          </div>
          <div className="topbar-status-grid">
            <div className="topbar-status-cell">
              <span>Status</span>
              <strong className={serverConnected ? 'status-green' : 'status-red'}>
                {serverConnected ? 'Live' : 'Offline'}
              </strong>
            </div>
            <div className="topbar-status-cell">
              <span>Mode</span>
              <strong className={`mode-text-${operatorMode}`}>{operatorModeLabel}</strong>
            </div>
            <div className="topbar-status-cell">
              <span>Safe</span>
              <strong className={operatorModeCanApply ? 'status-yellow' : 'status-green'}>
                {operatorModeCanApply ? 'Armed' : 'On'}
              </strong>
            </div>
            <div className="topbar-status-cell">
              <span>Connected to</span>
              <strong>{mixerConnected ? 'WING RACK' : 'Safe Offline'}</strong>
            </div>
          </div>
          <div className="topbar-user">
            <button title="Аудио">▥</button>
            <button title="Снимок">□</button>
            <button title="Настройки">⚙</button>
            <div><strong>Director</strong><span>Automixer</span></div>
          </div>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'dashboard' && (
            <DashboardView
              snapshot={dashboardSnapshot}
              channelInventory={channelInventory}
              decisionQueue={decisionQueue}
              operatorModeStatus={operatorModeStatus}
              proposalActions={proposalActions}
              operatorAnalysisReport={operatorAnalysisReport}
            />
          )}
          {activeTab === 'agentsConsole' && (
            <AgentsConsoleView
              decisionQueue={decisionQueue}
              operatorModeStatus={operatorModeStatus}
              proposalActions={proposalActions}
              operatorAnalysisReport={operatorAnalysisReport}
            />
          )}
          {activeTab === 'channelList' && (
            <ChannelListView
              inventory={channelInventory}
            />
          )}
          {activeTab === 'connections' && (
            <ConnectionsView
              serverConnected={serverConnected}
              mixerConnected={mixerConnected}
              topology={connectionTopology}
              statusMessage={statusMessage}
              mixerIp={mixerIp}
              mixerPort={mixerPort}
              setMixerIp={setMixerIp}
              setMixerPort={setMixerPort}
              handleConnect={handleConnect}
              connecting={connecting}
            />
          )}
          {activeTab === 'mixer' && (
            <div className="connect-page">
              <div className="connect-card">
                <div className="connect-row">
                  <div className="field">
                    <label>Wing IP</label>
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

              <PilotDeploymentCockpit
                status={pilotCockpit}
                onRefresh={() => websocketService.getSupervisedPilotStatus()}
              />

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
              onMixerSettingsChange={(key, value) => {
                if (key === 'mixerType') setMixerType(value);
                if (key === 'mixerIp') setMixerIp(value);
                if (key === 'mixerPort') setMixerPort(value);
              }}
            />
          )}
        </div>
        <TransportBar />
      </main>
    </div>
  );
}

export default App;
