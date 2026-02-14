import React, { useState, useEffect } from 'react';
import './App.css';
import websocketService from './services/websocket';
import VoiceControlTab from './components/VoiceControlTab';
import GainStagingTab from './components/GainStagingTab';
import AutoEQTab from './components/AutoEQTab';
import PhaseAlignmentTab from './components/PhaseAlignmentTab';
import AutoFaderTab from './components/AutoFaderTab';
import AutoSoundcheckTab from './components/AutoSoundcheckTab';
import AutoCompressorTab from './components/AutoCompressorTab';
import SettingsTab from './components/SettingsTab';

function App() {
  // Connection state
  const [serverConnected, setServerConnected] = useState(false);
  const [mixerConnected, setMixerConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  
  // Mixer settings
  const [mixerIp, setMixerIp] = useState('192.168.1.102');
  const [mixerPort, setMixerPort] = useState(2223);
  
  // Audio device settings
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [availableChannels, setAvailableChannels] = useState([]);
  const [selectedChannels, setSelectedChannels] = useState([]);
  
  // Connection in progress
  const [connecting, setConnecting] = useState(false);
  
  // Tab management
  const [activeTab, setActiveTab] = useState('mixer');

  useEffect(() => {
    // Register listeners FIRST before connecting
    
    // Handle connection status updates
    websocketService.on('connection_status', (data) => {
      setConnecting(false);
      setMixerConnected(data.connected);
      if (data.connected) {
        setStatusMessage(`Connected to Wing at ${mixerIp}`);
      } else {
        if (data.error) {
          setStatusMessage(`Error: ${data.error}`);
        } else {
          setStatusMessage('Disconnected from mixer');
        }
      }
    });

    // Handle audio devices list
    websocketService.on('audio_devices', (data) => {
      setAudioDevices(data.devices || []);
      if (data.devices && data.devices.length > 0) {
        // Try to find Dante device first
        const danteDevice = data.devices.find(device => 
          device.name && device.name.toLowerCase().includes('dante')
        );
        
        // Select Dante if found, otherwise select first device
        const deviceToSelect = danteDevice || data.devices[0];
        setSelectedDevice(deviceToSelect.id);
        setAvailableChannels(deviceToSelect.channels || []);
      }
    });

    // Handle bypass result
    websocketService.on('bypass_result', (data) => {
      if (data.error) {
        setStatusMessage(`Bypass error: ${data.error}`);
      } else if (data.success) {
        setStatusMessage(`Bypass completed: ${data.success_count}/40 channels processed`);
      } else {
        setStatusMessage('Bypass operation failed');
      }
    });

    // Helper function to check if channel name is a default name (not an instrument/vocal name)
    const isDefaultChannelName = (name) => {
      if (!name || !name.trim()) return true;
      const trimmedName = name.trim();
      
      // Check for default patterns: "Ch 1", "Ch 2", "Channel 1", "Channel 2", etc.
      const defaultPatterns = [
        /^ch\s*\d+$/i,           // "Ch 1", "ch 2", "CH 3", etc.
        /^channel\s*\d+$/i,      // "Channel 1", "channel 2", etc.
        /^\d+$/,                 // Just numbers "1", "2", "3", etc.
        /^input\s*\d+$/i,        // "Input 1", "input 2", etc.
        /^in\s*\d+$/i,           // "In 1", "in 2", etc.
      ];
      
      return defaultPatterns.some(pattern => pattern.test(trimmedName));
    };

    // Handle mixer channel names scan result
    websocketService.on('mixer_channel_names', (data) => {
      if (data.error) {
        setStatusMessage(`Error scanning mixer: ${data.error}`);
        return;
      }
      
      if (data.channel_names) {
        // Update channel names in availableChannels
        setAvailableChannels(prevChannels => {
          const updatedChannels = prevChannels.map(channel => {
            // Try to match by channel number
            // channel.id might be a number or string like "1", "2", etc.
            const channelNum = typeof channel.id === 'number' ? channel.id : parseInt(channel.id);
            
            // Try both numeric and string keys (JSON may serialize numeric keys as strings)
            let newName = null;
            if (!isNaN(channelNum)) {
              // Try numeric key first
              newName = data.channel_names[channelNum];
              // If not found, try string key
              if (!newName) {
                newName = data.channel_names[String(channelNum)];
              }
            }
            
            if (newName && newName.trim()) {
              return {
                ...channel,
                name: newName.trim()
              };
            }
            
            // Also try matching by channel name if it contains a number
            const nameMatch = channel.name?.match(/(\d+)/);
            if (nameMatch) {
              const numFromName = parseInt(nameMatch[1]);
              if (!isNaN(numFromName)) {
                // Try both numeric and string keys
                let nameFromMatch = data.channel_names[numFromName] || data.channel_names[String(numFromName)];
                if (nameFromMatch && nameFromMatch.trim()) {
                  return {
                    ...channel,
                    name: nameFromMatch.trim()
                  };
                }
              }
            }
            return channel;
          });
          
          // Automatically select channels with instrument/vocal names (not default names)
          const channelsWithNames = updatedChannels.filter(channel => {
            const channelName = channel.name || '';
            return !isDefaultChannelName(channelName);
          });
          
          if (channelsWithNames.length > 0) {
            const channelIds = channelsWithNames.map(ch => ch.id);
            setSelectedChannels(channelIds);
            const count = Object.keys(data.channel_names).length;
            setStatusMessage(`Scanned ${count} channel names. Auto-selected ${channelsWithNames.length} channels with instrument/vocal names.`);
          } else {
            const count = Object.keys(data.channel_names).length;
            setStatusMessage(`Scanned ${count} channel names from mixer. No channels with custom names found.`);
          }
          
          return updatedChannels;
        });
      } else {
        setStatusMessage('No channel names received from mixer');
      }
    });

    // Handle disconnection
    websocketService.on('disconnected', () => {
      setServerConnected(false);
      setMixerConnected(false);
      setStatusMessage('Backend connection lost. Reconnecting...');
    });

    // Handle saved settings for mixer connection
    const handleAllSettingsLoaded = (data) => {
      if (data.settings && data.settings.mixer) {
        const m = data.settings.mixer;
        if (m.mixerIp) setMixerIp(m.mixerIp);
        if (m.mixerPort) setMixerPort(m.mixerPort);
        console.log('App: Applied saved mixer settings:', m);
      }
    };
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);

    // NOW connect to backend WebSocket (after all listeners are registered)
    websocketService.connect()
      .then(() => {
        setServerConnected(true);
        setStatusMessage('Backend connected');
        // Request audio devices list
        websocketService.send({ type: 'get_audio_devices' });
        // Load saved settings
        websocketService.loadAllSettings();
      })
      .catch(err => {
        setStatusMessage('Failed to connect to backend');
        console.error(err);
      });

    return () => {
      // Clean up event listeners
      websocketService.off('connection_status', () => {});
      websocketService.off('audio_devices', () => {});
      websocketService.off('bypass_result', () => {});
      websocketService.off('mixer_channel_names', () => {});
      websocketService.off('disconnected', () => {});
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
      websocketService.disconnect();
    };
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
    setSelectedChannels(prev => {
      if (prev.includes(channelId)) {
        return prev.filter(id => id !== channelId);
      } else {
        return [...prev, channelId];
      }
    });
  };

  const handleSelectAllChannels = () => {
    if (selectedChannels.length === availableChannels.length) {
      setSelectedChannels([]);
    } else {
      setSelectedChannels(availableChannels.map(ch => ch.id));
    }
  };

  const handleScanMixer = () => {
    // #region agent log
    fetch('http://127.0.0.1:7249/ingest/4264ed61-ddd5-4beb-978e-d0eb0972f907',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'App.js:handleScanMixer',message:'Function called',data:{mixerConnected},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
    // #endregion
    if (!mixerConnected) {
      setStatusMessage('Please connect to mixer first');
      return;
    }
    setStatusMessage('Scanning mixer channel names...');
    websocketService.scanMixerChannelNames();
  };

  const handleBypass = () => {
    if (!mixerConnected) {
      setStatusMessage('Please connect to mixer first');
      return;
    }
    if (!window.confirm('This will disable all modules and set all faders to 0dB on all 40 channels. Continue?')) {
      return;
    }
    setStatusMessage('Bypassing mixer (disabling modules and setting faders to 0dB)...');
    websocketService.bypassMixer();
  };

  const handleConnect = () => {
    if (mixerConnected) {
      websocketService.disconnectMixer();
    } else {
      setConnecting(true);
      setStatusMessage('Connecting to Wing...');
      websocketService.connectWing(mixerIp, mixerPort, mixerPort);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Auto Mixer Tubeslave</h1>
        <div className="status-indicators">
          <span className={`indicator ${serverConnected ? 'connected' : 'disconnected'}`}>
            Backend: {serverConnected ? 'Online' : 'Offline'}
          </span>
          <span className={`indicator ${mixerConnected ? 'connected' : 'disconnected'}`}>
            Mixer: {mixerConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      <main className="App-main">
        {/* Tabs Navigation */}
        <div className="tabs-navigation">
          <button
            className={`tab-button ${activeTab === 'mixer' ? 'active' : ''}`}
            onClick={() => setActiveTab('mixer')}
          >
            Mixer Connection
          </button>
          <button
            className={`tab-button ${activeTab === 'gainStaging' ? 'active' : ''}`}
            onClick={() => setActiveTab('gainStaging')}
          >
            GAIN STAGING
          </button>
          <button
            className={`tab-button ${activeTab === 'phaseAlignment' ? 'active' : ''}`}
            onClick={() => setActiveTab('phaseAlignment')}
          >
            PHASE ALIGNMENT
          </button>
          <button
            className={`tab-button ${activeTab === 'autoEQ' ? 'active' : ''}`}
            onClick={() => setActiveTab('autoEQ')}
          >
            AUTO EQ
          </button>
          <button
            className={`tab-button ${activeTab === 'autoCompressor' ? 'active' : ''}`}
            onClick={() => setActiveTab('autoCompressor')}
          >
            AUTO COMPRESSOR
          </button>
          <button
            className={`tab-button ${activeTab === 'autoFader' ? 'active' : ''}`}
            onClick={() => setActiveTab('autoFader')}
          >
            AUTO FADER
          </button>
          <button
            className={`tab-button ${activeTab === 'autoSoundcheck' ? 'active' : ''}`}
            onClick={() => setActiveTab('autoSoundcheck')}
          >
            AUTO SOUNDCHECK
          </button>
          <button
            className={`tab-button ${activeTab === 'voice' ? 'active' : ''}`}
            onClick={() => setActiveTab('voice')}
          >
            VOICE CONTROL
          </button>
          <button
            className={`tab-button ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            SETTINGS
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'mixer' && (
          <>
            <section className="config-section">
              <h2>Mixer Connection</h2>
          
          <div className="config-grid">
            <div className="config-item">
              <label>Wing IP Address</label>
              <input
                type="text"
                value={mixerIp}
                onChange={(e) => setMixerIp(e.target.value)}
                disabled={mixerConnected}
                placeholder="192.168.1.102"
              />
            </div>
            
            <div className="config-item">
              <label>OSC Port</label>
              <input
                type="number"
                value={mixerPort}
                onChange={(e) => setMixerPort(parseInt(e.target.value))}
                disabled={mixerConnected}
              />
            </div>
          </div>
        </section>

        <section className="config-section">
          <h2>Audio Device</h2>
          
          <div className="config-grid">
            <div className="config-item full-width">
              <label>Select Audio Device</label>
              <select
                value={selectedDevice}
                onChange={(e) => handleDeviceChange(e.target.value)}
                disabled={audioDevices.length === 0}
              >
                {audioDevices.length === 0 ? (
                  <option value="">No audio devices found</option>
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

          {availableChannels.length > 0 && (
            <div className="channels-section">
              <div className="channels-header">
                <label>Select Channels to Process</label>
                <div className="channels-header-buttons">
                  <button 
                    className="btn-small"
                    onClick={handleSelectAllChannels}
                  >
                    {selectedChannels.length === availableChannels.length ? 'Deselect All' : 'Select All'}
                  </button>
                  <button 
                    className="btn-small"
                    onClick={handleBypass}
                    disabled={!mixerConnected}
                    title={mixerConnected ? "Disable all modules and set faders to 0dB" : "Connect to mixer first"}
                  >
                    Bypass
                  </button>
                  <button 
                    className="btn-small"
                    onClick={handleScanMixer}
                    disabled={!mixerConnected}
                    title={mixerConnected ? "Scan channel names from mixer" : "Connect to mixer first"}
                  >
                    Scan Mixer
                  </button>
                </div>
              </div>
              
              <div className="channels-grid">
                {availableChannels.map(channel => (
                  <label key={channel.id} className="channel-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedChannels.includes(channel.id)}
                      onChange={() => handleChannelToggle(channel.id)}
                    />
                    <span>{channel.name || `Ch ${channel.id}`}</span>
                  </label>
                ))}
              </div>
              
              <div className="selected-count">
                {selectedChannels.length} of {availableChannels.length} channels selected
              </div>
            </div>
          )}
        </section>

        <section className="actions-section">
          <button
            className={`btn-connect ${mixerConnected ? 'connected' : ''}`}
            onClick={handleConnect}
            disabled={!serverConnected || connecting}
          >
            {connecting ? 'Connecting...' : mixerConnected ? 'Disconnect' : 'Connect to Wing'}
          </button>
          
          <p className="status-message">{statusMessage}</p>
        </section>
          </>
        )}

        {activeTab === 'gainStaging' && (
          <GainStagingTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'phaseAlignment' && (
          <PhaseAlignmentTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'autoEQ' && (
          <AutoEQTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'autoFader' && (
          <AutoFaderTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'autoSoundcheck' && (
          <AutoSoundcheckTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'autoCompressor' && (
          <AutoCompressorTab 
            selectedChannels={selectedChannels}
            availableChannels={availableChannels}
            selectedDevice={selectedDevice}
            audioDevices={audioDevices}
          />
        )}

        {activeTab === 'voice' && (
          <VoiceControlTab />
        )}

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
      </main>
    </div>
  );
}

export default App;
