import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './VoiceControlTab.css';

function VoiceControlTab() {
  const [voiceControlActive, setVoiceControlActive] = useState(false);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [selectedChannel, setSelectedChannel] = useState(1);
  const [availableChannels, setAvailableChannels] = useState([]);
  const [statusMessage, setStatusMessage] = useState('');
  const [modelSize, setModelSize] = useState('small');
  const [language, setLanguage] = useState('ru');
  const [lastCommand, setLastCommand] = useState(null);

  // Debug: log state changes
  useEffect(() => {
    console.log('VoiceControlTab state:', {
      selectedDeviceId,
      audioDevicesCount: audioDevices.length,
      voiceControlActive
    });
  }, [selectedDeviceId, audioDevices.length, voiceControlActive]);

  useEffect(() => {
    console.log('VoiceControlTab mounted, requesting audio devices...');
    // Request audio devices list
    websocketService.send({ type: 'get_audio_devices' });

    // Handle audio devices list
    const handleAudioDevices = (data) => {
      console.log('Received audio devices:', data);
      setAudioDevices(data.devices || []);
      if (data.devices && data.devices.length > 0) {
        const firstDevice = data.devices[0];
        console.log('Setting first device:', firstDevice);
        setSelectedDeviceId(firstDevice.id);
        setAvailableChannels(firstDevice.channels || []);
        if (firstDevice.channels && firstDevice.channels.length > 0) {
          setSelectedChannel(firstDevice.channels[0].id);
        }
      } else {
        console.warn('No audio devices found');
      }
    };

    // Handle voice control status
    const handleVoiceControlStatus = (data) => {
      console.log('Received voice_control_status:', data);
      setVoiceControlActive(data.active || false);
      if (data.message) {
        setStatusMessage(data.message);
      }
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setVoiceControlActive(false);
      }
    };

    // Handle voice command executed
    const handleVoiceCommandExecuted = (data) => {
      setLastCommand({
        type: data.command,
        channel: data.channel,
        db_change: data.db_change,
        new_db: data.new_db,
        value: data.value,
        timestamp: new Date().toLocaleTimeString()
      });
      setStatusMessage(`Command executed: ${data.command}${data.channel ? ` (channel ${data.channel})` : ''}`);
    };

    // Handle voice command error
    const handleVoiceCommandError = (data) => {
      setStatusMessage(`Command error: ${data.error}`);
    };

    const handleAllSettingsLoaded = (data) => {
      if (data.settings && data.settings.voiceControl) {
        const vc = data.settings.voiceControl;
        if (vc.modelSize) setModelSize(vc.modelSize);
        if (vc.language !== undefined) setLanguage(vc.language);
        console.log('VoiceControl: Applied saved defaults:', vc);
      }
    };

    websocketService.on('audio_devices', handleAudioDevices);
    websocketService.on('voice_control_status', handleVoiceControlStatus);
    websocketService.on('voice_command_executed', handleVoiceCommandExecuted);
    websocketService.on('voice_command_error', handleVoiceCommandError);
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);

    // Check initial status
    websocketService.getVoiceControlStatus();
    websocketService.loadAllSettings();

    return () => {
      websocketService.off('audio_devices', handleAudioDevices);
      websocketService.off('voice_control_status', handleVoiceControlStatus);
      websocketService.off('voice_command_executed', handleVoiceCommandExecuted);
      websocketService.off('voice_command_error', handleVoiceCommandError);
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
    };
  }, []);

  const handleDeviceChange = (deviceId) => {
    setSelectedDeviceId(deviceId);
    const device = audioDevices.find(d => d.id === deviceId);
    if (device) {
      setAvailableChannels(device.channels || []);
      if (device.channels && device.channels.length > 0) {
        setSelectedChannel(device.channels[0].id);
      }
    }
  };

  const handleStartVoiceControl = () => {
    if (!selectedDeviceId) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    const params = {
      modelSize,
      language: language || null, // Convert empty string to null for auto-detect
      deviceId: selectedDeviceId,
      channel: selectedChannel - 1
    };
    
    console.log('Starting voice control with params:', params);
    setStatusMessage('Starting voice control...');
    
    try {
      websocketService.startVoiceControl(
        params.modelSize, 
        params.language, 
        params.deviceId, 
        params.channel
      );
    } catch (error) {
      console.error('Error starting voice control:', error);
      setStatusMessage(`Error: ${error.message}`);
    }
  };

  const handleStopVoiceControl = () => {
    setStatusMessage('Stopping voice control...');
    websocketService.stopVoiceControl();
  };

  const selectedDevice = audioDevices.find(d => d.id === selectedDeviceId);

  return (
    <div className="voice-control-tab">
      <div className="voice-control-section">
        <h2>VOICE CONTROL</h2>
        
        <div className="config-grid">
          <div className="config-item full-width">
            <label>Audio Device</label>
            <select
              value={selectedDeviceId}
              onChange={(e) => handleDeviceChange(e.target.value)}
              disabled={voiceControlActive || audioDevices.length === 0}
            >
              {audioDevices.length === 0 ? (
                <option value="">No audio devices found</option>
              ) : (
                audioDevices.map(device => (
                  <option key={device.id} value={device.id}>
                    {device.name} ({device.max_channels} channels)
                  </option>
                ))
              )}
            </select>
          </div>

          {selectedDevice && availableChannels.length > 0 && (
            <div className="config-item full-width">
              <label>Input Channel</label>
              <select
                value={selectedChannel}
                onChange={(e) => setSelectedChannel(parseInt(e.target.value))}
                disabled={voiceControlActive}
              >
                {availableChannels.map(channel => (
                  <option key={channel.id} value={channel.id}>
                    {channel.name || `Channel ${channel.id}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="config-item">
            <label>Model Size</label>
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value)}
              disabled={voiceControlActive}
            >
              <option value="tiny">Tiny (fastest)</option>
              <option value="base">Base</option>
              <option value="small">Small (recommended)</option>
              <option value="medium">Medium</option>
              <option value="large">Large (best quality)</option>
            </select>
          </div>

          <div className="config-item">
            <label>Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={voiceControlActive}
            >
              <option value="ru">Russian</option>
              <option value="en">English</option>
              <option value="">Auto-detect</option>
            </select>
          </div>
        </div>

        <div className="voice-control-status">
          <div className={`status-indicator ${voiceControlActive ? 'active' : 'inactive'}`}>
            <span className="status-dot"></span>
            <span>{voiceControlActive ? 'Voice Control Active' : 'Voice Control Inactive'}</span>
          </div>
        </div>

        <div className="voice-control-actions">
          <button
            className={`btn-voice-control ${voiceControlActive ? 'stop' : 'start'}`}
            onClick={voiceControlActive ? handleStopVoiceControl : handleStartVoiceControl}
            disabled={!selectedDeviceId || audioDevices.length === 0}
            title={!selectedDeviceId ? 'Please select an audio device' : ''}
          >
            {voiceControlActive ? 'Stop Voice Control' : 'Start Voice Control'}
          </button>
          {!selectedDeviceId && (
            <div style={{marginTop: '10px', color: '#ff4444', fontSize: '0.9em'}}>
              ⚠️ Please select an audio device first
            </div>
          )}
        </div>

        {statusMessage && (
          <div className="status-message">
            {statusMessage}
          </div>
        )}

        {lastCommand && (
          <div className="last-command">
            <h3>Last Command</h3>
            <div className="command-info">
              <span className="command-type">{lastCommand.type}</span>
              {lastCommand.channel && (
                <span className="command-channel">Channel {lastCommand.channel}</span>
              )}
              {lastCommand.db_change !== undefined && (
                <span className="command-db">
                  {lastCommand.type === 'volume_up' ? '+' : '-'}{lastCommand.db_change} dB
                </span>
              )}
              {lastCommand.new_db !== undefined && (
                <span className="command-value">→ {lastCommand.new_db.toFixed(1)} dB</span>
              )}
              <span className="command-time">{lastCommand.timestamp}</span>
            </div>
          </div>
        )}

        <div className="voice-control-help">
          <h3>Supported Commands</h3>
          <div className="commands-list">
            <div className="command-group">
              <h4>Russian:</h4>
              <ul>
                <li><code>канал 1</code> - Set fader for channel 1</li>
                <li><code>гейн 3</code> - Set gain for channel 3</li>
                <li><code>загрузить концерт</code> - Load snapshot "концерт"</li>
                <li><code>мут 2</code> - Mute channel 2</li>
                <li><code>громче 4</code> - Increase volume for channel 4</li>
                <li><code>тише 6</code> - Decrease volume for channel 6</li>
              </ul>
            </div>
            <div className="command-group">
              <h4>English:</h4>
              <ul>
                <li><code>channel 10</code> - Set fader for channel 10</li>
                <li><code>gain 5</code> - Set gain for channel 5</li>
                <li><code>load test</code> - Load snapshot "test"</li>
                <li><code>mute 1</code> - Mute channel 1</li>
                <li><code>louder 2</code> - Increase volume for channel 2</li>
                <li><code>quieter 3</code> - Decrease volume for channel 3</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VoiceControlTab;
