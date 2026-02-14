import React, { useState, useEffect, useCallback } from 'react';
import websocketService from '../services/websocket';
import './AutoFaderTab.css';

// Instrument type presets for balance
const INSTRUMENT_TYPES = {
  leadVocal: { name: 'Lead Vocal', isReference: true },
  backVocal: { name: 'Back Vocal', isReference: false },
  kick: { name: 'Kick', isReference: false },
  snare: { name: 'Snare', isReference: false },
  toms: { name: 'Toms', isReference: false },
  drums: { name: 'Drums / Kit', isReference: false },
  hihat: { name: 'Hi-Hat', isReference: false },
  ride: { name: 'Ride', isReference: false },
  overhead: { name: 'Overhead', isReference: false },
  room: { name: 'Room', isReference: false },
  bass: { name: 'Bass', isReference: false },
  guitar: { name: 'Guitar', isReference: false },
  acousticGuitar: { name: 'Acoustic Guitar', isReference: false },
  keys: { name: 'Keys / Piano', isReference: false },
  synth: { name: 'Synth', isReference: false },
  brass: { name: 'Brass', isReference: false },
  strings: { name: 'Strings', isReference: false },
  accordion: { name: 'Accordion', isReference: false },
  playback: { name: 'Playback', isReference: false },
  custom: { name: 'Custom', isReference: false }
};

// Genre profiles
const GENRE_PROFILES = {
  custom: { name: 'Custom', description: 'Manual settings' },
  pop_rock: { name: 'Pop/Rock', description: 'High vocal priority, heavy compression' },
  jazz: { name: 'Jazz', description: 'Wide dynamics, natural balance' },
  electronic: { name: 'Electronic', description: 'Punchy bass, controlled dynamics' },
  acoustic: { name: 'Acoustic', description: 'Natural sound, moderate dynamics' },
  classical: { name: 'Classical', description: 'Maximum dynamics, minimal processing' }
};

// Default settings
const DEFAULT_SETTINGS = {
  targetLufs: -18,
  ratio: 2,
  maxAdjustmentDb: 6,
  attackMs: 100,
  releaseMs: 1000,
  holdMs: 500,
  autoBalanceDuration: 15,
  bleedThreshold: -50
};

// Function to recognize instrument type from channel name
// Based on backend channel_recognizer.py patterns
const recognizeInstrumentType = (channelName) => {
  if (!channelName || !channelName.trim()) {
    return 'custom';
  }
  
  const nameLower = channelName.toLowerCase().trim();
  
  // Drums - specific (check more specific patterns first)
  // Room mic (check before generic patterns)
  if (/\b(room|рум)\b/i.test(nameLower)) {
    return 'room';
  }
  // Overheads (check before generic "oh" patterns)
  if (/\b(ohl|ohr|over[\s-]?head|overhead|оверхэд)\b/i.test(nameLower)) {
    return 'overhead';
  }
  // Snare (check before generic patterns)
  if (/\b(snare|sd|sn|малый|снэйр|снейр)\b/i.test(nameLower)) {
    return 'snare';
  }
  // Kick (check before generic drum patterns)
  if (/\b(kick|bd|bass\s*drum|бас[\s-]?бочка|бочка|кик)\b/i.test(nameLower)) {
    return 'kick';
  }
  // Toms (M TOM, F TOM, etc.) - separate type
  if (/\b(m\s*tom|f\s*tom|tom|том|floor\s*tom|флор\s*том|флор|rack\s*tom|рэк\s*том)\b/i.test(nameLower)) {
    return 'toms';
  }
  // Hi-Hat - separate type
  if (/\b(hi[\s-]?hat|hh|хай[\s-]?хэт|хэт)\b/i.test(nameLower)) {
    return 'hihat';
  }
  // Ride - separate type
  if (/\b(ride|райд)\b/i.test(nameLower)) {
    return 'ride';
  }
  // Cymbals (crash, splash, china) - map to drums
  if (/\b(crash|splash|china|крэш|сплэш|чайна|cymbal|тарелк)\b/i.test(nameLower)) {
    return 'drums';
  }
  // Generic drums
  if (/\b(drums|drum|ударн|oh\b)\b/i.test(nameLower)) {
    return 'drums';
  }
  
  // Bass
  if (/\b(bass|бас|sub|саб)\b(?![\s-]?(drum|бочка))/i.test(nameLower)) {
    return 'bass';
  }
  
  // Guitars
  if (/\b(acoustic|акустик|акуст[\s-]?гитар|акуст|agtr)\b/i.test(nameLower)) {
    return 'acousticGuitar';
  }
  if (/\b(electric|электро|egtr|e[\s-]?gtr|gtr|гитар|guitar)\b/i.test(nameLower)) {
    return 'guitar';
  }
  
  // Keys & Instruments
  if (/\b(accordion|accord|bayan|баян|аккордеон|гармонь|гармошка)\b/i.test(nameLower)) {
    return 'accordion';
  }
  if (/\b(synth|keys|keyboard|piano|клавиш|синт|пиано|орган|organ|rhodes|wurli)\b/i.test(nameLower)) {
    return 'synth';
  }
  if (/\b(playback|pb|track|backing|минус|фонограмма|плейбэк|трек)\b/i.test(nameLower)) {
    return 'playback';
  }
  
  // Vocals - check for lead vocals first
  if (/\b(lead\s*vox|lead\s*vocal|лид[\s-]?вок|main\s*vox|solo\s*vox|соло[\s-]?вок|vox\s*1|вок\s*1)\b/i.test(nameLower)) {
    return 'leadVocal';
  }
  if (/\b(back[\s-]?vox|backing[\s-]?vox|bvox|бэк[\s-]?вок|choir|хор|bgv|vox\s*[2-9]|вок\s*[2-9])\b/i.test(nameLower)) {
    return 'backVocal';
  }
  // Common names that should be recognized as vocals
  if (/\b(katya|катя|sergey|сергей|slava|слава|dima|дима|masha|маша|sasha|саша|pasha|паша|vova|вова|andrey|андрей|alex|алекс|misha|миша|natasha|наташа|olga|ольга|tanya|таня|vlad|влад|ivan|иван|max|макс|nikita|никита|dasha|даша|anya|аня|lena|лена|maria|мария|anna|анна|elena|елена)\b/i.test(nameLower)) {
    return 'leadVocal';
  }
  if (/\b(vox|vocal|вокал|голос|voice|mic|микрофон)\b/i.test(nameLower)) {
    return 'leadVocal'; // Generic vocal defaults to lead
  }
  
  // Brass and Strings
  if (/\b(brass|труб|тромбон|тромбон|horn|рог)\b/i.test(nameLower)) {
    return 'brass';
  }
  if (/\b(strings|струн|violin|скрипк|viola|альт|cello|виолончел)\b/i.test(nameLower)) {
    return 'strings';
  }
  
  return 'custom';
};

function AutoFaderTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  // Mode: 'realtime' or 'static'
  const [mode, setMode] = useState('realtime');
  
  // Channel settings
  const [channelSettings, setChannelSettings] = useState({});
  const [referenceChannel, setReferenceChannel] = useState(null);
  
  // Status
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);
  const [autoBalanceCollecting, setAutoBalanceCollecting] = useState(false);
  const [autoBalanceProgress, setAutoBalanceProgress] = useState(0);
  const [autoBalanceReady, setAutoBalanceReady] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  
  // Measured levels
  const [measuredLevels, setMeasuredLevels] = useState({});
  
  // Auto balance results
  const [autoBalanceResult, setAutoBalanceResult] = useState({});
  
  // Settings
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [profile, setProfile] = useState('custom');
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  // WebSocket event handlers
  useEffect(() => {
    const handleAutoFaderStatus = (data) => {
      console.log('Auto fader status update:', data);
      
      if (data.realtime_enabled !== undefined) {
        setRealtimeEnabled(data.realtime_enabled);
      }
      
      if (data.auto_balance_collecting !== undefined) {
        setAutoBalanceCollecting(data.auto_balance_collecting);
      }
      
      if (data.message) {
        setStatusMessage(data.message);
      }
      
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setRealtimeEnabled(false);
        setAutoBalanceCollecting(false);
      }
      
      // Handle specific status types
      switch (data.status_type) {
        case 'realtime_fader_started':
          setRealtimeEnabled(true);
          setStatusMessage('Real-Time Fader started');
          break;
        case 'realtime_fader_stopped':
          setRealtimeEnabled(false);
          setStatusMessage('Real-Time Fader stopped');
          break;
        case 'auto_balance_started':
          setAutoBalanceCollecting(true);
          setAutoBalanceReady(false);
          setAutoBalanceProgress(0);
          setStatusMessage(`Collecting audio data for ${data.duration || 15} seconds...`);
          break;
        case 'auto_balance_ready':
          setAutoBalanceCollecting(false);
          setAutoBalanceReady(true);
          setAutoBalanceResult(data.result || {});
          setStatusMessage('Auto Balance ready to apply');
          break;
        case 'auto_balance_applied':
          setStatusMessage(`Auto Balance applied to ${data.applied_count}/${data.total_count} channels`);
          break;
        case 'auto_balance_cancelled':
          setAutoBalanceCollecting(false);
          setAutoBalanceReady(false);
          setAutoBalanceProgress(0);
          setStatusMessage('Auto Balance cancelled');
          break;
        default:
          break;
      }
      
      // Handle levels update
      if (data.status_type === 'levels_update' && data.channels) {
        const newLevels = {};
        Object.entries(data.channels).forEach(([audioChStr, chData]) => {
          const audioChannel = parseInt(audioChStr);
          newLevels[audioChannel] = {
            lufs: chData.lufs ?? -60,
            avgLufs: chData.avg_lufs ?? -60,
            truePeak: chData.true_peak ?? -60,
            currentFader: chData.current_fader ?? 0,
            targetFader: chData.target_fader ?? 0,
            correction: chData.correction ?? 0,
            isActive: chData.is_active || false,
            isReference: chData.is_reference || false,
            status: chData.status ?? 'idle',
            progress: chData.progress ?? 0,
            bleedRatio: chData.bleed_ratio ?? 0,
            bleedSource: chData.bleed_source ?? null
          };
          
          // Update progress for auto balance
          if (chData.progress !== undefined) {
            setAutoBalanceProgress(chData.progress * 100);
          }
        });
        setMeasuredLevels(newLevels);
      }
    };
    
    const handleMixerChannelNames = (data) => {
      if (data.error || !data.channel_names) {
        return;
      }
      
      console.log('AutoFaderTab: Received mixer channel names', data.channel_names);
      
      // Auto-recognize instrument types from channel names
      setChannelSettings(prev => {
        const updated = { ...prev };
        let foundReference = false;
        let recognizedCount = 0;
        
        // Process all selected channels
        selectedChannels.forEach(channelId => {
          // Get channel name directly from scan result
          const channelNum = typeof channelId === 'number' ? channelId : parseInt(channelId);
          let channelName = null;
          
          if (!isNaN(channelNum)) {
            // Try both numeric and string keys
            channelName = data.channel_names[channelNum] || data.channel_names[String(channelNum)];
          }
          
          // If we have a channel name, recognize instrument type
          if (channelName && channelName.trim()) {
            const recognizedType = recognizeInstrumentType(channelName);
            const isReference = recognizedType === 'leadVocal';
            
            // Only update if we got a recognition or if settings don't exist
            if (recognizedType !== 'custom' || !updated[channelId]) {
              updated[channelId] = {
                instrumentType: recognizedType,
                isReference: isReference,
                scannedName: channelName.trim()
              };
              
              if (recognizedType !== 'custom') {
                recognizedCount++;
              }
              
              // Set first leadVocal as reference
              if (isReference && !foundReference) {
                setReferenceChannel(channelId);
                foundReference = true;
              }
            } else {
              // Preserve existing settings but update scanned name
              updated[channelId] = {
                ...updated[channelId],
                scannedName: channelName.trim()
              };
            }
          } else if (!updated[channelId]) {
            // Keep existing settings or set default
            updated[channelId] = {
              instrumentType: 'custom',
              isReference: false
            };
          }
        });
        
        if (recognizedCount > 0) {
          console.log(`AutoFaderTab: Recognized ${recognizedCount} instrument types from channel names`);
        }
        
        return updated;
      });
    };
    
    websocketService.on('auto_fader_status', handleAutoFaderStatus);
    websocketService.on('mixer_channel_names', handleMixerChannelNames);
    websocketService.getAutoFaderStatus();
    
    return () => {
      websocketService.off('auto_fader_status', handleAutoFaderStatus);
      websocketService.off('mixer_channel_names', handleMixerChannelNames);
    };
  }, [selectedChannels, availableChannels]);

  // Initialize channel settings when selectedChannels change
  useEffect(() => {
    const newSettings = {};
    let foundReference = false;
    
    selectedChannels.forEach(channelId => {
      // Preserve existing settings if they exist
      if (channelSettings[channelId]) {
        newSettings[channelId] = channelSettings[channelId];
        if (channelSettings[channelId].isReference) {
          foundReference = true;
        }
      } else {
        // Try to recognize from channel name if available
        const channel = availableChannels.find(ch => ch.id === channelId);
        if (channel?.name) {
          const recognizedType = recognizeInstrumentType(channel.name);
          const isReference = recognizedType === 'leadVocal';
          
          newSettings[channelId] = {
            instrumentType: recognizedType,
            isReference: isReference
          };
          
          if (isReference && !foundReference) {
            setReferenceChannel(channelId);
            foundReference = true;
          }
        } else {
          newSettings[channelId] = {
            instrumentType: 'custom',
            isReference: false
          };
        }
      }
    });
    
    setChannelSettings(newSettings);
  }, [selectedChannels, availableChannels]);

  const handleInstrumentTypeChange = (channelId, instrumentType) => {
    const isRefType = INSTRUMENT_TYPES[instrumentType]?.isReference || false;
    
    setChannelSettings(prev => {
      const updated = { ...prev };
      
      // If this is a reference type, remove reference from others
      if (isRefType) {
        Object.keys(updated).forEach(id => {
          updated[id] = { ...updated[id], isReference: false };
        });
        setReferenceChannel(channelId);
      }
      
      updated[channelId] = {
        ...updated[channelId],
        instrumentType,
        isReference: isRefType
      };
      
      return updated;
    });
  };

  const handleSetReference = (channelId) => {
    setChannelSettings(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(id => {
        updated[id] = { 
          ...updated[id], 
          isReference: parseInt(id) === channelId 
        };
      });
      return updated;
    });
    setReferenceChannel(channelId);
  };

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleStartRealtimeFader = () => {
    if (!selectedDevice) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    if (selectedChannels.length === 0) {
      setStatusMessage('Please select channels to process');
      return;
    }
    
    const channelMapping = {};
    selectedChannels.forEach(ch => {
      channelMapping[ch] = ch;
    });
    
    setStatusMessage('Starting Real-Time Fader...');
    websocketService.startRealtimeFader(
      selectedDevice,
      selectedChannels,
      channelSettings,
      channelMapping,
      settings
    );
  };

  const handleStopRealtimeFader = () => {
    setStatusMessage('Stopping Real-Time Fader...');
    websocketService.stopRealtimeFader();
  };

  const handleStartAutoBalance = () => {
    if (!selectedDevice) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    if (selectedChannels.length === 0) {
      setStatusMessage('Please select channels to process');
      return;
    }
    
    const channelMapping = {};
    selectedChannels.forEach(ch => {
      channelMapping[ch] = ch;
    });
    
    setStatusMessage('Starting Auto Balance collection...');
    websocketService.startAutoBalance(
      selectedDevice,
      selectedChannels,
      channelSettings,
      channelMapping,
      settings.autoBalanceDuration,
      settings.bleedThreshold ?? -50
    );
  };

  const handleApplyAutoBalance = () => {
    setStatusMessage('Applying Auto Balance...');
    websocketService.applyAutoBalance();
  };

  const handleCancelAutoBalance = () => {
    websocketService.cancelAutoBalance();
  };

  const handleProfileChange = (newProfile) => {
    setProfile(newProfile);
    websocketService.setAutoFaderProfile(newProfile);
  };

  const handleSettingChange = (setting, value) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const formatDb = (value) => {
    const num = typeof value === 'number' ? value : Number(value);
    if (value === undefined || value === null || isNaN(num) || num <= -60) return '--';
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(1)}`;
  };

  const formatLufs = (value) => {
    const num = typeof value === 'number' ? value : Number(value);
    if (value === undefined || value === null || isNaN(num) || num <= -60) return '--';
    return `${num.toFixed(1)} LUFS`;
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'adjusting': return 'status-adjusting';
      case 'active': return 'status-active';
      case 'idle': return 'status-idle';
      case 'high_bleed': return 'status-high-bleed';
      case 'calibrating': return 'status-calibrating';
      case 'inactive_bleed': return 'status-inactive';
      default: return 'status-idle';
    }
  };

  if (selectedChannels.length === 0) {
    return (
      <div className="auto-fader-tab">
        <div className="auto-fader-section">
          <h2>AUTO FADER</h2>
          <div className="no-channels-message">
            <p>No channels selected for processing.</p>
            <p>Please go to the "Mixer Connection" tab and select channels to process.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auto-fader-tab">
      <div className="auto-fader-section">
        <h2>AUTO FADER</h2>
        
        <div className="auto-fader-header">
          <p className="section-description">
            Automatic fader control for maintaining balance. Choose between Real-Time dynamic mixing 
            or Auto Balance for one-time setup.
          </p>
        </div>

        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            className={`mode-button ${mode === 'realtime' ? 'active' : ''}`}
            onClick={() => setMode('realtime')}
            disabled={autoBalanceCollecting}
          >
            <span className="mode-icon">&#x25B6;</span>
            Real-Time Fader
          </button>
          <button
            className={`mode-button ${mode === 'static' ? 'active' : ''}`}
            onClick={() => setMode('static')}
            disabled={realtimeEnabled}
          >
            <span className="mode-icon">&#x2630;</span>
            Auto Balance
          </button>
        </div>

        {/* Genre Profile Selector */}
        <div className="profile-selector">
          <label>Genre Profile</label>
          <div className="profile-buttons">
            {Object.entries(GENRE_PROFILES).map(([key, { name, description }]) => (
              <button
                key={key}
                className={`profile-button ${profile === key ? 'active' : ''}`}
                onClick={() => handleProfileChange(key)}
                title={description}
                disabled={realtimeEnabled || autoBalanceCollecting}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        {/* Settings Panel */}
        <div className="settings-panel">
          <div className="settings-header" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
            <h3>Settings</h3>
            <span className="toggle-icon">{showAdvancedSettings ? '▼' : '▶'}</span>
          </div>
          
          <div className="settings-main">
            <div className="setting-group">
              <label>Target LUFS</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="-24"
                  max="-12"
                  step="1"
                  value={settings.targetLufs}
                  onChange={(e) => handleSettingChange('targetLufs', parseInt(e.target.value))}
                  disabled={realtimeEnabled || autoBalanceCollecting}
                />
                <span className="setting-value">{settings.targetLufs} LUFS</span>
              </div>
            </div>

            <div className="setting-group">
              <label>Correction Ratio</label>
              <div className="setting-control">
                <select
                  value={settings.ratio}
                  onChange={(e) => handleSettingChange('ratio', parseFloat(e.target.value))}
                  disabled={realtimeEnabled || autoBalanceCollecting}
                >
                  <option value={1}>1:1 (Direct)</option>
                  <option value={1.5}>1.5:1 (Light)</option>
                  <option value={2}>2:1 (Normal)</option>
                  <option value={3}>3:1 (Moderate)</option>
                  <option value={4}>4:1 (Heavy)</option>
                </select>
              </div>
            </div>

            <div className="setting-group">
              <label>Max Adjustment</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="3"
                  max="12"
                  step="1"
                  value={settings.maxAdjustmentDb}
                  onChange={(e) => handleSettingChange('maxAdjustmentDb', parseInt(e.target.value))}
                  disabled={realtimeEnabled || autoBalanceCollecting}
                />
                <span className="setting-value">±{settings.maxAdjustmentDb} dB</span>
              </div>
            </div>
          </div>

          {showAdvancedSettings && (
            <div className="settings-advanced">
              <div className="setting-group">
                <label>Attack Time</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="20"
                    max="500"
                    step="10"
                    value={settings.attackMs}
                    onChange={(e) => handleSettingChange('attackMs', parseInt(e.target.value))}
                    disabled={realtimeEnabled || autoBalanceCollecting}
                  />
                  <span className="setting-value">{settings.attackMs} ms</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Release Time</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="200"
                    max="3000"
                    step="100"
                    value={settings.releaseMs}
                    onChange={(e) => handleSettingChange('releaseMs', parseInt(e.target.value))}
                    disabled={realtimeEnabled || autoBalanceCollecting}
                  />
                  <span className="setting-value">{settings.releaseMs} ms</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Hold Time</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="0"
                    max="1000"
                    step="50"
                    value={settings.holdMs}
                    onChange={(e) => handleSettingChange('holdMs', parseInt(e.target.value))}
                    disabled={realtimeEnabled || autoBalanceCollecting}
                  />
                  <span className="setting-value">{settings.holdMs} ms</span>
                </div>
              </div>

              {mode === 'static' && (
                <>
                  <div className="setting-group">
                    <label>Collection Duration</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="5"
                        max="30"
                        step="5"
                        value={settings.autoBalanceDuration}
                        onChange={(e) => handleSettingChange('autoBalanceDuration', parseInt(e.target.value))}
                        disabled={autoBalanceCollecting}
                      />
                      <span className="setting-value">{settings.autoBalanceDuration} sec</span>
                    </div>
                  </div>
                  <div className="setting-group">
                    <label>Silence Threshold (LUFS)</label>
                    <div className="setting-control">
                      <input
                        type="range"
                        min="-60"
                        max="-30"
                        step="5"
                        value={settings.bleedThreshold ?? -50}
                        onChange={(e) => handleSettingChange('bleedThreshold', parseInt(e.target.value))}
                        disabled={autoBalanceCollecting}
                      />
                      <span className="setting-value">{settings.bleedThreshold ?? -50} LUFS</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Control Buttons */}
        <div className="control-panel">
          {mode === 'realtime' ? (
            <>
              <div className="control-header">
                <h3>Real-Time Fader Control</h3>
                <div className="status-indicators">
                  <div className={`status-indicator ${realtimeEnabled ? 'active' : 'inactive'}`}>
                    <span className="status-dot"></span>
                    <span>{realtimeEnabled ? 'Active' : 'Inactive'}</span>
                  </div>
                </div>
              </div>
              
              <div className="control-buttons">
                <button
                  className={`btn-control ${realtimeEnabled ? 'stop' : 'start'}`}
                  onClick={realtimeEnabled ? handleStopRealtimeFader : handleStartRealtimeFader}
                  disabled={!selectedDevice || selectedChannels.length === 0}
                >
                  {realtimeEnabled ? 'Stop Real-Time Fader' : 'Start Real-Time Fader'}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="control-header">
                <h3>Auto Balance Control</h3>
                <div className="status-indicators">
                  <div className={`status-indicator ${autoBalanceCollecting ? 'collecting' : autoBalanceReady ? 'ready' : 'inactive'}`}>
                    <span className="status-dot"></span>
                    <span>
                      {autoBalanceCollecting ? 'Collecting...' : autoBalanceReady ? 'Ready to Apply' : 'Idle'}
                    </span>
                  </div>
                </div>
              </div>
              
              {autoBalanceCollecting && (
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${autoBalanceProgress}%` }}></div>
                  <span className="progress-text">{autoBalanceProgress.toFixed(0)}%</span>
                </div>
              )}
              
              <div className="control-buttons">
                {!autoBalanceCollecting && !autoBalanceReady && (
                  <button
                    className="btn-control start"
                    onClick={handleStartAutoBalance}
                    disabled={!selectedDevice || selectedChannels.length === 0}
                  >
                    Start Collection
                  </button>
                )}
                
                {autoBalanceCollecting && (
                  <button
                    className="btn-control cancel"
                    onClick={handleCancelAutoBalance}
                  >
                    Cancel
                  </button>
                )}
                
                {autoBalanceReady && (
                  <>
                    <button
                      className="btn-control apply"
                      onClick={handleApplyAutoBalance}
                    >
                      Apply Balance
                    </button>
                    <button
                      className="btn-control restart"
                      onClick={handleStartAutoBalance}
                    >
                      Restart Collection
                    </button>
                  </>
                )}
              </div>
            </>
          )}
          
          {statusMessage && (
            <div className="status-message">
              {statusMessage}
            </div>
          )}
        </div>

        {/* Channels Table */}
        <div className="channels-table">
          <div className="table-header">
            <div className="col-channel">Channel</div>
            <div className="col-type">Instrument Type</div>
            <div className="col-ref">Reference</div>
            <div className="col-lufs">LUFS</div>
            {mode === 'realtime' && realtimeEnabled && (
              <>
                <div className="col-fader">Current</div>
                <div className="col-target">Target</div>
                <div className="col-correction">Correction</div>
              </>
            )}
            {mode === 'static' && (autoBalanceCollecting || autoBalanceReady) && (
              <>
                <div className="col-avg-lufs">Avg LUFS</div>
                <div className="col-correction">Correction</div>
              </>
            )}
            <div className="col-status">Status</div>
          </div>

          <div className="table-body">
            {selectedChannels.map(channelId => {
              const chSettings = channelSettings[channelId] || {};
              const levels = measuredLevels[channelId] || {};
              const result = autoBalanceResult[channelId];
              const correction = (result && typeof result === 'object' && 'correction' in result)
                ? result.correction : (typeof result === 'number' ? result : 0);
              // Get bleed info from levels (realtime) or result (auto balance)
              const bleedRatio = levels.bleedRatio ?? result?.bleed_ratio ?? 0;
              const bleedSource = levels.bleedSource ?? result?.bleed_source ?? null;
              const hasBleed = bleedRatio > 0 && bleedSource != null;

              return (
                <div 
                  key={channelId} 
                  className={`table-row ${levels.isActive ? 'has-signal' : ''} ${chSettings.isReference ? 'is-reference' : ''}`}
                >
                  <div className="col-channel">
                    <span className="channel-name">{getChannelName(channelId)}</span>
                  </div>

                  <div className="col-type">
                    <select
                      value={chSettings.instrumentType || 'custom'}
                      onChange={(e) => handleInstrumentTypeChange(channelId, e.target.value)}
                      className="type-select"
                      disabled={realtimeEnabled || autoBalanceCollecting}
                    >
                      <optgroup label="Vocals">
                        <option value="leadVocal">Lead Vocal (Ref)</option>
                        <option value="backVocal">Back Vocal</option>
                      </optgroup>
                      <optgroup label="Drums">
                        <option value="kick">Kick</option>
                        <option value="snare">Snare</option>
                        <option value="toms">Toms</option>
                        <option value="drums">Drums / Kit</option>
                        <option value="hihat">Hi-Hat</option>
                        <option value="ride">Ride</option>
                        <option value="overhead">Overhead</option>
                        <option value="room">Room</option>
                      </optgroup>
                      <optgroup label="Bass & Guitars">
                        <option value="bass">Bass</option>
                        <option value="guitar">Electric Guitar</option>
                        <option value="acousticGuitar">Acoustic Guitar</option>
                      </optgroup>
                      <optgroup label="Keys & Other">
                        <option value="keys">Keys / Piano</option>
                        <option value="synth">Synth</option>
                        <option value="accordion">Accordion</option>
                        <option value="brass">Brass</option>
                        <option value="strings">Strings</option>
                      </optgroup>
                      <optgroup label="Other">
                        <option value="playback">Playback</option>
                        <option value="custom">Custom</option>
                      </optgroup>
                    </select>
                  </div>

                  <div className="col-ref">
                    <button
                      className={`ref-button ${chSettings.isReference ? 'active' : ''}`}
                      onClick={() => handleSetReference(channelId)}
                      disabled={realtimeEnabled || autoBalanceCollecting}
                      title="Set as reference channel"
                    >
                      {chSettings.isReference ? '★' : '☆'}
                    </button>
                  </div>

                  <div className="col-lufs">
                    <span className={`lufs-value ${levels.isActive ? 'active' : ''}`}>
                      {formatLufs(levels.lufs)}
                    </span>
                  </div>

                  {mode === 'realtime' && realtimeEnabled && (
                    <>
                      <div className="col-fader">
                        <span className="fader-value">
                          {formatDb(levels.currentFader)} dB
                        </span>
                      </div>
                      <div className="col-target">
                        <span className="target-value">
                          {formatDb(levels.targetFader)} dB
                        </span>
                      </div>
                      <div className="col-correction">
                        <span className={`correction-value ${levels.correction > 0 ? 'positive' : levels.correction < 0 ? 'negative' : ''}`}>
                          {formatDb(levels.correction)}
                        </span>
                      </div>
                    </>
                  )}

                  {mode === 'static' && (autoBalanceCollecting || autoBalanceReady) && (
                    <>
                      <div className="col-avg-lufs">
                        <span className="avg-lufs-value">
                          {formatLufs(levels.avgLufs)}
                        </span>
                      </div>
                      <div className="col-correction">
                        <span className={`correction-value ${correction > 0 ? 'positive' : correction < 0 ? 'negative' : ''}`}>
                          {autoBalanceReady ? formatDb(correction) : '--'}
                        </span>
                        {hasBleed && (
                          <span className="bleed-badge" title={`BLEED ${Math.round(bleedRatio * 100)}% from Ch${bleedSource}`}>
                            BLEED
                          </span>
                        )}
                      </div>
                    </>
                  )}

                  <div className={`col-status ${getStatusClass(levels.status)}`}>
                    <span className="status-text">{levels.status || 'idle'}</span>
                    {hasBleed && (
                      <span className="bleed-indicator" title={`BLEED ${Math.round(bleedRatio * 100)}% from Ch${bleedSource}`}>
                        ⚠ {Math.round(bleedRatio * 100)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AutoFaderTab;
