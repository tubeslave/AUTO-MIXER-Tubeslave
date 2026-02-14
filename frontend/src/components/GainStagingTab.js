import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './GainStagingTab.css';

const SIGNAL_PRESETS = {
  kick: { name: 'Kick' },
  snare: { name: 'Snare' },
  tom: { name: 'Tom' },
  hihat: { name: 'Hi-Hat' },
  ride: { name: 'Ride' },
  cymbals: { name: 'Cymbals' },
  overheads: { name: 'Overheads' },
  room: { name: 'Room' },
  bass: { name: 'Bass' },
  electricGuitar: { name: 'Electric Guitar' },
  acousticGuitar: { name: 'Acoustic Guitar' },
  accordion: { name: 'Accordion' },
  synth: { name: 'Synth / Keys' },
  playback: { name: 'Playback' },
  leadVocal: { name: 'Lead Vocal' },
  backVocal: { name: 'Back Vocal' },
  custom: { name: 'Custom' }
};

// Default AGC settings
const DEFAULT_AGC_SETTINGS = {
  targetLufs: -23,
  truePeakLimit: -1,
  ratio: 4,
  attackMs: 50,
  releaseMs: 500,
  holdMs: 200,
  maxGainDb: 12,
  learningDurationSec: 30
};

// Function to recognize signal type from channel name
// Based on backend channel_recognizer.py patterns
const recognizeSignalType = (channelName) => {
  if (!channelName || !channelName.trim()) {
    return 'custom';
  }
  
  const nameLower = channelName.toLowerCase().trim();
  
  // Drums - specific (check more specific patterns first)
  // Room mic
  if (/\b(room|рум)\b/i.test(nameLower)) {
    return 'room';
  }
  // Overheads
  if (/\b(ohl|ohr|over[\s-]?head|overhead|оверхэд)\b/i.test(nameLower)) {
    return 'overheads';
  }
  // Snare
  if (/\b(snare|sd|sn|малый|снэйр|снейр)\b/i.test(nameLower)) {
    return 'snare';
  }
  // Kick
  if (/\b(kick|bd|bass\s*drum|бас[\s-]?бочка|бочка|кик)\b/i.test(nameLower)) {
    return 'kick';
  }
  // Toms
  if (/\b(m\s*tom|f\s*tom|tom|том|floor\s*tom|флор\s*том|флор|rack\s*tom|рэк\s*том)\b/i.test(nameLower)) {
    return 'tom';
  }
  // Hi-Hat
  if (/\b(hi[\s-]?hat|hh|хай[\s-]?хэт|хэт)\b/i.test(nameLower)) {
    return 'hihat';
  }
  // Ride
  if (/\b(ride|райд)\b/i.test(nameLower)) {
    return 'ride';
  }
  // Cymbals
  if (/\b(crash|splash|china|крэш|сплэш|чайна|cymbal|тарелк)\b/i.test(nameLower)) {
    return 'cymbals';
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
    return 'electricGuitar';
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
  
  return 'custom';
};

function GainStagingTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [channelSettings, setChannelSettings] = useState({});
  const [realtimeCorrectionEnabled, setRealtimeCorrectionEnabled] = useState(false);
  const [measuredLevels, setMeasuredLevels] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  const [agcSettings, setAgcSettings] = useState(DEFAULT_AGC_SETTINGS);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  useEffect(() => {
    const handleGainStagingStatus = (data) => {
      console.log('Gain staging status update:', data);
      
      if (data.realtime_enabled !== undefined) {
        setRealtimeCorrectionEnabled(data.realtime_enabled);
      }
      
      if (data.message) {
        setStatusMessage(data.message);
      }
      
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setRealtimeCorrectionEnabled(false);
      }
      
      // Handle realtime correction events
      if (data.status_type === 'realtime_correction_started') {
        setRealtimeCorrectionEnabled(true);
        setStatusMessage('LUFS-based correction started');
      } else if (data.status_type === 'realtime_correction_stopped') {
        setRealtimeCorrectionEnabled(false);
        setStatusMessage('LUFS-based correction stopped');
      }
      
      // Handle level updates with new LUFS metrics
      if (data.status_type === 'levels_update' && data.channels) {
        const newLevels = {};
        Object.entries(data.channels).forEach(([audioChStr, chData]) => {
          const audioChannel = parseInt(audioChStr);
          newLevels[audioChannel] = {
            peak: chData.measured_peak ?? chData.true_peak ?? -60,
            lufs: chData.lufs ?? -60,
            truePeak: chData.true_peak ?? chData.measured_peak ?? -60,
            gain: chData.gain ?? 0,
            appliedGain: chData.applied_gain ?? 0,
            status: chData.status ?? 'idle',
            signalPresent: chData.signal_present || false
          };
        });
        setMeasuredLevels(newLevels);
      }
    };
    
    const handleChannelScanResult = (data) => {
      if (data.error) {
        setStatusMessage(`Scan error: ${data.error}`);
        return;
      }
      
      const results = data.results || {};
      setChannelSettings(prev => {
        const updated = { ...prev };
        
        Object.entries(results).forEach(([channelStr, result]) => {
          const channelId = parseInt(channelStr);
          
          if (result.recognized && result.preset && SIGNAL_PRESETS[result.preset]) {
            updated[channelId] = {
              ...updated[channelId],
              preset: result.preset,
              scannedName: result.name
            };
          } else if (result.name) {
            updated[channelId] = {
              ...updated[channelId],
              scannedName: result.name
            };
          }
        });
        return updated;
      });
      
      const recognizedCount = Object.values(results).filter(r => r.recognized).length;
      const totalCount = Object.keys(results).length;
      setStatusMessage(`Scanned ${totalCount} channels, recognized ${recognizedCount} instruments`);
    };
    
    const handleResetTrimResult = (data) => {
      if (data.error) {
        setStatusMessage(`Reset TRIM error: ${data.error}`);
        return;
      }
      
      if (data.message) {
        setStatusMessage(data.message);
      } else if (data.success_count !== undefined) {
        const successCount = data.success_count;
        const totalCount = data.total_count;
        if (successCount === totalCount) {
          setStatusMessage(`Successfully reset TRIM to 0dB for all ${totalCount} channels`);
        } else {
          setStatusMessage(`Reset TRIM: ${successCount}/${totalCount} channels successful`);
        }
      }
    };
    
    const handleMixerChannelNames = (data) => {
      if (data.error || !data.channel_names) {
        return;
      }
      
      // Auto-recognize signal types from channel names
      setChannelSettings(prev => {
        const updated = { ...prev };
        let recognizedCount = 0;
        
        // Process all selected channels
        selectedChannels.forEach(channelId => {
          // Get channel name from mixer scan result
          const channelNum = typeof channelId === 'number' ? channelId : parseInt(channelId);
          let channelName = null;
          
          if (!isNaN(channelNum)) {
            channelName = data.channel_names[channelNum] || data.channel_names[String(channelNum)];
          }
          
          // If we have a channel name, recognize signal type
          if (channelName && channelName.trim()) {
            const recognizedType = recognizeSignalType(channelName);
            
            if (recognizedType !== 'custom') {
              recognizedCount++;
            }
            
            updated[channelId] = {
              ...updated[channelId],
              preset: recognizedType,
              scannedName: channelName.trim()
            };
          } else if (!updated[channelId]) {
            // Keep existing settings or set default
            updated[channelId] = {
              preset: 'custom'
            };
          }
        });
        
        if (recognizedCount > 0) {
          console.log(`GainStagingTab: Recognized ${recognizedCount} signal types from channel names`);
        }
        
        return updated;
      });
    };
    
    const handleAllSettingsLoaded = (data) => {
      if (data.settings) {
        const updatedSettings = { ...DEFAULT_AGC_SETTINGS };
        
        // Загружаем настройки из gainStaging (старый формат)
        if (data.settings.gainStaging) {
          Object.assign(updatedSettings, data.settings.gainStaging);
        }
        
        // Загружаем learning_duration_sec из safe_gain_calibration (новый формат)
        if (data.settings.safe_gain_calibration && data.settings.safe_gain_calibration.learning_duration_sec !== undefined) {
          updatedSettings.learningDurationSec = data.settings.safe_gain_calibration.learning_duration_sec;
        }
        
        setAgcSettings(updatedSettings);
        console.log('GainStaging: Applied saved defaults:', updatedSettings);
      }
    };

    websocketService.on('gain_staging_status', handleGainStagingStatus);
    websocketService.on('channel_scan_result', handleChannelScanResult);
    websocketService.on('reset_trim_result', handleResetTrimResult);
    websocketService.on('mixer_channel_names', handleMixerChannelNames);
    websocketService.on('all_settings_loaded', handleAllSettingsLoaded);
    
    websocketService.getGainStagingStatus();
    websocketService.loadAllSettings();
    
    return () => {
      websocketService.off('gain_staging_status', handleGainStagingStatus);
      websocketService.off('channel_scan_result', handleChannelScanResult);
      websocketService.off('reset_trim_result', handleResetTrimResult);
      websocketService.off('mixer_channel_names', handleMixerChannelNames);
      websocketService.off('all_settings_loaded', handleAllSettingsLoaded);
    };
  }, [selectedChannels]);

  useEffect(() => {
    const newSettings = {};
    selectedChannels.forEach(channelId => {
      // Preserve existing settings if they exist
      if (channelSettings[channelId]) {
        newSettings[channelId] = channelSettings[channelId];
      } else {
        // Try to recognize from channel name if available
        const channel = availableChannels.find(ch => ch.id === channelId);
        if (channel?.name) {
          const recognizedType = recognizeSignalType(channel.name);
          newSettings[channelId] = {
            preset: recognizedType
          };
        } else {
          newSettings[channelId] = {
            preset: 'custom'
          };
        }
      }
    });
    setChannelSettings(newSettings);
  }, [selectedChannels, availableChannels]);

  const handlePresetChange = (channelId, presetKey) => {
    setChannelSettings(prev => {
      return {
        ...prev,
        [channelId]: {
          ...prev[channelId],
          preset: presetKey
        }
      };
    });
  };

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleStartRealtimeCorrection = () => {
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
    
    const completeChannelSettings = {};
    selectedChannels.forEach(channelId => {
      if (channelSettings[channelId]) {
        completeChannelSettings[channelId] = channelSettings[channelId];
      } else {
        completeChannelSettings[channelId] = {
          preset: 'custom'
        };
      }
    });
    
    setStatusMessage('Starting LUFS-based correction...');
    websocketService.startRealtimeCorrection(
      selectedDevice, 
      selectedChannels, 
      completeChannelSettings, 
      channelMapping,
      { learning_duration_sec: agcSettings.learningDurationSec }
    );
  };

  const handleStopRealtimeCorrection = () => {
    setStatusMessage('Stopping LUFS-based correction...');
    websocketService.stopRealtimeCorrection();
  };

  const handleResetTrim = () => {
    if (selectedChannels.length === 0) {
      setStatusMessage('Please select channels first');
      return;
    }
    
    setStatusMessage('Resetting TRIM to 0dB for selected channels...');
    websocketService.resetTrim(selectedChannels);
  };

  const handleAgcSettingChange = (setting, value) => {
    setAgcSettings(prev => ({
      ...prev,
      [setting]: value
    }));
    // Настройки применяются только при запуске нового измерения
  };

  const formatDb = (value) => {
    if (value === undefined || value === null || value === -60) return '--';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}`;
  };

  const formatLufs = (value) => {
    if (value === undefined || value === null || value <= -60) return '--';
    return `${value.toFixed(1)} LUFS`;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'adjusting': return 'status-adjusting';
      case 'limiting': return 'status-limiting';
      case 'measuring': return 'status-measuring';
      default: return 'status-idle';
    }
  };

  const getTruePeakClass = (truePeak) => {
    if (truePeak > -1) return 'peak-danger';
    if (truePeak > -3) return 'peak-warning';
    return '';
  };

  if (selectedChannels.length === 0) {
    return (
      <div className="gain-staging-tab">
        <div className="gain-staging-section">
          <h2>GAIN STAGING</h2>
          <div className="no-channels-message">
            <p>No channels selected for processing.</p>
            <p>Please go to the "Mixer Connection" tab and select channels to process.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="gain-staging-tab">
      <div className="gain-staging-section">
        <h2>LUFS GAIN STAGING</h2>
        
        <div className="gain-staging-header">
          <p className="section-description">
            Automatic gain control based on LUFS loudness measurement (ITU-R BS.1770 / EBU R128).
          </p>
        </div>

        {/* AGC Settings Panel */}
        <div className="agc-settings-panel">
          <div className="settings-header" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}>
            <h3>AGC Settings</h3>
            <span className="toggle-icon">{showAdvancedSettings ? '▼' : '▶'}</span>
          </div>
          
          <div className="settings-main">
            <div className="setting-group">
              <label>Target LUFS</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="-30"
                  max="-14"
                  step="1"
                  value={agcSettings.targetLufs}
                  onChange={(e) => handleAgcSettingChange('targetLufs', parseInt(e.target.value))}
                  disabled={realtimeCorrectionEnabled}
                />
                <span className="setting-value">{agcSettings.targetLufs} LUFS</span>
              </div>
            </div>

            <div className="setting-group">
              <label>Ratio</label>
              <div className="setting-control">
                <select
                  value={agcSettings.ratio}
                  onChange={(e) => handleAgcSettingChange('ratio', parseInt(e.target.value))}
                  disabled={realtimeCorrectionEnabled}
                >
                  <option value={2}>2:1 (Light)</option>
                  <option value={4}>4:1 (Normal)</option>
                  <option value={8}>8:1 (Heavy)</option>
                </select>
              </div>
            </div>

            <div className="setting-group">
              <label>True Peak Limit</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="-6"
                  max="0"
                  step="0.5"
                  value={agcSettings.truePeakLimit}
                  onChange={(e) => handleAgcSettingChange('truePeakLimit', parseFloat(e.target.value))}
                  disabled={realtimeCorrectionEnabled}
                />
                <span className="setting-value">{agcSettings.truePeakLimit} dBTP</span>
              </div>
            </div>

            <div className="setting-group">
              <label>Learning Duration (Analysis Time)</label>
              <div className="setting-control">
                <input
                  type="range"
                  min="30"
                  max="300"
                  step="5"
                  value={agcSettings.learningDurationSec}
                  onChange={(e) => handleAgcSettingChange('learningDurationSec', parseInt(e.target.value))}
                  disabled={realtimeCorrectionEnabled}
                />
                <span className="setting-value">{agcSettings.learningDurationSec} sec</span>
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
                    min="10"
                    max="200"
                    step="10"
                    value={agcSettings.attackMs}
                    onChange={(e) => handleAgcSettingChange('attackMs', parseInt(e.target.value))}
                    disabled={realtimeCorrectionEnabled}
                  />
                  <span className="setting-value">{agcSettings.attackMs} ms</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Release Time</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="100"
                    max="2000"
                    step="100"
                    value={agcSettings.releaseMs}
                    onChange={(e) => handleAgcSettingChange('releaseMs', parseInt(e.target.value))}
                    disabled={realtimeCorrectionEnabled}
                  />
                  <span className="setting-value">{agcSettings.releaseMs} ms</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Hold Time</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="0"
                    max="500"
                    step="50"
                    value={agcSettings.holdMs}
                    onChange={(e) => handleAgcSettingChange('holdMs', parseInt(e.target.value))}
                    disabled={realtimeCorrectionEnabled}
                  />
                  <span className="setting-value">{agcSettings.holdMs} ms</span>
                </div>
              </div>

              <div className="setting-group">
                <label>Max Gain</label>
                <div className="setting-control">
                  <input
                    type="range"
                    min="6"
                    max="18"
                    step="1"
                    value={agcSettings.maxGainDb}
                    onChange={(e) => handleAgcSettingChange('maxGainDb', parseInt(e.target.value))}
                    disabled={realtimeCorrectionEnabled}
                  />
                  <span className="setting-value">+{agcSettings.maxGainDb} dB</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="auto-gain-staging-control">
          <div className="control-header">
            <h3>Real-time LUFS Correction</h3>
            <div className="status-indicators">
              <div className={`status-indicator ${realtimeCorrectionEnabled ? 'active' : 'inactive'}`}>
                <span className="status-dot"></span>
                <span>{realtimeCorrectionEnabled ? 'AGC Active' : 'AGC Inactive'}</span>
              </div>
            </div>
          </div>
          
          <div className="control-buttons">
            <button
              className={`btn-realtime ${realtimeCorrectionEnabled ? 'stop' : 'start'}`}
              onClick={(e) => {
                e.preventDefault();
                if (realtimeCorrectionEnabled) {
                  handleStopRealtimeCorrection();
                } else {
                  handleStartRealtimeCorrection();
                }
              }}
              disabled={!selectedDevice || selectedChannels.length === 0}
            >
              {realtimeCorrectionEnabled ? 'Stop AGC' : 'Start AGC'}
            </button>
            
            <button
              className="btn-reset-trim"
              onClick={handleResetTrim}
              disabled={selectedChannels.length === 0 || realtimeCorrectionEnabled}
            >
              Reset TRIM to 0dB
            </button>
          </div>
          
          {statusMessage && (
            <div className="status-message">
              {statusMessage}
            </div>
          )}
        </div>

        <div className="gain-staging-table">
          <div className="table-header">
            <div className="col-channel">Channel</div>
            <div className="col-preset">Signal Type</div>
            <div className="col-measured-peak">Peak Level</div>
            {realtimeCorrectionEnabled && (
              <>
                <div className="col-lufs">LUFS</div>
                <div className="col-true-peak">True Peak</div>
                <div className="col-gain">Gain</div>
                <div className="col-status">Status</div>
              </>
            )}
          </div>

          <div className="table-body">
            {selectedChannels.map(channelId => {
              const settings = channelSettings[channelId] || {};
              const preset = settings.preset || 'custom';
              const levels = measuredLevels[channelId] || {};

              return (
                <div key={channelId} className={`table-row ${levels.signalPresent ? 'has-signal' : ''}`}>
                  <div className="col-channel">
                    <span className="channel-name">{getChannelName(channelId)}</span>
                    {settings?.scannedName && (
                      <span className="scanned-name">({settings.scannedName})</span>
                    )}
                  </div>

                  <div className="col-preset">
                    <select
                      value={preset}
                      onChange={(e) => handlePresetChange(channelId, e.target.value)}
                      className="preset-select"
                      disabled={realtimeCorrectionEnabled}
                    >
                      <optgroup label="Drums">
                        <option value="kick">Kick</option>
                        <option value="snare">Snare</option>
                        <option value="tom">Tom</option>
                        <option value="hihat">Hi-Hat</option>
                        <option value="ride">Ride</option>
                        <option value="cymbals">Cymbals</option>
                        <option value="overheads">Overheads</option>
                      </optgroup>
                      <optgroup label="Bass & Guitars">
                        <option value="bass">Bass</option>
                        <option value="electricGuitar">Electric Guitar</option>
                        <option value="acousticGuitar">Acoustic Guitar</option>
                      </optgroup>
                      <optgroup label="Keys & Other">
                        <option value="accordion">Accordion</option>
                        <option value="synth">Synth / Keys</option>
                        <option value="playback">Playback</option>
                      </optgroup>
                      <optgroup label="Vocals">
                        <option value="leadVocal">Lead Vocal</option>
                        <option value="backVocal">Back Vocal</option>
                      </optgroup>
                      <optgroup label="Other">
                        <option value="custom">Custom</option>
                      </optgroup>
                    </select>
                  </div>
                  
                  <div className="col-measured-peak">
                    {levels.peak !== undefined ? (
                      <span className={`measured-value ${levels.peak > -6 ? 'hot' : ''}`}>
                        {formatDb(levels.peak)} dB
                      </span>
                    ) : (
                      <span className="no-data">--</span>
                    )}
                  </div>

                  {realtimeCorrectionEnabled && (
                    <>
                      <div className="col-lufs">
                        <span className="lufs-value">
                          {formatLufs(levels.lufs)}
                        </span>
                      </div>

                      <div className={`col-true-peak ${getTruePeakClass(levels.truePeak)}`}>
                        <span className="true-peak-value">
                          {formatDb(levels.truePeak)} dBTP
                        </span>
                      </div>

                      <div className="col-gain">
                        <span className={`gain-value ${levels.gain > 0 ? 'positive' : levels.gain < 0 ? 'negative' : ''}`}>
                          {formatDb(levels.gain)}
                        </span>
                      </div>

                      <div className={`col-status ${getStatusColor(levels.status)}`}>
                        <span className="status-text">{levels.status || 'idle'}</span>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default GainStagingTab;
