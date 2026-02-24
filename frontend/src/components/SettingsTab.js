import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './SettingsTab.css';

// All default settings organized by section
const ALL_DEFAULTS = {
  mixer: {
    mixerIp: '192.168.1.102',
    mixerPort: 2223
  },
  gainStaging: {
    targetLufs: -23,
    truePeakLimit: -1,
    ratio: 4,
    attackMs: 50,
    releaseMs: 500,
    holdMs: 200,
    maxGainDb: 12
  },
  autoFader: {
    targetLufs: -18,
    ratio: 2,
    maxAdjustmentDb: 6,
    attackMs: 100,
    releaseMs: 1000,
    holdMs: 500,
    autoBalanceDuration: 30,
    bleedThreshold: -50
  },
  voiceControl: {
    modelSize: 'small',
    language: 'ru'
  }
};

function SettingsTab({ mixerIp, mixerPort, onMixerSettingsChange }) {
  const [allSettings, setAllSettings] = useState(ALL_DEFAULTS);
  const [saveStatus, setSaveStatus] = useState(''); // '', 'saving', 'saved', 'error'
  const [loadStatus, setLoadStatus] = useState('loading');
  const [hasChanges, setHasChanges] = useState(false);
  const [savedSnapshot, setSavedSnapshot] = useState(null);

  // Load saved settings on mount
  useEffect(() => {
    const handleLoaded = (data) => {
      console.log('All settings loaded:', data);
      if (data.settings && Object.keys(data.settings).length > 0) {
        setAllSettings(prev => {
          const merged = { ...prev };
          Object.keys(data.settings).forEach(section => {
            if (merged[section]) {
              merged[section] = { ...merged[section], ...data.settings[section] };
            } else {
              merged[section] = data.settings[section];
            }
          });
          return merged;
        });
        setSavedSnapshot(data.settings);
      }
      setLoadStatus('loaded');
    };

    const handleSaved = (data) => {
      console.log('All settings saved:', data);
      if (data.success) {
        setSaveStatus('saved');
        setHasChanges(false);
        setSavedSnapshot(JSON.parse(JSON.stringify(allSettings)));
        setTimeout(() => setSaveStatus(''), 2000);
      } else {
        setSaveStatus('error');
        setTimeout(() => setSaveStatus(''), 3000);
      }
    };

    websocketService.on('all_settings_loaded', handleLoaded);
    websocketService.on('all_settings_saved', handleSaved);
    websocketService.loadAllSettings();

    return () => {
      websocketService.off('all_settings_loaded', handleLoaded);
      websocketService.off('all_settings_saved', handleSaved);
    };
  }, []);

  // Sync mixer IP/port from App.js props
  useEffect(() => {
    if (mixerIp !== undefined && mixerPort !== undefined) {
      setAllSettings(prev => ({
        ...prev,
        mixer: { ...prev.mixer, mixerIp, mixerPort }
      }));
    }
  }, [mixerIp, mixerPort]);

  const handleChange = (section, key, value) => {
    setAllSettings(prev => ({
      ...prev,
      [section]: { ...prev[section], [key]: value }
    }));
    setHasChanges(true);

    // Propagate mixer settings changes to App.js
    if (section === 'mixer' && onMixerSettingsChange) {
      onMixerSettingsChange(key, value);
    }
  };

  const handleSave = () => {
    setSaveStatus('saving');
    websocketService.saveAllSettings(allSettings);
  };

  const handleReset = () => {
    setAllSettings({ ...ALL_DEFAULTS, mixer: { mixerIp, mixerPort } });
    setHasChanges(true);
  };

  const handleRevert = () => {
    if (savedSnapshot) {
      setAllSettings(prev => {
        const reverted = { ...ALL_DEFAULTS };
        Object.keys(savedSnapshot).forEach(section => {
          if (reverted[section]) {
            reverted[section] = { ...reverted[section], ...savedSnapshot[section] };
          }
        });
        return reverted;
      });
      setHasChanges(false);
    }
  };

  if (loadStatus === 'loading') {
    return <div className="settings-tab"><div className="settings-loading">Loading settings...</div></div>;
  }

  return (
    <div className="settings-tab">
      <div className="settings-tab-header">
        <h2>Settings</h2>
        <div className="settings-tab-actions">
          {hasChanges && (
            <button className="btn-settings revert" onClick={handleRevert} title="Revert to last saved">
              Revert
            </button>
          )}
          <button className="btn-settings reset" onClick={handleReset} title="Reset all to factory defaults">
            Factory Reset
          </button>
          <button 
            className={`btn-settings save ${saveStatus}`}
            onClick={handleSave}
            disabled={saveStatus === 'saving'}
          >
            {saveStatus === 'saving' ? 'Saving...' : 
             saveStatus === 'saved' ? 'Saved!' : 
             saveStatus === 'error' ? 'Error!' : 
             'Save All'}
          </button>
        </div>
      </div>

      {/* Mixer Connection */}
      <div className="settings-section">
        <h3>Mixer Connection</h3>
        <div className="settings-grid">
          <div className="setting-item">
            <label>Wing IP Address</label>
            <input
              type="text"
              value={allSettings.mixer.mixerIp}
              onChange={(e) => handleChange('mixer', 'mixerIp', e.target.value)}
              placeholder="192.168.1.102"
            />
          </div>
          <div className="setting-item">
            <label>OSC Port</label>
            <input
              type="number"
              value={allSettings.mixer.mixerPort}
              onChange={(e) => handleChange('mixer', 'mixerPort', parseInt(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Gain Staging */}
      <div className="settings-section">
        <h3>Gain Staging</h3>
        <div className="settings-grid">
          <div className="setting-item">
            <label>Target LUFS</label>
            <div className="setting-slider">
              <input
                type="range" min="-30" max="-14" step="1"
                value={allSettings.gainStaging.targetLufs}
                onChange={(e) => handleChange('gainStaging', 'targetLufs', parseInt(e.target.value))}
              />
              <span>{allSettings.gainStaging.targetLufs} LUFS</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Ratio</label>
            <select
              value={allSettings.gainStaging.ratio}
              onChange={(e) => handleChange('gainStaging', 'ratio', parseInt(e.target.value))}
            >
              <option value={2}>2:1 (Light)</option>
              <option value={4}>4:1 (Normal)</option>
              <option value={8}>8:1 (Heavy)</option>
            </select>
          </div>
          <div className="setting-item">
            <label>True Peak Limit</label>
            <div className="setting-slider">
              <input
                type="range" min="-6" max="0" step="0.5"
                value={allSettings.gainStaging.truePeakLimit}
                onChange={(e) => handleChange('gainStaging', 'truePeakLimit', parseFloat(e.target.value))}
              />
              <span>{allSettings.gainStaging.truePeakLimit} dBTP</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Attack Time</label>
            <div className="setting-slider">
              <input
                type="range" min="10" max="200" step="10"
                value={allSettings.gainStaging.attackMs}
                onChange={(e) => handleChange('gainStaging', 'attackMs', parseInt(e.target.value))}
              />
              <span>{allSettings.gainStaging.attackMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Release Time</label>
            <div className="setting-slider">
              <input
                type="range" min="100" max="2000" step="100"
                value={allSettings.gainStaging.releaseMs}
                onChange={(e) => handleChange('gainStaging', 'releaseMs', parseInt(e.target.value))}
              />
              <span>{allSettings.gainStaging.releaseMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Hold Time</label>
            <div className="setting-slider">
              <input
                type="range" min="0" max="500" step="50"
                value={allSettings.gainStaging.holdMs}
                onChange={(e) => handleChange('gainStaging', 'holdMs', parseInt(e.target.value))}
              />
              <span>{allSettings.gainStaging.holdMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Max Gain</label>
            <div className="setting-slider">
              <input
                type="range" min="6" max="18" step="1"
                value={allSettings.gainStaging.maxGainDb}
                onChange={(e) => handleChange('gainStaging', 'maxGainDb', parseInt(e.target.value))}
              />
              <span>+{allSettings.gainStaging.maxGainDb} dB</span>
            </div>
          </div>
        </div>
      </div>

      {/* Auto Fader */}
      <div className="settings-section">
        <h3>Auto Fader</h3>
        <div className="settings-grid">
          <div className="setting-item">
            <label>Target LUFS</label>
            <div className="setting-slider">
              <input
                type="range" min="-24" max="-12" step="1"
                value={allSettings.autoFader.targetLufs}
                onChange={(e) => handleChange('autoFader', 'targetLufs', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.targetLufs} LUFS</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Correction Ratio</label>
            <select
              value={allSettings.autoFader.ratio}
              onChange={(e) => handleChange('autoFader', 'ratio', parseFloat(e.target.value))}
            >
              <option value={1}>1:1 (Direct)</option>
              <option value={1.5}>1.5:1 (Light)</option>
              <option value={2}>2:1 (Normal)</option>
              <option value={3}>3:1 (Moderate)</option>
              <option value={4}>4:1 (Heavy)</option>
            </select>
          </div>
          <div className="setting-item">
            <label>Max Adjustment</label>
            <div className="setting-slider">
              <input
                type="range" min="3" max="12" step="1"
                value={allSettings.autoFader.maxAdjustmentDb}
                onChange={(e) => handleChange('autoFader', 'maxAdjustmentDb', parseInt(e.target.value))}
              />
              <span>&plusmn;{allSettings.autoFader.maxAdjustmentDb} dB</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Attack Time</label>
            <div className="setting-slider">
              <input
                type="range" min="20" max="500" step="10"
                value={allSettings.autoFader.attackMs}
                onChange={(e) => handleChange('autoFader', 'attackMs', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.attackMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Release Time</label>
            <div className="setting-slider">
              <input
                type="range" min="200" max="3000" step="100"
                value={allSettings.autoFader.releaseMs}
                onChange={(e) => handleChange('autoFader', 'releaseMs', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.releaseMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Hold Time</label>
            <div className="setting-slider">
              <input
                type="range" min="0" max="1000" step="50"
                value={allSettings.autoFader.holdMs}
                onChange={(e) => handleChange('autoFader', 'holdMs', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.holdMs} ms</span>
            </div>
          </div>
          <div className="setting-item">
            <label>LEARN Duration</label>
            <div className="setting-slider">
              <input
                type="range" min="10" max="120" step="5"
                value={allSettings.autoFader.autoBalanceDuration}
                onChange={(e) => handleChange('autoFader', 'autoBalanceDuration', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.autoBalanceDuration} sec</span>
            </div>
          </div>
          <div className="setting-item">
            <label>Bleed Threshold</label>
            <div className="setting-slider">
              <input
                type="range" min="-60" max="-20" step="1"
                value={allSettings.autoFader.bleedThreshold}
                onChange={(e) => handleChange('autoFader', 'bleedThreshold', parseInt(e.target.value))}
              />
              <span>{allSettings.autoFader.bleedThreshold} LUFS</span>
            </div>
          </div>
        </div>
      </div>

      {/* Voice Control */}
      <div className="settings-section">
        <h3>Voice Control</h3>
        <div className="settings-grid">
          <div className="setting-item">
            <label>Model Size</label>
            <select
              value={allSettings.voiceControl.modelSize}
              onChange={(e) => handleChange('voiceControl', 'modelSize', e.target.value)}
            >
              <option value="tiny">Tiny (fast, less accurate)</option>
              <option value="base">Base</option>
              <option value="small">Small (recommended)</option>
              <option value="medium">Medium</option>
              <option value="large">Large (slow, most accurate)</option>
            </select>
          </div>
          <div className="setting-item">
            <label>Language</label>
            <select
              value={allSettings.voiceControl.language}
              onChange={(e) => handleChange('voiceControl', 'language', e.target.value)}
            >
              <option value="ru">Russian</option>
              <option value="en">English</option>
              <option value="">Auto-detect</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsTab;
