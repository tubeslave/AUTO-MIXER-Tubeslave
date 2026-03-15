import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './SettingsTab.css';

const DEFAULTS = {
  mixer: { mixerIp: '192.168.1.102', mixerPort: 2223 },
  gainStaging: { targetLufs: -23, truePeakLimit: -1, ratio: 4, learningDurationSec: 30 },
  autoFader: { bleedThreshold: -50, targetLoudness: -18, smoothingAlpha: 0.3 },
  voiceControl: { language: 'ru', modelSize: 'small' },
};

function SettingsTab({ mixerIp, mixerPort, onMixerSettingsChange }) {
  const [allSettings, setAllSettings] = useState(JSON.parse(JSON.stringify(DEFAULTS)));
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    const handle = (data) => {
      if (data.settings) {
        setAllSettings(prev => {
          const merged = { ...prev };
          Object.keys(data.settings).forEach(section => {
            if (merged[section]) {
              merged[section] = { ...merged[section], ...data.settings[section] };
            }
          });
          return merged;
        });
      }
    };
    websocketService.on('all_settings_loaded', handle);
    websocketService.loadAllSettings();
    return () => websocketService.off('all_settings_loaded', handle);
  }, []);

  const handleChange = (section, key, value) => {
    setAllSettings(prev => ({
      ...prev,
      [section]: { ...prev[section], [key]: value }
    }));
    if (section === 'mixer') onMixerSettingsChange(key, value);
  };

  const handleSave = () => {
    websocketService.saveAllSettings(allSettings);
    setStatusMessage('Сохранено');
    setTimeout(() => setStatusMessage(''), 2000);
  };

  const handleReset = () => {
    setAllSettings(JSON.parse(JSON.stringify(DEFAULTS)));
    setStatusMessage('Сброшено к заводским');
  };

  return (
    <div className="settings-tab">
      <div className="module-card">
        <div className="module-actions">
          <button className="btn-start go" onClick={handleSave}>Сохранить</button>
          <button className="btn-sm" onClick={handleReset}>Сброс</button>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        <h3 className="settings-section-title">Mixer</h3>
        <div className="settings-body">
          <div className="setting-row">
            <label>Wing IP</label>
            <input type="text" value={allSettings.mixer.mixerIp}
              onChange={e => handleChange('mixer', 'mixerIp', e.target.value)}
              style={{flex:1, padding: '4px 8px', border: '1px solid #30363d', borderRadius: '4px', background: '#0d1117', color: '#e6e6e6'}} />
          </div>
          <div className="setting-row">
            <label>OSC Port</label>
            <input type="number" value={allSettings.mixer.mixerPort}
              onChange={e => handleChange('mixer', 'mixerPort', parseInt(e.target.value))}
              style={{width: '80px', padding: '4px 8px', border: '1px solid #30363d', borderRadius: '4px', background: '#0d1117', color: '#e6e6e6'}} />
          </div>
        </div>

        <h3 className="settings-section-title">Gain Staging</h3>
        <div className="settings-body">
          <div className="setting-row">
            <label>Target LUFS</label>
            <input type="range" min="-30" max="-14" value={allSettings.gainStaging.targetLufs}
              onChange={e => handleChange('gainStaging', 'targetLufs', parseInt(e.target.value))} />
            <span className="val">{allSettings.gainStaging.targetLufs}</span>
          </div>
          <div className="setting-row">
            <label>True Peak</label>
            <input type="range" min="-6" max="0" step="0.5" value={allSettings.gainStaging.truePeakLimit}
              onChange={e => handleChange('gainStaging', 'truePeakLimit', parseFloat(e.target.value))} />
            <span className="val">{allSettings.gainStaging.truePeakLimit} dBTP</span>
          </div>
          <div className="setting-row">
            <label>Ratio</label>
            <select value={allSettings.gainStaging.ratio} onChange={e => handleChange('gainStaging', 'ratio', parseInt(e.target.value))}>
              <option value={2}>2:1</option><option value={4}>4:1</option><option value={8}>8:1</option>
            </select>
          </div>
        </div>

        <h3 className="settings-section-title">Auto Fader</h3>
        <div className="settings-body">
          <div className="setting-row">
            <label>Bleed Threshold</label>
            <input type="range" min="-70" max="-30" value={allSettings.autoFader.bleedThreshold}
              onChange={e => handleChange('autoFader', 'bleedThreshold', parseInt(e.target.value))} />
            <span className="val">{allSettings.autoFader.bleedThreshold} dB</span>
          </div>
          <div className="setting-row">
            <label>Target Loudness</label>
            <input type="range" min="-30" max="-12" value={allSettings.autoFader.targetLoudness}
              onChange={e => handleChange('autoFader', 'targetLoudness', parseInt(e.target.value))} />
            <span className="val">{allSettings.autoFader.targetLoudness} LUFS</span>
          </div>
        </div>

        <h3 className="settings-section-title">Voice Control</h3>
        <div className="settings-body">
          <div className="setting-row">
            <label>Язык</label>
            <select value={allSettings.voiceControl.language} onChange={e => handleChange('voiceControl', 'language', e.target.value)}>
              <option value="ru">Русский</option><option value="en">English</option><option value="">Auto</option>
            </select>
          </div>
          <div className="setting-row">
            <label>Модель</label>
            <select value={allSettings.voiceControl.modelSize} onChange={e => handleChange('voiceControl', 'modelSize', e.target.value)}>
              <option value="tiny">Tiny</option><option value="base">Base</option>
              <option value="small">Small</option><option value="medium">Medium</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsTab;
