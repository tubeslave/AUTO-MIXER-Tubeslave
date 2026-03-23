import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './VoiceControlTab.css';

function VoiceControlTab({ globalMode }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [lastCommand, setLastCommand] = useState('');
  const [lastResponse, setLastResponse] = useState('');
  const [language, setLanguage] = useState('ru');
  const [modelSize, setModelSize] = useState('small');

  useEffect(() => {
    const handle = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) setStatusMessage(`Ошибка: ${data.error}`);
      if (data.command) setLastCommand(data.command);
      if (data.response) setLastResponse(data.response);
    };
    const handleLoaded = (data) => {
      if (data.settings?.voiceControl) {
        const vc = data.settings.voiceControl;
        if (vc.language) setLanguage(vc.language);
        if (vc.modelSize) setModelSize(vc.modelSize);
      }
    };
    websocketService.on('voice_control_status', handle);
    websocketService.on('all_settings_loaded', handleLoaded);
    websocketService.getVoiceControlStatus();
    return () => {
      websocketService.off('voice_control_status', handle);
      websocketService.off('all_settings_loaded', handleLoaded);
    };
  }, []);

  const handleToggle = () => {
    if (active) {
      websocketService.stopVoiceControl();
    } else {
      websocketService.startVoiceControl(modelSize, language);
    }
  };

  return (
    <div className="voice-control-tab">
      <div className="module-card">
        <div className="module-actions">
          <button className={`btn-start ${active ? 'stop' : 'go'}`} onClick={handleToggle}>
            {active ? 'Стоп' : 'Старт'}
          </button>
          <select value={language} onChange={e => setLanguage(e.target.value)} disabled={active}
            style={{padding: '4px 8px', border: '1px solid #30363d', borderRadius: '4px', background: '#0d1117', color: '#e6e6e6', fontSize: '0.85em'}}>
            <option value="ru">Русский</option>
            <option value="en">English</option>
            <option value="">Auto</option>
          </select>
          <select value={modelSize} onChange={e => setModelSize(e.target.value)} disabled={active}
            style={{padding: '4px 8px', border: '1px solid #30363d', borderRadius: '4px', background: '#0d1117', color: '#e6e6e6', fontSize: '0.85em'}}>
            <option value="tiny">Tiny</option>
            <option value="base">Base</option>
            <option value="small">Small</option>
            <option value="medium">Medium</option>
          </select>
        </div>
        {statusMessage && <div className="module-status">{statusMessage}</div>}

        {lastCommand && (
          <div className="voice-last">
            <div className="voice-cmd">🎙 {lastCommand}</div>
            {lastResponse && <div className="voice-resp">→ {lastResponse}</div>}
          </div>
        )}
      </div>
    </div>
  );
}

export default VoiceControlTab;
