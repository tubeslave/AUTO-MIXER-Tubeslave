import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoReverbTab.css';

// Reverb types
const REVERB_TYPES = [
  { id: 'plate', name: 'Plate', icon: '🔷', desc: 'Bright, dense, metallic' },
  { id: 'hall', name: 'Hall', icon: '🏛️', desc: 'Natural concert hall' },
  { id: 'room', name: 'Room', icon: '🏠', desc: 'Small space ambience' },
  { id: 'chamber', name: 'Chamber', icon: '📦', desc: 'Warm, enclosed' },
  { id: 'spring', name: 'Spring', icon: '🌀', desc: 'Vintage, boingy' }
];

// Instrument icons
const INSTRUMENT_ICONS = {
  leadVocal: '🎤',
  lead_vocal: '🎤',
  backingVocal: '🎤',
  backing_vocal: '🎤',
  kick: '🥁',
  snare: '🥁',
  tom: '🥁',
  drums: '🥁',
  hihat: '🎵',
  ride: '🎵',
  overhead: '🎙️',
  percussion: '🥁',
  bass: '🎸',
  electricGuitar: '🎸',
  acousticGuitar: '🎸',
  piano: '🎹',
  keys: '🎹',
  synth: '🎹',
  pads: '🎼',
  strings: '🎻',
  brass: '🎺',
  sax: '🎷',
  woodwinds: '🎵',
  room: '🎙️',
  unknown: '🎵'
};

function AutoReverbTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({});
  const [showDetails, setShowDetails] = useState(false);
  
  // Settings
  const [settings, setSettings] = useState({
    baseLevel: -14,
    drySideMargin: 2,
    predelayMin: 30,
    predelayMax: 50,
    hpfHz: 200,
    lpfHz: 5000
  });

  useEffect(() => {
    const handleAutoReverbStatus = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      if (data.channel_data) {
        setChannelData(data.channel_data);
      }
    };

    websocketService.on('auto_reverb_status', handleAutoReverbStatus);
    websocketService.getAutoReverbStatus();
    
    return () => {
      websocketService.off('auto_reverb_status', handleAutoReverbStatus);
    };
  }, []);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const getInstrumentIcon = (instrument) => {
    return INSTRUMENT_ICONS[instrument] || INSTRUMENT_ICONS.unknown;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    const instrumentTypes = {};
    selectedChannels.forEach(ch => {
      const chName = getChannelName(ch).toLowerCase();
      if (chName.includes('vocal') || chName.includes('vox')) {
        instrumentTypes[ch] = chName.includes('back') ? 'backing_vocal' : 'lead_vocal';
      } else if (chName.includes('kick') || chName.includes('бочка')) {
        instrumentTypes[ch] = 'kick';
      } else if (chName.includes('snare') || chName.includes('малый')) {
        instrumentTypes[ch] = 'snare';
      } else if (chName.includes('tom')) {
        instrumentTypes[ch] = 'tom';
      } else if (chName.includes('drum')) {
        instrumentTypes[ch] = 'drums';
      } else if (chName.includes('bass') || chName.includes('бас')) {
        instrumentTypes[ch] = 'bass';
      } else if (chName.includes('guitar') || chName.includes('гитара')) {
        instrumentTypes[ch] = 'electricGuitar';
      } else if (chName.includes('piano') || chName.includes('рояль')) {
        instrumentTypes[ch] = 'piano';
      } else if (chName.includes('key') || chName.includes('synth')) {
        instrumentTypes[ch] = 'keys';
      } else {
        instrumentTypes[ch] = 'unknown';
      }
    });

    websocketService.startAutoReverb(
      selectedDevice,
      selectedChannels,
      instrumentTypes,
      {},
      {}
    );
    
    setStatusMessage('Starting Auto Reverb analysis...');
  };

  const handleStop = () => {
    websocketService.stopAutoReverb();
    setStatusMessage('Stopping Auto Reverb...');
  };

  const handleCalculate = () => {
    if (!selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select channels first.');
      return;
    }
    
    const instrumentTypes = {};
    selectedChannels.forEach(ch => {
      instrumentTypes[ch] = 'unknown';
    });
    
    websocketService.calculateAutoReverb(selectedChannels, instrumentTypes, {}, {});
    setStatusMessage('Calculating reverb settings...');
  };

  const handleApply = () => {
    websocketService.applyAutoReverb();
    setStatusMessage('Applying reverb settings to mixer...');
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-reverb-tab">
      <section className="auto-reverb-section">
        <h2>Auto Reverb (Intelligent)</h2>
        
        <p className="section-description">
          Intelligent reverberation control based on IMP 7.5 / PSL rules [70].
          Automatically sets reverb type, decay, level, and predelay per instrument.
        </p>

        <div className="best-practices">
          <h4>Best Practices (IMP 7.5)</h4>
          <div className="practices-list">
            <div className="practice">Base level: -14 LU (middle ground)</div>
            <div className="practice">Err on the dry side (+2dB margin)</div>
            <div className="practice">Predelay: 30-50ms (Haas zone)</div>
            <div className="practice">Filters: 200Hz HPF, 5kHz LPF</div>
          </div>
        </div>

        <div className="reverb-types">
          <h4>Reverb Types</h4>
          <div className="types-grid">
            {REVERB_TYPES.map(type => (
              <div key={type.id} className="type-card">
                <span className="type-icon">{type.icon}</span>
                <span className="type-name">{type.name}</span>
                <small>{type.desc}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="control-buttons">
          {!active ? (
            <button className="btn-primary" onClick={handleStart} disabled={!canStart}>
              Start Auto Reverb
            </button>
          ) : (
            <button className="btn-stop" onClick={handleStop}>
              Stop Auto Reverb
            </button>
          )}
          
          <button className="btn-secondary" onClick={handleCalculate} disabled={channels.length === 0}>
            Calculate
          </button>
          
          <button className="btn-secondary" onClick={handleApply} disabled={!active || Object.keys(channelData).length === 0}>
            Apply to Mixer
          </button>
          
          <button className="btn-secondary" onClick={() => setShowDetails(!showDetails)} disabled={Object.keys(channelData).length === 0}>
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {Object.keys(channelData).length > 0 && (
          <div className="channels-reverb">
            <h4>Channel Reverb Settings</h4>
            <div className="reverb-table">
              <div className="table-header">
                <div>Channel</div>
                <div>Type</div>
                <div>Decay</div>
                <div>Level</div>
              </div>
              {Object.entries(channelData).map(([chId, data]) => (
                <div key={chId} className="table-row">
                  <div className="channel-cell">
                    <span>{getInstrumentIcon(data.instrument)}</span>
                    <span>{getChannelName(parseInt(chId))}</span>
                  </div>
                  <div><span className="reverb-type-badge">{data.reverb_type}</span></div>
                  <div>{data.decay_time_s?.toFixed(2)}s</div>
                  <div>{data.reverb_level_lu?.toFixed(1)} LU</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {showDetails && Object.keys(channelData).length > 0 && (
          <div className="detailed-view">
            <h4>Detailed Settings</h4>
            {Object.entries(channelData).map(([chId, data]) => (
              <div key={chId} className="detail-card">
                <h5>{getInstrumentIcon(data.instrument)} {getChannelName(parseInt(chId))}</h5>
                <div className="detail-grid">
                  <div><label>Type:</label> <span>{data.reverb_type}</span></div>
                  <div><label>Decay:</label> <span>{data.decay_time_s?.toFixed(2)}s</span></div>
                  <div><label>Level:</label> <span>{data.reverb_level_lu?.toFixed(1)} LU</span></div>
                  <div><label>Predelay:</label> <span>{data.predelay_ms?.toFixed(1)}ms</span></div>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li>Instrument Detection — Identify instrument type from channel name</li>
            <li>Profile Lookup — Get base settings from PSL rules [70]</li>
            <li>Decay Adaptation — High spectral flux = shorter decay [12, 314]</li>
            <li>Brightness Adaptation — Dull sounds = brighter reverb [119]</li>
            <li>Dry-Side Safety — Apply margin to avoid over-reverb [82]</li>
          </ol>
        </div>
      </section>
    </div>
  );
}

export default AutoReverbTab;
