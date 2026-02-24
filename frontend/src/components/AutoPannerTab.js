import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoPannerTab.css';

// Genre definitions
const GENRES = [
  { id: 'rock', name: 'Rock', icon: '🎸' },
  { id: 'pop', name: 'Pop', icon: '🎤' },
  { id: 'jazz', name: 'Jazz', icon: '🎷' },
  { id: 'electronic', name: 'Electronic', icon: '🎹' }
];

// 9 discrete pan positions
const PAN_POSITIONS = [
  { angle: -90, name: 'L90', label: 'Full Left' },
  { angle: -67, name: 'L67', label: 'Left Wide' },
  { angle: -45, name: 'L45', label: 'Left' },
  { angle: -22, name: 'L22', label: 'Left Mid' },
  { angle: 0, name: 'CENTER', label: 'Center' },
  { angle: 22, name: 'R22', label: 'Right Mid' },
  { angle: 45, name: 'R45', label: 'Right' },
  { angle: 67, name: 'R67', label: 'Right Wide' },
  { angle: 90, name: 'R90', label: 'Full Right' }
];

// Instrument icons
const INSTRUMENT_ICONS = {
  kick: '🥁',
  snare: '🥁',
  bass: '🎸',
  vocal: '🎤',
  guitar: '🎸',
  keys: '🎹',
  cymbals: '🎵'
};

function AutoPannerTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [active, setActive] = useState(false);
  const [genre, setGenre] = useState('rock');
  const [statusMessage, setStatusMessage] = useState('');
  const [channelData, setChannelData] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Settings
  const [settings, setSettings] = useState({
    lowFreq: 250,
    highFreq: 4000,
    smoothingMs: 50,
    useMultiband: true
  });

  useEffect(() => {
    const handleAutoPannerStatus = (data) => {
      if (data.active !== undefined) setActive(data.active);
      if (data.genre !== undefined) setGenre(data.genre);
      if (data.message) setStatusMessage(data.message);
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setActive(false);
      }
      if (data.channel_data) {
        setChannelData(data.channel_data);
      }
    };

    websocketService.on('auto_panner_status', handleAutoPannerStatus);
    websocketService.getAutoPannerStatus();
    
    return () => {
      websocketService.off('auto_panner_status', handleAutoPannerStatus);
    };
  }, []);

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const handleStart = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Select audio device and channels first.');
      return;
    }

    // Build instrument types from channel names
    const instrumentTypes = {};
    selectedChannels.forEach(ch => {
      const chName = getChannelName(ch).toLowerCase();
      if (chName.includes('kick') || chName.includes('бочка')) {
        instrumentTypes[ch] = 'kick';
      } else if (chName.includes('snare') || chName.includes('малый')) {
        instrumentTypes[ch] = 'snare';
      } else if (chName.includes('bass') || chName.includes('бас')) {
        instrumentTypes[ch] = 'bass';
      } else if (chName.includes('vocal') || chName.includes('vox') || chName.includes('вокал')) {
        instrumentTypes[ch] = 'vocal';
      } else if (chName.includes('guitar') || chName.includes('гитара')) {
        instrumentTypes[ch] = 'guitar';
      } else if (chName.includes('key') || chName.includes('пиано')) {
        instrumentTypes[ch] = 'keys';
      } else if (chName.includes('cymbal') || chName.includes('overhead')) {
        instrumentTypes[ch] = 'cymbals';
      } else {
        instrumentTypes[ch] = 'unknown';
      }
    });

    websocketService.startAutoPanner(
      selectedDevice,
      selectedChannels,
      instrumentTypes,
      {},  // spectral centroids
      genre
    );
    
    setStatusMessage('Starting Auto Panner...');
  };

  const handleStop = () => {
    websocketService.stopAutoPanner();
    setStatusMessage('Stopping Auto Panner...');
  };

  const getPanPositionInfo = (angle) => {
    return PAN_POSITIONS.find(p => p.angle === angle) || PAN_POSITIONS[4];
  };

  const channels = selectedChannels || [];
  const canStart = selectedDevice && channels.length > 0;

  return (
    <div className="auto-panner-tab">
      <section className="auto-panner-section">
        <h2>🎧 Auto Panner (Multi-band Adaptive)</h2>
        
        <p className="section-description">
          3-band adaptive panning with 9 discrete positions and genre-specific templates.
          Based on Intelligent Music Production (Perez Gonzalez & Reiss, 2007).
        </p>

        {/* 3-Band Scheme */}
        <div className="bands-scheme">
          <h4>3-Band Panning Scheme</h4>
          <div className="bands-visual">
            <div className="band low">
              <span className="band-name">Low</span>
              <span className="band-range">&lt; 250 Hz</span>
              <span className="band-pan">→ CENTER</span>
            </div>
            <div className="band mid">
              <span className="band-name">Mid</span>
              <span className="band-range">250 - 4000 Hz</span>
              <span className="band-pan">→ Position-based</span>
            </div>
            <div className="band high">
              <span className="band-name">High</span>
              <span className="band-range">&gt; 4000 Hz</span>
              <span className="band-pan">→ Wide stereo</span>
            </div>
          </div>
        </div>

        {/* Genre Selection */}
        <div className="genre-selection">
          <h4>Genre Template</h4>
          <div className="genre-buttons">
            {GENRES.map(g => (
              <button
                key={g.id}
                className={`genre-btn ${genre === g.id ? 'active' : ''}`}
                onClick={() => setGenre(g.id)}
                disabled={active}
              >
                <span className="genre-icon">{g.icon}</span>
                <span className="genre-name">{g.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Pan Positions Visual */}
        <div className="pan-positions">
          <h4>9 Discrete Positions (Sine/Cosine Law)</h4>
          <div className="pan-ruler">
            {PAN_POSITIONS.map(pos => (
              <div 
                key={pos.angle} 
                className={`pan-mark ${pos.angle === 0 ? 'center' : ''}`}
                style={{ left: `${(pos.angle + 90) / 1.8}%` }}
              >
                <span className="mark-name">{pos.name}</span>
                <span className="mark-angle">{pos.angle}°</span>
              </div>
            ))}
            <div className="pan-line"></div>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="control-buttons">
          {!active ? (
            <button
              className="btn-primary"
              onClick={handleStart}
              disabled={!canStart}
            >
              Start Auto Panner
            </button>
          ) : (
            <button
              className="btn-stop"
              onClick={handleStop}
            >
              Stop Auto Panner
            </button>
          )}
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        {/* Channel Pan Display */}
        {active && Object.keys(channelData).length > 0 && (
          <div className="channels-pan">
            <h4>Channel Panning</h4>
            {channels.map(ch => {
              const data = channelData[ch];
              if (!data) return null;
              
              const posInfo = getPanPositionInfo(data.position);
              const instIcon = INSTRUMENT_ICONS[data.instrument] || '🎵';
              
              return (
                <div key={ch} className="channel-pan-row">
                  <div className="channel-info">
                    <span className="channel-name">{getChannelName(ch)}</span>
                    <span className="instrument">
                      {instIcon} {data.instrument}
                    </span>
                  </div>
                  
                  <div className="pan-visual">
                    <div className="pan-slider">
                      <div 
                        className="pan-indicator"
                        style={{ left: `${(data.position + 90) / 1.8}%` }}
                      >
                        <span className="indicator-triangle">▲</span>
                      </div>
                    </div>
                    <div className="pan-labels">
                      <span>L</span>
                      <span className="position-name">{posInfo.name}</span>
                      <span>R</span>
                    </div>
                  </div>
                  
                  <div className="pan-details">
                    <span className="angle">{data.position}°</span>
                    <span className="confidence">conf: {data.confidence?.toFixed(2)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Genre Templates */}
        <div className="genre-templates">
          <h4>Genre Templates</h4>
          <div className="templates-grid">
            <div className="template-card">
              <h5>🎸 Rock</h5>
              <ul>
                <li>Kick/Snare/Bass/Vocal → Center</li>
                <li>Guitars → L45 / R45</li>
                <li>Keys → L22</li>
                <li>Cymbals → L67 / R67</li>
              </ul>
            </div>
            <div className="template-card">
              <h5>🎤 Pop</h5>
              <ul>
                <li>Rhythm Section → Center</li>
                <li>Guitars → L22 / R67</li>
                <li>Keys → L45 / R22</li>
                <li>Effects → L90 / R90</li>
              </ul>
            </div>
            <div className="template-card">
              <h5>🎷 Jazz</h5>
              <ul>
                <li>Kick → Center</li>
                <li>Bass → L22</li>
                <li>Snare → R22</li>
                <li>Piano → L67</li>
                <li>Horns → Wide spread</li>
              </ul>
            </div>
            <div className="template-card">
              <h5>🎹 Electronic</h5>
              <ul>
                <li>Drums/Bass → Center</li>
                <li>Synths → Wide stereo</li>
                <li>Effects → Extreme positions</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Algorithm Info */}
        <div className="algorithm-info">
          <h4>Algorithm Overview</h4>
          <div className="algorithm-steps">
            <div className="step">
              <span className="step-num">1</span>
              <span className="step-name">3-Band Split</span>
              <small>LR4 crossover: Low/Mid/High</small>
            </div>
            <div className="step">
              <span className="step-num">2</span>
              <span className="step-name">Feature Extraction</span>
              <small>3-tier: RMS/ZCR → Centroid/ILD → Bark</small>
            </div>
            <div className="step">
              <span className="step-num">3</span>
              <span className="step-name">Classification</span>
              <small>Instrument type detection</small>
            </div>
            <div className="step">
              <span className="step-num">4</span>
              <span className="step-name">Genre Template</span>
              <small>Default position lookup</small>
            </div>
            <div className="step">
              <span className="step-num">5</span>
              <span className="step-name">Sine/Cosine Pan</span>
              <small>9 discrete positions</small>
            </div>
          </div>
        </div>

        {/* OSC Commands */}
        <div className="osc-commands">
          <h4>OSC Commands</h4>
          <div className="osc-list">
            <code>/ch/{'{n}'}/mix/pan f [-1 to 1]</code>
            <code>/track/{'{n}'}/position s [L90/L67/L45/L22/CENTER/R22/R45/R67/R90]</code>
            <code>/track/{'{n}'}/instrument s [kick/snare/bass/vocal/guitar/keys/cymbals]</code>
          </div>
        </div>

        {/* Help Text */}
        <div className="help-text">
          <h4>How it works:</h4>
          <ol>
            <li><strong>3-Band Split</strong> — Low (&lt;250Hz) to center, Mid position-based, High wide</li>
            <li><strong>Feature Extraction</strong> — 3-tier: fast (10ms), medium (100ms), slow (optional)</li>
            <li><strong>Instrument Classification</strong> — Based on spectral centroid, crest factor, ZCR</li>
            <li><strong>Genre Template</strong> — Lookup default position for instrument type</li>
            <li><strong>Sine/Cosine Panning</strong> — 9 discrete positions with constant power</li>
          </ol>
          
          <h4>Key Features:</h4>
          <ul>
            <li><strong>Multi-band:</strong> Different panning strategies per frequency band</li>
            <li><strong>Genre Templates:</strong> Rock, Pop, Jazz, Electronic presets</li>
            <li><strong>9 Positions:</strong> Discrete steps from L90 to R90 with sine/cosine law</li>
            <li><strong>Smoothing:</strong> 50ms smoothing to prevent jumps</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default AutoPannerTab;
