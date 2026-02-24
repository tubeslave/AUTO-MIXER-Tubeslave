import React, { useState, useEffect, useRef } from 'react';
import websocketService from '../services/websocket';
import './SystemMeasurementTab.css';

const TARGET_BUSES = [
  { id: 'master', name: 'Master', icon: '🔊' },
  { id: 'group', name: 'Group', icon: '👥' },
  { id: 'matrix', name: 'Matrix', icon: '🔀' }
];

const RT60_BANDS = [
  { freq: 63, name: '63 Hz' },
  { freq: 125, name: '125 Hz' },
  { freq: 250, name: '250 Hz' },
  { freq: 500, name: '500 Hz' },
  { freq: 1000, name: '1 kHz' },
  { freq: 2000, name: '2 kHz' },
  { freq: 4000, name: '4 kHz' },
  { freq: 8000, name: '8 kHz' }
];

function SystemMeasurementTab({ selectedDevice, mixerClient }) {
  const [state, setState] = useState('idle');
  const [numPositions, setNumPositions] = useState(6);
  const [currentPosition, setCurrentPosition] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [measurementResult, setMeasurementResult] = useState(null);
  const [selectedBus, setSelectedBus] = useState('master');
  const [busId, setBusId] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const canvasRef = useRef(null);

  useEffect(() => {
    console.log('SystemMeasurementTab mounted');
    
    const handleMeasurementStatus = (data) => {
      console.log('System measurement status:', data);
      if (data.state) setState(data.state);
      if (data.message) setStatusMessage(data.message);
      if (data.current_position !== undefined) setCurrentPosition(data.current_position);
      if (data.result) {
        setMeasurementResult(data.result);
        drawFrequencyResponse(data.result);
      }
    };

    websocketService.on('system_measurement_status', handleMeasurementStatus);
    
    return () => {
      websocketService.off('system_measurement_status', handleMeasurementStatus);
    };
  }, []);

  const drawFrequencyResponse = (result) => {
    if (!canvasRef.current || !result?.frequencies || !result?.magnitude) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    
    [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].forEach(freq => {
      const x = (Math.log10(freq) - Math.log10(20)) / (Math.log10(20000) - Math.log10(20)) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      
      ctx.fillStyle = '#888';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(freq >= 1000 ? `${freq/1000}k` : freq, x, height - 5);
    });
    
    [-30, -20, -10, 0, 10, 20, 30].forEach(db => {
      const y = height - ((db + 40) / 80) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      
      ctx.fillStyle = '#888';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${db}dB`, 30, y + 3);
    });
    
    if (result.frequencies.length > 0 && result.magnitude.length > 0) {
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      result.frequencies.forEach((freq, i) => {
        if (freq >= 20 && freq <= 20000 && i < result.magnitude.length) {
          const x = (Math.log10(freq) - Math.log10(20)) / (Math.log10(20000) - Math.log10(20)) * width;
          const y = height - ((result.magnitude[i] + 40) / 80) * height;
          
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }
    
    if (result.corrections) {
      result.corrections.forEach(corr => {
        const x = (Math.log10(corr.frequency) - Math.log10(20)) / (Math.log10(20000) - Math.log10(20)) * width;
        const y = height / 2;
        
        ctx.fillStyle = corr.gain_db > 0 ? '#00c851' : '#ff4444';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  };

  const handleStartMeasurement = () => {
    websocketService.send({
      type: 'start_system_measurement',
      num_positions: numPositions,
      duration: 15
    });
    setState('measuring');
    setStatusMessage('Starting system measurement...');
  };

  const handlePlaySweep = () => {
    setIsPlaying(true);
    websocketService.send({ type: 'play_sine_sweep' });
    setTimeout(() => setIsPlaying(false), 15000);
  };

  const handleRecordPosition = () => {
    websocketService.send({
      type: 'record_measurement_position',
      position: [currentPosition * 2, 0, 1.5]
    });
    setCurrentPosition(prev => prev + 1);
    setStatusMessage(`Recorded position ${currentPosition + 1}/${numPositions}`);
  };

  const handleAnalyze = () => {
    websocketService.send({ type: 'analyze_measurement' });
    setState('analyzing');
    setStatusMessage('Analyzing measurements...');
  };

  const handleApplyCorrections = () => {
    websocketService.send({
      type: 'apply_measurement_corrections',
      target_bus: selectedBus,
      bus_id: busId
    });
    setStatusMessage(`Applying corrections to ${selectedBus} ${busId}...`);
  };

  const getStatusColor = () => {
    switch (state) {
      case 'measuring': return '#ff9800';
      case 'analyzing': return '#00d4ff';
      case 'complete': return '#00c851';
      case 'error': return '#ff4444';
      default: return '#888';
    }
  };

  console.log('SystemMeasurementTab rendering, state:', state);

  return (
    <div className="system-measurement-tab">
      <section className="measurement-section">
        <h2>📊 System Measurement</h2>
        
        <p className="section-description">
          Sine sweep measurement for room analysis and EQ correction.
          Measure at multiple positions and apply corrections to Master/Group/Matrix.
        </p>

        <div className="instructions-panel">
          <h4>Instructions</h4>
          <ol>
            <li>Place reference measurement microphone at FOH position</li>
            <li>Height: 1.2-1.5m (ear level in audience area)</li>
            <li>Distance from speakers: 1/3 to 1/2 of room depth</li>
            <li>Ensure quiet environment (no audience)</li>
            <li>Measure at {numPositions} positions in audience area</li>
            <li>Use moderate playback level (-12dBFS)</li>
          </ol>
        </div>

        {state === 'idle' && (
          <div className="setup-panel">
            <div className="setting-group">
              <label>Number of Positions:</label>
              <select value={numPositions} onChange={(e) => setNumPositions(parseInt(e.target.value))}>
                <option value={3}>3 positions (small room)</option>
                <option value={6}>6 positions (medium)</option>
                <option value={9}>9 positions (large)</option>
                <option value={12}>12 positions (arena)</option>
              </select>
              <small>More positions = better spatial averaging</small>
            </div>

            <div className="setting-group">
              <label>Sweep Duration:</label>
              <span className="value">15 seconds</span>
              <small>10-20s recommended (15s optimal)</small>
            </div>

            <button className="btn-primary" onClick={handleStartMeasurement}>
              Start Measurement
            </button>
          </div>
        )}

        {state === 'measuring' && (
          <div className="measurement-progress">
            <div className="status-indicator" style={{ color: getStatusColor() }}>
              <span className="status-dot"></span>
              Measuring...
            </div>

            <div className="position-counter">
              Position {Math.min(currentPosition + 1, numPositions)} of {numPositions}
            </div>

            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${(currentPosition / numPositions) * 100}%` }}/>
            </div>

            <div className="measurement-controls">
              <button className={`btn-play ${isPlaying ? 'playing' : ''}`} onClick={handlePlaySweep} disabled={isPlaying}>
                {isPlaying ? '🔊 Playing...' : '▶️ Play Sweep'}
              </button>

              <button className="btn-record" onClick={handleRecordPosition} disabled={isPlaying || currentPosition >= numPositions}>
                ● Record Position
              </button>
            </div>

            {currentPosition >= numPositions && (
              <button className="btn-analyze" onClick={handleAnalyze}>
                Analyze Measurements
              </button>
            )}
          </div>
        )}

        {state === 'analyzing' && (
          <div className="analyzing-panel">
            <div className="spinner"></div>
            <p>Analyzing measurements...</p>
            <ul>
              <li>Calculating impulse response</li>
              <li>Computing frequency response</li>
              <li>Analyzing RT60 per band</li>
              <li>Calculating EQ corrections</li>
            </ul>
          </div>
        )}

        {measurementResult && (
          <div className="results-panel">
            <h4>Measurement Results</h4>
            
            <div className="quality-score">
              <label>Quality Score:</label>
              <span className="score">{(measurementResult.quality * 100).toFixed(0)}%</span>
            </div>

            <div className="graph-container">
              <canvas ref={canvasRef} width={600} height={200} className="frequency-graph"/>
              <div className="graph-labels">
                <span>20 Hz</span>
                <span>100 Hz</span>
                <span>1 kHz</span>
                <span>10 kHz</span>
                <span>20 kHz</span>
              </div>
            </div>

            {measurementResult.rt60 && (
              <div className="rt60-panel">
                <h5>RT60 (Reverb Time)</h5>
                <div className="rt60-bars">
                  {measurementResult.rt60.bands.map((band, i) => (
                    <div key={band} className="rt60-bar">
                      <div className="bar-fill" style={{ 
                        height: `${Math.min(100, (measurementResult.rt60.rt60[i] / 3) * 100)}%`,
                        backgroundColor: measurementResult.rt60.rt60[i] > 2 ? '#ff4444' : 
                                        measurementResult.rt60.rt60[i] > 1.5 ? '#ff9800' : '#00c851'
                      }}/>
                      <span className="band-label">{band >= 1000 ? `${band/1000}k` : band}</span>
                      <span className="rt60-value">{measurementResult.rt60.rt60[i].toFixed(2)}s</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {measurementResult.corrections && measurementResult.corrections.length > 0 && (
              <div className="corrections-panel">
                <h5>Recommended EQ Corrections ({measurementResult.corrections.length})</h5>
                <div className="corrections-list">
                  {measurementResult.corrections.slice(0, 8).map((corr, i) => (
                    <div key={i} className={`correction-item ${corr.gain_db > 0 ? 'boost' : 'cut'}`}>
                      <span className="freq">{Math.round(corr.frequency)} Hz</span>
                      <span className="gain">{corr.gain_db > 0 ? '+' : ''}{corr.gain_db.toFixed(1)} dB</span>
                      <span className="q">Q={corr.q}</span>
                    </div>
                  ))}
                </div>
                {measurementResult.corrections.length > 8 && (
                  <small>+{measurementResult.corrections.length - 8} more corrections</small>
                )}
              </div>
            )}

            <div className="target-selection">
              <h5>Apply Corrections To:</h5>
              <div className="target-buses">
                {TARGET_BUSES.map(bus => (
                  <button key={bus.id} className={`bus-btn ${selectedBus === bus.id ? 'active' : ''}`} onClick={() => setSelectedBus(bus.id)}>
                    <span className="bus-icon">{bus.icon}</span>
                    <span className="bus-name">{bus.name}</span>
                  </button>
                ))}
              </div>

              {selectedBus !== 'master' && (
                <div className="bus-id-input">
                  <label>{selectedBus === 'group' ? 'Group' : 'Matrix'} Number:</label>
                  <input type="number" min={1} max={16} value={busId} onChange={(e) => setBusId(parseInt(e.target.value))}/>
                </div>
              )}

              <button className="btn-apply" onClick={handleApplyCorrections}>
                Apply EQ Corrections
              </button>
            </div>
          </div>
        )}

        {statusMessage && <div className="status-message">{statusMessage}</div>}

        <div className="technical-info">
          <h4>Technical Details</h4>
          <div className="info-grid">
            <div className="info-item"><label>Method:</label><span>Exponential Sine Sweep (Farina, 2000)</span></div>
            <div className="info-item"><label>Frequency Range:</label><span>20 Hz - 20 kHz</span></div>
            <div className="info-item"><label>FFT Size:</label><span>65536 (high resolution)</span></div>
            <div className="info-item"><label>Smoothing:</label><span>1/12 octave</span></div>
            <div className="info-item"><label>Max Correction:</label><span>±10 dB cuts, ±6 dB boost</span></div>
            <div className="info-item"><label>Q Range:</label><span>1.4 - 2.0 (gentle)</span></div>
          </div>
        </div>

        <div className="osc-commands">
          <h4>OSC Commands</h4>
          <code>/systest/start i [num_positions]</code>
          <code>/systest/play</code>
          <code>/systest/record i [position_id]</code>
          <code>/systest/analyze</code>
          <code>/bus/{'{id}'}/eq/{'{band}'}/gain f [gain_db]</code>
        </div>
      </section>
    </div>
  );
}

export default SystemMeasurementTab;
