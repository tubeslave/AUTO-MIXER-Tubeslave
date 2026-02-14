import React, { useState, useEffect } from 'react';
import websocketService from '../services/websocket';
import './AutoSoundcheckTab.css';

// Default timings for each function
const DEFAULT_TIMINGS = {
  gain_staging: 30,
  phase_alignment: 30,
  auto_eq: 30,
  auto_fader: 30
};

// Step definitions
const STEPS = [
  { id: 'reset', name: 'Reset to Defaults', key: 'reset' },
  { id: 'gain_staging', name: 'GAIN STAGING', key: 'gain_staging' },
  { id: 'phase_alignment', name: 'PHASE ALIGNMENT', key: 'phase_alignment' },
  { id: 'auto_eq', name: 'AUTO EQ', key: 'auto_eq' },
  { id: 'auto_fader', name: 'AUTO FADER', key: 'auto_fader' }
];

function AutoSoundcheckTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(null);
  const [stepProgress, setStepProgress] = useState(0);
  const [stepTimeRemaining, setStepTimeRemaining] = useState(0);
  const [timings, setTimings] = useState(DEFAULT_TIMINGS);
  const [statusMessage, setStatusMessage] = useState('');
  const [stepStatuses, setStepStatuses] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [logs, setLogs] = useState([]);

  // Initialize step statuses
  useEffect(() => {
    const initialStatuses = {};
    STEPS.forEach(step => {
      initialStatuses[step.key] = 'pending';
    });
    setStepStatuses(initialStatuses);
  }, []);

  // WebSocket event handlers
  useEffect(() => {
    const handleAutoSoundcheckStatus = (data) => {
      console.log('Auto soundcheck status update:', data);
      
      if (data.is_running !== undefined) {
        setIsRunning(data.is_running);
      }
      
      if (data.current_step !== undefined) {
        setCurrentStep(data.current_step);
        
        // Update step statuses
        setStepStatuses(prev => {
          const updated = { ...prev };
          // Mark previous steps as complete
          if (data.current_step) {
            const stepIndex = STEPS.findIndex(s => s.key === data.current_step);
            if (stepIndex > 0) {
              for (let i = 0; i < stepIndex; i++) {
                if (updated[STEPS[i].key] === 'running') {
                  updated[STEPS[i].key] = 'complete';
                }
              }
            }
            // Mark current step as running
            updated[data.current_step] = 'running';
          }
          return updated;
        });
      }
      
      if (data.step_progress !== undefined) {
        setStepProgress(data.step_progress);
      }
      
      if (data.step_time_remaining !== undefined) {
        setStepTimeRemaining(data.step_time_remaining);
      }
      
      if (data.message) {
        setStatusMessage(data.message);
        // Add to logs
        setLogs(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: data.message
        }]);
      }
      
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setIsRunning(false);
        setCurrentStep(null);
        setStepProgress(0);
        setStepTimeRemaining(0);
        
        // Mark current step as error
        if (data.current_step) {
          setStepStatuses(prev => ({
            ...prev,
            [data.current_step]: 'error'
          }));
        }
        
        setLogs(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: `ERROR: ${data.error}`,
          isError: true
        }]);
      }
      
      // Handle completion
      if (data.complete) {
        setIsRunning(false);
        setCurrentStep(null);
        setStepProgress(100);
        setStepTimeRemaining(0);
        
        // Mark all steps as complete
        setStepStatuses(prev => {
          const updated = { ...prev };
          STEPS.forEach(step => {
            if (updated[step.key] !== 'error') {
              updated[step.key] = 'complete';
            }
          });
          return updated;
        });
        
        setLogs(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: 'Soundcheck cycle complete!'
        }]);
      }
    };

    websocketService.on('auto_soundcheck_status', handleAutoSoundcheckStatus);
    websocketService.getAutoSoundcheckStatus();
    
    return () => {
      websocketService.off('auto_soundcheck_status', handleAutoSoundcheckStatus);
    };
  }, []);

  const handleStartAutoCheck = () => {
    if (!selectedDevice) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    if (selectedChannels.length === 0) {
      setStatusMessage('Please select channels to process');
      return;
    }
    
    // Reset logs and statuses
    setLogs([]);
    const initialStatuses = {};
    STEPS.forEach(step => {
      initialStatuses[step.key] = 'pending';
    });
    setStepStatuses(initialStatuses);
    setStepProgress(0);
    setStepTimeRemaining(0);
    
    // Build channel settings (similar to other tabs)
    const channelSettings = {};
    selectedChannels.forEach(channelId => {
      channelSettings[channelId] = {
        preset: 'custom'
      };
    });
    
    const channelMapping = {};
    selectedChannels.forEach(ch => {
      channelMapping[ch] = ch;
    });
    
    console.log('Starting AUTO SOUNDCHECK with:', {
      device: selectedDevice,
      channels: selectedChannels,
      channelSettings,
      channelMapping,
      timings
    });
    
    setStatusMessage('Starting AUTO SOUNDCHECK...');
    websocketService.startAutoSoundcheck(
      selectedDevice,
      selectedChannels,
      channelSettings,
      channelMapping,
      timings
    );
  };

  const handleStopAutoCheck = () => {
    setStatusMessage('Stopping AUTO SOUNDCHECK...');
    websocketService.stopAutoSoundcheck();
  };

  const handleTimingChange = (functionKey, value) => {
    setTimings(prev => ({
      ...prev,
      [functionKey]: parseInt(value) || 0
    }));
  };

  const getChannelName = (channelId) => {
    const channel = availableChannels.find(ch => ch.id === channelId);
    return channel?.name || `Channel ${channelId}`;
  };

  const formatTime = (seconds) => {
    if (seconds < 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (selectedChannels.length === 0) {
    return (
      <div className="auto-soundcheck-tab">
        <div className="auto-soundcheck-section">
          <h2>AUTO SOUNDCHECK</h2>
          <div className="no-channels-message">
            <p>No channels selected for processing.</p>
            <p>Please go to the "Mixer Connection" tab and select channels to process.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auto-soundcheck-tab">
      <div className="auto-soundcheck-section">
        <h2>AUTO SOUNDCHECK</h2>
        
        <div className="auto-soundcheck-header">
          <p className="section-description">
            Automatic sequential execution of GAIN STAGING → PHASE ALIGNMENT → AUTO EQ → AUTO FADER.
            All functions are reset to defaults before starting the cycle.
          </p>
        </div>

        {/* Settings Panel */}
        <div className="settings-panel">
          <div className="settings-header" onClick={() => setShowSettings(!showSettings)}>
            <h3>Measurement Timings</h3>
            <span className="toggle-icon">{showSettings ? '▼' : '▶'}</span>
          </div>
          
          {showSettings && (
            <div className="settings-content">
              <div className="timing-group">
                <label>GAIN STAGING Duration</label>
                <div className="timing-control">
                  <input
                    type="range"
                    min="10"
                    max="120"
                    step="5"
                    value={timings.gain_staging}
                    onChange={(e) => handleTimingChange('gain_staging', e.target.value)}
                    disabled={isRunning}
                  />
                  <span className="timing-value">{timings.gain_staging} sec</span>
                </div>
              </div>

              <div className="timing-group">
                <label>PHASE ALIGNMENT Duration</label>
                <div className="timing-control">
                  <input
                    type="range"
                    min="10"
                    max="120"
                    step="5"
                    value={timings.phase_alignment}
                    onChange={(e) => handleTimingChange('phase_alignment', e.target.value)}
                    disabled={isRunning}
                  />
                  <span className="timing-value">{timings.phase_alignment} sec</span>
                </div>
              </div>

              <div className="timing-group">
                <label>AUTO EQ Duration</label>
                <div className="timing-control">
                  <input
                    type="range"
                    min="10"
                    max="120"
                    step="5"
                    value={timings.auto_eq}
                    onChange={(e) => handleTimingChange('auto_eq', e.target.value)}
                    disabled={isRunning}
                  />
                  <span className="timing-value">{timings.auto_eq} sec</span>
                </div>
              </div>

              <div className="timing-group">
                <label>AUTO FADER Duration</label>
                <div className="timing-control">
                  <input
                    type="range"
                    min="10"
                    max="120"
                    step="5"
                    value={timings.auto_fader}
                    onChange={(e) => handleTimingChange('auto_fader', e.target.value)}
                    disabled={isRunning}
                  />
                  <span className="timing-value">{timings.auto_fader} sec</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <div className="control-header">
            <h3>Auto Soundcheck Control</h3>
            <div className="status-indicators">
              <div className={`status-indicator ${isRunning ? 'active' : 'inactive'}`}>
                <span className="status-dot"></span>
                <span>{isRunning ? 'Running' : 'Idle'}</span>
              </div>
            </div>
          </div>
          
          {isRunning && (
            <div className="progress-section">
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${stepProgress}%` }}></div>
                <span className="progress-text">
                  {stepProgress.toFixed(1)}% {stepTimeRemaining > 0 ? `- ${formatTime(stepTimeRemaining)} remaining` : ''}
                </span>
              </div>
            </div>
          )}
          
          <div className="control-buttons">
            {!isRunning ? (
              <button
                className="btn-control start"
                onClick={handleStartAutoCheck}
                disabled={!selectedDevice || selectedChannels.length === 0}
              >
                AUTO CHECK
              </button>
            ) : (
              <button
                className="btn-control stop"
                onClick={handleStopAutoCheck}
              >
                STOP
              </button>
            )}
          </div>
          
          {statusMessage && (
            <div className="status-message">
              {statusMessage}
            </div>
          )}
        </div>

        {/* Steps Status */}
        <div className="steps-status">
          <h3>Cycle Steps</h3>
          <div className="steps-list">
            {STEPS.map((step, index) => {
              const status = stepStatuses[step.key] || 'pending';
              return (
                <div key={step.key} className={`step-item ${status} ${currentStep === step.key ? 'current' : ''}`}>
                  <div className="step-number">{index + 1}</div>
                  <div className="step-content">
                    <div className="step-name">{step.name}</div>
                    {currentStep === step.key && isRunning && stepProgress > 0 && (
                      <div className="step-progress">
                        {stepProgress.toFixed(1)}% {stepTimeRemaining > 0 ? `- ${formatTime(stepTimeRemaining)} remaining` : ''}
                      </div>
                    )}
                  </div>
                  <div className="step-status-icon">
                    {status === 'complete' && '✓'}
                    {status === 'running' && '⟳'}
                    {status === 'error' && '✗'}
                    {status === 'pending' && '○'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Logs */}
        {logs.length > 0 && (
          <div className="logs-section">
            <h3>Operation Log</h3>
            <div className="logs-container">
              {logs.slice(-20).map((log, index) => (
                <div key={index} className={`log-entry ${log.isError ? 'error' : ''}`}>
                  <span className="log-time">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoSoundcheckTab;
