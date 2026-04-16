import React, { useEffect, useState } from 'react';
import websocketService from '../services/websocket';
import './AIAgentTab.css';

function formatParams(params = {}) {
  const entries = Object.entries(params).slice(0, 4);
  if (!entries.length) return 'no params';
  return entries.map(([key, value]) => {
    const rendered = typeof value === 'object' ? JSON.stringify(value) : value;
    return `${key}: ${rendered}`;
  }).join(', ');
}

function meterPercent(rmsDb) {
  if (rmsDb === undefined || rmsDb === null) return 0;
  return Math.max(0, Math.min(100, ((rmsDb + 90) / 72) * 100));
}

function AIAgentTab({ selectedChannels, availableChannels }) {
  const [status, setStatus] = useState({ is_running: false, mode: 'auto' });
  const [pending, setPending] = useState([]);
  const [history, setHistory] = useState([]);
  const [audioStatus, setAudioStatus] = useState({ running: false });
  const [meters, setMeters] = useState({});
  const [mode, setMode] = useState('auto');
  const [useLlm, setUseLlm] = useState(true);
  const [allowAutoApply, setAllowAutoApply] = useState(true);
  const [message, setMessage] = useState('');

  useEffect(() => {
    const handleStatus = (data) => {
      setStatus(data);
      if (data.mode) setMode(data.mode);
      if (data.error) setMessage(data.error);
    };
    const handleStarted = (data) => {
      setStatus(data);
      if (data.mode) setMode(data.mode);
      setMessage('AI Agent started');
      websocketService.getPendingActions();
    };
    const handleStopped = () => {
      setStatus(prev => ({ ...prev, is_running: false }));
      setMessage('AI Agent stopped');
    };
    const handleEmergencyStopped = () => {
      setStatus(prev => ({ ...prev, is_running: false, emergency_stop: true, mode: 'manual' }));
      setMode('manual');
      setMessage('Emergency stop: agent is in manual mode');
      websocketService.getActionHistory(20);
    };
    const handlePending = (data) => setPending(data.actions || []);
    const handleHistory = (data) => setHistory(data.history || []);
    const handleAudioStatus = (data) => setAudioStatus(data);
    const handleMeters = (data) => setMeters(data.channels || {});
    const handleMode = (data) => {
      if (data.mode) setMode(data.mode);
      setMessage(`Mode: ${data.mode}`);
    };
    const refreshHistory = () => websocketService.getActionHistory(20);

    websocketService.on('agent_status', handleStatus);
    websocketService.on('agent_started', handleStarted);
    websocketService.on('agent_stopped', handleStopped);
    websocketService.on('agent_emergency_stopped', handleEmergencyStopped);
    websocketService.on('pending_actions', handlePending);
    websocketService.on('action_history', handleHistory);
    websocketService.on('audio_capture_status', handleAudioStatus);
    websocketService.on('channel_meters', handleMeters);
    websocketService.on('agent_mode_changed', handleMode);
    websocketService.on('action_approved', refreshHistory);
    websocketService.on('all_actions_approved', refreshHistory);

    websocketService.getAgentStatus();
    websocketService.getPendingActions();
    websocketService.getActionHistory(20);
    websocketService.getAudioCaptureStatus();
    websocketService.getChannelMeters();

    const meterTimer = setInterval(() => {
      websocketService.getAudioCaptureStatus();
      websocketService.getChannelMeters();
    }, 2000);

    return () => {
      clearInterval(meterTimer);
      websocketService.off('agent_status', handleStatus);
      websocketService.off('agent_started', handleStarted);
      websocketService.off('agent_stopped', handleStopped);
      websocketService.off('agent_emergency_stopped', handleEmergencyStopped);
      websocketService.off('pending_actions', handlePending);
      websocketService.off('action_history', handleHistory);
      websocketService.off('audio_capture_status', handleAudioStatus);
      websocketService.off('channel_meters', handleMeters);
      websocketService.off('agent_mode_changed', handleMode);
      websocketService.off('action_approved', refreshHistory);
      websocketService.off('all_actions_approved', refreshHistory);
    };
  }, []);

  const channels = selectedChannels || [];
  const visibleChannels = channels.slice(0, 16);
  const startAgent = () => {
    websocketService.startAgent(mode, channels, useLlm, allowAutoApply);
    setMessage('Starting AI Agent...');
  };
  const stopAgent = () => websocketService.stopAgent();
  const refresh = () => {
    websocketService.updateAgentState(channels);
    websocketService.getPendingActions();
    websocketService.getActionHistory(20);
    websocketService.getAudioCaptureStatus();
    websocketService.getChannelMeters();
  };
  const changeMode = (nextMode) => {
    setMode(nextMode);
    websocketService.setAgentMode(nextMode, channels, useLlm, allowAutoApply, status.is_running);
  };

  return (
    <div className="ai-agent-tab">
      <div className="module-card agent-hero">
        <div>
          <h3>AI Mixing Agent</h3>
          <p>
            Rules + LLM recommendations. Test mode default: agent applies corrections to the real mixer immediately.
          </p>
        </div>
        <div className={`agent-state ${status.is_running ? 'on' : 'off'}`}>
          {status.is_running ? 'Running' : 'Idle'}
        </div>
      </div>

      <div className="module-card">
        <div className="module-actions">
          <select value={mode} disabled={status.is_running} onChange={e => changeMode(e.target.value)}>
            <option value="suggest">Suggest</option>
            <option value="manual">Manual</option>
            <option value="auto">Auto</option>
          </select>
          <label className="agent-toggle">
            <span>Use LLM</span>
            <input type="checkbox" checked={useLlm} disabled={status.is_running} onChange={e => setUseLlm(e.target.checked)} />
          </label>
          <label className="agent-toggle danger">
            <span>Allow Auto Apply</span>
            <input
              type="checkbox"
              checked={allowAutoApply}
              disabled={status.is_running}
              onChange={e => setAllowAutoApply(e.target.checked)}
            />
          </label>
          <button className={`btn-start ${status.is_running ? 'stop' : 'go'}`} onClick={status.is_running ? stopAgent : startAgent}>
            {status.is_running ? 'Stop Agent' : 'Start Agent'}
          </button>
          <button className="btn-sm danger" onClick={() => websocketService.emergencyStopAgent()}>
            Emergency Stop
          </button>
          <button className="btn-sm" onClick={refresh}>Refresh State</button>
        </div>
        <div className="module-status">
          Mode: {status.mode || mode} | Channels tracked: {status.channels_tracked || 0} | Pending: {status.pending_actions || pending.length}
        </div>
        {message && <div className="module-status">{message}</div>}
      </div>

      <div className="module-card">
        <div className="agent-section-head">
          <h3>Audio Capture</h3>
          <button className="btn-sm" onClick={() => websocketService.getChannelMeters()}>Refresh Meters</button>
        </div>
        <div className="module-status">
          Capture: {audioStatus.running ? 'Running' : 'Stopped'} | Active meters: {Object.keys(meters).length}
        </div>
        {!visibleChannels.length && <div className="module-status">Select channels on Connect to inspect agent input meters.</div>}
        {visibleChannels.length > 0 && (
          <div className="agent-meter-grid">
            {visibleChannels.map(ch => {
              const meter = meters[ch] || meters[String(ch)];
              const channelName = availableChannels?.find(c => c.id === ch)?.name || `Ch ${ch}`;
              const rms = meter?.rms_db;
              return (
                <div className="agent-meter" key={ch}>
                  <div className="agent-meter-head">
                    <span>{channelName}</span>
                    <span>{rms !== undefined ? `${rms} dB` : '--'}</span>
                  </div>
                  <div className="agent-meter-bar">
                    <div className="agent-meter-fill" style={{ width: `${meterPercent(rms)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="module-card">
        <div className="agent-section-head">
          <h3>Pending Actions</h3>
          <div>
            <button className="btn-sm" onClick={() => websocketService.approveAllActions()} disabled={!pending.length}>Approve All</button>
            <button className="btn-sm danger" onClick={() => websocketService.dismissAllActions()} disabled={!pending.length}>Dismiss All</button>
          </div>
        </div>
        {!pending.length && <div className="module-status">No pending actions yet. Refresh state or wait for the next agent cycle.</div>}
        {pending.map(action => (
          <div className="agent-action" key={`${action.index}-${action.timestamp}`}>
            <div>
              <div className="agent-action-title">Ch {action.channel}: {action.type}</div>
              <div className="agent-action-reason">{action.reason || 'No reason provided'}</div>
              <div className="agent-action-meta">
                {action.source} | confidence {action.confidence} | {formatParams(action.parameters)}
              </div>
            </div>
            <div className="agent-action-buttons">
              <button className="btn-sm" onClick={() => websocketService.approveAction(action.index)}>Approve</button>
              <button className="btn-sm danger" onClick={() => websocketService.dismissAction(action.index)}>Dismiss</button>
            </div>
          </div>
        ))}
      </div>

      <div className="module-card">
        <h3>Action History</h3>
        {!history.length && <div className="module-status">No approved/applied actions yet.</div>}
        {history.map((action, idx) => (
          <div className="agent-history" key={`${idx}-${action.timestamp}`}>
            Ch {action.channel}: {action.type} | {action.reason}
          </div>
        ))}
      </div>
    </div>
  );
}

export default AIAgentTab;
