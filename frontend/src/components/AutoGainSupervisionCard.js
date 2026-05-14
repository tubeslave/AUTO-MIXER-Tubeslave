import React from 'react';

function AutoGainSupervisionCard({ summary }) {
  if (!summary) return null;

  return (
    <div className="autogain-supervision-card">
      <div className="autogain-supervision-header">
        <div className="autogain-supervision-title">AutoGain Supervision</div>
        <span className={`autogain-badge ${summary.mode.tone}`}>{summary.mode.label}</span>
      </div>

      <div className="autogain-supervision-status-row">
        <span className={`autogain-status-pill ${summary.status.tone}`}>{summary.status.label}</span>
        <span className="autogain-status-text">Last blocked reason: {summary.lastBlockedReason}</span>
      </div>

      {summary.confirmationRequired && (
        <div className="autogain-confirmation-banner">
          Confirmation required. No console write sent.
        </div>
      )}

      <div className="autogain-metadata">
        <span className="autogain-chip">ready {summary.counts.ready}</span>
        <span className="autogain-chip">waiting {summary.counts.waiting}</span>
        <span className="autogain-chip">blocked {summary.counts.blocked}</span>
        <span className={`autogain-chip ${summary.meterTrusted ? 'trusted' : 'untrusted'}`}>
          meter {summary.meterTrusted ? 'trusted' : 'untrusted'}
        </span>
        <span className={`autogain-chip ${summary.readyForLive ? 'trusted' : ''}`}>
          {summary.readyForLive ? 'live checklist ready' : 'live checklist pending'}
        </span>
        {summary.frozen && <span className="autogain-chip warning">automation frozen</span>}
      </div>
    </div>
  );
}

export default AutoGainSupervisionCard;

