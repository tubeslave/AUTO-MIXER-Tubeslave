import React from 'react';
import MODULE_SIGNAL_INFO from '../signalInfo';

/**
 * SignalHint — shows a compact banner at the top of each module tab
 * indicating which Dante channels / tap points the module expects.
 *
 * Usage: <SignalHint moduleKey="gain_staging" />
 */
function SignalHint({ moduleKey }) {
  const info = MODULE_SIGNAL_INFO[moduleKey];
  if (!info) return null;

  return (
    <div className="signal-hint">
      <span className="signal-hint-icon">{info.icon}</span>
      <div className="signal-hint-content">
        <span className="signal-hint-signal">{info.signal}</span>
        <span className="signal-hint-desc">{info.description}</span>
      </div>
    </div>
  );
}

export default SignalHint;
