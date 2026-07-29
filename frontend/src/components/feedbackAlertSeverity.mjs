// Severity classification for live feedback alerts (`feedback_alert` frames).
// Isolated in its own module so the thresholds can be swapped in one place
// when the backend starts publishing a native severity field
// (see backend/feedback_live_monitor.py).
// NOTE: thresholds are in dBFS (post f1b8bb8af). Real feedback lives at
// −30..0 dBFS; monitor default peak_height_db = −40 dBFS.

export const FEEDBACK_ALERT_CRITICAL_DB = -12;
export const FEEDBACK_ALERT_WARNING_DB = -25;
export const FEEDBACK_ALERT_CRITICAL_CONFIDENCE = 0.9;

export function feedbackAlertSeverity(magnitudeDb, confidence) {
  const magnitude = Number(magnitudeDb);
  const conf = Number(confidence);
  if (!Number.isFinite(magnitude)) return 'info';
  if (Number.isFinite(conf) && conf >= FEEDBACK_ALERT_CRITICAL_CONFIDENCE) return 'critical';
  if (magnitude >= FEEDBACK_ALERT_CRITICAL_DB) return 'critical';
  if (magnitude >= FEEDBACK_ALERT_WARNING_DB) return 'warning';
  return 'info';
}
