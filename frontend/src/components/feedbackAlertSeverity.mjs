// Severity classification for live feedback alerts (`feedback_alert` frames).
// Isolated in its own module so the thresholds can be swapped in one place
// when the backend starts publishing a native severity field
// (see backend/feedback_live_monitor.py).

export const FEEDBACK_ALERT_CRITICAL_DB = 6;
export const FEEDBACK_ALERT_WARNING_DB = 3;

export function feedbackAlertSeverity(magnitudeDb) {
  const magnitude = Number(magnitudeDb);
  if (!Number.isFinite(magnitude)) return 'info';
  if (magnitude >= FEEDBACK_ALERT_CRITICAL_DB) return 'critical';
  if (magnitude >= FEEDBACK_ALERT_WARNING_DB) return 'warning';
  return 'info';
}
