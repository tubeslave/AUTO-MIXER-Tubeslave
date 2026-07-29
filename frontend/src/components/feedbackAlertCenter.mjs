import { feedbackAlertSeverity } from './feedbackAlertSeverity.mjs';

// Journal cap for the operator Alert Center. The backend already rate-limits
// `feedback_alert` frames (per-channel min interval + global cap/min), so the
// UI renders every arrived alert without additional throttling.
export const FEEDBACK_ALERT_JOURNAL_LIMIT = 50;

// Detection-only v1: the live monitor never writes to the console.
export const FEEDBACK_ALERT_READ_ONLY_NOTE = 'auto_action: none / read-only — no automatic suppression';

export function feedbackAlertSeverityTone(severity) {
  if (severity === 'critical') return 'red';
  if (severity === 'warning') return 'yellow';
  return 'blue';
}

export function feedbackAlertTimeLabel(timestampSeconds) {
  const seconds = Number(timestampSeconds);
  if (!Number.isFinite(seconds) || seconds <= 0) return 'time unknown';
  const date = new Date(seconds * 1000);
  const pad = value => String(value).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function frequencyLabel(value) {
  const rounded = Number(value.toFixed(1));
  return `${Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)} Hz`;
}

// Maps one backend `feedback_alert` frame (see feedback_live_monitor.py) to
// the Alert Center view model. Returns null for frames that fail the schema.
export function adaptFeedbackAlert(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const channel = Number(payload.channel);
  const frequencyHz = Number(payload.frequency_hz);
  const magnitudeDb = Number(payload.magnitude_db);
  if (!Number.isInteger(channel) || channel < 1) return null;
  if (!Number.isFinite(frequencyHz) || !Number.isFinite(magnitudeDb)) return null;

  const severity = feedbackAlertSeverity(magnitudeDb);
  const recommendedAction = String(payload.recommended_action ?? '').trim() || 'manual review';
  const autoAction = String(payload.auto_action ?? 'none').trim() || 'none';
  const rawConfidence = Number(payload.confidence);
  const confidence = Number.isFinite(rawConfidence) ? Math.min(1, Math.max(0, rawConfidence)) : 0;
  const timestamp = Number(payload.timestamp);

  return {
    channel,
    frequencyHz,
    magnitudeDb,
    recommendedAction,
    confidence,
    autoAction,
    readOnly: payload.read_only !== false,
    timestamp: Number.isFinite(timestamp) ? timestamp : 0,
    severity,
    severityTone: feedbackAlertSeverityTone(severity),
    channelLabel: `Ch ${channel}`,
    frequencyLabel: frequencyLabel(frequencyHz),
    magnitudeLabel: `${magnitudeDb.toFixed(2)} dB`,
    confidenceLabel: `${Math.round(confidence * 100)}%`,
    actionLabel: recommendedAction.replace(/_/g, ' '),
    timeLabel: feedbackAlertTimeLabel(timestamp),
    readOnlyNote: FEEDBACK_ALERT_READ_ONLY_NOTE
  };
}

// Appends one raw `feedback_alert` frame to the journal, newest first, capped
// at FEEDBACK_ALERT_JOURNAL_LIMIT entries. Invalid frames leave the journal
// untouched.
export function appendFeedbackAlert(journal, payload, limit = FEEDBACK_ALERT_JOURNAL_LIMIT) {
  const current = Array.isArray(journal) ? journal : [];
  const entry = adaptFeedbackAlert(payload);
  if (!entry) return current;
  return [entry, ...current].slice(0, limit);
}
