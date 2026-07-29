import test from 'node:test';
import assert from 'node:assert/strict';

import { feedbackAlertSeverity } from '../feedbackAlertSeverity.mjs';
import {
  FEEDBACK_ALERT_JOURNAL_LIMIT,
  FEEDBACK_ALERT_READ_ONLY_NOTE,
  adaptFeedbackAlert,
  appendFeedbackAlert,
  feedbackAlertSeverityTone,
  feedbackAlertTimeLabel
} from '../feedbackAlertCenter.mjs';

function sampleAlert(overrides = {}) {
  return {
    type: 'feedback_alert',
    channel: 3,
    frequency_hz: 2512.5,
    magnitude_db: -10,
    recommended_action: 'notch',
    confidence: 0.87,
    auto_action: 'none',
    read_only: true,
    timestamp: 1753780000.123,
    ...overrides
  };
}

test('feedbackAlertSeverity classifies boundary magnitudes in dBFS', () => {
  const cases = [
    [-10, 'critical'],
    [-12, 'critical'],
    [-12.01, 'warning'],
    [-20, 'warning'],
    [-25, 'warning'],
    [-25.01, 'info'],
    [-40, 'info'],
    ['-10', 'critical'],
    ['not-a-number', 'info'],
    [Number.NaN, 'info'],
    [undefined, 'info']
  ];
  cases.forEach(([magnitude, expected]) => {
    assert.equal(feedbackAlertSeverity(magnitude), expected, String(magnitude));
  });
});

test('feedbackAlertSeverity promotes to critical on high confidence', () => {
  assert.equal(feedbackAlertSeverity(-40, 0.95), 'critical', 'low magnitude + high confidence');
  assert.equal(feedbackAlertSeverity(-25, 0.9), 'critical', 'exact threshold confidence');
  assert.equal(feedbackAlertSeverity(-10, 0.89), 'critical', 'high magnitude + sub-threshold confidence');
  assert.equal(feedbackAlertSeverity(-40, 0.89), 'info', 'low magnitude + sub-threshold confidence');
  assert.equal(feedbackAlertSeverity(-40, undefined), 'info', 'low magnitude + no confidence');
});

test('feedbackAlertSeverityTone maps severities to operator tones', () => {
  assert.equal(feedbackAlertSeverityTone('critical'), 'red');
  assert.equal(feedbackAlertSeverityTone('warning'), 'yellow');
  assert.equal(feedbackAlertSeverityTone('info'), 'blue');
});

test('adaptFeedbackAlert maps the backend schema to the view model', () => {
  const entry = adaptFeedbackAlert(sampleAlert());

  assert.equal(entry.channel, 3);
  assert.equal(entry.frequencyHz, 2512.5);
  assert.equal(entry.magnitudeDb, -10);
  assert.equal(entry.recommendedAction, 'notch');
  assert.equal(entry.confidence, 0.87);
  assert.equal(entry.autoAction, 'none');
  assert.equal(entry.readOnly, true);
  assert.equal(entry.timestamp, 1753780000.123);
  assert.equal(entry.severity, 'critical');
  assert.equal(entry.severityTone, 'red');
  assert.equal(entry.channelLabel, 'Ch 3');
  assert.equal(entry.frequencyLabel, '2512.5 Hz');
  assert.equal(entry.magnitudeLabel, '-10.00 dB');
  assert.equal(entry.confidenceLabel, '87%');
  assert.equal(entry.actionLabel, 'notch');
  assert.match(entry.timeLabel, /^\d{2}:\d{2}:\d{2}$/);
});

test('adaptFeedbackAlert formats wall-clock time from epoch seconds', () => {
  const timestamp = 1753780000.123;
  const date = new Date(timestamp * 1000);
  const pad = value => String(value).padStart(2, '0');
  const expected = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;

  assert.equal(feedbackAlertTimeLabel(timestamp), expected);
  assert.equal(adaptFeedbackAlert(sampleAlert()).timeLabel, expected);
  assert.equal(feedbackAlertTimeLabel(0), 'time unknown');
  assert.equal(feedbackAlertTimeLabel('not-a-number'), 'time unknown');
});

test('adaptFeedbackAlert carries the detection-only read-only note', () => {
  const entry = adaptFeedbackAlert(sampleAlert());

  assert.equal(entry.readOnlyNote, FEEDBACK_ALERT_READ_ONLY_NOTE);
  assert.match(entry.readOnlyNote, /auto_action: none/);
  assert.match(entry.readOnlyNote, /read-only/);
  assert.match(entry.readOnlyNote, /no automatic suppression/);
});

test('adaptFeedbackAlert normalizes warning/info severities and defaults', () => {
  const warning = adaptFeedbackAlert(sampleAlert({ magnitude_db: -20, recommended_action: 'fader' }));
  assert.equal(warning.severity, 'warning');
  assert.equal(warning.severityTone, 'yellow');
  assert.equal(warning.actionLabel, 'fader');

  const info = adaptFeedbackAlert(sampleAlert({ magnitude_db: -30 }));
  assert.equal(info.severity, 'info');
  assert.equal(info.severityTone, 'blue');

  const defaults = adaptFeedbackAlert(sampleAlert({
    recommended_action: '',
    auto_action: undefined,
    read_only: undefined,
    confidence: undefined
  }));
  assert.equal(defaults.recommendedAction, 'manual review');
  assert.equal(defaults.autoAction, 'none');
  assert.equal(defaults.readOnly, true);
  assert.equal(defaults.confidenceLabel, '0%');

  const clamped = adaptFeedbackAlert(sampleAlert({ confidence: 1.4 }));
  assert.equal(clamped.confidence, 1);
  assert.equal(clamped.confidenceLabel, '100%');
});

test('adaptFeedbackAlert rejects frames that fail the schema', () => {
  assert.equal(adaptFeedbackAlert(null), null);
  assert.equal(adaptFeedbackAlert('feedback_alert'), null);
  assert.equal(adaptFeedbackAlert(sampleAlert({ channel: 0 })), null);
  assert.equal(adaptFeedbackAlert(sampleAlert({ channel: 2.5 })), null);
  assert.equal(adaptFeedbackAlert(sampleAlert({ channel: 'three' })), null);
  assert.equal(adaptFeedbackAlert(sampleAlert({ frequency_hz: 'bad' })), null);
  assert.equal(adaptFeedbackAlert(sampleAlert({ magnitude_db: Number.NaN })), null);
});

test('appendFeedbackAlert keeps the journal newest-first', () => {
  const older = sampleAlert({ channel: 3, timestamp: 1753780000.123 });
  const newer = sampleAlert({ channel: 7, timestamp: 1753780005.5 });

  const journal = appendFeedbackAlert(appendFeedbackAlert([], older), newer);

  assert.equal(journal.length, 2);
  assert.equal(journal[0].channel, 7);
  assert.equal(journal[1].channel, 3);
});

test('appendFeedbackAlert caps the journal at 50 entries', () => {
  assert.equal(FEEDBACK_ALERT_JOURNAL_LIMIT, 50);

  let journal = [];
  for (let index = 1; index <= 55; index += 1) {
    journal = appendFeedbackAlert(journal, sampleAlert({
      channel: ((index - 1) % 16) + 1,
      timestamp: 1753780000 + index
    }));
  }

  assert.equal(journal.length, 50);
  assert.equal(journal[0].timestamp, 1753780000 + 55);
  assert.equal(journal[49].timestamp, 1753780000 + 6);
});

test('appendFeedbackAlert ignores invalid frames and tolerates empty journals', () => {
  const journal = appendFeedbackAlert([], sampleAlert());
  const unchanged = appendFeedbackAlert(journal, sampleAlert({ channel: 0 }));

  assert.equal(unchanged, journal);
  assert.equal(unchanged.length, 1);
  assert.equal(appendFeedbackAlert(null, sampleAlert()).length, 1);
  assert.deepEqual(appendFeedbackAlert(undefined, null), []);
});
