import test from 'node:test';
import assert from 'node:assert/strict';

import { buildAutofohSessionSummary } from '../autoSoundcheckSummary.mjs';

test('buildAutofohSessionSummary returns null without report data', () => {
  assert.equal(buildAutofohSessionSummary({}), null);
});

test('buildAutofohSessionSummary builds chips and detail lines from report payload', () => {
  const summary = buildAutofohSessionSummary({
    autofoh_session_report_summary: 'AutoFOH session report: events=12; sent=4; blocked=3; guard_blocks=2',
    autofoh_report_path: '/tmp/autofoh_session_report.json',
    autofoh_session_report: {
      action_sent_count: 4,
      action_blocked_count: 3,
      guard_block_count: 2,
      evaluation_count: 5,
      rollback_count: 1,
      channels_with_guard_blocks: [1, 7, 12],
      guard_blocks_by_reason: {
        phase_target_guard_blocked: 2,
        low_confidence: 1,
      },
      guard_blocks_by_action_type: {
        ChannelFaderMove: 2,
        ChannelEQMove: 1,
      },
      guard_blocks_by_phase: {
        FULL_BAND_LEARNING: 2,
      },
      guard_blocks_by_runtime_state: {
        CHORUS: 1,
        SNAPSHOT_LOCK: 1,
      },
    },
  });

  assert.equal(summary.title, 'AutoFOH Session Summary');
  assert.equal(summary.summaryText, 'AutoFOH session report: events=12; sent=4; blocked=3; guard_blocks=2');
  assert.equal(summary.hasExpandableDetails, true);
  assert.deepEqual(summary.chips, [
    { label: 'Sent', value: 4 },
    { label: 'Blocked', value: 3 },
    { label: 'Guard Blocks', value: 2 },
    { label: 'Evaluations', value: 5 },
    { label: 'Rollbacks', value: 1 },
  ]);
  assert.equal(summary.detailLines[0], 'Guarded channels: Ch 1, Ch 7, Ch 12');
  assert.match(summary.detailLines[1], /Top guard reasons: phase_target_guard_blocked \(2\), low_confidence \(1\)/);
  assert.match(summary.detailLines[2], /Blocked action types: ChannelFaderMove \(2\), ChannelEQMove \(1\)/);
  assert.equal(summary.detailLines[3], 'Most restrictive phase: FULL_BAND_LEARNING (2)');
  assert.equal(summary.detailLines[4], 'Report: /tmp/autofoh_session_report.json');
  assert.deepEqual(summary.sections, [
    {
      title: 'Guarded Channels',
      items: ['Ch 1', 'Ch 7', 'Ch 12'],
    },
    {
      title: 'Top Guard Reasons',
      items: ['phase_target_guard_blocked (2)', 'low_confidence (1)'],
    },
    {
      title: 'Blocked Action Types',
      items: ['ChannelFaderMove (2)', 'ChannelEQMove (1)'],
    },
    {
      title: 'Runtime Phases',
      items: ['FULL_BAND_LEARNING (2)'],
    },
    {
      title: 'Runtime States',
      items: ['CHORUS (1)', 'SNAPSHOT_LOCK (1)'],
    },
    {
      title: 'Session Report',
      items: ['/tmp/autofoh_session_report.json'],
    },
  ]);
});

test('buildAutofohSessionSummary falls back to summary text when report body is missing', () => {
  const summary = buildAutofohSessionSummary({
    autofoh_session_report_summary: 'AutoFOH session report: events=4; sent=1; blocked=0; guard_blocks=0',
  });

  assert.equal(summary.summaryText, 'AutoFOH session report: events=4; sent=1; blocked=0; guard_blocks=0');
  assert.deepEqual(summary.chips, []);
  assert.deepEqual(summary.detailLines, []);
  assert.deepEqual(summary.sections, []);
  assert.equal(summary.hasExpandableDetails, false);
});
