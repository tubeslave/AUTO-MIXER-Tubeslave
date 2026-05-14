import { deriveAutoGainSupervision } from './autoGainSupervision';

describe('deriveAutoGainSupervision', () => {
  test('stays idle by default', () => {
    const summary = deriveAutoGainSupervision();

    expect(summary.mode.label).toBe('Idle');
    expect(summary.status.label).toBe('Idle');
    expect(summary.lastBlockedReason).toBe('None');
  });

  test('shows dry run when active but analysis only', () => {
    const summary = deriveAutoGainSupervision({
      liveInputTrimState: {
        active: true,
        analysis_only_mode: true,
        live_apply_enabled: false,
        waiting_channels: [1, 2],
        ready_channels: [],
        applied: [],
        blocked: {},
      },
    });

    expect(summary.mode.label).toBe('Dry Run');
    expect(summary.status.label).toBe('Waiting for signal');
  });

  test('shows live ready before any real apply occurs', () => {
    const summary = deriveAutoGainSupervision({
      liveInputTrimState: {
        active: true,
        analysis_only_mode: false,
        live_apply_enabled: true,
        waiting_channels: [],
        ready_channels: [3],
        applied: [],
        blocked: {},
      },
    });

    expect(summary.mode.label).toBe('Live Ready');
    expect(summary.status.label).toBe('Pending apply');
  });

  test('shows live applying only after applied activity exists', () => {
    const summary = deriveAutoGainSupervision({
      liveInputTrimState: {
        active: true,
        analysis_only_mode: false,
        live_apply_enabled: true,
        waiting_channels: [],
        ready_channels: [],
        applied: [{ channel: 1 }],
        blocked: {},
      },
    });

    expect(summary.mode.label).toBe('Live Applying');
    expect(summary.status.label).toBe('Applied');
  });

  test('surfaces confirmation required without implying apply', () => {
    const summary = deriveAutoGainSupervision({
      liveInputTrimState: {
        active: true,
        analysis_only_mode: false,
        live_apply_enabled: true,
        waiting_channels: [],
        ready_channels: [1],
        applied: [],
        blocked: {},
      },
      latestLiveApplyResult: {
        blocked_reason: 'confirm_live_apply_required',
        message_for_user: 'Live apply blocked: confirmation required.',
      },
    });

    expect(summary.mode.label).toBe('Live Ready');
    expect(summary.status.label).toBe('Waiting for confirmation');
    expect(summary.confirmationRequired).toBe(true);
    expect(summary.lastBlockedReason).toBe('Live apply blocked: confirmation required.');
  });
});
