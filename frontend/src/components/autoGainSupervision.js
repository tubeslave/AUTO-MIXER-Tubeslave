const MODE_META = {
  idle: { label: 'Idle', tone: 'idle' },
  dryRun: { label: 'Dry Run', tone: 'dry-run' },
  liveReady: { label: 'Live Ready', tone: 'live-ready' },
  liveApplying: { label: 'Live Applying', tone: 'live-applying' },
};

const STATUS_META = {
  idle: { label: 'Idle', tone: 'idle' },
  monitoring: { label: 'Monitoring', tone: 'neutral' },
  waitingForSignal: { label: 'Waiting for signal', tone: 'neutral' },
  pendingApply: { label: 'Pending apply', tone: 'warning' },
  waitingForConfirmation: { label: 'Waiting for confirmation', tone: 'warning' },
  blocked: { label: 'Blocked', tone: 'danger' },
  applied: { label: 'Applied', tone: 'success' },
};

function firstBlockedReasonFromBlockedMap(blockedMap = {}) {
  const firstEntry = Object.values(blockedMap || {})[0];
  if (!firstEntry || typeof firstEntry !== 'object') return '';
  return (
    firstEntry.reason ||
    firstEntry.blocked_reason ||
    firstEntry.failure_reason ||
    firstEntry.waiting_reason ||
    ''
  );
}

function firstBlockedReasonFromChannels(channels = {}) {
  const firstChannel = Object.values(channels || {}).find((channel) => channel?.blocked_reason);
  return firstChannel?.blocked_reason || '';
}

export function deriveAutoGainSupervision({
  liveInputTrimState,
  latestLiveApplyResult,
  readyForLive,
  freezeStatus,
} = {}) {
  const state = liveInputTrimState || {};
  const active = Boolean(state.active);
  const appliedCount = Array.isArray(state.applied) ? state.applied.length : 0;
  const waitingCount = Array.isArray(state.waiting_channels) ? state.waiting_channels.length : 0;
  const readyCount = Array.isArray(state.ready_channels) ? state.ready_channels.length : 0;
  const blockedMap = state.blocked || {};
  const blockedCount = Object.keys(blockedMap).length;
  const channels = state.channels || {};
  const confirmationRequired = latestLiveApplyResult?.blocked_reason === 'confirm_live_apply_required';

  let modeKey = 'idle';
  if (active) {
    if (state.analysis_only_mode === true || state.live_apply_enabled === false) {
      modeKey = 'dryRun';
    } else if (appliedCount > 0) {
      modeKey = 'liveApplying';
    } else {
      modeKey = 'liveReady';
    }
  }

  let statusKey = 'idle';
  if (active) {
    if (confirmationRequired) {
      statusKey = 'waitingForConfirmation';
    } else if (blockedCount > 0 || firstBlockedReasonFromChannels(channels)) {
      statusKey = 'blocked';
    } else if (appliedCount > 0) {
      statusKey = 'applied';
    } else if (readyCount > 0) {
      statusKey = 'pendingApply';
    } else if (waitingCount > 0 && readyCount === 0) {
      statusKey = 'waitingForSignal';
    } else {
      statusKey = 'monitoring';
    }
  }

  const lastBlockedReason = (
    (latestLiveApplyResult?.blocked_reason ? latestLiveApplyResult?.message_for_user : '') ||
    firstBlockedReasonFromBlockedMap(blockedMap) ||
    firstBlockedReasonFromChannels(channels) ||
    'None'
  );

  return {
    mode: MODE_META[modeKey],
    status: STATUS_META[statusKey],
    confirmationRequired,
    lastBlockedReason,
    counts: {
      applied: appliedCount,
      waiting: waitingCount,
      ready: readyCount,
      blocked: blockedCount,
    },
    meterTrusted: Boolean(state?.meter_status?.trusted),
    readyForLive: Boolean(readyForLive?.ready),
    frozen: Boolean(
      freezeStatus?.gain_staging_frozen ||
      freezeStatus?.auto_fader?.automation_frozen ||
      freezeStatus?.auto_fader_v2?.automation_frozen
    ),
  };
}

