import type { Channel, Tone } from '../../types';
import type { ChannelDashboardTelemetry } from '../../mock/dashboardMock';

function finiteNumber(value: number): value is number {
  return Number.isFinite(value);
}

function clippedLevelTone(level: number, fallback: Tone): Tone {
  if (level >= -3) return 'red';
  if (level >= -10) return 'yellow';
  if (level <= -35) return 'blue';
  return fallback;
}

function safeStateForLevel(level: number): string {
  if (level >= -3) return 'CLIP';
  if (level >= -10) return 'HOT';
  if (level <= -35) return 'LOW';
  return 'OK';
}

export function dashboardTelemetryForChannel(
  channel: Channel,
  fallback: ChannelDashboardTelemetry,
  options: { suppressMockSolo?: boolean } = {}
): ChannelDashboardTelemetry {
  const hasRuntimeLevel = finiteNumber(channel.level);
  const level = hasRuntimeLevel ? channel.level : fallback.meterL;
  const muted = channel.status === 'MUTED';
  const inactive = channel.status === 'INACTIVE';

  return {
    ...fallback,
    meterL: hasRuntimeLevel ? level : fallback.meterL,
    meterR: hasRuntimeLevel ? level : fallback.meterR,
    muted,
    solo: options.suppressMockSolo ? false : fallback.solo,
    safeState: muted ? 'MUTE' : inactive ? 'OFF' : safeStateForLevel(level),
    accent: muted
      ? 'red'
      : inactive
        ? 'muted'
        : clippedLevelTone(level, fallback.accent)
  };
}
