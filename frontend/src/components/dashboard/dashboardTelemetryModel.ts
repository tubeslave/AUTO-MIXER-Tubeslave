import type { Channel, DashboardRuntime, Tone } from '../../types';
import type { AnalysisCardMock, ChannelDashboardTelemetry } from '../../mock/dashboardMock';

function finiteNumber(value: unknown): value is number {
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

function tracePoint(value: number): number {
  return Math.max(8, Math.min(88, value));
}

function fallbackSpectrum(values: number[]): number[] {
  return values.map((value, index) => tracePoint(value + (index % 3) * 4));
}

function backendTone(label: string): Tone {
  return label.toLowerCase().includes('backend') ? 'green' : 'blue';
}

function backendLabel(label: string): string {
  return label.toLowerCase().includes('backend') ? 'Backend DTO' : label;
}

export function dashboardSpectrumTrace(
  runtime: Pick<DashboardRuntime, 'sourceLabel' | 'spectrumBins'>,
  fallback: number[]
): { values: number[]; sourceLabel: string; sourceTone: Tone; caption: string } {
  const runtimeBins = Array.isArray(runtime.spectrumBins)
    ? runtime.spectrumBins.filter(value => Number.isFinite(value))
    : [];

  if (runtimeBins.length >= 4) {
    const backendSource = runtime.sourceLabel.toLowerCase().includes('backend');
    return {
      values: runtimeBins.map(tracePoint),
      sourceLabel: backendSource ? 'Backend DTO' : runtime.sourceLabel,
      sourceTone: backendSource ? 'green' : 'blue',
      caption: 'Read-only analyzer trace from dashboard DTO'
    };
  }

  return {
    values: fallbackSpectrum(fallback),
    sourceLabel: 'Display-only',
    sourceTone: 'yellow',
    caption: 'Display-only analyzer trace'
  };
}

export function dashboardAnalysisCards(
  runtime: Pick<DashboardRuntime, 'sourceLabel' | 'analysisTraces'>,
  fallbackCards: AnalysisCardMock[]
): AnalysisCardMock[] {
  return fallbackCards.map(card => {
    const runtimeTrace = runtime.analysisTraces?.[card.id as keyof NonNullable<DashboardRuntime['analysisTraces']>];
    const runtimeValues = Array.isArray(runtimeTrace?.values)
      ? runtimeTrace.values.filter(value => Number.isFinite(value)).map(tracePoint)
      : [];

    if (!runtimeValues.length) {
      return card;
    }

    const sourceLabel = backendLabel(runtimeTrace?.sourceLabel ?? runtime.sourceLabel);
    return {
      ...card,
      values: runtimeValues,
      primary: runtimeTrace?.primary || card.primary,
      secondary: runtimeTrace?.secondary || card.secondary,
      sourceLabel,
      sourceTone: runtimeTrace?.sourceTone ?? backendTone(sourceLabel)
    };
  });
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
    gateLabel: channel.gateLabel || fallback.gateLabel,
    compLabel: channel.compLabel || fallback.compLabel,
    pan: finiteNumber(channel.pan) ? channel.pan : fallback.pan,
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
