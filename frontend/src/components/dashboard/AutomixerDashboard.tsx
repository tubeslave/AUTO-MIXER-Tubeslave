import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  analysisCards,
  channelTelemetryById,
  dashboardTabs,
  spectrumBins
} from '../../mock/dashboardMock';
import type { AnalysisCardMock, ChannelDashboardTelemetry, DashboardTab } from '../../mock/dashboardMock';
import type { Channel, DashboardRuntime, Tone, UiActions, UiState } from '../../types';
import { dashboardAnalysisCards, dashboardSpectrumTrace, dashboardTelemetryForChannel } from './dashboardTelemetryModel';

const toneColors: Record<Tone, string> = {
  green: '#00d26a',
  blue: '#1687ff',
  purple: '#a855f7',
  yellow: '#ff9f1a',
  red: '#ff4d4f',
  muted: '#5f6b78'
};

const spectralBandLabels = ['Sub', 'Bass', 'Low M', 'Mid', 'High M', 'Presence', 'Brilliance'];
const dynamicsGroupLabels = ['Drums', 'Bass', 'Guitars', 'Vocals', 'Other'];

function safeStateTextClass(safeState: string) {
  if (safeState === 'MUTE' || safeState === 'CLIP') return 'text-danger';
  if (safeState === 'HOT') return 'text-warn';
  if (safeState === 'FOCUS') return 'text-ai';
  if (safeState === 'LOW') return 'text-audio';
  if (safeState === 'OFF' || safeState === 'NO METER' || safeState === 'NO SIGNAL') return 'text-muted';
  return 'text-online';
}

function formatNumber(value: number, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function parseLabelNumber(label: string, fallback: number) {
  const match = label.match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : fallback;
}

function metricDisplay(value: number, available: boolean, noSignalBelow?: number) {
  if (!available) return '—';
  if (noSignalBelow !== undefined && value <= noSignalBelow) return 'NO SIGNAL';
  return formatNumber(value);
}

const energyGroups = [
  { label: 'Drums', color: '#1687ff', matches: ['DRUM', 'KICK', 'SNARE', 'TOM', 'OVERHEAD'] },
  { label: 'Bass', color: '#ff9f1a', matches: ['BASS'] },
  { label: 'Guitars', color: '#00d26a', matches: ['GUITAR'] },
  { label: 'Vocals', color: '#a855f7', matches: ['VOCAL', 'BACK'] },
  { label: 'Other', color: '#5f6b78', matches: [] }
];

function liveEnergySegments(channels: Channel[]) {
  const powers = new Map(energyGroups.map(group => [group.label, 0]));
  channels.filter(channel => channel.meterAvailable !== false && channel.level > -90).forEach(channel => {
    const token = `${channel.group} ${channel.type} ${channel.name}`.toUpperCase();
    const group = energyGroups.find(candidate => candidate.matches.some(match => token.includes(match))) ?? energyGroups[energyGroups.length - 1];
    powers.set(group.label, (powers.get(group.label) ?? 0) + Math.pow(10, channel.level / 10));
  });
  const total = Array.from(powers.values()).reduce((sum, value) => sum + value, 0);
  return energyGroups.map(group => ({
    ...group,
    value: total > 0 ? Math.round(((powers.get(group.label) ?? 0) / total) * 100) : 0
  }));
}

function meterPercent(value: number) {
  return Math.max(3, Math.min(100, ((value + 60) / 60) * 100));
}

function MiniTrace({ values, tone = 'green', filled = false, ghost = false }: { values: number[]; tone?: Tone; filled?: boolean; ghost?: boolean }) {
  const color = toneColors[tone];
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${100 - value}`).join(' ');
  const ghostPoints = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${100 - Math.min(100, value * 1.06)}`).join(' ');
  const fillPoints = `0,100 ${points} 100,100`;
  return (
    <svg className="h-full w-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
      {filled && <polygon fill={color} opacity="0.2" points={fillPoints} />}
      {ghost && <polyline fill="none" opacity="0.45" points={ghostPoints} stroke="#9fb6c9" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.2" />}
      <polyline fill="none" points={points} stroke={color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" />
    </svg>
  );
}

function GridLines() {
  return (
    <div className="pointer-events-none absolute inset-2 flex flex-col justify-between">
      {[0, 1, 2, 3, 4].map(line => (
        <span className="h-px w-full bg-line/40" key={line} />
      ))}
    </div>
  );
}

function SegmentedMeter({ value, secondary, available = true }: { value: number; secondary: number; available?: boolean }) {
  return (
    <div className="relative flex h-full w-[19px] items-end gap-[2px] rounded-[4px] border border-line bg-[#05080d] p-[2px]" title={available ? 'Real read-only audio meter' : 'Audio meter unavailable'}>
      {[value, secondary].map((item, index) => (
        <span className="relative block h-full flex-1 overflow-hidden rounded-[2px]" key={`${item}-${index}`}>
          <span
            className="absolute inset-x-0 bottom-0"
            style={{
              height: `${available ? meterPercent(item) : 0}%`,
              background: 'linear-gradient(to top, #00d26a 0%, #00d26a 74%, #ffb020 74%, #ffb020 91%, #ff4d4f 91%, #ff4d4f 100%)'
            }}
          />
          <span
            className="absolute inset-0"
            style={{ background: 'repeating-linear-gradient(to top, transparent 0 3px, #05080d 3px 4px)' }}
          />
        </span>
      ))}
      {!available && <span className="absolute inset-0 grid place-items-center text-[6px] font-black text-dim">—</span>}
    </div>
  );
}

function FaderTrack({ value, available = true }: { value: number; available?: boolean }) {
  const bottom = Math.max(8, Math.min(86, 74 + value));
  return (
    <div className="relative mx-auto h-full w-[20px]">
      <div className="absolute inset-y-1 left-1/2 w-[6px] -translate-x-1/2 rounded-full border border-line bg-[#04070b] shadow-[inset_0_0_8px_rgba(0,0,0,.7)]" />
      <div className="pointer-events-none absolute inset-y-2 left-0 flex flex-col justify-between">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map(tick => (
          <span className="h-px w-1.5 bg-line2/70" key={tick} />
        ))}
      </div>
      {available && <span
        className="absolute left-1/2 h-[12px] w-[17px] -translate-x-1/2 rounded-[2.5px] border border-[#e8eef5] bg-gradient-to-b from-[#f2f6fa] via-[#c7d2dd] to-[#eef3f8] shadow-[0_2px_8px_rgba(0,0,0,.55),0_0_14px_rgba(230,237,243,.14)]"
        style={{ bottom: `${bottom}%` }}
      >
        <i className="absolute inset-x-1 top-1/2 h-px -translate-y-1/2 bg-[#31404f]" />
      </span>}
      {!available && <span className="absolute inset-0 grid place-items-center text-[7px] font-black text-dim">N/A</span>}
    </div>
  );
}

function SourcePill({ label, tone = 'blue' }: { label: string; tone?: Tone }) {
  const color = toneColors[tone];
  return (
    <span
      className="max-w-[64px] shrink-0 truncate rounded-[3px] border px-1 py-[2px] text-[6px] font-black uppercase tracking-[0.04em]"
      style={{ borderColor: `${color}70`, color, backgroundColor: `${color}18` }}
      title={label}
    >
      {label}
    </span>
  );
}

function runtimeSourcePill(runtime: DashboardRuntime) {
  const isBackend = runtime.sourceLabel.toLowerCase().includes('backend');
  return {
    label: isBackend ? 'Backend DTO' : runtime.sourceLabel,
    tone: isBackend ? 'green' as Tone : 'blue' as Tone
  };
}

function MetricShell({
  title,
  caption,
  accent,
  sourceLabel,
  sourceTone = 'blue',
  children
}: {
  title: string;
  caption: string;
  accent: Tone;
  sourceLabel?: string;
  sourceTone?: Tone;
  children: ReactNode;
}) {
  return (
    <section className="min-h-0 overflow-hidden rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <header className="mb-1.5">
        <div className="flex items-center justify-between gap-1.5">
          <h3 className="truncate text-[9px] font-black uppercase tracking-[0.09em] text-muted">{title}</h3>
          <span className="h-2 w-2 shrink-0 rounded-full shadow-[0_0_14px_currentColor]" style={{ backgroundColor: toneColors[accent], color: toneColors[accent] }} />
        </div>
        <div className="mt-0.5 flex items-center justify-between gap-1.5">
          <p className="truncate text-[7.5px] text-dim">{caption}</p>
          {sourceLabel ? <SourcePill label={sourceLabel} tone={sourceTone} /> : null}
        </div>
      </header>
      {children}
    </section>
  );
}

function ConfidenceRing({ percent, available = true }: { percent: number; available?: boolean }) {
  const clamped = Math.max(0, Math.min(100, percent));
  return (
    <div
      className="grid h-[66px] w-[66px] place-items-center rounded-full"
      style={{ background: `conic-gradient(#00d26a ${clamped * 3.6}deg, #12202b ${clamped * 3.6}deg 360deg)` }}
    >
      <div className="grid h-[52px] w-[52px] place-items-center rounded-full bg-panel shadow-[0_0_24px_rgba(0,210,106,.1)]">
        <div className="text-center leading-none">
          <div className="text-[14px] font-black text-text">{available ? `${clamped}%` : '—'}</div>
          <div className={`mt-0.5 text-[7px] font-bold uppercase tracking-[0.1em] ${available ? 'text-online' : 'text-muted'}`}>{available ? 'Runtime' : 'No data'}</div>
        </div>
      </div>
    </div>
  );
}

function AnalyticsRow({ runtime, channels }: { runtime: DashboardRuntime; channels: Channel[] }) {
  const energySegments = liveEnergySegments(channels);
  const conic = energySegments.reduce((parts, segment, index) => {
    const previous = energySegments.slice(0, index).reduce((sum, item) => sum + item.value, 0);
    const next = previous + segment.value;
    return `${parts}, ${segment.color} ${previous}% ${next}%`;
  }, 'transparent 0 0');
  const runtimeSource = runtimeSourcePill(runtime);
  const spectrumTrace = dashboardSpectrumTrace(runtime, spectrumBins);
  const activeMeterChannels = channels.filter(channel => channel.meterAvailable !== false && channel.level > -90).length;

  return (
    <div className="am-scrollbar grid h-[150px] min-w-0 grid-cols-[178px_minmax(210px,1.2fr)_132px_192px_minmax(196px,1fr)] gap-1.5 overflow-x-auto">
      <MetricShell title="Master Bus" caption={runtime.audioCaptureRunning ? 'Main L/R read-only meter' : 'Audio capture is stopped'} accent={runtime.masterAvailable ? 'green' : 'yellow'} sourceLabel={runtime.masterAvailable ? 'Real audio' : 'Unavailable'} sourceTone={runtime.masterAvailable ? 'green' : 'muted'}>
        <div>
          <div className="flex items-end gap-4 pt-1.5">
            <div>
              <div className={`${runtime.integratedLufsAvailable && runtime.integratedLufs > -90 ? 'text-[27px]' : 'text-[15px]'} font-black leading-none text-text`}>{metricDisplay(runtime.integratedLufs, runtime.integratedLufsAvailable, -90)} {runtime.integratedLufs > -90 && <span className="text-[10px] font-bold text-muted">LUFS</span>}</div>
              <div className="mt-1 text-[8px] uppercase tracking-[0.08em] text-dim">Integrated</div>
            </div>
            <div className="pb-0.5">
              <div className="whitespace-nowrap text-[15px] font-black leading-none text-text">{metricDisplay(runtime.truePeakDb, runtime.truePeakAvailable, -150)} {runtime.truePeakDb > -150 && <span className="text-[8px] font-bold text-muted">dBTP</span>}</div>
              <div className="mt-1 whitespace-nowrap text-[7px] uppercase tracking-[0.06em] text-dim">True Peak</div>
            </div>
          </div>
          <div className="mt-2 flex items-center justify-between text-[7px] uppercase tracking-[0.06em] text-dim"><span>Main fader</span><span className="font-mono text-text">{runtime.mainFaderAvailable ? `${formatNumber(runtime.mainFaderDb ?? 0)} dB` : '—'}</span></div>
          <div className="relative mt-3 h-[11px] overflow-hidden rounded-[3px] border border-line/60 bg-[#05080d]">
            <span className="absolute inset-0 bg-gradient-to-r from-[#134ad8] via-[#00d26a] via-55% to-[#ff4d4f]" />
            <span className="absolute inset-0" style={{ background: 'repeating-linear-gradient(to right, transparent 0 3px, #05080d 3px 4px)' }} />
          </div>
          <div className="mt-1 flex justify-between font-mono text-[7px] text-dim"><span>-60</span><span>-48</span><span>-36</span><span>-24</span><span>-12</span><span>-6</span><span>0</span></div>
        </div>
      </MetricShell>

      <MetricShell title="Spectrum" caption={spectrumTrace.caption} accent="blue" sourceLabel={spectrumTrace.sourceLabel} sourceTone={spectrumTrace.sourceTone}>
        <div className="relative h-[104px] rounded-[5px] border border-line/60 bg-[#060c14] p-2">
          <GridLines />
          <div className="absolute inset-y-2 left-2 z-10 flex flex-col justify-between font-mono text-[7px] text-dim"><span>-12</span><span>-24</span><span>-36</span><span>-48</span><span>-60</span></div>
          {spectrumTrace.values.length ? <div className="ml-6 h-[calc(100%-10px)]"><MiniTrace values={spectrumTrace.values} tone="purple" filled ghost /></div> : <div className="absolute inset-0 grid place-items-center px-6 text-center text-[9px] font-bold uppercase tracking-[0.08em] text-muted">{runtime.audioCaptureRunning ? 'No measurable signal' : 'Audio meters stopped'}</div>}
          <div className="absolute bottom-1 left-8 right-2 flex justify-between font-mono text-[7px] text-dim"><span>20</span><span>100</span><span>500</span><span>2k</span><span>10k</span><span>20k</span></div>
        </div>
      </MetricShell>

      <MetricShell title="Loudness" caption="Product Layer loudness" accent="yellow" sourceLabel={runtimeSource.label} sourceTone={runtimeSource.tone}>
        <div className="space-y-[4px]">
          <div className="border-b border-line/50 pb-[3px]">
            <div className="text-[7px] uppercase tracking-[0.06em] text-dim">Short Term</div>
            <div className="text-[12px] font-black leading-tight text-text">{metricDisplay(runtime.shortTermLufs, runtime.shortTermLufsAvailable, -90)} {runtime.shortTermLufs > -90 && <span className="text-[7px] font-bold text-muted">LUFS</span>}</div>
          </div>
          <div className="border-b border-line/50 pb-[3px]">
            <div className="text-[7px] uppercase tracking-[0.06em] text-dim">Momentary</div>
            <div className="text-[12px] font-black leading-tight text-text">{metricDisplay(runtime.momentaryLufs, runtime.momentaryLufsAvailable, -90)} {runtime.momentaryLufs > -90 && <span className="text-[7px] font-bold text-muted">LUFS</span>}</div>
          </div>
          <div>
            <div className="text-[7px] uppercase tracking-[0.06em] text-dim">LRA</div>
            <div className="text-[12px] font-black leading-tight text-text">{metricDisplay(runtime.lraLu, runtime.lraAvailable)} {runtime.lraAvailable && <span className="text-[7px] font-bold text-muted">LU</span>}</div>
          </div>
        </div>
      </MetricShell>

      <MetricShell title="Live Energy" caption="Real meter energy by source group" accent="purple" sourceLabel={activeMeterChannels ? 'Real meters' : 'No signal'} sourceTone={activeMeterChannels ? 'green' : 'muted'}>
        <div className="grid grid-cols-[84px_1fr] items-center gap-3 pt-1">
          <div className="grid h-[84px] w-[84px] place-items-center rounded-full p-[11px]" style={{ background: `conic-gradient(${conic})` }}>
            <div className="grid h-full w-full place-items-center rounded-full bg-panel text-center text-[9px] font-black text-text">{activeMeterChannels ? `${activeMeterChannels} live` : 'NO SIGNAL'}</div>
          </div>
          <div className="space-y-[5px] text-[9px]">
            {energySegments.map(segment => (
              <div className="grid grid-cols-[9px_1fr_26px] items-center gap-1.5" key={segment.label}>
                <i className="h-2 w-2 rounded-full" style={{ backgroundColor: segment.color }} />
                <span className="truncate text-muted">{segment.label}</span>
                <b className="text-right text-text">{segment.value}%</b>
              </div>
            ))}
          </div>
        </div>
      </MetricShell>

      <MetricShell title="AI Confidence" caption="Latest runtime agent confidence" accent={runtime.aiConfidenceAvailable ? 'green' : 'yellow'} sourceLabel={runtime.aiConfidenceAvailable ? runtimeSource.label : 'Unavailable'} sourceTone={runtime.aiConfidenceAvailable ? runtimeSource.tone : 'muted'}>
        <div className="grid grid-cols-[66px_1fr] items-center gap-2.5 pt-1">
          <ConfidenceRing available={runtime.aiConfidenceAvailable} percent={runtime.aiConfidencePercent} />
          <div className="min-w-0">
            <div className="grid grid-cols-3 gap-1 text-center"><div className="rounded border border-line bg-row px-1 py-1"><b className="block text-[11px] text-warn">{runtime.pendingDecisions}</b><span className="text-[6px] uppercase text-dim">Pending</span></div><div className="rounded border border-line bg-row px-1 py-1"><b className="block text-[11px] text-online">{runtime.appliedDecisions}</b><span className="text-[6px] uppercase text-dim">Applied</span></div><div className="rounded border border-line bg-row px-1 py-1"><b className="block text-[11px] text-danger">{runtime.blockedDecisions}</b><span className="text-[6px] uppercase text-dim">Blocked</span></div></div>
            <div className="mt-2 truncate text-right text-[7px] uppercase tracking-[0.06em] text-dim">{runtime.readCoverageStatus} readback · {runtime.cacheFreshnessStatus}</div>
          </div>
        </div>
      </MetricShell>
    </div>
  );
}

function DashboardChannelStrip({
  channel,
  selected,
  telemetry,
  onSelect
}: {
  channel: Channel;
  selected: boolean;
  telemetry: ChannelDashboardTelemetry;
  onSelect: (id: number) => void;
}) {
  const accent = toneColors[telemetry.accent];
  const faderLabel = channel.faderAvailable === false ? '—' : channel.fader <= -140 ? '-∞' : channel.fader.toFixed(1);
  return (
    <button
      className={`flex min-w-0 flex-col overflow-hidden rounded-[7px] border bg-[#0a1017] p-1.5 text-left transition hover:border-audio/70 hover:bg-[#0e1620] ${
        selected ? 'border-ai shadow-[0_0_0_1px_rgba(168,85,247,.82),0_0_22px_rgba(168,85,247,.22)]' : 'border-line'
      }`}
      onClick={() => onSelect(channel.id)}
      type="button"
      aria-label={`Select channel ${channel.id} ${channel.name}; read-only console state`}
    >
      <div className="min-h-[30px]">
        <div className="flex items-center justify-between gap-1">
          <span className="min-w-0 flex-1 truncate text-[9.5px] font-bold leading-tight text-text">{channel.name}</span>
          <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: accent, boxShadow: `0 0 10px ${accent}` }} />
        </div>
        <div className="truncate text-[7.5px] uppercase tracking-[0.07em] text-dim">{channel.type}</div>
      </div>

      <div className="mb-1 flex items-baseline justify-between gap-0.5 whitespace-nowrap">
        <span className="font-mono text-[9.5px] font-bold text-text">{faderLabel}{channel.faderAvailable !== false && <span className="ml-0.5 text-[6.5px] text-muted">dB</span>}</span>
        <span className={`text-[7px] font-bold ${safeStateTextClass(telemetry.safeState)}`}>{telemetry.safeState}</span>
      </div>

      <div className="relative h-7 overflow-hidden rounded-[4px] border border-line/60 bg-[#05080d] p-0.5">
        {channel.meterAvailable === false ? <span className="absolute inset-0 grid place-items-center text-[6.5px] font-black uppercase tracking-[0.06em] text-dim">No meter</span> : <MiniTrace values={telemetry.waveform} tone={telemetry.accent} filled />}
      </div>

      <div className="my-1 grid grid-cols-2 gap-1">
        <span className={`rounded-[3px] border px-0.5 py-[3px] text-center text-[6.5px] font-black uppercase tracking-[0.05em] ${channel.gateAvailable ? 'border-audio/40 bg-audio/10 text-audio' : 'border-line bg-panel2 text-dim'}`} title={channel.gateAvailable ? telemetry.gateLabel : 'Gate readback unavailable'}>{channel.gateAvailable ? telemetry.gateLabel : 'G —'}</span>
        <span className={`rounded-[3px] border px-0.5 py-[3px] text-center text-[6.5px] font-black uppercase tracking-[0.05em] ${channel.compAvailable ? 'border-warn/40 bg-warn/10 text-warn' : 'border-line bg-panel2 text-dim'}`} title={channel.compAvailable ? telemetry.compLabel : 'Compressor readback unavailable'}>{channel.compAvailable ? telemetry.compLabel : 'C —'}</span>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[19px_1fr] gap-0.5">
        <SegmentedMeter available={channel.meterAvailable !== false} value={telemetry.meterL} secondary={telemetry.meterR} />
        <div className="grid min-h-0 grid-cols-[11px_1fr] gap-0.5">
          <div className="flex h-full flex-col justify-between py-1 text-right font-mono text-[6px] leading-none text-dim">
            <span>10</span><span>0</span><span>-10</span><span>-20</span><span>-40</span><span>-60</span>
          </div>
          <FaderTrack available={channel.faderAvailable !== false} value={channel.fader} />
        </div>
      </div>

      <div className="mt-1 grid grid-cols-2 gap-1">
        <span className={`grid h-[18px] place-items-center rounded-[3px] border text-[8px] font-black ${channel.muteAvailable === false ? 'border-line bg-panel2 text-dim' : telemetry.muted ? 'border-danger bg-danger text-white shadow-[0_0_10px_rgba(255,77,79,.4)]' : 'border-danger/40 bg-danger/12 text-danger'}`}>{channel.muteAvailable === false ? 'M —' : 'M'}</span>
        <span className="grid h-[18px] place-items-center rounded-[3px] border border-line bg-panel2 text-[8px] font-black text-dim" title="Solo readback is not available in the current DTO">S —</span>
      </div>

      {channel.panAvailable ? <div className="mt-1 h-[3px] rounded-full bg-[#05080d]" title={`Pan readback ${telemetry.pan}`}><span className="block h-full rounded-full bg-audio/80" style={{ width: `${50 + telemetry.pan}%` }} /></div> : <div className="mt-1 text-center text-[6px] font-black uppercase tracking-[0.05em] text-dim">Pan —</div>}
      <div className="mt-0.5 text-center font-mono text-[7.5px] text-dim">{String(channel.id).padStart(2, '0')}</div>
    </button>
  );
}

function ViewChip({ label, onInfo }: { label: string; onInfo?: (message: string) => void }) {
  return (
    <button
      className="flex items-center gap-1 rounded-[4px] border border-line bg-panel2 px-2 py-1 text-[9px] font-bold uppercase tracking-[0.05em] text-muted transition hover:border-line2 hover:text-text"
      onClick={() => onInfo?.(`${label} view control is display-only.`)}
      type="button"
    >
      {label}
      <span className="text-[7px] text-dim">▾</span>
    </button>
  );
}

function MixerArea({ channels, runtime, ui, actions }: { channels: Channel[]; runtime: DashboardRuntime; ui: UiState; actions: UiActions }) {
  const [activeTab, setActiveTab] = useState<DashboardTab>('Channels');
  const visibleChannels = channels.slice(0, 14);
  const selectedChannel = visibleChannels.find(channel => channel.id === ui.selectedChannelId) ?? visibleChannels[0];
  const suppressMockSolo = runtime.sourceLabel.toLowerCase().includes('backend');

  return (
    <section className="min-h-0 overflow-hidden rounded-[9px] border border-line bg-panel shadow-panel">
      <header className="flex h-[38px] items-center justify-between border-b border-line bg-[#0c141d] px-2.5">
        <div className="flex min-w-0 gap-1">
          {dashboardTabs.map(tab => (
            <button
              className={`whitespace-nowrap rounded-[5px] px-2.5 py-1.5 text-[9px] font-black uppercase tracking-[0.05em] transition ${
                activeTab === tab
                  ? 'border border-audio/45 bg-audio/12 text-audio shadow-[0_0_14px_rgba(22,135,255,.14)]'
                  : 'border border-transparent text-muted hover:bg-panel2 hover:text-text'
              }`}
              key={tab}
              onClick={() => setActiveTab(tab)}
              type="button"
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="hidden items-center gap-1.5 xl:flex">
          <span className="mr-0.5 text-[8px] font-black uppercase tracking-[0.12em] text-dim">View</span>
          <ViewChip label="Small" onInfo={message => actions.pushToast(message, 'blue')} />
          <ViewChip label="All" onInfo={message => actions.pushToast(message, 'blue')} />
          <span className="mx-1 h-4 w-px bg-line" />
          <span className="whitespace-nowrap text-[9px] uppercase text-muted">{visibleChannels.length} strips</span>
          <span className="whitespace-nowrap text-[9px] uppercase text-ai">Selected: {selectedChannel ? `CH ${String(selectedChannel.id).padStart(2, '0')}` : 'None'}</span>
        </div>
      </header>

      <div className="am-scrollbar grid h-[calc(100%-38px)] min-w-0 grid-flow-col auto-cols-[minmax(68px,1fr)] gap-1 overflow-x-auto p-1">
        {visibleChannels.map(channel => (
          <DashboardChannelStrip
            channel={channel}
            key={channel.id}
            onSelect={actions.selectChannel}
            selected={channel.id === ui.selectedChannelId}
            telemetry={dashboardTelemetryForChannel(channel, channelTelemetryById[channel.id] ?? channelTelemetryById[1], { suppressMockSolo })}
          />
        ))}
      </div>
    </section>
  );
}

function AnalysisCardHeader({ card }: { card: AnalysisCardMock }) {
  return (
    <div className="mb-1.5 flex items-center justify-between gap-2">
      <h3 className="min-w-0 truncate text-[8.5px] font-black uppercase tracking-[0.07em] text-muted">{card.title}</h3>
      <SourcePill label={card.sourceLabel ?? 'Display-only'} tone={card.sourceTone ?? 'yellow'} />
    </div>
  );
}

function PhaseScopeCard({ card }: { card: AnalysisCardMock }) {
  const points = card.values
    .map((value, index) => {
      const theta = (index / card.values.length) * Math.PI * 2;
      const radius = 8 + (value / 100) * 20;
      const x = 31 + Math.cos(theta) * radius * 0.55;
      const y = 31 - Math.sin(theta) * radius;
      return `${x},${y}`;
    })
    .join(' ');
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="grid grid-cols-[64px_1fr] items-center gap-3">
        <div className="relative h-[64px] w-[64px] overflow-hidden rounded-[5px] border border-line/70 bg-[#04110a]">
          <svg className="h-full w-full" viewBox="0 0 62 62">
            <line stroke="#0f3d26" strokeWidth="0.8" x1="31" x2="31" y1="4" y2="58" />
            <line stroke="#0f3d26" strokeWidth="0.8" x1="4" x2="58" y1="31" y2="31" />
            <line stroke="#0f3d26" strokeDasharray="2 2" strokeWidth="0.6" x1="10" x2="52" y1="52" y2="10" />
            <line stroke="#0f3d26" strokeDasharray="2 2" strokeWidth="0.6" x1="10" x2="52" y1="10" y2="52" />
            <polygon fill="#00d26a" opacity="0.14" points={points} />
            <polyline fill="none" points={`${points} ${points.split(' ')[0]}`} stroke="#00d26a" strokeWidth="1" opacity="0.85" />
          </svg>
          <span className="absolute bottom-0.5 left-1 font-mono text-[7px] text-dim">L</span>
          <span className="absolute bottom-0.5 right-1 font-mono text-[7px] text-dim">R</span>
        </div>
        <div className="min-w-0 text-[9.5px]">
          <div className="truncate font-bold text-text">{card.primary}</div>
          <div className="truncate text-muted">{card.secondary}</div>
        </div>
      </div>
    </section>
  );
}

function CorrelationCard({ card }: { card: AnalysisCardMock }) {
  const measured = card.values.length > 0 && card.primary !== 'Not measured';
  const value = measured ? parseLabelNumber(card.primary, 0) : 0;
  const position = Math.max(2, Math.min(98, ((value + 1) / 2) * 100));
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="pt-1 text-center">
        <div className="text-[21px] font-black leading-none text-text">{measured ? (value > 0 ? `+${value.toFixed(2)}` : value.toFixed(2)) : '—'}</div>
        <div className="relative mt-2.5 h-[9px] overflow-hidden rounded-full border border-line/60 bg-[#05080d]">
          <span className="absolute inset-0 bg-gradient-to-r from-danger via-warn to-online" />
          {measured && <span className="absolute top-1/2 h-[13px] w-[3px] -translate-y-1/2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,.8)]" style={{ left: `calc(${position}% - 1px)` }} />}
        </div>
        <div className="mt-1 flex justify-between font-mono text-[7px] text-dim"><span>-1</span><span>0</span><span>+1</span></div>
        <div className="mt-1 truncate text-[9px] text-muted">{card.secondary}</div>
      </div>
    </section>
  );
}

function SpectralBalanceCard({ card }: { card: AnalysisCardMock }) {
  const bars = card.values.slice(0, spectralBandLabels.length);
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="flex h-[52px] items-end gap-1">
        {bars.map((value, index) => (
          <span
            className="flex-1 rounded-t-[3px]"
            key={spectralBandLabels[index]}
            style={{ height: `${value}%`, background: 'linear-gradient(180deg, #37a8ff, #135a9e 78%, #0d2136)' }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between gap-0.5">
        {spectralBandLabels.map(label => (
          <span className="flex-1 truncate text-center text-[6px] uppercase tracking-[0.02em] text-dim" key={label}>{label}</span>
        ))}
      </div>
      <div className="mt-1 flex justify-between gap-2 text-[8.5px]"><b className="truncate text-text">{card.primary}</b><span className="truncate text-muted">{card.secondary}</span></div>
    </section>
  );
}

function DynamicsOverviewCard({ card }: { card: AnalysisCardMock }) {
  const rows = card.values.slice(0, dynamicsGroupLabels.length);
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="space-y-[4px]">
        {rows.map((value, index) => (
          <div className="grid grid-cols-[34px_1fr] items-center gap-1.5" key={dynamicsGroupLabels[index]}>
            <span className="truncate text-[7px] uppercase tracking-[0.03em] text-muted">{dynamicsGroupLabels[index]}</span>
            <div className="h-[7px] overflow-hidden rounded-[2px] border border-line/50 bg-[#05080d]">
              <span
                className="block h-full rounded-[2px]"
                style={{ width: `${Math.max(6, value)}%`, background: 'linear-gradient(90deg, #00d26a, #ffb020 70%, #ff4d4f)' }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="ml-[40px] mt-1 flex justify-between font-mono text-[6.5px] text-dim"><span>0</span><span>6</span><span>12</span><span>18</span><span>24 dB</span></div>
      <div className="mt-0.5 flex justify-between gap-2 text-[8.5px]"><b className="truncate text-text">{card.primary}</b><span className="truncate text-muted">{card.secondary}</span></div>
    </section>
  );
}

function LoudnessHistoryCard({ card }: { card: AnalysisCardMock }) {
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="relative h-[46px] rounded-[4px] border border-line/50 bg-[#060c14] p-1">
        <GridLines />
        <MiniTrace values={card.values} tone="green" filled />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[7px] text-dim"><span>-30m</span><span>-20m</span><span>-10m</span><span>Now</span></div>
      <div className="mt-1 flex items-center justify-between gap-2 text-[8.5px]">
        <span className="flex items-center gap-1 truncate text-muted"><i className="h-1.5 w-1.5 rounded-full bg-online" />{card.primary}</span>
        <span className="truncate text-dim">-- {card.secondary}</span>
      </div>
    </section>
  );
}

function AnalysisCard({ card }: { card: AnalysisCardMock }) {
  if (card.id === 'phase') return <PhaseScopeCard card={card} />;
  if (card.id === 'correlation') return <CorrelationCard card={card} />;
  if (card.id === 'spectral') return <SpectralBalanceCard card={card} />;
  if (card.id === 'dynamics') return <DynamicsOverviewCard card={card} />;
  if (card.id === 'loudness') return <LoudnessHistoryCard card={card} />;

  const color = toneColors[card.accent];
  return (
    <section className="min-h-0 rounded-[9px] border border-line bg-panel p-2.5 shadow-panel">
      <AnalysisCardHeader card={card} />
      <div className="h-12"><MiniTrace values={card.values} tone={card.accent} filled /></div>
      <div className="mt-2 flex justify-between gap-2 text-[9px]"><b className="truncate text-text" style={{ color }}>{card.primary}</b><span className="truncate text-muted">{card.secondary}</span></div>
    </section>
  );
}

function AnalysisRow({ runtime }: { runtime: DashboardRuntime }) {
  const cards = dashboardAnalysisCards(runtime, analysisCards).filter(card => card.id !== 'suggestions');

  return (
    <div className="am-scrollbar grid h-[128px] min-w-0 grid-flow-col auto-cols-[minmax(186px,1fr)] gap-1.5 overflow-x-auto">
      {cards.map(card => <AnalysisCard card={card} key={card.id} />)}
    </div>
  );
}

export function AutomixerDashboard({
  channels,
  runtime,
  ui,
  actions
}: {
  channels: Channel[];
  runtime: DashboardRuntime;
  ui: UiState;
  actions: UiActions;
}) {
  const normalizedChannels = useMemo(() => channels.slice(0, 14), [channels]);

  return (
    <div className="grid h-full min-h-0 grid-rows-[150px_minmax(0,1fr)_128px] gap-1.5">
      <AnalyticsRow channels={normalizedChannels} runtime={runtime} />
      <MixerArea actions={actions} channels={normalizedChannels} runtime={runtime} ui={ui} />
      <AnalysisRow runtime={runtime} />
    </div>
  );
}
