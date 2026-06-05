import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  analysisCards,
  channelTelemetryById,
  dashboardTabs,
  mixBalanceSegments,
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

function safeStateTextClass(safeState: string) {
  if (safeState === 'MUTE' || safeState === 'CLIP') return 'text-danger';
  if (safeState === 'HOT') return 'text-warn';
  if (safeState === 'FOCUS') return 'text-ai';
  if (safeState === 'LOW') return 'text-audio';
  if (safeState === 'OFF') return 'text-muted';
  return 'text-online';
}

function formatNumber(value: number, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function meterPercent(value: number) {
  return Math.max(3, Math.min(100, ((value + 60) / 60) * 100));
}

function MiniTrace({ values, tone = 'green', filled = false }: { values: number[]; tone?: Tone; filled?: boolean }) {
  const color = toneColors[tone];
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${100 - value}`).join(' ');
  const fillPoints = `0,100 ${points} 100,100`;
  return (
    <svg className="h-full w-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
      {filled && <polygon fill={color} opacity="0.18" points={fillPoints} />}
      <polyline fill="none" points={points} stroke={color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.4" />
    </svg>
  );
}

function VerticalMeter({ value, secondary }: { value: number; secondary: number }) {
  return (
    <div className="flex h-full w-6 items-end gap-1 rounded-[4px] border border-line bg-[#070b10] p-1">
      {[value, secondary].map((item, index) => (
        <span
          className="block flex-1 rounded-t-[3px] bg-gradient-to-t from-online via-warn to-danger"
          key={`${item}-${index}`}
          style={{ height: `${meterPercent(item)}%` }}
        />
      ))}
    </div>
  );
}

function FaderTrack({ value }: { value: number }) {
  const bottom = Math.max(9, Math.min(88, 74 + value));
  return (
    <div className="relative mx-auto h-full w-[18px] rounded-full border border-line bg-[#070b10] shadow-[inset_0_0_12px_rgba(0,0,0,.55)]">
      <span className="absolute left-1/2 top-2 h-[calc(100%-16px)] w-px -translate-x-1/2 bg-line2" />
      <span
        className="absolute left-1/2 h-3 w-9 -translate-x-1/2 rounded-[3px] border border-[#d9e4ef] bg-[#d9e4ef] shadow-[0_0_12px_rgba(230,237,243,.18)]"
        style={{ bottom: `${bottom}%` }}
      />
    </div>
  );
}

function SourcePill({ label, tone = 'blue' }: { label: string; tone?: Tone }) {
  const color = toneColors[tone];
  return (
    <span
      className="max-w-[94px] truncate rounded-[3px] border px-1.5 py-0.5 text-[7px] font-black uppercase tracking-[0.08em]"
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
    <section className="min-h-0 overflow-hidden rounded-[4px] border border-line bg-panel p-2 shadow-panel">
      <header className="mb-1.5 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[10px] font-black uppercase tracking-[0.13em] text-muted">{title}</h3>
          <p className="mt-0.5 truncate text-[8px] text-dim">{caption}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {sourceLabel ? <SourcePill label={sourceLabel} tone={sourceTone} /> : null}
          <span className="h-2 w-2 rounded-full shadow-[0_0_14px_currentColor]" style={{ backgroundColor: toneColors[accent], color: toneColors[accent] }} />
        </div>
      </header>
      {children}
    </section>
  );
}

function AnalyticsRow({ runtime }: { runtime: DashboardRuntime }) {
  const conic = mixBalanceSegments.reduce((parts, segment, index) => {
    const previous = mixBalanceSegments.slice(0, index).reduce((sum, item) => sum + item.value, 0);
    const next = previous + segment.value;
    return `${parts}, ${segment.color} ${previous}% ${next}%`;
  }, 'transparent 0 0');
  const runtimeSource = runtimeSourcePill(runtime);
  const spectrumTrace = dashboardSpectrumTrace(runtime, spectrumBins);

  return (
    <div className="grid h-[145px] grid-cols-[172px_250px_128px_188px_minmax(0,1fr)] gap-1">
      <MetricShell title="Master Bus" caption="Main L/R safety window" accent="green" sourceLabel={runtimeSource.label} sourceTone={runtimeSource.tone}>
        <div>
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div>
              <div className="text-[26px] font-black leading-none text-text">{formatNumber(runtime.integratedLufs)} <span className="text-[10px] text-muted">LUFS</span></div>
              <div className="mt-1 text-[9px] text-muted">Integrated</div>
            </div>
            <div>
              <div className="text-[17px] font-black leading-none text-text">{formatNumber(runtime.truePeakDb)} <span className="text-[9px] text-muted">dBTP</span></div>
              <div className="mt-1 text-[9px] text-muted">True Peak</div>
            </div>
          </div>
          <div className="mt-2.5 h-3 rounded-sm bg-gradient-to-r from-[#134ad8] via-[#00d26a] via-55% to-[#ff4d4f]" />
          <div className="mt-1 flex justify-between font-mono text-[8px] text-dim"><span>-60</span><span>-48</span><span>-36</span><span>-24</span><span>-12</span><span>0</span></div>
          <div className="mt-1 text-right font-mono text-[9px] text-danger">{formatNumber(runtime.truePeakDb)}</div>
        </div>
      </MetricShell>

      <MetricShell title="Spectrum" caption={spectrumTrace.caption} accent="blue" sourceLabel={spectrumTrace.sourceLabel} sourceTone={spectrumTrace.sourceTone}>
        <div className="relative h-[102px] rounded-[3px] border border-line/60 bg-[#07101a] p-2">
          <div className="absolute inset-y-2 left-2 flex flex-col justify-between font-mono text-[7px] text-dim"><span>-12</span><span>-24</span><span>-36</span><span>-48</span><span>-60</span></div>
          <div className="ml-6 h-full">
            <MiniTrace values={spectrumTrace.values} tone="purple" filled />
          </div>
          <div className="absolute bottom-1 left-8 right-2 flex justify-between font-mono text-[7px] text-dim"><span>20</span><span>50</span><span>100</span><span>200</span><span>500</span><span>1k</span><span>2k</span><span>5k</span><span>10k</span><span>20k</span></div>
        </div>
      </MetricShell>

      <MetricShell title="Loudness" caption="Product Layer loudness when connected" accent="yellow" sourceLabel={runtimeSource.label} sourceTone={runtimeSource.tone}>
        <div className="space-y-2 pt-1 text-[11px]">
          <div className="flex justify-between"><span className="text-muted">Short Term</span><b>{formatNumber(runtime.shortTermLufs)} LUFS</b></div>
          <div className="flex justify-between"><span className="text-muted">Momentary</span><b>{formatNumber(runtime.momentaryLufs)} LUFS</b></div>
          <div className="h-8 rounded-[5px] border border-line bg-[#070b10] p-1">
            <span className="block h-full rounded-[4px] bg-gradient-to-r from-online via-warn to-danger" style={{ width: `${meterPercent(runtime.shortTermLufs)}%` }} />
          </div>
        </div>
      </MetricShell>

      <MetricShell title="Mix Balance" caption="Runtime channel count, static group split" accent="purple" sourceLabel="Mixed source" sourceTone="yellow">
        <div className="grid grid-cols-[78px_1fr] gap-3 pt-1">
          <div className="grid h-[78px] w-[78px] place-items-center rounded-full p-[10px]" style={{ background: `conic-gradient(${conic})` }}>
            <div className="grid h-full w-full place-items-center rounded-full bg-panel text-[11px] font-black text-text">{runtime.activeChannels}/{runtime.totalChannels}</div>
          </div>
          <div className="space-y-1.5 text-[9px]">
            {mixBalanceSegments.map(segment => (
              <div className="grid grid-cols-[9px_1fr_24px] items-center gap-1.5" key={segment.label}>
                <i className="h-2 w-2 rounded-full" style={{ backgroundColor: segment.color }} />
                <span className="truncate text-muted">{segment.label}</span>
                <b className="text-right text-text">{segment.value}%</b>
              </div>
            ))}
          </div>
        </div>
      </MetricShell>

      <MetricShell title="AI Confidence" caption="Runtime value, display trend" accent="green" sourceLabel={runtimeSource.label} sourceTone={runtimeSource.tone}>
        <div className="grid grid-cols-[76px_1fr] items-center gap-3 pt-2">
          <div className="grid h-[76px] w-[76px] place-items-center rounded-full border-[8px] border-online bg-online/10 shadow-[0_0_24px_rgba(0,210,106,.12)]">
            <span className="text-[18px] font-black text-text">{runtime.aiConfidencePercent}%</span>
          </div>
          <div className="space-y-1 text-[10px] text-muted">
            <div className="text-right font-bold text-text">High</div>
            <div className="text-right font-bold text-muted">High</div>
            <div className="mt-2 h-7"><MiniTrace values={[38, 41, 39, 46, 43, 49, 45, 52, 48, 55, 51, 58]} tone="green" /></div>
            <div className="flex justify-between font-mono text-[7px] text-dim"><span>-5m</span><span>-4m</span><span>-3m</span><span>-2m</span><span>-1m</span><span>Now</span></div>
            <div className="text-right text-[7px] uppercase tracking-[0.08em] text-dim">trend display-only</div>
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
  return (
    <button
      className={`flex min-w-0 flex-col overflow-hidden rounded-[4px] border bg-[#0b1118] p-1.5 text-left transition hover:border-audio hover:bg-[#101820] ${
        selected ? 'border-ai shadow-[0_0_0_1px_rgba(168,85,247,.82),0_0_22px_rgba(168,85,247,.2)]' : 'border-line'
      }`}
      onClick={() => onSelect(channel.id)}
      type="button"
    >
      <div className="mb-1 min-h-[33px] border-b border-line/70 pb-0.5">
        <div className="flex items-center justify-between gap-1">
          <span className="font-mono text-[9px] text-dim">{String(channel.id).padStart(2, '0')}</span>
          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: accent, boxShadow: `0 0 10px ${accent}` }} />
        </div>
        <div className="mt-0.5 truncate text-[10px] font-black text-text">{channel.name}</div>
        <div className="truncate text-[8px] uppercase tracking-[0.06em] text-muted">{channel.type}</div>
      </div>

      <div className="mb-0.5 flex items-center justify-between text-[9px]">
        <span className="font-mono font-bold text-text">{channel.fader.toFixed(1)} dB</span>
        <span className={safeStateTextClass(telemetry.safeState)}>{telemetry.safeState}</span>
      </div>

      <div className="relative h-7 rounded-[4px] border border-line/70 bg-[#070b10] p-0.5">
        <span className="absolute right-1 top-0.5 z-10 rounded-[2px] bg-[#070b10]/85 px-1 text-[6px] font-black uppercase tracking-[0.08em] text-dim">display</span>
        <MiniTrace values={telemetry.waveform} tone={telemetry.accent} filled />
      </div>

      <div className="my-0.5 grid grid-cols-2 gap-1 text-[8px] font-bold uppercase tracking-[0.04em]">
        <span className="rounded border border-line bg-panel2 px-1 py-0.5 text-muted">{telemetry.gateLabel}</span>
        <span className="rounded border border-line bg-panel2 px-1 py-0.5 text-muted">{telemetry.compLabel}</span>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[24px_1fr] gap-1.5">
        <VerticalMeter value={telemetry.meterL} secondary={telemetry.meterR} />
        <div className="grid min-h-0 grid-cols-[14px_1fr] gap-1">
          <div className="flex h-full flex-col justify-between py-1 text-right font-mono text-[7px] text-dim">
            <span>10</span><span>0</span><span>-10</span><span>-20</span><span>-40</span><span>-60</span>
          </div>
          <FaderTrack value={channel.fader} />
        </div>
      </div>

      <div className="mt-1 grid grid-cols-2 gap-1">
        <span className={`grid h-5 place-items-center rounded-[3px] border text-[10px] font-black ${telemetry.muted ? 'border-danger bg-danger text-white' : 'border-danger/45 bg-danger/18 text-danger'}`}>M</span>
        <span className={`grid h-5 place-items-center rounded-[3px] border text-[10px] font-black ${telemetry.solo ? 'border-warn bg-warn text-[#070b10]' : 'border-warn/45 bg-warn/15 text-warn'}`}>S</span>
      </div>

      <div className="mt-1 h-1 rounded-full bg-[#070b10]">
        <span className="block h-full rounded-full bg-audio" style={{ width: `${50 + telemetry.pan}%` }} />
      </div>
    </button>
  );
}

function MixerArea({ channels, runtime, ui, actions }: { channels: Channel[]; runtime: DashboardRuntime; ui: UiState; actions: UiActions }) {
  const [activeTab, setActiveTab] = useState<DashboardTab>('Channels');
  const visibleChannels = channels.slice(0, 14);
  const selectedChannel = visibleChannels.find(channel => channel.id === ui.selectedChannelId) ?? visibleChannels[0];
  const suppressMockSolo = runtime.sourceLabel.toLowerCase().includes('backend');

  return (
    <section className="min-h-0 overflow-hidden rounded-[4px] border border-line bg-panel shadow-panel">
      <header className="flex h-[38px] items-center justify-between border-b border-line bg-[#0d151e] px-3">
        <div className="flex min-w-0 gap-1">
          {dashboardTabs.map(tab => (
            <button
              className={`rounded-[5px] px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.06em] transition ${
                activeTab === tab ? 'bg-audio/10 text-audio shadow-[inset_0_-2px_0_#1687ff]' : 'text-muted hover:bg-panel2 hover:text-text'
              }`}
              key={tab}
              onClick={() => setActiveTab(tab)}
              type="button"
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="hidden items-center gap-3 text-[10px] uppercase text-muted xl:flex">
          <span>{visibleChannels.length} strips</span>
          <span>Small fader step</span>
          <span className="text-ai">Selected: {selectedChannel ? `CH ${String(selectedChannel.id).padStart(2, '0')}` : 'None'}</span>
        </div>
      </header>

      <div className="grid h-[calc(100%-38px)] grid-cols-[repeat(14,minmax(48px,1fr))] gap-1 p-1">
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

function AnalysisCard({ card }: { card: AnalysisCardMock }) {
  const color = toneColors[card.accent];

  if (card.id === 'phase') {
    return (
      <section className="rounded-[4px] border border-line bg-panel p-2.5 shadow-panel">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="min-w-0 truncate text-[10px] font-black uppercase tracking-[0.13em] text-muted">{card.title}</h3>
          <SourcePill label={card.sourceLabel ?? 'Display-only'} tone={card.sourceTone ?? 'yellow'} />
        </div>
        <div className="grid grid-cols-[62px_1fr] items-center gap-3">
          <div className="grid h-[62px] w-[62px] place-items-center rounded-full border border-online/40 bg-[repeating-conic-gradient(rgba(0,210,106,.28)_0_8deg,transparent_8deg_16deg)]">
            <span className="h-8 w-8 rounded-full border border-online/50" />
          </div>
          <div className="min-w-0 text-[10px]">
            <div className="truncate font-bold text-text">{card.primary}</div>
            <div className="truncate text-muted">{card.secondary}</div>
          </div>
        </div>
      </section>
    );
  }

  if (card.id === 'spectral' || card.id === 'dynamics') {
    return (
      <section className="rounded-[4px] border border-line bg-panel p-2.5 shadow-panel">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="min-w-0 truncate text-[10px] font-black uppercase tracking-[0.13em] text-muted">{card.title}</h3>
          <SourcePill label={card.sourceLabel ?? 'Display-only'} tone={card.sourceTone ?? 'yellow'} />
        </div>
        <div className="flex h-12 items-end gap-1">
          {card.values.map((value, index) => (
            <span className="flex-1 rounded-t-[3px]" key={`${card.id}-${index}`} style={{ height: `${value}%`, background: `linear-gradient(180deg, ${color}, #101820)` }} />
          ))}
        </div>
        <div className="mt-2 flex justify-between gap-2 text-[9px]"><b className="truncate text-text">{card.primary}</b><span className="truncate text-muted">{card.secondary}</span></div>
      </section>
    );
  }

  return (
    <section className="rounded-[4px] border border-line bg-panel p-2.5 shadow-panel">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="min-w-0 truncate text-[10px] font-black uppercase tracking-[0.13em] text-muted">{card.title}</h3>
        <SourcePill label={card.sourceLabel ?? 'Display-only'} tone={card.sourceTone ?? 'yellow'} />
      </div>
      <div className="h-12"><MiniTrace values={card.values} tone={card.accent} filled /></div>
      <div className="mt-2 flex justify-between gap-2 text-[9px]"><b className="truncate text-text">{card.primary}</b><span className="truncate text-muted">{card.secondary}</span></div>
    </section>
  );
}

function AnalysisRow({ runtime }: { runtime: DashboardRuntime }) {
  const cards = dashboardAnalysisCards(runtime, analysisCards);

  return (
    <div className="grid h-[100px] grid-cols-6 gap-1">
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
    <div className="grid h-full min-h-0 grid-rows-[145px_333px_100px] gap-1">
      <AnalyticsRow runtime={runtime} />
      <MixerArea actions={actions} channels={normalizedChannels} runtime={runtime} ui={ui} />
      <AnalysisRow runtime={runtime} />
    </div>
  );
}
