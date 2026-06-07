import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { Agent, AutomixerMode, Channel, DirectorRuntime, LogRecord, Metric, NavItem, RouteId, Tone, UiActions, UiState, ViewModel } from '../../types';
import { getModeDisplayLabel } from '../../utils/prototypeTone';

const toneClass: Record<Tone, string> = {
  green: 'text-online border-online/30 bg-online/10',
  blue: 'text-audio border-audio/30 bg-audio/10',
  purple: 'text-ai border-ai/30 bg-ai/10',
  yellow: 'text-warn border-warn/30 bg-warn/10',
  red: 'text-danger border-danger/30 bg-danger/10',
  muted: 'text-muted border-line2 bg-panel2'
};

const modePolicies: Record<AutomixerMode, { tone: Tone; title: string; apply: string; writes: string; rollback: string; command: string }> = {
  Analyze: {
    tone: 'blue',
    title: 'Read-only analysis',
    apply: 'Blocked',
    writes: 'Disabled',
    rollback: 'N/A',
    command: 'Analyze only'
  },
  Assist: {
    tone: 'purple',
    title: 'Suggest and review',
    apply: 'Review',
    writes: 'Disabled',
    rollback: 'Local',
    command: 'Suggest'
  },
  Supervised: {
    tone: 'yellow',
    title: 'Operator approval',
    apply: 'Backend gate',
    writes: 'Supervised',
    rollback: 'Backend',
    command: 'Approve'
  },
  Auto: {
    tone: 'red',
    title: 'Simulated automatic',
    apply: 'Local auto',
    writes: 'Disabled',
    rollback: 'Local',
    command: 'Sim auto'
  }
};

function policyForMode(mode: AutomixerMode) {
  return modePolicies[mode];
}

function statusToneForAgent(status: Agent['status']): Tone {
  if (status === 'BLOCKED') return 'red';
  if (status === 'WARNING' || status === 'UNAVAILABLE') return 'yellow';
  if (status === 'STANDBY') return 'blue';
  return 'green';
}

export function StatusBadge({ children, tone = 'green' }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5 text-[10px] font-semibold uppercase ${toneClass[tone]}`}>
      <i className="h-1.5 w-1.5 rounded-full bg-current" />
      {children}
    </span>
  );
}

export function IconButton({ label, children, onClick }: { label: string; children: ReactNode; onClick?: () => void }) {
  return (
    <button
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid h-8 w-8 place-items-center rounded-[5px] border border-line bg-panel2 text-[13px] font-black text-muted transition hover:border-audio hover:text-audio"
    >
      {children}
    </button>
  );
}

export function DirectorAvatar({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`relative grid place-items-center rounded-full border border-ai bg-[#07111a] shadow-[0_0_28px_rgba(163,113,247,.16)] ${compact ? 'h-8 w-8' : 'h-20 w-20'}`}>
      <div className={`relative rounded-[38%] border border-audio/70 bg-gradient-to-b from-[#172536] to-[#08111a] ${compact ? 'h-5 w-5' : 'h-12 w-12'}`}>
        <span className={`absolute rounded-full bg-audio shadow-[0_0_10px_rgba(0,212,255,.75)] ${compact ? 'left-1 top-2 h-1 w-1' : 'left-3 top-5 h-1.5 w-1.5'}`} />
        <span className={`absolute rounded-full bg-ai shadow-[0_0_10px_rgba(163,113,247,.75)] ${compact ? 'right-1 top-2 h-1 w-1' : 'right-3 top-5 h-1.5 w-1.5'}`} />
        <span className={`absolute left-1/2 -translate-x-1/2 rounded-full bg-online/80 ${compact ? 'bottom-1 h-0.5 w-2.5' : 'bottom-3 h-1 w-5'}`} />
      </div>
      {!compact && (
        <>
          <span className="absolute -left-1 top-9 h-3 w-1.5 rounded-full border border-audio/60 bg-[#07111a]" />
          <span className="absolute -right-1 top-9 h-3 w-1.5 rounded-full border border-audio/60 bg-[#07111a]" />
          <span className="absolute bottom-1 h-5 w-9 rounded-t-full border border-line bg-panel" />
        </>
      )}
    </div>
  );
}

export function Panel({ title, children, className = '', action }: { title?: string; children: ReactNode; className?: string; action?: ReactNode }) {
  return (
    <section className={`am-panel min-h-0 overflow-hidden p-3 ${className}`}>
      {title && (
        <header className="mb-2 flex items-center justify-between gap-3">
          <h3 className="am-label">{title}</h3>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function MetricCard({ metric }: { metric: Metric }) {
  const colorClass = metric.tone === 'yellow'
    ? 'text-warn'
    : metric.tone === 'blue'
      ? 'text-audio'
      : metric.tone === 'purple'
        ? 'text-ai'
        : metric.tone === 'red'
          ? 'text-danger'
          : 'text-text';
  const dotClass = metric.tone === 'yellow'
    ? 'bg-warn'
    : metric.tone === 'blue'
      ? 'bg-audio'
      : metric.tone === 'purple'
        ? 'bg-ai'
        : metric.tone === 'red'
          ? 'bg-danger'
          : 'bg-online';

  return (
    <Panel className="flex flex-col justify-between">
      <div className="am-label">{metric.label}</div>
      <div className={`mt-2 text-[25px] font-bold leading-none ${colorClass}`}>
        {metric.value}
      </div>
      <div className="mt-1 flex items-center gap-1 text-[10px] text-muted">
        <i className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
        {metric.sub}
      </div>
    </Panel>
  );
}

export function Sparkline({ values = [32, 68, 42, 80, 46, 72, 54, 88, 60, 36], tone = 'green', filled = false }: { values?: number[]; tone?: Tone; filled?: boolean }) {
  const color = tone === 'purple' ? '#a371f7' : tone === 'blue' ? '#00d4ff' : tone === 'yellow' ? '#f0a22e' : tone === 'red' ? '#f85149' : '#2dd36f';
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${100 - value}`).join(' ');
  const fillPath = `0,100 ${points} 100,100`;
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full overflow-visible">
      {filled && <polygon points={fillPath} fill={color} opacity="0.16" />}
      <polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function LevelMeter({ value, horizontal = true }: { value: number; horizontal?: boolean }) {
  const pct = Math.max(4, Math.min(100, ((value + 70) / 70) * 100));
  if (!horizontal) {
    return (
      <div className="flex h-full w-3 items-end overflow-hidden rounded-full border border-line bg-bg">
        <span className="w-full rounded-full bg-gradient-to-t from-online via-online to-danger" style={{ height: `${pct}%` }} />
      </div>
    );
  }
  return (
    <div className="h-2 overflow-hidden rounded-full border border-line bg-bg">
      <span className="block h-full rounded-full bg-gradient-to-r from-online via-warn to-danger" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Knob({ value, label }: { value: number; label: string }) {
  const angle = Math.max(20, Math.min(320, value * 3.2));
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-9 w-9 rounded-full border border-line2 bg-bg shadow-glow">
        <div className="absolute left-1/2 top-1/2 h-[15px] w-[2px] origin-bottom -translate-x-1/2 -translate-y-full rounded-full bg-audio" style={{ transform: `translate(-50%, -100%) rotate(${angle}deg)` }} />
        <div className="absolute inset-2 rounded-full bg-panel2" />
      </div>
      <div>
        <div className="text-[11px] font-semibold text-text">{label}</div>
        <div className="text-[10px] text-muted">{value}%</div>
      </div>
    </div>
  );
}

type TableFilter<T> = {
  label: string;
  predicate: (row: T) => boolean;
};

type TableColumn<T> = {
  key: string;
  label: string;
  width?: string;
  render?: (row: T) => ReactNode;
};

export function ActionCluster({
  actions = ['Inspect', 'Meter', 'Route', 'More'],
  onAction
}: {
  actions?: string[];
  onAction?: (action: string) => void;
}) {
  const glyphs: Record<string, string> = {
    Inspect: '○',
    Meter: '▥',
    Route: '⌁',
    More: '⋮',
    Apply: '✓',
    Reject: '×',
    Snapshot: '□',
    Detail: 'i'
  };
  const actionClass: Record<string, string> = {
    Apply: 'border-online/30 bg-online/10 text-online hover:border-online hover:text-white',
    Reject: 'border-danger/35 bg-danger/10 text-danger hover:border-danger hover:text-white',
    Meter: 'border-audio/30 bg-audio/10 text-audio hover:border-audio hover:text-white',
    Route: 'border-ai/30 bg-ai/10 text-ai hover:border-ai hover:text-white',
    Detail: 'border-audio/25 bg-panel2 text-audio hover:border-audio hover:text-white',
    Inspect: 'border-audio/25 bg-panel2 text-muted hover:border-audio hover:text-audio',
    More: 'border-line bg-panel2 text-muted hover:border-line2 hover:text-text'
  };

  return (
    <span className="flex justify-end gap-1">
      {actions.map(action => (
        <button
          aria-label={action}
          className={`grid h-6 w-6 place-items-center rounded-[4px] border text-[11px] font-bold transition ${actionClass[action] ?? 'border-line bg-panel2 text-muted hover:border-audio hover:text-audio'}`}
          key={action}
          title={action}
          type="button"
          onClick={event => {
            event.stopPropagation();
            onAction?.(action);
          }}
        >
          {glyphs[action] ?? action.slice(0, 1)}
        </button>
      ))}
    </span>
  );
}

export function DataTable<T>({
  columns,
  rows,
  className = '',
  searchable = false,
  searchPlaceholder = 'Search...',
  filters = [],
  getSearchText,
  rowKey,
  selectedRow,
  onSelect,
  emptyState
}: {
  columns: TableColumn<T>[];
  rows: T[];
  className?: string;
  searchable?: boolean;
  searchPlaceholder?: string;
  filters?: TableFilter<T>[];
  getSearchText?: (row: T) => string;
  rowKey?: (row: T, index: number) => string | number;
  selectedRow?: T;
  onSelect?: (row: T) => void;
  emptyState?: ReactNode;
}) {
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState(filters[0]?.label ?? 'All');
  const template = columns.map(column => column.width ?? 'minmax(0, 1fr)').join(' ');
  const selectedKey = selectedRow && rowKey ? String(rowKey(selectedRow, -1)) : undefined;
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return rows.filter(row => {
      const filter = filters.find(item => item.label === activeFilter);
      const passesFilter = filter ? filter.predicate(row) : true;
      if (!passesFilter) return false;
      if (!normalizedQuery) return true;
      const haystack = (getSearchText ? getSearchText(row) : Object.values(row as Record<string, unknown>).join(' ')).toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  }, [activeFilter, filters, getSearchText, query, rows]);

  return (
    <div className={`am-panel min-h-0 overflow-hidden ${className}`}>
      {(searchable || filters.length > 0) && (
        <div className="flex h-10 items-center justify-between gap-3 border-b border-line bg-[#101923] px-3">
          {searchable ? (
            <div className="flex min-w-[210px] items-center gap-2 rounded-[5px] border border-line bg-bg px-2 py-1">
              <span className="text-[10px] text-audio">⌕</span>
              <input
                value={query}
                onChange={event => setQuery(event.target.value)}
                placeholder={searchPlaceholder}
                className="w-full bg-transparent text-[10px] text-text outline-none placeholder:text-dim"
              />
            </div>
          ) : <span />}
          {filters.length > 0 && (
            <div className="flex shrink-0 gap-1">
              {filters.map(filter => (
                <button
                  className={`rounded-[4px] border px-2 py-1 text-[9px] font-semibold uppercase transition ${activeFilter === filter.label ? 'border-audio bg-audio/10 text-audio' : 'border-line bg-row text-muted hover:text-text'}`}
                  key={filter.label}
                  onClick={() => setActiveFilter(filter.label)}
                >
                  {filter.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="am-table-header" style={{ gridTemplateColumns: template }}>
        {columns.map(column => <span key={column.key}>{column.label}</span>)}
      </div>
      <div className="am-scrollbar max-h-full overflow-auto">
        {filteredRows.map((row, index) => {
          const key = rowKey ? String(rowKey(row, index)) : String(index);
          const selected = selectedKey === key;
          return (
          <div
            role={onSelect ? 'button' : undefined}
            tabIndex={onSelect ? 0 : undefined}
            className={`am-table-row w-full text-left ${onSelect ? 'cursor-pointer' : 'cursor-default'} ${selected ? 'bg-audio/10 text-text shadow-[inset_2px_0_0_#00d4ff]' : ''}`}
            style={{ gridTemplateColumns: template }}
            key={key}
            onClick={() => onSelect?.(row)}
            onKeyDown={event => {
              if (!onSelect) return;
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onSelect(row);
              }
            }}
          >
            {columns.map(column => <span className="truncate" key={column.key}>{column.render ? column.render(row) : String((row as Record<string, unknown>)[column.key] ?? '')}</span>)}
          </div>
          );
        })}
        {!filteredRows.length && (
          <div className="px-3 py-5 text-[10px] text-muted">
            {emptyState ?? 'No rows to display.'}
          </div>
        )}
      </div>
    </div>
  );
}

export function ChannelStrip({ channel, selected = false, onSelect }: { channel: Channel; selected?: boolean; onSelect?: (channel: Channel) => void }) {
  const fader = Math.max(10, Math.min(86, 74 + channel.fader));
  const waveTone: Tone = channel.group === 'VOCALS' ? 'purple' : channel.group === 'GUITARS' ? 'blue' : channel.group === 'BASS' ? 'yellow' : 'green';
  return (
    <button type="button" onClick={() => onSelect?.(channel)} className={`flex min-w-0 flex-col rounded-[6px] border bg-[#0e1721] p-1.5 text-left transition hover:border-audio ${selected ? 'border-ai shadow-[0_0_0_1px_rgba(163,113,247,0.65)]' : 'border-line'}`}>
      <div className="min-h-[34px]">
        <div className="truncate text-[10px] font-bold text-text">{channel.name}</div>
        <div className="truncate text-[9px] text-muted">{channel.type}</div>
      </div>
      <div className="text-[10px] font-semibold text-text">{channel.fader.toFixed(1)} dB</div>
      <div className="h-8"><Sparkline tone={waveTone} filled /></div>
      <div className="mb-1 flex justify-between text-[8px] font-semibold text-muted"><span>GATE</span><span>COMP</span></div>
      <div className="grid flex-1 grid-cols-[13px_1fr] gap-2">
        <LevelMeter value={channel.level} horizontal={false} />
        <div className="relative mx-auto h-full w-4 rounded-full border border-line bg-bg">
          <span className="absolute left-1/2 h-2.5 w-7 -translate-x-1/2 rounded-[2px] bg-[#c9d1d9]" style={{ bottom: `${fader}%` }} />
        </div>
      </div>
      <div className="mt-1 flex justify-center gap-1">
        <span className="grid h-5 w-5 place-items-center rounded border border-danger/40 bg-danger/10 text-[10px] text-danger">M</span>
        <span className="grid h-5 w-5 place-items-center rounded border border-warn/40 bg-warn/10 text-[10px] text-warn">S</span>
      </div>
      <footer className="mt-1 text-center text-[10px] text-muted">{String(channel.id).padStart(2, '0')}</footer>
    </button>
  );
}

export function AgentCard({
  agent,
  selected = false,
  onSelect,
  onAction
}: {
  agent: Agent;
  selected?: boolean;
  onSelect?: (agent: Agent) => void;
  onAction?: (agent: Agent, action: string) => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect?.(agent)}
      onKeyDown={event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect?.(agent);
        }
      }}
      className={`grid w-full cursor-pointer grid-cols-[36px_1.15fr_0.8fr_1.1fr_74px_1fr_64px_86px] items-center gap-3 border-b border-line px-3 py-2 text-left text-[11px] transition hover:bg-panel2 ${selected ? 'bg-audio/10 shadow-[inset_2px_0_0_#00d4ff]' : ''}`}
    >
      <div className={`grid h-8 w-8 place-items-center rounded-full border ${toneClass[agent.tone]}`}>{agent.name.slice(0, 1)}</div>
      <div>
        <div className="font-semibold text-text">{agent.name}</div>
        <div className="text-[9px] text-muted">v1.2.{agent.confidence % 10}</div>
      </div>
      <div className="text-muted">{agent.role}</div>
      <StatusBadge tone={statusToneForAgent(agent.status)}>{agent.status}</StatusBadge>
      <div className="truncate text-muted">{agent.task}</div>
      <div className="h-7"><Sparkline values={agent.activity} tone={agent.tone} /></div>
      <div className="text-right">
        <div className="font-bold text-online">{agent.confidence}%</div>
        <div className="text-[9px] text-muted">High</div>
      </div>
      <ActionCluster actions={['Inspect', 'Meter', 'More']} onAction={action => onAction?.(agent, action)} />
    </div>
  );
}

export function LogRow({ log }: { log: LogRecord }) {
  const tone: Tone = log.level === 'ERROR' ? 'red' : log.level === 'WARN' ? 'yellow' : log.level === 'SUCCESS' ? 'green' : 'blue';
  return (
    <div className="grid grid-cols-[58px_62px_90px_1fr] items-center gap-2 border-b border-line px-3 py-2 text-[11px]">
      <span className="font-mono text-muted">{log.time}</span>
      <StatusBadge tone={tone}>{log.level}</StatusBadge>
      <span className="text-muted">{log.source}</span>
      <span className="text-text">{log.message}</span>
    </div>
  );
}

export function NetworkGraph({ mode = 'system' }: { mode?: 'system' | 'agents' }) {
  const nodes = mode === 'agents'
    ? [
        [50, 48, 'Director', 'green'], [22, 34, 'Analyzer', 'blue'], [76, 30, 'Safety', 'green'], [35, 70, 'Critic', 'purple'], [68, 74, 'OSC', 'yellow'], [88, 52, 'Memory', 'muted']
      ]
    : [
        [15, 22, 'Director', 'blue'], [15, 48, 'Tablet', 'blue'], [15, 74, 'FOH', 'blue'], [47, 36, 'Core', 'purple'], [74, 36, 'WING', 'green'], [34, 78, 'DB', 'green'], [56, 78, 'Analytics', 'yellow'], [78, 78, 'Plugin', 'green']
      ];
  return (
    <div className="relative h-full min-h-[210px] overflow-hidden rounded-[8px] border border-line bg-[#0a1119]">
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <filter id="glow"><feGaussianBlur stdDeviation="1.4" result="coloredBlur" /><feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {nodes.slice(1).map((node, index) => (
          <line key={index} x1={nodes[0][0]} y1={nodes[0][1]} x2={node[0]} y2={node[1]} stroke={index % 2 ? '#00d4ff' : '#2dd36f'} strokeWidth="0.5" strokeDasharray={index % 2 ? '2 1.5' : '0'} opacity="0.65" filter="url(#glow)" />
        ))}
      </svg>
      {nodes.map(([x, y, label, tone]) => (
        <div key={String(label)} className={`absolute grid -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border text-[9px] font-bold ${toneClass[tone as Tone]} ${label === 'Core' || label === 'Director' ? 'h-16 w-16' : 'h-12 w-12'}`} style={{ left: `${x}%`, top: `${y}%` }}>
          {label}
        </div>
      ))}
    </div>
  );
}

export function Sidebar({
  items,
  active,
  onNavigate,
  runtime,
  dashboardMode = false
}: {
  items: NavItem[];
  active: RouteId;
  onNavigate: (id: RouteId) => void;
  runtime: DirectorRuntime;
  dashboardMode?: boolean;
}) {
  const dashboardItems: NavItem[] = [
    { id: 'dashboard', label: 'Dashboard', section: 'Overview', mark: 'OV' },
    { id: 'channels', label: 'Channel List', section: 'Channels', mark: 'CH' },
    { id: 'groups', label: 'Groups / Buses', section: 'Channels', mark: 'GB' },
    { id: 'agents', label: 'Agents Console', section: 'AI Agents', mark: 'AI' },
    { id: 'task-queue', label: 'Decision Log', section: 'AI Agents', mark: 'DL' },
    { id: 'training', label: 'Training Sessions', section: 'Learning', mark: 'TR' },
    { id: 'rules', label: 'Rules Library', section: 'Learning', mark: 'RL' },
    { id: 'system', label: 'Connections', section: 'System', mark: 'CN' },
    { id: 'settings', label: 'Settings', section: 'System', mark: 'ST' },
    { id: 'logs', label: 'Logs', section: 'System', mark: 'LG' }
  ];
  const visibleItems = dashboardMode ? dashboardItems : items;
  const sections = Array.from(new Set(visibleItems.map(item => item.section)));
  return (
    <aside className="row-span-3 flex min-h-0 flex-col border-r border-line bg-[#060a10]">
      <div className="flex h-[50px] items-center gap-2 border-b border-line px-3">
        <div className="relative grid h-7 w-7 place-items-center rounded-full border border-audio text-[10px] font-black text-audio">
          {dashboardMode ? (
            <span className="relative h-4 w-4 rounded-[5px] border border-muted">
              <i className="absolute left-1 top-1.5 h-1 w-1 rounded-full bg-audio" />
              <i className="absolute right-1 top-1.5 h-1 w-1 rounded-full bg-ai" />
            </span>
          ) : 'AM'}
        </div>
        <div>
          <div className="text-[11px] font-black uppercase tracking-[0.15em] text-text">Automixer</div>
          <div className="text-[8px] font-semibold uppercase tracking-[0.2em] text-muted">Tubeslave</div>
        </div>
      </div>
      <nav className="am-scrollbar min-h-0 flex-1 overflow-auto px-2 py-3">
        {sections.map(section => (
          <div className="mb-4" key={section}>
            <div className="mb-1 px-2 text-[9px] font-bold uppercase tracking-[0.14em] text-dim">{section}</div>
            <div className="space-y-1">
              {visibleItems.filter(item => item.section === section).map(item => (
                <button
                  className={`flex w-full items-center gap-2 rounded-[4px] px-2 py-1.5 text-left text-[10px] transition ${active === item.id ? 'bg-[#1a2532] text-text shadow-[inset_2px_0_0_#1687ff]' : 'text-muted hover:bg-panel hover:text-text'}`}
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                >
                  <span className={`grid h-4.5 w-4.5 place-items-center rounded border text-[8px] ${active === item.id ? 'border-audio text-audio' : 'border-line2 text-dim'}`}>{item.mark}</span>
                  <span className="truncate">{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </nav>
      <div className="mx-2 rounded-[4px] border border-line bg-panel p-2 text-[9px]">
        <div className="mb-1 font-bold uppercase tracking-[0.08em] text-muted">System Health</div>
        {dashboardMode ? (
          <div className="space-y-1 text-muted">
            <div className="font-bold text-online">✓ Display-only health</div>
            <div>CPU/RAM: <span className="text-text">visual baseline</span></div>
            <div>DISK: <span className="text-text">visual baseline</span></div>
          </div>
        ) : (
          <>
            <StatusBadge tone={runtime.readyTone}>{runtime.readyLabel}</StatusBadge>
            <div className="mt-2 grid grid-cols-2 gap-1 text-muted">
              <span>Source</span><span className="truncate text-text">{runtime.sourceLabel}</span>
              <span>Audio</span><span className="truncate text-text">{runtime.audioLabel}</span>
              <span>Gate</span><span className="truncate text-text">{runtime.manualGateLabel}</span>
            </div>
          </>
        )}
      </div>
      <div className="px-3 py-2 text-[9px] text-dim">Automixer v0.8.7<br />Build 2025.05.20</div>
    </aside>
  );
}

export function TopStatusBar({
  title,
  subtitle,
  mode,
  onModeChange,
  ui,
  actions,
  mockDataOnly = false
}: {
  title: string;
  subtitle: string;
  mode: AutomixerMode;
  onModeChange: (mode: AutomixerMode) => void;
  ui: UiState;
  actions: UiActions;
  mockDataOnly?: boolean;
}) {
  const statusLabel = mockDataOnly ? 'Live' : ui.backend.statusLabel;
  const connectionLabel = mockDataOnly ? 'WING Rack' : ui.backend.connectionLabel;
  const sourceLabel = mockDataOnly ? 'Mock' : ui.backend.sourceLabel;
  const safeValue = mockDataOnly
    ? 'On'
    : (ui.backend.canApplyToConsole ? 'Gate' : 'Read-only');
  const statusItems: { label: string; value: string; tone: Tone; interactive?: boolean }[] = [
    ...(!mockDataOnly ? [{ label: 'Session', value: 'Live Club', tone: 'blue' as Tone }] : []),
    { label: 'Status', value: statusLabel, tone: mockDataOnly ? 'green' : ui.backend.statusTone },
    { label: 'Data', value: mockDataOnly ? 'Mock fallback' : ui.backend.dataStateLabel, tone: mockDataOnly ? 'blue' : ui.backend.dataStateTone },
    { label: 'Mode', value: getModeDisplayLabel(mode), tone: policyForMode(mode).tone, interactive: true },
    { label: 'Safe', value: safeValue, tone: mockDataOnly ? 'green' : (ui.backend.canApplyToConsole ? 'yellow' : 'green') },
    { label: 'Connected To', value: connectionLabel, tone: mockDataOnly ? 'green' : (ui.backend.connectionState === 'connected' ? 'green' : 'blue') }
  ];

  return (
    <header className="col-span-2 flex items-center justify-between border-b border-line bg-[#060a10] px-3">
      <div className="flex min-w-0 items-center gap-5">
        <div>
          <div className="text-[13px] font-black uppercase tracking-[0.08em] text-text">{title}</div>
          <div className="text-[9px] text-muted">{subtitle}</div>
        </div>
        <div className="hidden h-7 w-px bg-line lg:block" />
        <div className="hidden items-center gap-4 lg:flex">
          {statusItems.map(item => (
            <div key={item.label}>
              <div className="mb-0.5 text-[7px] font-black uppercase tracking-[0.13em] text-dim">{item.label}</div>
              {item.interactive ? (
                <select
                  aria-label="Automation mode"
                  value={mode}
                  onChange={event => onModeChange(event.target.value as AutomixerMode)}
                  className={`rounded-[3px] border bg-[#0a1119] px-1.5 py-0.5 text-[9px] font-semibold uppercase outline-none ${toneClass[item.tone]}`}
                >
                  <option value="Analyze">Analysis Mode</option>
                  <option value="Assist">Assisted Mixing</option>
                  <option value="Supervised">Supervised AutoMix</option>
                  <option value="Auto">Full Autonomous Mixing</option>
                </select>
              ) : (
                <StatusBadge tone={item.tone}>{item.value}</StatusBadge>
              )}
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted">
        <IconButton label="Input monitor" onClick={() => actions.pushToast('Input monitor opened locally.', 'blue')}>~</IconButton>
        <IconButton label="Refresh backend" onClick={actions.refreshBackend}>↻</IconButton>
        <IconButton label="Run backend analysis" onClick={actions.runBackendAnalysis}>◇</IconButton>
        <div className="ml-2 text-right">
          <div className="font-bold text-text">Director</div>
          <div>{mockDataOnly ? 'Automixer' : sourceLabel}</div>
        </div>
        <DirectorAvatar compact />
      </div>
    </header>
  );
}

export function BottomTransportBar({ ui, actions, dashboardMode = false }: { ui: UiState; actions: UiActions; dashboardMode?: boolean }) {
  const policy = policyForMode(ui.mode);
  const writesLabel = ui.backend.connectionState === 'connected'
    ? (ui.backend.canApplyToConsole ? 'Gate' : 'Read-only')
    : policy.writes;

  if (dashboardMode) {
    return (
      <footer className="col-start-2 col-span-1 grid grid-cols-[100px_84px_178px_158px_124px_78px_76px_minmax(92px,1fr)] items-center border-t border-line bg-[#0a1017] text-[8px]">
        <div className="flex h-full items-center gap-2 border-r border-line px-3">
          <button className="flex h-9 items-center gap-2 rounded-[3px] bg-online px-4 text-[11px] font-black uppercase text-white" type="button">
            <span className="h-0 w-0 border-y-[5px] border-l-[8px] border-y-transparent border-l-white" />
            Play
          </button>
        </div>
        <div className="flex h-full items-center gap-2 border-r border-line px-3">
          <div className="font-mono text-[18px] font-bold text-text">00:34:27</div>
          <div className="hidden uppercase text-muted xl:block">Elapsed</div>
        </div>
        <div className="border-r border-line px-3">
          <div className="am-label">Markers</div>
          <div className="mt-1 grid grid-cols-[1fr_52px] gap-2 text-muted"><span>Soundcheck Start</span><span>00:00:00</span><span>Set 1 Start</span><span>00:25:00</span></div>
        </div>
        <div className="border-r border-line px-3">
          <div className="am-label">Record / Snapshot</div>
          <div className="mt-1 flex items-center gap-2"><button className="am-button px-2 text-danger" type="button">Rec</button><button className="am-button px-2" onClick={actions.createSnapshot} type="button">Snapshot</button><button className="am-button px-2" type="button">□</button></div>
        </div>
        <div className="border-r border-line px-3">
          <div className="am-label">Automation</div>
          <div className="mt-1 flex items-center gap-2"><span className="rounded-[3px] border border-warn/40 bg-warn/10 px-2 py-1 text-warn">Display</span><span className="font-bold text-text">No write</span></div>
        </div>
        <div className="border-r border-line px-3"><div className="whitespace-nowrap text-[9px] font-semibold uppercase tracking-[0.1em] text-muted">Safe Limit</div><div className="mt-1 font-bold uppercase text-text">Display</div></div>
        <div className="border-r border-line px-3"><div className="am-label">Step</div><div className="mt-1 font-bold uppercase text-text">Display</div></div>
        <div className="flex h-full items-center justify-between px-3"><div><div className="am-label">Rollback</div><div className="mt-1 text-muted">Display-only<br /><span className="text-text">no live action</span></div></div><button className="am-button px-2 text-lg leading-none" onClick={actions.rollbackLastAction}>↻</button></div>
      </footer>
    );
  }

  return (
    <footer className="col-span-2 grid grid-cols-[162px_1.2fr_1.25fr_1.2fr_.86fr_.86fr_1fr] items-center gap-0 border-t border-line bg-[#0b121b] text-[10px]">
      <div className="flex h-full items-center gap-3 border-r border-line px-3">
        <button className="flex items-center gap-1.5 whitespace-nowrap rounded-[5px] bg-online px-4 py-2 text-[11px] font-bold uppercase text-white">
          <span className="h-0 w-0 border-y-[4px] border-l-[7px] border-y-transparent border-l-white" />
          Play
        </button>
        <div><div className="text-[18px] font-semibold text-text">00:34:27</div><div className="uppercase text-muted">Elapsed</div></div>
      </div>
      <div className="border-r border-line px-3"><div className="am-label">Markers</div><div className="mt-1 flex items-center gap-4 text-text"><span>Soundcheck Start</span><span className="text-muted">00:00:00</span><button className="text-online" onClick={() => actions.pushToast('Marker added locally.', 'green')}>+ Add</button></div></div>
      <div className="border-r border-line px-3"><div className="am-label">Record / Snapshot</div><div className="mt-1 flex items-center gap-2"><button className="am-button text-danger" onClick={() => actions.pushToast('Record armed in mock UI only.', 'red')}>Rec</button><button className="am-button" onClick={actions.createSnapshot}>Snapshot</button><span className="font-mono text-[10px] text-muted">#{String(ui.snapshotCount).padStart(2, '0')}</span></div></div>
      <div className="border-r border-line px-3">
        <div className="am-label">Automation</div>
        <div className="mt-1 flex min-w-0 items-center gap-2">
          <StatusBadge tone={policy.tone}>{getModeDisplayLabel(ui.mode)}</StatusBadge>
          <span className="truncate text-text">Ride On</span>
          <span className="text-muted">Writes <b className={ui.backend.canApplyToConsole ? 'text-warn' : 'text-danger'}>{writesLabel}</b></span>
        </div>
      </div>
      <div className="border-r border-line px-3"><div className="am-label">Safe Limit</div><div className="mt-1 font-bold text-online">-1.0 dBTP</div></div>
      <div className="border-r border-line px-3"><div className="am-label">Step Size</div><div className="mt-1 text-text">0.5 dB</div></div>
      <div className="flex items-center justify-between px-3"><div><div className="am-label">Rollback</div><div className="mt-1 max-w-[120px] truncate text-text">{ui.lastAction}</div></div><button className="am-button px-2 text-lg leading-none" onClick={actions.rollbackLastAction}>↻</button></div>
    </footer>
  );
}

export function ToastLayer({ toast, onDismiss }: { toast: UiState['toast']; onDismiss: () => void }) {
  if (!toast) return null;
  return (
    <button
      type="button"
      onClick={onDismiss}
      className={`absolute right-[266px] top-[56px] z-30 max-w-[320px] rounded-[7px] border px-3 py-2 text-left text-[11px] shadow-panel ${toneClass[toast.tone]}`}
      style={{ background: '#151f2b' }}
    >
      <div className="mb-0.5 text-[9px] font-bold uppercase tracking-[0.12em]">Local UI Event</div>
      <div className="text-text">{toast.message}</div>
    </button>
  );
}

function CompactOperatorView({
  items,
  active,
  onNavigate,
  mode,
  onModeChange,
  ui,
  actions,
  runtime
}: {
  items: NavItem[];
  active: RouteId;
  onNavigate: (id: RouteId) => void;
  mode: AutomixerMode;
  onModeChange: (mode: AutomixerMode) => void;
  ui: UiState;
  actions: UiActions;
  runtime: DirectorRuntime;
}) {
  const compactRoutes = items.filter(item => ['dashboard', 'task-queue', 'channels', 'agents', 'settings'].includes(item.id));
  const compactDataLabel = ui.backend.dataState === 'connected' ? 'PL live' : ui.backend.dataStateLabel;
  const compactReadyLabel = runtime.readyLabel === 'Ready for live' ? 'Ready' : runtime.readyLabel;
  const compactAudioLabel = runtime.audioLabel === 'Audio capture running' ? 'Audio running' : runtime.audioLabel;
  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-bg font-ui text-text">
      <header className="border-b border-line bg-[#060a10] px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[13px] font-black uppercase tracking-[0.12em] text-text">Automixer</div>
            <div className="mt-1 text-[10px] text-muted">Compact Operator</div>
          </div>
          <StatusBadge tone={ui.backend.dataStateTone}>{compactDataLabel}</StatusBadge>
        </div>
        <select
          aria-label="Compact mode"
          value={mode}
          onChange={event => onModeChange(event.target.value as AutomixerMode)}
          className={`mt-3 w-full rounded-[5px] border bg-[#0a1119] px-2 py-2 text-[11px] font-semibold uppercase outline-none ${toneClass[policyForMode(mode).tone]}`}
        >
          <option value="Analyze">Analysis Mode</option>
          <option value="Assist">Assisted Mixing</option>
          <option value="Supervised">Supervised AutoMix</option>
          <option value="Auto">Full Autonomous Mixing</option>
        </select>
      </header>
      <main className="am-scrollbar flex-1 space-y-2 overflow-auto p-3">
        <Panel title="Status">
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <StatusBadge tone={runtime.readyTone}>{compactReadyLabel}</StatusBadge>
            <StatusBadge tone={runtime.audioTone}>{compactAudioLabel}</StatusBadge>
            <StatusBadge tone={runtime.proposalQueueTone}>{runtime.proposalQueueLabel}</StatusBadge>
            <StatusBadge tone={runtime.manualGateTone}>{runtime.manualGateLabel}</StatusBadge>
          </div>
          <p className="mt-3 text-[11px] leading-5 text-muted">{runtime.currentGoal}</p>
        </Panel>
        <Panel title="Selected Channel">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[20px] font-bold text-text">CH {String(ui.selectedChannelId).padStart(2, '0')}</div>
              <div className="text-[10px] text-muted">Open Channel List for full details</div>
            </div>
            <button className="am-button" onClick={() => onNavigate('channels')}>Open</button>
          </div>
        </Panel>
        <Panel title="Queue">
          <div className="space-y-2 text-[10px]">
            <div className="flex items-center justify-between border-b border-line pb-2"><span className="text-muted">Current route</span><b>{active}</b></div>
            <div className="flex items-center justify-between border-b border-line pb-2"><span className="text-muted">Writes</span><b className={ui.backend.canApplyToConsole ? 'text-warn' : 'text-danger'}>{ui.backend.canApplyToConsole ? 'Supervised gate' : 'Blocked'}</b></div>
            <div className="flex items-center justify-between border-b border-line pb-2"><span className="text-muted">Backend</span><b className="text-text">{compactDataLabel}</b></div>
          </div>
        </Panel>
        <Panel title="Safe Mode">
          <div className="grid grid-cols-2 gap-2">
            <button className="am-button" onClick={actions.refreshBackend}>Refresh</button>
            <button className="am-button" onClick={actions.rollbackLastAction}>Rollback UI</button>
          </div>
          <p className="mt-3 text-[10px] leading-5 text-muted">Compact view remains read-only unless the supervised backend gate explicitly allows an approved apply.</p>
        </Panel>
      </main>
      <nav className="grid grid-cols-5 border-t border-line bg-[#0b121b]">
        {compactRoutes.map(item => (
          <button
            className={`px-1 py-2 text-[9px] font-semibold ${active === item.id ? 'bg-audio/10 text-audio' : 'text-muted'}`}
            key={item.id}
            onClick={() => onNavigate(item.id)}
          >
            <span className="block text-[10px]">{item.mark}</span>
            <span className="block truncate">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

function ModePolicyTray({ mode }: { mode: AutomixerMode }) {
  const policy = policyForMode(mode);

  return (
    <Panel title="Mode Guard" className="shrink-0" action={<StatusBadge tone={policy.tone}>{getModeDisplayLabel(mode)}</StatusBadge>}>
      <div className="text-[12px] font-bold text-text">{policy.title}</div>
      <div className="mt-2 grid grid-cols-3 gap-1 text-[9px]">
        <div className="rounded-[5px] border border-line bg-row px-1.5 py-1">
          <div className="text-dim">Apply</div>
          <div className="truncate font-bold text-text">{policy.apply}</div>
        </div>
        <div className="rounded-[5px] border border-line bg-row px-1.5 py-1">
          <div className="text-dim">Writes</div>
          <div className="truncate font-bold text-danger">{policy.writes}</div>
        </div>
        <div className="rounded-[5px] border border-line bg-row px-1.5 py-1">
          <div className="text-dim">Rollback</div>
          <div className="truncate font-bold text-audio">{policy.rollback}</div>
        </div>
      </div>
    </Panel>
  );
}

function ActionFocusTray({ focus, onClear }: { focus: UiState['actionFocus']; onClear: () => void }) {
  if (!focus) return null;

  return (
    <Panel
      title="Action Focus"
      className="shrink-0"
      action={<button className="text-[10px] font-bold uppercase text-muted hover:text-text" onClick={onClear} type="button">Clear</button>}
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <StatusBadge tone={focus.tone}>{focus.action}</StatusBadge>
        <span className="font-mono text-[10px] text-muted">{focus.timestamp}</span>
      </div>
      <div className="truncate text-[13px] font-bold text-text">{focus.target}</div>
      <p className="mt-2 text-[11px] leading-5 text-muted">{focus.detail}</p>
      <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
        <div className="rounded-[5px] border border-line bg-row px-2 py-1.5">
          <div className="text-dim">Write Path</div>
          <div className="font-bold text-danger">Guarded</div>
        </div>
        <div className="rounded-[5px] border border-line bg-row px-2 py-1.5">
          <div className="text-dim">Scope</div>
          <div className="font-bold text-audio">Local UI</div>
        </div>
      </div>
    </Panel>
  );
}

export function AppShell({ items, active, onNavigate, view, mode, onModeChange, ui, actions, runtime }: { items: NavItem[]; active: RouteId; onNavigate: (id: RouteId) => void; view: ViewModel; mode: AutomixerMode; onModeChange: (mode: AutomixerMode) => void; ui: UiState; actions: UiActions; runtime: DirectorRuntime }) {
  const shellRuntime = view.runtime ?? runtime;
  const dashboardMode = Boolean(view.mockDataOnly);

  return (
    <>
      <div className="lg:hidden">
        <CompactOperatorView items={items} active={active} onNavigate={onNavigate} mode={mode} onModeChange={onModeChange} ui={ui} actions={actions} runtime={shellRuntime} />
      </div>
      <div className={`relative hidden h-screen w-screen overflow-hidden bg-bg font-ui text-text lg:grid ${dashboardMode ? 'min-w-[1280px] grid-cols-[128px_minmax(0,1fr)_256px] grid-rows-[48px_minmax(0,1fr)_78px]' : 'min-w-[1280px] grid-cols-[150px_minmax(0,1fr)_318px] grid-rows-[50px_minmax(0,1fr)_62px]'}`}>
        <Sidebar items={items} active={active} onNavigate={onNavigate} runtime={shellRuntime} dashboardMode={dashboardMode} />
        <TopStatusBar title={view.title} subtitle={view.subtitle} mode={mode} onModeChange={onModeChange} ui={ui} actions={actions} mockDataOnly={view.mockDataOnly} />
        <main className={`min-h-0 min-w-0 overflow-hidden bg-bg2 ${dashboardMode ? 'p-1.5' : 'p-2'}`}>{view.content}</main>
        <aside className={`flex min-h-0 flex-col gap-2 overflow-hidden border-l border-line bg-[#0a1017] ${dashboardMode ? 'row-start-2 row-span-2 col-start-3 p-1.5' : 'p-2'}`}>
          <div className="am-scrollbar min-h-0 flex-1 overflow-auto">{view.rightPanel}</div>
          {!dashboardMode && <ModePolicyTray mode={mode} />}
          {!dashboardMode && <ActionFocusTray focus={ui.actionFocus} onClear={actions.clearActionFocus} />}
        </aside>
        <BottomTransportBar ui={ui} actions={actions} dashboardMode={dashboardMode} />
        <ToastLayer toast={ui.toast} onDismiss={actions.dismissToast} />
      </div>
    </>
  );
}
