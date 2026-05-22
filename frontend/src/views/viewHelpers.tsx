import type { ReactNode } from 'react';
import { LevelMeter, StatusBadge } from '../components/prototype/AutomixerComponents';
import { buses, rules, trainingSessions } from '../mock/automixerData';
import type { LogRecord, MemoryRecord, RoutingSegment, Task, Tone, TopologyLink, TopologyNode, UiActions, ViewModel } from '../types';
import type { OperatorViewData } from '../services/useOperatorBackend';

export function MiniBarRow({ label, value, meter, tone = 'green' }: { label: string; value: string; meter: number; tone?: Tone }) {
  return (
    <div className="grid grid-cols-[92px_1fr_52px] items-center gap-2 text-[10px]">
      <span className="truncate text-muted">{label}</span>
      <LevelMeter value={meter} />
      <span className={`text-right font-mono ${tone === 'yellow' ? 'text-warn' : tone === 'purple' ? 'text-ai' : tone === 'blue' ? 'text-audio' : 'text-online'}`}>{value}</span>
    </div>
  );
}

export function TechnicalChecklist({ rows }: { rows: { label: string; value: string; tone?: Tone }[] }) {
  return (
    <div className="space-y-2">
      {rows.map(row => (
        <div className="flex items-center justify-between border-b border-line pb-2 text-[10px]" key={`${row.label}-${row.value}`}>
          <span className="text-muted">{row.label}</span>
          <StatusBadge tone={row.tone ?? 'green'}>{row.value}</StatusBadge>
        </div>
      ))}
    </div>
  );
}

export function statusToneForBus(status?: string): Tone {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('inactive')) return 'red';
  if (normalized.includes('muted')) return 'yellow';
  if (normalized.includes('active')) return 'green';
  return 'blue';
}

export function statusToneForDevice(status?: string): Tone {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('error') || normalized.includes('offline')) return 'red';
  if (normalized.includes('warn') || normalized.includes('standby')) return 'yellow';
  if (normalized.includes('online') || normalized.includes('running') || normalized.includes('connected')) return 'green';
  return 'blue';
}

export function statusToneForLog(level: LogRecord['level']): Tone {
  if (level === 'ERROR') return 'red';
  if (level === 'WARN') return 'yellow';
  if (level === 'SUCCESS') return 'green';
  return 'blue';
}

export function logKey(log: LogRecord, index = 0) {
  return `${log.time}-${log.source}-${index}`;
}

export const fallbackTopologyNodes: TopologyNode[] = [
  { id: 'director_control', label: 'Director Control', kind: 'client', status: 'online', detail: 'reference client surface' },
  { id: 'automixer_core', label: 'Automixer Core', kind: 'core', status: 'running', detail: 'mode=assist' },
  { id: 'mixer', label: 'WING Rack', kind: 'mixer', status: 'online', detail: 'reference mixer target' },
  { id: 'audio_capture', label: 'Audio Capture', kind: 'audio', status: 'running', detail: '48 channel(s)' },
  { id: 'decision_queue', label: 'Decision Queue', kind: 'agent', status: 'online', detail: 'proposal flow' }
];

export const fallbackTopologyLinks: TopologyLink[] = [
  { from: 'director_control', to: 'automixer_core', kind: 'websocket', status: 'online' },
  { from: 'automixer_core', to: 'mixer', kind: 'osc_control', status: 'online' },
  { from: 'audio_capture', to: 'automixer_core', kind: 'audio_analysis', status: 'online' },
  { from: 'decision_queue', to: 'automixer_core', kind: 'proposal_flow', status: 'online' }
];

export function runtimeTopologyNodes(nodes: TopologyNode[]) {
  return nodes.length ? nodes : fallbackTopologyNodes;
}

export function runtimeTopologyLinks(links: TopologyLink[]) {
  return links.length ? links : fallbackTopologyLinks;
}

export function fallbackRoutingSegments(): RoutingSegment[] {
  return buses.map((bus, index) => {
    const start = index * 2 + 1;
    return {
      id: `routing-${start}-${start + 1}`,
      start,
      end: start + 1,
      role: bus.role,
      tapPoint: bus.role === 'Return' ? 'FX return tap' : 'Post fader group',
      labelShort: bus.name,
      labelFull: `${bus.name} · ${bus.role}`,
      wingRoutingHint: `WING route group ${start}-${start + 1}`,
      stereo: true,
      required: bus.role !== 'Return',
      usedBy: [bus.name]
    };
  });
}

export function runtimeRoutingSegments(data: OperatorViewData) {
  return data.routingSegments.length ? data.routingSegments : fallbackRoutingSegments();
}

export function formatDb(value: number) {
  return `${value.toFixed(1)} dB`;
}

export function formatLufs(value: number) {
  return `${value.toFixed(1)} LUFS`;
}

export function runtimeMemoryTypes(records: MemoryRecord[]) {
  const grouped = new Map<string, MemoryRecord[]>();
  records.forEach(record => {
    const rows = grouped.get(record.type) ?? [];
    rows.push(record);
    grouped.set(record.type, rows);
  });

  return Array.from(grouped.entries()).slice(0, 6).map(([name, rows], index) => {
    const confidence = Math.round(rows.reduce((sum, row) => sum + row.confidence, 0) / Math.max(rows.length, 1));
    return {
      name,
      records: String(rows.length),
      freshness: `${confidence}%`,
      level: -8 - index * 4
    };
  });
}

export type TrainingSessionRow = (typeof trainingSessions)[number];
export type RuleRow = (typeof rules)[number];

export function runtimeTrainingSessions(data: OperatorViewData): TrainingSessionRow[] {
  if (data.trainingSessions.length) return data.trainingSessions;
  if (!data.memories.length && !data.tasks.length && !data.agents.length) return trainingSessions;
  const avgAgentConfidence = Math.round(data.agents.reduce((sum, agent) => sum + agent.confidence, 0) / Math.max(data.agents.length, 1));
  const reviewTasks = data.tasks.filter(task => task.status === 'Review' || task.status === 'Pending').length;
  return [
    {
      name: 'Agent roster calibration',
      status: data.agents.length ? 'Running' : 'Queued',
      examples: data.agents.length,
      score: `${avgAgentConfidence}%`,
      updated: 'runtime'
    },
    {
      name: 'Decision queue learning set',
      status: reviewTasks ? 'Review' : 'Complete',
      examples: data.tasks.length,
      score: `${Math.max(80, 96 - reviewTasks * 2)}%`,
      updated: 'runtime'
    },
    {
      name: 'Channel profile refresh',
      status: data.channels.length ? 'Complete' : 'Queued',
      examples: data.channels.length,
      score: data.channels.length ? '92%' : 'N/A',
      updated: 'inventory'
    },
    ...trainingSessions.slice(0, 4)
  ];
}

export function runtimeRules(data: OperatorViewData): RuleRow[] {
  if (data.rules.length) return data.rules;
  if (!data.memories.length && !data.tasks.length) return rules;
  const taskRules: RuleRow[] = data.tasks.slice(0, 4).map((task: Task) => ({
    name: `${task.title} requires director review`,
    category: task.priority === 'High' ? 'Safety' : 'Decision',
    confidence: task.priority === 'High' ? '96%' : '88%',
    state: task.status === 'Applied' ? 'Active' : 'Review'
  }));
  return [
    { name: 'Runtime memory writes stay disabled', category: 'Safety', confidence: '100%', state: 'Locked' },
    { name: `${data.directorRuntime.liveWritePolicy} apply policy`, category: 'Control', confidence: '100%', state: 'Locked' },
    ...taskRules,
    ...rules.slice(0, 6)
  ];
}

export function openActionFocus(actions: UiActions, action: string, target: string, detail: string, tone: Tone = 'blue') {
  actions.focusAction({
    action,
    target,
    detail,
    tone,
    timestamp: '10:24:15'
  });
}

export function SettingRows({ rows }: { rows: string[] }) {
  return (
    <div className="space-y-3">
      {rows.map(row => {
        const words = row.split(' ');
        const unitValue = ['kHz', 'samples', 'dBTP', 'dB', 'sec', 'only'].includes(words[words.length - 1]) && words.length > 2;
        const valueWords = unitValue ? words.slice(-2) : words.slice(-1);
        const labelWords = words.slice(0, words.length - valueWords.length);
        return (
          <div className="flex justify-between gap-3 border-b border-line pb-2 text-[11px]" key={row}>
            <span className="truncate text-muted">{labelWords.join(' ')}</span>
            <b className="shrink-0 text-text">{valueWords.join(' ')}</b>
          </div>
        );
      })}
    </div>
  );
}

export function tableView(title: string, subtitle: string, content: ReactNode, rightPanel?: ReactNode): ViewModel {
  return {
    title,
    subtitle,
    rightPanel,
    content: <div className="h-full min-h-0">{content}</div>
  };
}
