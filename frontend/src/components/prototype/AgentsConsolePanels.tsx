import {
  ActionCluster,
  AgentCard,
  DataTable,
  MetricCard,
  NetworkGraph,
  Panel,
  Sparkline,
  StatusBadge
} from './AutomixerComponents';
import { toneForPriority } from '../../utils/prototypeTone';
import type { Agent, Communication, Metric, Task, Tone, UiActions } from '../../types';

function focusAgentAction(actions: UiActions, agent: Agent, action: string) {
  actions.focusAction({
    action,
    target: agent.name,
    detail: `${action} panel for ${agent.role}. Agent control remains local and cannot send mixer commands.`,
    tone: agent.tone,
    timestamp: '10:24:15'
  });
}

function focusCommunicationAction(actions: UiActions, item: Communication, action: string) {
  actions.focusAction({
    action,
    target: `${item.from} to ${item.to}`,
    detail: item.topic,
    tone: item.tone,
    timestamp: '10:24:15'
  });
}

function communicationKey(item: Communication) {
  return `${item.time}-${item.from}-${item.to}-${item.topic}`;
}

type CommunicationRow = Communication & { key: string };

function communicationRows(communications: Communication[]): CommunicationRow[] {
  return communications.map((item, index) => ({ ...item, key: `${communicationKey(item)}-${index}` }));
}

function MiniBarRow({ label, value, meter, tone = 'green' }: { label: string; value: string; meter: number; tone?: Tone }) {
  return (
    <div className="grid grid-cols-[92px_1fr_52px] items-center gap-2 text-[10px]">
      <span className="truncate text-muted">{label}</span>
      <div className="h-2 overflow-hidden rounded-full border border-line bg-bg">
        <span className="block h-full rounded-full bg-gradient-to-r from-online via-warn to-danger" style={{ width: `${Math.max(4, Math.min(100, ((meter + 70) / 70) * 100))}%` }} />
      </div>
      <span className={`text-right font-mono ${tone === 'yellow' ? 'text-warn' : tone === 'purple' ? 'text-ai' : tone === 'blue' ? 'text-audio' : 'text-online'}`}>{value}</span>
    </div>
  );
}

function TechnicalChecklist({ rows }: { rows: { label: string; value: string; tone?: Tone }[] }) {
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

function statusToneForAgent(status: Agent['status']): Tone {
  if (status === 'BLOCKED') return 'red';
  if (status === 'WARNING' || status === 'UNAVAILABLE') return 'yellow';
  if (status === 'STANDBY') return 'blue';
  return 'green';
}

function AgentInspector({ agent }: { agent: Agent }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-full border border-audio bg-audio/10 text-sm font-black text-audio">{agent.name.slice(0, 1)}</div>
        <div className="min-w-0">
          <div className="truncate text-[13px] font-bold text-text">{agent.name}</div>
          <div className="text-[10px] text-muted">{agent.role}</div>
        </div>
      </div>
      <div className="grid grid-cols-[76px_1fr] gap-y-2 text-[10px]">
        <span className="text-muted">Status</span><StatusBadge tone={statusToneForAgent(agent.status)}>{agent.status}</StatusBadge>
        <span className="text-muted">Task</span><span className="truncate text-text">{agent.task}</span>
        <span className="text-muted">Confidence</span><span className="font-bold text-online">{agent.confidence}%</span>
        <span className="text-muted">Last</span><span className="truncate text-text">{agent.lastAction}</span>
      </div>
      <div className="h-14"><Sparkline values={agent.activity} tone={agent.tone} filled /></div>
    </div>
  );
}

export function AgentsConsoleRightPanel({
  agents,
  communications,
  tasks,
  selectedAgentId
}: {
  agents: Agent[];
  communications: Communication[];
  tasks: Task[];
  selectedAgentId: string;
}) {
  const selectedAgent = agents.find(item => item.id === selectedAgentId) ?? agents[0];

  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Selected Agent" className="h-[182px]"><AgentInspector agent={selectedAgent} /></Panel>
      <Panel title="Task Queue" className="flex-1">{tasks.slice(0, 5).map(task => <div className="mb-2 grid grid-cols-[52px_1fr_36px] text-[10px]" key={task.id}><StatusBadge tone={toneForPriority(task.priority)}>{task.priority}</StatusBadge><span className="truncate text-muted">{task.title}</span><span>{task.time}</span></div>)}</Panel>
      <Panel title="Agent Messages" className="flex-1">{communications.slice(0, 4).map((item, index) => <p className="mb-2 text-[10px] text-muted" key={`${item.time}-${item.from}-${item.to}-${index}`}><b className="text-text">{item.from}</b> to {item.to}<br />{item.topic}</p>)}</Panel>
    </div>
  );
}

export function AgentsConsoleWorkspace({
  agents,
  metrics,
  selectedAgentId,
  actions
}: {
  agents: Agent[];
  metrics: Metric[];
  selectedAgentId: string;
  actions: UiActions;
}) {
  return (
    <div className="grid h-full grid-rows-[76px_36px_1fr] gap-2">
      <div className="grid grid-cols-6 gap-2">{metrics.map(metric => <MetricCard metric={metric} key={metric.label} />)}</div>
      <div className="am-panel flex items-center justify-between px-3"><div className="flex gap-7 text-[10px] font-semibold uppercase text-muted"><span className="text-text">Agents</span><span>Communication</span><span>Task Queue</span><span>Memory</span><span>Performance</span><span>Settings</span></div><button className="am-button">Create Agent</button></div>
      <Panel className="p-0">{agents.map(agent => (
        <AgentCard
          agent={agent}
          selected={agent.id === selectedAgentId}
          onSelect={selected => actions.selectAgent(selected.id)}
          onAction={(selected, action) => focusAgentAction(actions, selected, action)}
          key={agent.id}
        />
      ))}</Panel>
    </div>
  );
}

export function CommunicationsWorkspace({
  agents,
  communications,
  actions,
  selectedCommunicationKey,
  source = 'mock'
}: {
  agents: Agent[];
  communications: Communication[];
  actions: UiActions;
  selectedCommunicationKey: string;
  source?: string;
}) {
  const approvedCount = communications.filter(item => item.status === 'Approved').length;
  const reviewCount = communications.filter(item => item.status === 'Review' || item.status === 'Pending').length;
  const sourceLabel = source.includes('_snapshot') ? 'Backend DTO' : source === 'mock' ? 'Mock' : 'Runtime-derived';
  const communicationMetrics: Metric[] = [
    { label: 'Messages', value: String(communications.length), sub: 'runtime stream', tone: 'blue' },
    { label: 'Approved', value: String(approvedCount), sub: 'success/info', tone: 'green' },
    { label: 'In Review', value: String(reviewCount), sub: 'pending/review', tone: 'yellow' },
    { label: 'Source', value: sourceLabel, sub: source, tone: sourceLabel === 'Backend DTO' ? 'green' : 'blue' }
  ];
  const rows = communicationRows(communications);
  const selectedCommunication = rows.find(item => item.key === selectedCommunicationKey || communicationKey(item) === selectedCommunicationKey) ?? rows[0];

  return (
    <div className="grid h-full grid-rows-[76px_1fr_108px] gap-2">
      <div className="grid grid-cols-4 gap-2">
        {communicationMetrics.map(metric => <MetricCard metric={metric} key={metric.label} />)}
      </div>
      <DataTable
        rows={rows}
        searchable
        searchPlaceholder="Search agent message..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Approved', predicate: row => row.status === 'Approved' },
          { label: 'Review', predicate: row => row.status === 'Review' || row.status === 'Pending' },
          { label: 'Blocked', predicate: row => row.status === 'Blocked' || row.status === 'Rejected' }
        ]}
        getSearchText={row => `${row.time} ${row.from} ${row.to} ${row.topic} ${row.status}`}
        selectedRow={selectedCommunication}
        onSelect={(row) => {
          actions.selectCommunication(row.key);
        }}
        rowKey={row => row.key}
        columns={[
          { key: 'time', label: 'Time', width: '60px' },
          { key: 'from', label: 'From', width: '1fr' },
          { key: 'to', label: 'To', width: '1fr' },
          { key: 'topic', label: 'Message', width: '1.7fr' },
          { key: 'status', label: 'Status', width: '0.75fr', render: row => <StatusBadge tone={row.tone}>{row.status}</StatusBadge> },
          { key: 'actions', label: 'Actions', width: '86px', render: row => <ActionCluster actions={['Inspect', 'Detail', 'More']} onAction={action => {
            actions.selectCommunication(row.key);
            focusCommunicationAction(actions, row, action);
          }} /> }
        ]}
      />
      <div className="grid grid-cols-3 gap-2">
        <Panel title="Traffic by Agent">{agents.slice(0, 4).map(agent => <MiniBarRow label={agent.name} value={`${agent.confidence}%`} meter={-8 - (100 - agent.confidence)} tone={agent.tone} key={agent.id} />)}</Panel>
        <Panel title="Message Filters"><TechnicalChecklist rows={[{ label: 'Write commands', value: 'Blocked', tone: 'red' }, { label: 'DTO source', value: source, tone: sourceLabel === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Director broadcast', value: 'Read-only', tone: 'blue' }]} /></Panel>
        <Panel title="Last Broadcast"><p className="text-[11px] leading-5 text-muted">{communications[0]?.topic ?? 'Director keeps the system in supervised prototype mode. Decisions are displayed locally and no console write is sent.'}</p></Panel>
      </div>
    </div>
  );
}

export function CommunicationsRightPanel({ agents, communications, selectedCommunicationKey }: { agents: Agent[]; communications: Communication[]; selectedCommunicationKey: string }) {
  const rows = communicationRows(communications);
  const selectedMessage = rows.find(item => item.key === selectedCommunicationKey || communicationKey(item) === selectedCommunicationKey) ?? rows[0];

  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Active Conversations">{agents.slice(0, 4).map(agent => <div className="mb-3 flex items-center gap-2 text-[11px]" key={agent.id}><div className="h-7 w-7 rounded-full border border-ai bg-ai/10" /><div><div>{agent.name}</div><div className="text-muted">{agent.lastAction}</div></div></div>)}</Panel>
      <Panel title="Agent Communications" className="flex-1"><NetworkGraph mode="agents" /></Panel>
      <Panel title="Selected Message">
        <div className="mb-2 flex items-center justify-between text-[10px]">
          <span className="font-mono text-muted">{selectedMessage?.time ?? 'now'}</span>
          <StatusBadge tone={selectedMessage?.tone ?? 'blue'}>{selectedMessage?.status ?? 'Info'}</StatusBadge>
        </div>
        <p className="text-[11px] leading-5 text-muted">{selectedMessage ? `${selectedMessage.from} to ${selectedMessage.to}: ${selectedMessage.topic}` : 'Supervised mode only. No write commands are sent from this prototype.'}</p>
        <button className="am-button mt-3">Open Broadcast</button>
      </Panel>
    </div>
  );
}
