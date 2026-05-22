import { ActionCluster, DataTable, LevelMeter, MetricCard, Panel, Sparkline, StatusBadge } from '../components/prototype/AutomixerComponents';
import { memories, memoryTypes } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { Metric, UiActions, UiState } from '../types';
import { toneFromState } from '../utils/prototypeTone';
import {
  openActionFocus,
  runtimeMemoryTypes,
  runtimeRules,
  runtimeTrainingSessions,
  tableView,
  TechnicalChecklist
} from './viewHelpers';

function sourceDisplay(source: string) {
  if (source === 'mock') return 'Mock';
  if (source.includes('_snapshot')) return 'Backend DTO';
  return 'Runtime-derived';
}

export function Memory(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const memoryRows = data.memories.length ? data.memories : memories;
  const typeRows = runtimeMemoryTypes(memoryRows).length ? runtimeMemoryTypes(memoryRows) : memoryTypes;
  const avgConfidence = Math.round(memoryRows.reduce((sum, row) => sum + row.confidence, 0) / Math.max(memoryRows.length, 1));
  const ruleLikeCount = memoryRows.filter(row => /rule|decision|safety/i.test(row.type)).length;
  const selectedMemory = memoryRows.find(row => row.name === ui.selectedMemoryName) ?? memoryRows[0] ?? memories[0];
  const memorySource = sourceDisplay(data.readModelSources.memory);
  const memoryMetrics: Metric[] = [
    { label: 'Memory Rows', value: String(memoryRows.length), sub: 'runtime-derived', tone: 'blue' },
    { label: 'Types', value: String(typeRows.length), sub: 'knowledge groups', tone: 'green' },
    { label: 'Rules / Decisions', value: String(ruleLikeCount), sub: 'review records', tone: 'purple' },
    { label: 'Confidence', value: `${avgConfidence}%`, sub: 'retrieval quality', tone: avgConfidence > 88 ? 'green' : 'yellow' }
  ];
  return tableView('5. Memory / Knowledge Base', 'Memory and knowledge base', (
    <div className="grid h-full grid-rows-[86px_1fr_154px] gap-2">
      <div className="grid grid-cols-4 gap-2">{memoryMetrics.map(metric => <MetricCard metric={metric} key={metric.label} />)}</div>
      <div className="grid grid-cols-[1fr_1.2fr_1fr] gap-2">
        <Panel title="Memory Types">{typeRows.map(type => <div className="mb-3 grid grid-cols-[1fr_74px_56px] items-center gap-2 text-[11px]" key={type.name}><span className="truncate">{type.name}</span><LevelMeter value={type.level} /><b className="text-right text-online">{type.freshness}</b></div>)}</Panel>
        <Panel title="Memory Timeline"><div className="flex h-full items-end gap-3">{memoryRows.slice(0, 7).map((memory, i) => <div className="flex h-full flex-1 flex-col justify-end gap-1" key={`${memory.name}-${i}`}><i className="block w-full rounded-t bg-ai" style={{ height: `${Math.max(12, memory.confidence * 0.55)}%` }} /><i className="block w-full rounded-t bg-audio" style={{ height: `${Math.max(8, memory.confidence * 0.35)}%` }} /><i className="block w-full rounded-t bg-online" style={{ height: `${Math.max(6, memory.confidence * 0.25)}%` }} /></div>)}</div></Panel>
        <Panel title="Knowledge Health"><TechnicalChecklist rows={[{ label: 'Runtime source', value: memorySource, tone: memorySource === 'Backend DTO' ? 'green' : 'blue' }, { label: 'DTO source', value: data.readModelSources.memory, tone: memorySource === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Write learning', value: 'Disabled', tone: 'red' }, { label: 'Rule conflicts', value: ruleLikeCount ? 'Review' : 'Clear', tone: ruleLikeCount ? 'yellow' : 'green' }]} /></Panel>
      </div>
      <DataTable
        rows={memoryRows}
        selectedRow={selectedMemory}
        onSelect={memory => actions.selectMemory(memory.name)}
        searchable
        searchPlaceholder="Search memory..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Agents', predicate: memory => memory.type.includes('Agent') },
          { label: 'Decisions', predicate: memory => memory.type.includes('Decision') },
          { label: 'Channels', predicate: memory => memory.type.includes('Profile') }
        ]}
        getSearchText={memory => `${memory.name} ${memory.type} ${memory.confidence}`}
        rowKey={memory => memory.name}
        columns={[
          { key: 'name', label: 'Recent Memories', width: '1.6fr' },
          { key: 'type', label: 'Type', width: '1fr' },
          { key: 'size', label: 'Size', width: '0.7fr' },
          { key: 'confidence', label: 'Confidence', width: '0.8fr', render: row => <StatusBadge tone={row.confidence > 95 ? 'green' : row.confidence > 90 ? 'blue' : 'yellow'}>{row.confidence}%</StatusBadge> },
          { key: 'updated', label: 'Updated', width: '0.65fr' },
          { key: 'actions', label: 'Actions', width: '86px', render: row => <ActionCluster actions={['Inspect', 'Detail', 'More']} onAction={action => {
            actions.selectMemory(row.name);
            openActionFocus(actions, action, row.name, `${row.type} memory, ${row.confidence}% confidence. Read-only inspection only.`, row.confidence > 95 ? 'green' : 'blue');
          }} /> }
        ]}
      />
    </div>
  ), (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Memory Inspector">
        <h3 className="text-[15px] font-bold text-text">{selectedMemory.name}</h3>
        <p className="mt-2 text-[11px] leading-5 text-muted">{selectedMemory.type} · {selectedMemory.confidence}% confidence. This prototype does not save new learning data to backend storage.</p>
      </Panel>
      <Panel title="Top Memories" className="flex-1">{memoryRows.slice(0, 6).map(memory => <div className="mb-3 grid grid-cols-[1fr_44px] gap-2 text-[11px]" key={memory.name}><span className="truncate text-muted">{memory.name}</span><b className="text-right text-online">{memory.confidence}%</b></div>)}</Panel>
      <Panel title="Sync Status"><TechnicalChecklist rows={[{ label: 'Database', value: memorySource, tone: memorySource === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Embeddings', value: 'Display only', tone: 'blue' }, { label: 'Live writes', value: 'Off', tone: 'red' }]} /></Panel>
    </div>
  ));
}

export function Training(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const sessionRows = runtimeTrainingSessions(data);
  const trainingSource = sourceDisplay(data.readModelSources.training);
  const selectedSession = sessionRows.find(row => row.name === ui.selectedTrainingName) ?? sessionRows[0];
  const runningCount = sessionRows.filter(row => row.status === 'Running').length;
  const reviewCount = sessionRows.filter(row => row.status === 'Review' || row.status === 'Queued').length;
  const examples = sessionRows.reduce((sum, row) => sum + Number(row.examples || 0), 0);
  const scoredRows = sessionRows
    .map(row => Number(String(row.score).replace('%', '')))
    .filter(value => Number.isFinite(value));
  const avgScore = Math.round(scoredRows.reduce((sum, value) => sum + value, 0) / Math.max(scoredRows.length, 1));
  return tableView('15. Training Sessions', 'Training datasets and model sessions', (
    <div className="grid h-full grid-rows-[76px_1fr_108px] gap-2">
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Datasets', value: String(sessionRows.length), sub: trainingSource, tone: trainingSource === 'Backend DTO' ? 'green' : 'blue' },
          { label: 'Examples', value: String(examples), sub: 'approved samples', tone: 'green' },
          { label: 'Running', value: String(runningCount), sub: 'training queue', tone: runningCount ? 'yellow' : 'green' },
          { label: 'Avg Score', value: `${avgScore}%`, sub: 'read-only metric', tone: avgScore > 88 ? 'green' : 'yellow' }
        ].map(metric => <MetricCard metric={metric as Metric} key={metric.label} />)}
      </div>
      <DataTable
        rows={sessionRows}
        selectedRow={selectedSession}
        onSelect={session => actions.selectTraining(session.name)}
        searchable
        searchPlaceholder="Search training session..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Complete', predicate: row => row.status === 'Complete' },
          { label: 'Running', predicate: row => row.status === 'Running' },
          { label: 'Review', predicate: row => row.status === 'Review' }
        ]}
        getSearchText={row => `${row.name} ${row.status} ${row.score}`}
        rowKey={row => row.name}
        columns={[
          { key: 'name', label: 'Session' },
          { key: 'status', label: 'Status', render: row => <StatusBadge tone={toneFromState(row.status)}>{row.status}</StatusBadge> },
          { key: 'examples', label: 'Examples' },
          { key: 'score', label: 'Score' },
          { key: 'updated', label: 'Updated' }
        ]}
      />
      <div className="grid grid-cols-3 gap-2">
        <Panel title="Training Sources"><TechnicalChecklist rows={[{ label: 'DTO source', value: data.readModelSources.training, tone: trainingSource === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Approved snapshots', value: 'Read-only', tone: 'blue' }, { label: 'Live recording', value: 'Off', tone: 'red' }]} /></Panel>
        <Panel title="Model Drift"><Sparkline tone="yellow" filled values={sessionRows.slice(0, 10).map(row => Number(String(row.score).replace('%', '')) || 50)} /></Panel>
        <Panel title="Next Import"><p className="text-[11px] leading-5 text-muted">Training import remains operator-approved and local-only. This prototype displays candidate training sets but does not write learning data.</p></Panel>
      </div>
    </div>
  ), <Panel title="Training Queue" className="h-full"><h3 className="text-[15px] font-bold text-text">{selectedSession?.name ?? 'No training session'}</h3><p className="mt-2 text-[11px] leading-5 text-muted">{selectedSession?.status ?? 'Idle'} · {selectedSession?.examples ?? 0} examples · {selectedSession?.score ?? 'N/A'} score.</p><div className="mt-4 h-20"><Sparkline tone="purple" filled values={sessionRows.slice(0, 10).map(row => Number(row.examples) || 20)} /></div><div className="mt-4"><TechnicalChecklist rows={[{ label: 'Auto learn', value: 'Off', tone: 'red' }, { label: 'Dataset writes', value: 'Disabled', tone: 'red' }, { label: 'Review queue', value: String(reviewCount), tone: reviewCount ? 'yellow' : 'green' }]} /></div></Panel>);
}

export function RulesLibrary(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const ruleRows = runtimeRules(data);
  const selectedRule = ruleRows.find(row => row.name === ui.selectedRuleName) ?? ruleRows[0];
  const rulesSource = sourceDisplay(data.readModelSources.rules);
  return tableView('16. Rules Library', 'Operational mix rules and safety policy library', (
    <div className="grid h-full grid-rows-[76px_1fr_108px] gap-2">
      <div className="grid grid-cols-5 gap-2">
        {[
          { label: 'Rules', value: String(ruleRows.length), sub: rulesSource, tone: rulesSource === 'Backend DTO' ? 'green' : 'blue' },
          { label: 'Active', value: String(ruleRows.filter(rule => rule.state === 'Active').length), sub: 'usable rules', tone: 'green' },
          { label: 'Locked', value: String(ruleRows.filter(rule => rule.state === 'Locked').length), sub: 'safety policy', tone: 'purple' },
          { label: 'Review', value: String(ruleRows.filter(rule => rule.state === 'Review').length), sub: 'needs operator', tone: 'yellow' },
          { label: 'Conflicts', value: String(ruleRows.filter(rule => rule.state === 'Review').length), sub: 'read-only watch', tone: 'yellow' }
        ].map(metric => <MetricCard metric={metric as Metric} key={metric.label} />)}
      </div>
      <DataTable
        rows={ruleRows}
        selectedRow={selectedRule}
        onSelect={rule => actions.selectRule(rule.name)}
        searchable
        searchPlaceholder="Search rules..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Safety', predicate: row => row.category === 'Safety' },
          { label: 'Active', predicate: row => row.state === 'Active' },
          { label: 'Locked', predicate: row => row.state === 'Locked' }
        ]}
        getSearchText={row => `${row.name} ${row.category} ${row.state}`}
        rowKey={row => row.name}
        columns={[
          { key: 'name', label: 'Rule' },
          { key: 'category', label: 'Category' },
          { key: 'confidence', label: 'Confidence' },
          { key: 'state', label: 'State', render: row => <StatusBadge tone={toneFromState(row.state)}>{row.state}</StatusBadge> }
        ]}
      />
      <div className="grid grid-cols-3 gap-2">
        <Panel title="Policy Layers"><TechnicalChecklist rows={[{ label: 'DTO source', value: data.readModelSources.rules, tone: rulesSource === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Safety limits', value: 'Locked', tone: 'purple' }, { label: 'Learning rules', value: 'Review', tone: 'yellow' }]} /></Panel>
        <Panel title="Rule Confidence"><Sparkline tone="green" filled values={ruleRows.slice(0, 10).map(rule => Number(rule.confidence.replace('%', '')) || 80)} /></Panel>
        <Panel title="Conflict Watch"><p className="text-[11px] leading-5 text-muted">Rules that imply a gain increase remain blocked until headroom and rollback proof exist.</p></Panel>
      </div>
    </div>
  ), <Panel title="Rule Inspector" className="h-full"><h3 className="text-base font-bold">{selectedRule?.name ?? 'No boost without headroom'}</h3><p className="mt-2 text-[11px] leading-5 text-muted">{selectedRule?.category ?? 'Safety'} · {selectedRule?.confidence ?? '100%'} confidence · {selectedRule?.state ?? 'Locked'} state. Locked/read-only policy rule.</p><div className="mt-5"><TechnicalChecklist rows={[{ label: 'Priority', value: 'Safety wins', tone: 'purple' }, { label: 'Write path', value: 'Disabled', tone: 'red' }, { label: 'Rollback', value: 'Required', tone: 'yellow' }]} /></div></Panel>);
}
